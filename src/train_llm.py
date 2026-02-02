import os
import csv
import time
import math
import random
import numpy as np
from itertools import cycle
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader

from tqdm import tqdm
from dataclasses import dataclass
from datasets import load_dataset
from transformers import AutoTokenizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
# ------------------------- 
# Model
# ------------------------- 
class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # minimal: manual attention + causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()
        head_dim = C // self.n_head
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size
        pos = torch.arange(0, T, device=idx.device, dtype=torch.long)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss

# ------------------------- 
# Data: FineWeb + GPT-2 BPE
# ------------------------- 
class TokenBlockDataset(Dataset):
    """
    Takes a 1D LongTensor of token ids and returns contiguous blocks:
    x = tokens[i : i+block_size]
    y = tokens[i+1 : i+block_size+1]
    """
    def __init__(self, tokens_1d: torch.LongTensor, block_size: int):
        assert tokens_1d.dim() == 1
        self.tokens = tokens_1d
        self.block_size = block_size
        # number of full (block_size+1) windows, stepped by block_size
        self.n_blocks = (len(self.tokens) - 1) // block_size

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, i):
        start = i * self.block_size
        end = start + self.block_size + 1
        chunk = self.tokens[start:end]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

class FineWebIterableDataset(IterableDataset):
    """
    Stream FineWeb and emit token blocks without storing the corpus in memory.
    """
    def __init__(
        self,
        tokenizer,
        block_size: int,
        split: str,
        subset: str,
        text_field: str,
        shuffle_buffer: int,
        seed: int,
        dataset_name: str = "HuggingFaceFW/fineweb",
        max_tokens: int | None = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.split = split
        self.subset = subset
        self.text_field = text_field
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.dataset_name = dataset_name
        self.max_tokens = max_tokens

    def __iter__(self):
        ds = load_dataset(
            self.dataset_name,
            self.subset,
            split=self.split,
            streaming=True,
        )
        if self.shuffle_buffer and self.shuffle_buffer > 0:
            ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)

        token_buffer: list[int] = []
        emitted_tokens = 0
        eos = self.tokenizer.eos_token_id

        for row in ds:
            text = row.get(self.text_field)
            if not text:
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue

            token_buffer.extend(ids)
            token_buffer.append(eos)

            while len(token_buffer) > self.block_size:
                x = token_buffer[:self.block_size]
                y = token_buffer[1 : self.block_size + 1]
                emitted_tokens += self.block_size
                yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
                del token_buffer[: self.block_size]

                if self.max_tokens is not None and emitted_tokens >= self.max_tokens:
                    return

def build_fineweb_val_tokens(
    tokenizer,
    subset: str,
    text_field: str,
    max_tokens: int,
    dataset_name: str = "HuggingFaceFW/fineweb",
    split: str = "train",
):
    """
    Stream a slice of FineWeb and materialize enough tokens for validation.
    """
    ds = load_dataset(
        dataset_name,
        subset,
        split=split,
        streaming=True,
    )
    tokens: list[int] = []
    eos = tokenizer.eos_token_id

    for row in ds:
        text = row.get(text_field)
        if not text:
            continue
        ids = self_tokenize(tokenizer, text)
        tokens.extend(ids)
        tokens.append(eos)
        if len(tokens) >= max_tokens:
            break

    return torch.tensor(tokens[:max_tokens], dtype=torch.long)


def self_tokenize(tokenizer, text: str):
    """Helper to allow build_fineweb_val_tokens to stay TorchScript friendly if needed."""
    return tokenizer.encode(text, add_special_tokens=False)

# ------------------------- 
# Validation function
# ------------------------- 
@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=None):
    """Calculate average validation loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    for i, (x, y) in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _, loss = model(x, y)
        total_loss += loss.item()
        num_batches += 1
    model.train()
    if num_batches == 0:
        print("Warning: No validation batches processed!")
        return float('inf')
    return total_loss / num_batches

# ------------------------- 
# Training loop
# ------------------------- 
def main(experiment_name: str = "baseline"):
    @dataclass
    class GPTConfig:
        block_size: int = 256
        vocab_size: int = 50304  # larger vocab size, multiple of 128
        n_layer: int = 12
        n_head: int = 12
        n_embd: int = 768
        bias: bool = True

    @dataclass
    class TrainingConfig:
        batch_size: int = 8
        steps: int = 10000
        lr: float = 3e-4
        eval_interval: int = 100
        eval_batches: int = 50
        precision: str = "bf16"  # choices: fp32 | fp16 | bf16

    @dataclass
    class DataConfig:
        fineweb_subset: str = "sample-10BT"
        fineweb_text_field: str = "text"
        fineweb_split: str = "train"
        val_max_tokens: int = 500_000  # ~2k blocks at block_size=256
        shuffle_buffer: int = 10_000
        seed: int = 1337
        dataset_name: str = "HuggingFaceFW/fineweb"

    model_cfg = GPTConfig()
    train_cfg = TrainingConfig()
    data_cfg = DataConfig()

    # Set random seed for reproducibility
    set_seed(data_cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    # precision setup
    if device_type == "cuda" and train_cfg.precision == "fp16":
        autocast_dtype = torch.float16
    elif train_cfg.precision == "bf16":
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = None

    scaler = torch.amp.GradScaler(enabled=(device_type == "cuda" and train_cfg.precision == "fp16"))

    # Setup logging with experiment name in filename
    os.makedirs("../logs", exist_ok=True)
    log_file_path = f"../logs/training_log_{experiment_name}.csv"
    log_file = open(log_file_path, 'w', newline='')
    csv_writer = csv.writer(log_file)
    # Write header
    csv_writer.writerow(['step', 'train_loss', 'val_loss', 'tokens_seen', 'toks_per_s_avg', 'toks_per_s_win'])

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    # Update vocab_size to match tokenizer
    #model_cfg.vocab_size = tokenizer.vocab_size  # 50257 for GPT-2

    train_dataset = FineWebIterableDataset(
        tokenizer=tokenizer,
        block_size=model_cfg.block_size,
        split=data_cfg.fineweb_split,
        subset=data_cfg.fineweb_subset,
        text_field=data_cfg.fineweb_text_field,
        shuffle_buffer=data_cfg.shuffle_buffer,
        seed=data_cfg.seed,
        dataset_name=data_cfg.dataset_name,
    )
    val_tokens = build_fineweb_val_tokens(
        tokenizer=tokenizer,
        subset=data_cfg.fineweb_subset,
        text_field=data_cfg.fineweb_text_field,
        max_tokens=data_cfg.val_max_tokens,
        dataset_name=data_cfg.dataset_name,
        split=data_cfg.fineweb_split,
    )
    val_dataset = TokenBlockDataset(val_tokens, block_size=model_cfg.block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,  # streaming shuffle handled inside dataset
        num_workers=0,
        pin_memory=(device_type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        pin_memory=(device_type == "cuda"),
    )

    model = GPT(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)

    model.train()
    train_iter = cycle(train_loader)
    train_start = time.perf_counter()
    tokens_seen = 0
    last_log_time = train_start
    last_log_tokens = 0

    pbar = tqdm(range(train_cfg.steps), desc="Training")
    for step in pbar:
        x, y = next(train_iter)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        autocast_ctx = torch.autocast(device_type=device_type, dtype=autocast_dtype) if autocast_dtype else nullcontext()
        with autocast_ctx:
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        tokens_this_batch = x.numel()
        tokens_seen += tokens_this_batch

        now_time = time.perf_counter()
        interval_elapsed = now_time - last_log_time
        interval_tokens = tokens_seen - last_log_tokens
        throughput_window = interval_tokens / interval_elapsed if interval_elapsed > 0 else 0.0
        throughput_avg = tokens_seen / (now_time - train_start + 1e-9)
        last_log_time = now_time
        last_log_tokens = tokens_seen

        current_train_loss = loss.item()
        pbar.set_postfix({'train_loss': f'{current_train_loss:.4f}', 'toks/s': f'{throughput_window:.0f}'})

        if step % train_cfg.eval_interval == 0 and step > 0:
            eval_ctx = torch.autocast(device_type=device_type, dtype=autocast_dtype) if autocast_dtype else nullcontext()
            with eval_ctx:
                val_loss = evaluate(model, val_loader, device, max_batches=train_cfg.eval_batches)
            pbar.set_postfix({'train_loss': f'{current_train_loss:.4f}', 'val_loss': f'{val_loss:.4f}'})
            # Log to CSV
            os.makedirs("../logs", exist_ok=True)
            csv_writer.writerow([
                step,
                f'{current_train_loss:.6f}',
                f'{val_loss:.6f}',
                tokens_seen,
                f'{throughput_avg:.2f}',
                f'{throughput_window:.2f}',
            ])
            log_file.flush()

    train_end = time.perf_counter()
    train_time = train_end - train_start

    # Final validation loss
    eval_ctx = torch.autocast(device_type=device_type, dtype=autocast_dtype) if autocast_dtype else nullcontext()
    with eval_ctx:
        final_val_loss = evaluate(model, val_loader, device, max_batches=train_cfg.eval_batches)

    end_time = time.perf_counter()
    final_throughput_avg = tokens_seen / (end_time - train_start + 1e-9)
    final_interval_elapsed = end_time - last_log_time
    final_interval_tokens = tokens_seen - last_log_tokens
    final_throughput_window = final_interval_tokens / final_interval_elapsed if final_interval_elapsed > 0 else 0.0

    # Log final results
    csv_writer.writerow([
        train_cfg.steps,
        'N/A',
        f'{final_val_loss:.6f}',
        tokens_seen,
        f'{final_throughput_avg:.2f}',
        f'{final_throughput_window:.2f}',
    ])
    log_file.flush()
    log_file.close()

    return final_val_loss, train_time, final_throughput_avg

if __name__ == "__main__":
    experiment_name = "mixed_precision_bf16"
    val_loss, train_time, avg_throughput = main(experiment_name)
    print(f"Returned validation loss: {val_loss:.4f}, training time: {train_time:.2f}s, average throughput: {avg_throughput:.2f} tokens/sec")
