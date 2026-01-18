import csv
import time
import math
from itertools import cycle

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

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
# Data: WikiText-2 + GPT-2 BPE
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

def build_wikitext2_tokens(tokenizer, split: str, max_examples: int | None = None):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    texts = ds["text"]
    if max_examples is not None:
        texts = texts[:max_examples]
    eos = tokenizer.eos_token_id
    all_ids = []
    for t in texts:
        t = t.strip()
        if not t:
            continue
        ids = tokenizer.encode(t, add_special_tokens=False)
        all_ids.extend(ids)
        all_ids.append(eos)  # separate documents
    return torch.tensor(all_ids, dtype=torch.long)

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
        x = x.to(device)
        y = y.to(device)
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
        vocab_size: int = 50257
        n_layer: int = 12
        n_head: int = 12
        n_embd: int = 768
        bias: bool = True

    @dataclass
    class TrainingConfig:
        batch_size: int = 8
        steps: int = 1000
        lr: float = 3e-4
        eval_interval: int = 100
        eval_batches: int = 50

    model_cfg = GPTConfig()
    train_cfg = TrainingConfig()
    
    max_train_examples = None
    max_val_examples = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup logging with experiment name in filename
    log_file_path = f"logs/training_log_{experiment_name}.csv"
    log_file = open(log_file_path, 'w', newline='')
    csv_writer = csv.writer(log_file)
    # Write header
    csv_writer.writerow(['step', 'train_loss', 'val_loss'])

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    # Update vocab_size to match tokenizer
    model_cfg.vocab_size = tokenizer.vocab_size  # 50257 for GPT-2

    train_tokens = build_wikitext2_tokens(tokenizer, split="train", max_examples=max_train_examples)
    val_tokens = build_wikitext2_tokens(tokenizer, split="validation", max_examples=max_val_examples)

    train_dataset = TokenBlockDataset(train_tokens, block_size=model_cfg.block_size)
    val_dataset = TokenBlockDataset(val_tokens, block_size=model_cfg.block_size)

    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, shuffle=False)

    model = GPT(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)

    model.train()
    train_iter = cycle(train_loader)
    train_start = time.perf_counter()

    pbar = tqdm(range(train_cfg.steps), desc="Training")
    for step in pbar:
        x, y = next(train_iter)
        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_train_loss = loss.item()
        pbar.set_postfix({'train_loss': f'{current_train_loss:.4f}'})

        if step % train_cfg.eval_interval == 0 and step > 0:
            val_loss = evaluate(model, val_loader, device, max_batches=train_cfg.eval_batches)
            pbar.set_postfix({'train_loss': f'{current_train_loss:.4f}', 'val_loss': f'{val_loss:.4f}'})
            # Log to CSV
            csv_writer.writerow([step, f'{current_train_loss:.6f}', f'{val_loss:.6f}'])
            log_file.flush()

    train_end = time.perf_counter()
    train_time = train_end - train_start

    # Final validation loss
    final_val_loss = evaluate(model, val_loader, device, max_batches=train_cfg.eval_batches)

    # Log final results
    csv_writer.writerow([train_cfg.steps, 'N/A', f'{final_val_loss:.6f}'])
    log_file.flush()
    log_file.close()

    return final_val_loss, train_time

if __name__ == "__main__":
    val_loss, train_time = main("baseline")
    print(f"Returned validation loss: {val_loss:.4f}, training time: {train_time:.2f}s")