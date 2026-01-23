# LLM Speedrun

The point of this repo is to start from an implementation of LLM (GPT-2 124M) training on a single GPU and gradually apply optimizations that will make the training as efficient as possible. Once I am satisfied with the single GPU performance I plan to run it on multiple GPUs. I will do testing on the FineWeb dataset. Once I am satisfied I will try to see how my optimizations compare to the record of the [nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt).

I plan to track 3 metrics; the time it takes to train for 5000 steps, the validation loss and the throughput of the training.
