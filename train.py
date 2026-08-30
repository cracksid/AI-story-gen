"""The training loop, written by hand.

Five operations repeated: predict, score, clear old gradients, compute new
gradients, update parameters. No Trainer class, no Lightning - the loop is
visible because understanding it is the point of the exercise.

Run:  python train.py
Produces checkpoint.pt, which generate.py loads.
"""

import sys
import time
from pathlib import Path

import torch

from dataset import build_data, get_batch
from tokenizer import GENRES
from model import GPT, GPTConfig

# Training settings, separate from the architecture settings in GPTConfig.
MAX_STEPS = 3000
BATCH_SIZE = 32
LEARNING_RATE = 1e-3      # 1e-3 suits a model this small; 3e-4 is the usual
                          # figure for models a hundred times larger
EVAL_INTERVAL = 250       # how often to measure train and validation loss
EVAL_ITERS = 25           # batches averaged per measurement
CHECKPOINT = Path(__file__).parent / "checkpoint.pt"


@torch.no_grad()   # measuring, not learning: build no gradient graph
def estimate_loss(model: GPT, splits: dict[str, dict[str, torch.Tensor]],
                  block_size: int) -> dict[str, float]:
    """Average the loss over several batches of each split.

    A single batch is noisy enough to hide the trend, so this averages
    EVAL_ITERS of them. model.eval() disables dropout: leaving it on would
    measure a randomly damaged model and report validation loss as worse than
    it is. model.train() puts it back afterwards.
    """
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(EVAL_ITERS)
        for i in range(EVAL_ITERS):
            x, y, g = get_batch(data, BATCH_SIZE, block_size)
            losses[i] = model(x, g, y)[1].item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(1337)

    tokenizer, train_data, val_data = build_data()
    splits = {"train": train_data, "val": val_data}

    cfg = GPTConfig(vocab_size=tokenizer.vocab_size)
    model = GPT(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    # AdamW keeps a running estimate of each gradient's magnitude and scales
    # that parameter's step by it, so one badly scaled layer cannot force a
    # tiny learning rate on the whole model. Weight decay pulls parameters
    # gently toward zero, discouraging reliance on a few very large weights.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"parameters     {n_params:,}")
    print(f"context        {cfg.block_size} characters")
    print(f"steps          {MAX_STEPS} at batch size {BATCH_SIZE}")
    print(f"learning rate  {LEARNING_RATE}")
    print(f"\n{'step':>6}  {'train':>7}  {'val':>7}  {'gap':>6}  {'elapsed':>8}")

    start = time.time()
    for step in range(MAX_STEPS + 1):
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, splits, cfg.block_size)
            gap = losses["val"] - losses["train"]
            mins = (time.time() - start) / 60
            print(f"{step:>6}  {losses['train']:>7.4f}  {losses['val']:>7.4f}  "
                  f"{gap:>+6.3f}  {mins:>7.1f}m")

        x, y, g = get_batch(train_data, BATCH_SIZE, cfg.block_size)

        # The five lines this whole file exists for.
        logits, loss = model(x, g, y)  # forward: predict and score
        optimizer.zero_grad()          # clear gradients from the previous step;
                                       # backward ADDS to them, it does not replace
        loss.backward()                # compute a gradient for every parameter
        optimizer.step()               # nudge every parameter down its gradient

    # Save the weights AND the config, so generation rebuilds the identical
    # architecture. A state_dict loaded into a differently shaped model either
    # errors or, worse, silently mismatches.
    torch.save({"config": cfg, "state_dict": model.state_dict(),
                "chars": tokenizer.chars}, CHECKPOINT)
    print(f"\nsaved {CHECKPOINT.name} "
          f"({CHECKPOINT.stat().st_size / 1e6:.1f} MB, {n_params:,} parameters)")

    print("\n=== SAMPLE FROM THE TRAINED MODEL ===")
    for i, genre in enumerate(GENRES):
        start_ids = torch.zeros((1, 1), dtype=torch.long)
        gid = torch.tensor([i], dtype=torch.long)
        out = model.generate(start_ids, gid, 300, temperature=0.8)[0].tolist()
        print(f"\n--- {genre} ---")
        print(tokenizer.decode(out))


if __name__ == "__main__":
    main()
