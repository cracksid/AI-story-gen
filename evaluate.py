"""Measure the trained model against baselines, per genre and overall.

A loss of 1.48 means nothing on its own. This reports it beside two baselines
that bracket it - a model with no knowledge at all, and the best possible
model that looks only at the previous character - so the number can be read.

Run:  python evaluate.py
"""

import sys
from collections import Counter

import torch

from dataset import build_data, get_batch
from generate import load
from tokenizer import GENRES

EVAL_BATCHES = 60
BATCH_SIZE = 32


@torch.no_grad()
def model_loss(model, data: dict[str, torch.Tensor], told: str | None = None) -> float:
    """Average loss over EVAL_BATCHES batches drawn from data.

    told=None uses each sequence's true genre. Passing a genre name forces
    that conditioning vector instead, which is how the confusion matrix
    below checks whether the signal is used at all.
    """
    torch.manual_seed(0)   # same batches for every configuration measured
    losses = []
    for _ in range(EVAL_BATCHES):
        x, y, g = get_batch(data, BATCH_SIZE, model.cfg.block_size)
        if told is not None:
            g = torch.full_like(g, GENRES.index(told))
        losses.append(model(x, g, y)[1].item())
    return sum(losses) / len(losses)


def bigram_loss(train: dict[str, torch.Tensor], val: dict[str, torch.Tensor],
                vocab_size: int) -> float:
    """Loss of the BEST POSSIBLE bigram model, computed by counting.

    Not trained by gradient descent: counting every character pair in the
    training text and normalising gives the optimal bigram directly. Add-one
    smoothing keeps an unseen pair from costing infinite loss.

    This is the bar a model with actual context has to clear.
    """
    counts = torch.ones(vocab_size, vocab_size)   # add-one smoothing
    for data in train.values():
        pairs = torch.stack([data[:-1], data[1:]])
        for a, b in zip(pairs[0].tolist(), pairs[1].tolist()):
            counts[a, b] += 1
    probs = counts / counts.sum(dim=1, keepdim=True)

    total, n = 0.0, 0
    for data in val.values():
        p = probs[data[:-1], data[1:]]
        total += -torch.log(p).sum().item()
        n += len(p)
    return total / n


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    model, tokenizer = load()
    _, train, val = build_data()
    V = tokenizer.vocab_size

    def show(name: str, loss: float) -> None:
        print(f"  {name:<34} {loss:>7.4f}   {torch.exp(torch.tensor(loss)):>7.2f}")

    print("=== BASELINES AND MODEL ===")
    print(f"  {'':<34} {'loss':>7}   {'ppl':>7}")
    # A model with no preference at all: every character equally likely.
    show(f"uniform over {V} characters", torch.log(torch.tensor(float(V))).item())
    show("best possible bigram (counted)", bigram_loss(train, val, V))
    show("this model (validation)", model_loss(model, val))
    show("this model (training)", model_loss(model, train))

    print("\n=== PER GENRE (validation) ===")
    print(f"  {'':<34} {'loss':>7}   {'ppl':>7}")
    for genre in GENRES:
        show(genre, model_loss(model, {genre: val[genre]}))

    print("\n=== IS THE CONDITIONING USED? ===")
    print("  rows: real held-out text. columns: the genre the model was told.")
    print(f"  {'real text':<12}" + "".join(f"{g:>11}" for g in GENRES))
    correct = 0
    for real in GENRES:
        row = [model_loss(model, {real: val[real]}, told=t) for t in GENRES]
        good = GENRES[row.index(min(row))] == real
        correct += good
        print(f"  {real:<12}" + "".join(f"{v:>11.4f}" for v in row)
              + ("   correct" if good else "   WRONG"))
    print(f"  {correct}/{len(GENRES)} genres identified by lowest loss")


if __name__ == "__main__":
    main()
