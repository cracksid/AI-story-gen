"""Turn the cleaned corpus into batches of integer tensors for training.

Owns three things: encoding the whole corpus once, splitting it into training
and validation sets, and sampling random (input, target) pairs from them.

The model never touches raw text - it only ever sees tensors produced here.
"""

import sys

import torch

from tokenizer import GENRES, CharTokenizer, load_corpus

# Defaults. block_size is how many characters of context the model can see;
# batch_size is how many independent sequences are processed at once. Both are
# revisited in Stage 5 when we measure how long training actually takes.
BLOCK_SIZE = 128
BATCH_SIZE = 32
VAL_FRACTION = 0.1


def build_data(val_fraction: float = VAL_FRACTION
               ) -> tuple[CharTokenizer, dict[str, torch.Tensor],
                          dict[str, torch.Tensor]]:
    """Encode the corpus and split it into training and validation tensors.

    Returns one tensor PER GENRE rather than one combined tensor, because
    Stage 7 conditions on genre and therefore has to know which genre every
    training sequence came from.

    The split is done per genre. Holding out the last 10% of a joined corpus
    would put the entire validation set inside one genre, so validation loss
    would measure the wrong thing.
    """
    corpus = load_corpus()
    tokenizer = CharTokenizer("".join(corpus.values()))

    train, val = {}, {}
    for genre, text in corpus.items():
        # dtype=torch.long because these are ids used to index a table, not
        # measurements. Indexing requires an integer type.
        ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        split_at = int(len(ids) * (1 - val_fraction))
        train[genre] = ids[:split_at]
        val[genre] = ids[split_at:]

    return tokenizer, train, val


def get_batch(splits: dict[str, torch.Tensor],
              batch_size: int = BATCH_SIZE,
              block_size: int = BLOCK_SIZE
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a random batch of (input, target, genre) triples.

    Each sequence in the batch is drawn from a randomly chosen genre, rather
    than all of them coming from one genre per batch. Mixing them keeps the
    gradient from being dominated by a single genre on any given step.

    Returns x and y of shape (batch_size, block_size) and genre ids of shape
    (batch_size,). y is x shifted one position left, so y[i] is the character
    following x[i]. Because the model predicts at every position at once, one
    sequence of length block_size supplies block_size training examples.
    """
    # Only the genres actually present, so a caller can pass a subset (the
    # evaluation does this). Ids stay indices into GENRES either way, because
    # they select rows of the genre embedding table.
    names = [g for g in GENRES if g in splits]
    picks = torch.randint(len(names), (batch_size,))
    xs, ys = [], []
    for g in picks.tolist():
        data = splits[names[g]]
        # The upper bound leaves room for the target, one character further.
        start = torch.randint(len(data) - block_size - 1, (1,)).item()
        xs.append(data[start:start + block_size])
        ys.append(data[start + 1:start + block_size + 1])
    ids = torch.tensor([GENRES.index(names[g]) for g in picks.tolist()])
    return torch.stack(xs), torch.stack(ys), ids


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(1337)  # fixed seed so this demo is reproducible

    tokenizer, train_data, val_data = build_data()
    print("=== DATA ===")
    print(f"vocab size   {tokenizer.vocab_size}")
    for genre in GENRES:
        print(f"{genre:12} train {len(train_data[genre]):>9,}  "
              f"val {len(val_data[genre]):>8,}")

    print("\n=== ONE BATCH ===")
    x, y, g = get_batch(train_data, batch_size=4, block_size=16)
    print(f"x {tuple(x.shape)}   y {tuple(y.shape)}   genre {tuple(g.shape)}")
    print(f"genre ids  {g.tolist()}  -> {[GENRES[i] for i in g.tolist()]}")
    print(f"x[0] as text {tokenizer.decode(x[0].tolist())!r}")
    print(f"y[0] as text {tokenizer.decode(y[0].tolist())!r}   (shifted by one)")

    print("\n=== THE EXAMPLES HIDDEN INSIDE ONE SEQUENCE ===")
    for t in range(6):
        context = tokenizer.decode(x[0][:t + 1].tolist())
        target = tokenizer.decode([y[0][t].item()])
        print(f"  given {context!r:<12} predict {target!r}")
    print(f"  ... {x.shape[1]} examples in this one sequence")
