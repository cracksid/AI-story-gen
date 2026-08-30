"""The transformer language model, built from the pieces of Stage 4.

Reading order matches the order it was built in:
    BigramLM            - the baseline: next character from current one alone
    Head                - one head of causal self-attention
    MultiHeadAttention  - several heads in parallel
    FeedForward         - per-position computation after attention gathers
    Block               - attention + feed-forward, with residuals and norms
    GPT                 - embeddings, a stack of blocks, and an output layer

Nothing here is pretrained. Every parameter starts random and is learned by
the training loop in train.py.
"""

import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """Every architectural choice in one object.

    @dataclass is a decorator: it rewrites the class to generate __init__ and
    friends from these annotated fields, so a plain settings holder needs no
    boilerplate. Keeping the settings together means train.py and generate.py
    cannot disagree about the shape of the model.
    """
    vocab_size: int
    block_size: int = 64     # characters of context the model can see;
                             # 64 not 128 because attention cost grows with
                             # T squared - measured 302 vs 726 ms/step here
    n_embd: int = 128        # width of every vector inside the model
    n_head: int = 4          # attention heads per block
    n_layer: int = 4         # number of blocks stacked
    dropout: float = 0.1     # fraction of activations zeroed while training
    n_genres: int = 3        # conditioning signal: one learned vector each


class BigramLM(nn.Module):
    """Baseline: predicts the next character from the current one alone.

    Kept as the number every later addition has to beat. For a bigram the
    embedding table IS the model - row i holds the scores for whatever
    follows character i - hence a vocab_size x vocab_size table.
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.token_embedding(idx)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss


class Head(nn.Module):
    """One head of causal self-attention.

    Each position produces a query ("what am I looking for"), a key ("what do
    I contain") and a value ("what I pass on if selected"). Affinity between a
    query and a key is their dot product; softmax over affinities gives
    weights; the output is the weighted sum of values.

    Causal means a position may attend to earlier positions and itself, never
    to later ones. Without that, the model can read its own target - targets
    are the input shifted by one - and would learn to copy, not to predict.
    """

    def __init__(self, n_embd: int, head_size: int, block_size: int,
                 dropout: float) -> None:
        super().__init__()
        # bias=False: these are pure projections, an offset adds nothing.
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        # Lower-triangular ones, used as the causal mask. register_buffer
        # stores it in the module without making it a trainable parameter.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x is (B, T, n_embd). Returns (B, T, head_size)."""
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)

        # Every query against every key. The 1/sqrt(head_size) scaling keeps
        # the variance near 1; without it, large dot products drive softmax
        # towards one-hot and gradients stop flowing.
        weights = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (B, T, T)
        # Sliced to T so shorter sequences work, which generation produces.
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = self.dropout(F.softmax(weights, dim=-1))
        return weights @ self.value(x)                            # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """Several attention heads in parallel, concatenated.

    One head learns one kind of relationship. Four heads of size 32 cost the
    same as one head of size 128 but give four independent views, so this
    partitions capacity rather than adding any. The final projection lets the
    heads' outputs mix; without it they sit in separate slots forever.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        # ModuleList registers each head so its parameters are tracked. A
        # plain Python list would leave them invisible to the optimiser.
        self.heads = nn.ModuleList(
            Head(cfg.n_embd, head_size, cfg.block_size, cfg.dropout)
            for _ in range(cfg.n_head)
        )
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, n_embd)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Per-position computation.

    Attention gathers information but barely processes it - its output is a
    weighted average. This is where a position does something with what it
    gathered. Attention is communication; this is computation.

    The 4x widening in the middle is the convention from the original
    transformer paper: room to compute in before compressing back.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """One transformer block: attention, then feed-forward.

    Both sublayers are wrapped in a residual connection - x = x + f(x) rather
    than x = f(x). This is what makes a deep stack trainable: gradients get an
    unobstructed path back past every layer, and each block starts near the
    identity and only has to learn a modification.

    LayerNorm is applied BEFORE each sublayer. The 2017 paper applied it
    after; GPT-2 moved it before because it trains more stably.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ffwd = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    """Token and position embeddings, a stack of blocks, and an output layer."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # A second table indexed by POSITION, not by character. Attention is a
        # weighted sum, and sums ignore order, so without this the model
        # cannot tell "dog bit man" from "man bit dog".
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        # A THIRD table, indexed by genre: one learned vector per genre, added
        # to every position. This is the conditioning signal. A control token
        # placed in the text would be cheaper still, but with a 64-character
        # context it appears in only 12-29% of training windows and scrolls
        # out of the window during generation; an embedding cannot be lost.
        self.genre_embedding = nn.Embedding(cfg.n_genres, cfg.n_embd)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def forward(self, idx: torch.Tensor, genre: torch.Tensor,
                targets: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        tok = self.token_embedding(idx)                                    # (B,T,n_embd)
        pos = self.position_embedding(torch.arange(T, device=idx.device))  # (T,n_embd)
        # (B, n_embd) -> (B, 1, n_embd) so it broadcasts across all T
        # positions: every position is told which genre it belongs to.
        gen = self.genre_embedding(genre).unsqueeze(1)
        # Added, not concatenated: in n_embd dimensions there is room for the
        # network to use different directions for identity, position and genre.
        x = self.blocks(tok + pos + gen)
        logits = self.lm_head(self.ln_f(x))                                # (B,T,vocab)

        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()   # generation is not training: build no gradient graph
    def generate(self, idx: torch.Tensor, genre: torch.Tensor,
                 max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None) -> torch.Tensor:
        """Extend idx by max_new_tokens characters, sampling one at a time."""
        self.eval()    # disables dropout
        for _ in range(max_new_tokens):
            # The position embedding table has only block_size rows, so the
            # context must be cropped or the lookup goes out of range.
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond, genre)
            # Only the last position predicts the next character.
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                # Keep the k highest scores, set the rest to -inf so softmax
                # gives them zero probability.
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            # Sample rather than take the argmax, which loops on a few chars.
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(1337)

    from dataset import build_data, get_batch

    tokenizer, train_data, _ = build_data()
    cfg = GPTConfig(vocab_size=tokenizer.vocab_size)
    x, y, g = get_batch(train_data, batch_size=32, block_size=cfg.block_size)
    uniform = torch.log(torch.tensor(float(tokenizer.vocab_size))).item()

    print("=== BASELINE: BIGRAM ===")
    bigram = BigramLM(tokenizer.vocab_size)
    print(f"parameters {sum(p.numel() for p in bigram.parameters()):>10,}")
    print(f"loss       {bigram(x, y)[1].item():>10.4f}   "
          f"(-ln(1/{tokenizer.vocab_size}) = {uniform:.4f})")

    print("\n=== FULL MODEL ===")
    cfg = GPTConfig(vocab_size=tokenizer.vocab_size)
    model = GPT(cfg)
    print(f"parameters {sum(p.numel() for p in model.parameters()):>10,}")
    logits, loss = model(x, g, y)
    print(f"logits     {tuple(logits.shape)}")
    print(f"loss       {loss.item():>10.4f}   (untrained)")

    print("\n=== PARAMETERS BY PART ===")
    for name, module in [("token embedding", model.token_embedding),
                         ("position embedding", model.position_embedding),
                         ("genre embedding", model.genre_embedding),
                         ("blocks", model.blocks),
                         ("output layer", model.lm_head)]:
        print(f"  {name:<20} {sum(p.numel() for p in module.parameters()):>9,}")

    print("\n=== GENERATION FROM THE UNTRAINED MODEL ===")
    start = torch.zeros((1, 1), dtype=torch.long)
    genre0 = torch.zeros(1, dtype=torch.long)
    print(repr(tokenizer.decode(
        model.generate(start, genre0, 200)[0].tolist())))
