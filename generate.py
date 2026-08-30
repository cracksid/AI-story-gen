"""Generate text from a trained checkpoint.

Sampling is a separate choice from the model. The model supplies a probability
distribution over the next character; temperature and top-k decide how that
distribution is turned into an actual character, and they change the output
far more than most people expect.

Run:  python generate.py --prompt "Once upon a time" --temperature 0.8 --top-k 40
      python generate.py --sweep        (compare temperatures side by side)
"""

import argparse
import sys
from pathlib import Path

import torch

from model import GPT
from tokenizer import GENRES, CharTokenizer, clean_text, load_corpus

CHECKPOINT = Path(__file__).parent / "checkpoint.pt"


def load() -> tuple[GPT, CharTokenizer]:
    """Rebuild the model and tokenizer from the saved checkpoint."""
    # weights_only=False because the checkpoint holds a GPTConfig object, not
    # only tensors. Safe here: we wrote this file ourselves.
    saved = torch.load(CHECKPOINT, weights_only=False)
    model = GPT(saved["config"])
    model.load_state_dict(saved["state_dict"])
    model.eval()

    tokenizer = CharTokenizer("".join(load_corpus().values()))
    # A checkpoint trained on a different vocabulary would silently produce
    # nonsense, so check rather than trust.
    if tokenizer.chars != saved["chars"]:
        raise ValueError("checkpoint vocabulary does not match the corpus")
    return model, tokenizer


def sample(model: GPT, tokenizer: CharTokenizer, genre: str, prompt: str,
           tokens: int, temperature: float, top_k: int | None) -> str:
    """Generate text continuing from prompt."""
    # Prompts come from a human, so they may contain characters the model has
    # never seen. Running them through the same cleaning the corpus went
    # through keeps encode from raising KeyError.
    prompt = clean_text(prompt) if prompt else "\n"
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    # The genre id selects which learned conditioning vector is added to
    # every position for the whole of this generation.
    gid = torch.tensor([GENRES.index(genre)], dtype=torch.long)
    out = model.generate(idx, gid, tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Generate text from the model")
    parser.add_argument("--genre", default="fairytale", choices=GENRES,
                        help="conditioning signal")
    parser.add_argument("--prompt", default="", help="text to continue from")
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="<1 sharpens the distribution, >1 flattens it")
    parser.add_argument("--top-k", type=int, default=None,
                        help="sample only from the k most likely characters")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--sweep", action="store_true",
                        help="compare several temperatures on the same prompt")
    parser.add_argument("--compare", action="store_true",
                        help="same prompt and seed under every genre")
    args = parser.parse_args()

    model, tokenizer = load()

    if args.sweep:
        for temp in [0.5, 0.8, 1.0, 1.5]:
            # Same seed each time, so differences come from temperature alone.
            torch.manual_seed(args.seed)
            print(f"\n=== temperature {temp} ===")
            print(sample(model, tokenizer, args.genre, args.prompt, 300,
                         temp, args.top_k))
        return

    if args.compare:
        for genre in GENRES:
            # Same seed and prompt, so any difference comes from the
            # conditioning vector alone.
            torch.manual_seed(args.seed)
            print(f"\n=== {genre} ===")
            print(sample(model, tokenizer, genre, args.prompt, 300,
                         args.temperature, args.top_k))
        return

    torch.manual_seed(args.seed)
    print(sample(model, tokenizer, args.genre, args.prompt, args.tokens,
                 args.temperature, args.top_k))


if __name__ == "__main__":
    main()
