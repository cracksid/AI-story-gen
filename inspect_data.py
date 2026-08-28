"""Report the corpus statistics we need before designing the tokenizer.

Read-only. Answers: how much text is there, what characters occur in it, and
how rare are the rare ones - which is what decides whether the character
vocabulary needs cleaning in Stage 2.
"""

import sys
from collections import Counter
from pathlib import Path

# The Windows console defaults to the legacy cp1252 encoding, which cannot
# represent every character in the corpus. Without this, printing the
# vocabulary raises UnicodeEncodeError.
sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent / "data"


def main() -> None:
    texts = {p.stem: p.read_text(encoding="utf-8") for p in sorted(DATA_DIR.glob("*.txt"))}

    print("=== SIZE ===")
    total = 0
    for genre, text in texts.items():
        words = len(text.split())
        print(f"{genre:10} {len(text):>9,} chars  {words:>8,} words")
        total += len(text)
    print(f"{'TOTAL':10} {total:>9,} chars")

    # Counter counts how many times each character appears across all genres.
    counts = Counter()
    for text in texts.values():
        counts.update(text)

    print(f"\n=== VOCABULARY === {len(counts)} distinct characters")
    # sorted() on the set of characters puts them in Unicode order, which
    # groups whitespace, punctuation, digits, then letters.
    print(repr("".join(sorted(counts))))

    print("\n=== RARE CHARACTERS (fewer than 100 occurrences) ===")
    rare = [(char, n) for char, n in counts.most_common() if n < 100]
    for char, n in rare:
        print(f"  {repr(char):>10}  {n:>5}")
    print(f"  {len(rare)} of {len(counts)} characters are this rare")

    print("\n=== SAMPLE: first 300 characters of each genre ===")
    for genre, text in texts.items():
        print(f"\n--- {genre} ---")
        print(text[:300])


if __name__ == "__main__":
    main()
