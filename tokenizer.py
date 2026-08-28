"""Clean the raw corpus and map characters to integers.

A neural network consumes integers, not text. This file owns both halves of
that bridge: the text normalisation that decides WHICH characters exist, and
the CharTokenizer that maps them to integer ids and back.

Cleaning lives here rather than in its own file because the vocabulary is
defined as "whatever survives cleaning" - separating them would let the two
drift apart, and a tokenizer built from a different text than the model was
trained on produces silent garbage.
"""

import re
import sys
import unicodedata
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
GENRES = ["fairytale", "fable", "mystery"]

# Characters with an exact English equivalent that differs only in glyph.
# Merging them loses no meaning and removes rows from the embedding table.
REPLACEMENTS = {
    "‘": "'", "’": "'",   # curly single quotes
    "“": '"', "”": '"',   # curly double quotes
    "–": "-", "—": "-",   # en dash, em dash
    "æ": "ae", "Æ": "AE",  # ae ligature
    "œ": "oe", "Œ": "OE",  # oe ligature
    "½": " 1/2",
    "£": " pounds ",
}

# Everything the model is allowed to see. Anything outside this set is deleted.
# An explicit allow-list rather than a deny-list: a character we never thought
# of gets dropped, instead of silently becoming a vocabulary entry.
ALLOWED = set(
    "\n abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".,;:!?'\"-()"
)


def is_prose(paragraph: str) -> bool:
    """True for a real narrative paragraph, False for front matter.

    Used only to find where the book's actual text begins.
    """
    letters = [c for c in paragraph if c.isalpha()]
    if len(paragraph) < 200 or not letters:
        return False
    lowercase_ratio = sum(1 for c in letters if c.islower()) / len(letters)
    return lowercase_ratio > 0.90 and paragraph.endswith((".", "!", "?", '"'))


def clean_text(text: str) -> str:
    """Normalise one book: fix glyphs, strip accents, unwrap lines, drop noise."""
    # 1. Map typographic variants to their plain equivalents.
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)

    # 2. Strip accents. NFKD splits 'e' into 'e' plus a separate combining
    #    accent character; category 'Mn' means "mark, nonspacing" - the accent.
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")

    # 3. Undo hard wrapping, so that \n means only "new paragraph" - real
    #    structure worth learning - rather than "the typesetter hit column 75".
    #    Measured on these three books, the text is DOUBLE-SPACED: one blank
    #    line is a wrap inside a paragraph (17,292 of them), three blank lines
    #    is a real paragraph break (3,510), five or nine is a story break.
    #    So the split is on three-or-more newlines, not on any blank line.
    paragraphs = re.split(r"\n(?:[ \t]*\n){2,}", text)
    paragraphs = [" ".join(p.split()) for p in paragraphs]

    # 4. Drop front matter: title pages, tables of contents, publisher
    #    addresses. Measured, these differ from prose on three axes at once -
    #    they are Title Case or ALL CAPS, and they do not end in a sentence.
    #    Length alone is not enough: Aesop's contents list is 7,857 characters.
    paragraphs = [p for p in paragraphs if p]
    first_prose = next(
        (i for i, p in enumerate(paragraphs) if is_prose(p)), 0
    )
    paragraphs = paragraphs[first_prose:]

    text = "\n\n".join(p for p in paragraphs if p)

    # 5. Delete anything still outside the allow-list.
    return "".join(c for c in text if c in ALLOWED)


def load_corpus() -> dict[str, str]:
    """Return {genre: cleaned text} for every genre."""
    return {g: clean_text((DATA_DIR / f"{g}.txt").read_text(encoding="utf-8"))
            for g in GENRES}


class CharTokenizer:
    """Maps single characters to integer ids and back.

    The vocabulary is built from the text itself, sorted, so the same corpus
    always produces the same ids. That matters because a saved model's weights
    are meaningless under a different id ordering.
    """

    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        # Two dicts pointing at each other: character -> id, and id -> character.
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        """Text -> list of integer ids."""
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        """List of integer ids -> text."""
        return "".join(self.itos[i] for i in ids)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    corpus = load_corpus()
    joined = "\n\n".join(corpus.values())
    tok = CharTokenizer(joined)

    print("=== AFTER CLEANING ===")
    for genre, text in corpus.items():
        print(f"{genre:10} {len(text):>9,} chars")
    print(f"{'TOTAL':10} {len(joined):>9,} chars")

    print(f"\n=== VOCABULARY === {tok.vocab_size} characters")
    print(repr("".join(tok.chars)))

    print("\n=== ROUND TRIP ===")
    sample = "Once upon a time, in a far kingdom!"
    ids = tok.encode(sample)
    print(f"text   {sample!r}")
    print(f"ids    {ids}")
    print(f"back   {tok.decode(ids)!r}")
    print(f"lossless: {tok.decode(ids) == sample}")

    print("\n=== SAMPLE: first 400 chars of each genre, cleaned ===")
    for genre, text in corpus.items():
        print(f"\n--- {genre} ---\n{text[:400]}")
