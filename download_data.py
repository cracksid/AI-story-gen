"""Download the training corpus from Project Gutenberg and strip its boilerplate.

Run once. Produces one plain-text file per genre in data/, which every later
stage reads. Kept separate from the model code because it touches the network:
you should be able to delete data/ and rebuild it without re-running anything else.
"""

import urllib.request
from pathlib import Path

# Genre label -> (Gutenberg ebook id, human-readable title).
# The label is what Stage 7's control token will be built from, so it is
# chosen to be short and lowercase.
BOOKS = {
    "fairytale": (2591, "Grimm's Fairy Tales"),
    "fable": (21, "Aesop's Fables"),
    "mystery": (1661, "The Adventures of Sherlock Holmes"),
}

# Gutenberg wraps every book in a licence header and footer. These lines mark
# where the real text starts and ends.
START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"

DATA_DIR = Path(__file__).parent / "data"


def download(book_id: int) -> str:
    """Fetch one ebook as UTF-8 text."""
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    # Gutenberg blocks requests with no User-Agent, so we set one.
    request = urllib.request.Request(url, headers={"User-Agent": "story-gpt/0.1"})
    with urllib.request.urlopen(request) as response:
        return response.read().decode("utf-8")


def strip_boilerplate(text: str) -> str:
    """Return only the text between Gutenberg's start and end markers."""
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    if start == -1 or end == -1:
        # Fail loudly. Silently returning the raw text would train the model
        # on legalese and we would not notice until the samples looked odd.
        raise ValueError("Gutenberg markers not found - the file format changed")
    # Skip past the marker line itself, not just the marker text.
    start = text.index("\n", start) + 1
    return text[start:end].strip()


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for genre, (book_id, title) in BOOKS.items():
        out_path = DATA_DIR / f"{genre}.txt"
        if out_path.exists():
            print(f"{genre:10} already downloaded, skipping")
            continue
        print(f"{genre:10} downloading {title} (id {book_id})...")
        cleaned = strip_boilerplate(download(book_id))
        out_path.write_text(cleaned, encoding="utf-8")
        print(f"{genre:10} {len(cleaned):,} characters -> {out_path.name}")


if __name__ == "__main__":
    main()
