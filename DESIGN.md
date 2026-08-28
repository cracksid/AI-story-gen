# DESIGN.md — AI Story Generator

A character-level GPT trained from scratch in PyTorch on a public-domain
multi-genre story corpus, with genre conditioning via a control token.

This document answers the six assignment questions. Each answer states the
decision, the alternatives considered, and why this one was chosen given the
constraints. Numbers are from actual runs, not estimates. It is updated at the
end of every stage; sections for unbuilt stages say so plainly.

**Constraints that drove every decision:** must train on a CPU (no GPU
available) in under ~20 minutes, no paid APIs, no training frameworks, and the
whole system must be small enough that I can explain every file.

---

## 1. What is your data?

**Decision.** 1,354,274 characters (~250,000 words) of public-domain English
prose from Project Gutenberg, split across three genres:

| Genre       | Source                                          | Characters | Words   |
|-------------|-------------------------------------------------|-----------:|--------:|
| `fairytale` | Grimm's Fairy Tales (Gutenberg #2591)           |    530,039 | 101,111 |
| `fable`     | Aesop's Fables (Gutenberg #21)                  |    250,112 |  44,877 |
| `mystery`   | The Adventures of Sherlock Holmes (#1661)       |    574,123 | 104,506 |
| **Total**   |                                                 | **1,354,274** | **250,494** |

Downloaded and cleaned by `download_data.py`, which strips Gutenberg's licence
header and footer using the `*** START/END OF THE PROJECT GUTENBERG EBOOK ***`
markers. It raises rather than falling back if those markers are missing —
silently training on legalese is a failure that would only surface as strange
samples much later.

**Alternatives considered.**

- *A single book.* Simpler, and the standard choice for a teaching model
  (tinyshakespeare is one work, 1.1M characters). Rejected because question 5
  requires a conditioning signal, and a control token with only one possible
  value gives the model nothing to learn. Multi-genre makes conditioning
  demonstrable rather than merely described.
- *A large modern corpus (WikiText, OpenWebText).* Rejected on two grounds:
  CPU training time, and register. A model this small has no capacity to
  represent several styles at once, so heterogeneous text trains to a worse
  average than a smaller, stylistically consistent corpus.
- *Scraped contemporary fiction.* Rejected — copyright, and it could not be
  committed to the repository.

**Why these three genres specifically.** They must be separable in *surface*
statistics, because surface statistics is all a small character-level model can
represent. Fables are short with formulaic moral endings; fairy tales use a
heavy formulaic register; Sherlock Holmes is first-person with long sentences
and different punctuation density. Choosing two similar genres (e.g. fairy
tales and folk tales) would produce a control token the model correctly learns
to ignore.

**Known compromises, stated plainly.**

- The corpus is unbalanced: `fable` is roughly half the size of the other two.
  Not corrected, because truncating the larger books discards real training
  text to fix a problem not yet measured. Revisit if per-genre validation loss
  in Stage 8 shows `fable` doing markedly worse.
- 1.35M characters is very small in absolute terms. This bounds output quality
  hard, and no amount of architecture work compensates for it. The goal is a
  system that is fully understood, not one that writes well.
- The corpus is Victorian-era translated English. Generated text will sound
  archaic. This is a property of the data, not a bug in the model.

**Corpus problems found by inspection** (`inspect_data.py`), each carried into
the Stage 2 tokenizer decision:

- **131 distinct characters, with a long unusable tail.** 65 of them occur
  fewer than 100 times; 43 occur five times or fewer, together accounting for
  91 occurrences out of 1.35 million. This includes Greek script quoted by
  Aesop's translator (`ΑΔΜΣΦαβγδε…`) and stray accented and symbol characters
  (`£ ½ Æ œ ü`). At character level every distinct character gets its own
  learned embedding vector, so a character seen once receives one gradient
  update in the whole run and its vector stays effectively random — one third
  of the vocabulary would be untrained noise consuming model capacity.
- **Hard line wrapping.** Text is wrapped at 75 columns and double-spaced:
  33,574 of 55,213 lines are blank, and `\n` is the 10th most frequent
  character at 55,210 occurrences (4% of the corpus). Trained as-is, the model
  learns "emit a newline every ~70 characters" and generated stories come out
  shredded into ragged lines for reasons unrelated to storytelling.
- **Front matter.** Each book opens with a title page and a table of contents
  in Roman numerals, which is formatting to be imitated rather than prose.

## 2. How will you tokenize/process it?

**Decision.** Character-level tokenization over a cleaned 75-character
vocabulary. `tokenizer.py` owns both the cleaning and the mapping, because the
vocabulary is defined as "whatever survives cleaning" - splitting them across
two files would let them drift, and a tokenizer built from different text than
the model was trained on produces silent garbage rather than an error.

`CharTokenizer` builds `stoi` and `itos` dictionaries from `sorted(set(text))`.
Sorting matters: it makes ids deterministic, and a saved model's weights are
meaningless under a different id ordering.

**Cleaning, and what it was worth.**

| | Before | After |
|---|---:|---:|
| Characters | 1,354,274 | 1,311,454 |
| Vocabulary size | 131 | **75** |
| `
` as a share of text | 4.08% | **0.66%** |

Three fixes, each answering a problem measured in Stage 1:

1. **Typographic normalisation.** Curly quotes to straight, en/em dashes to
   hyphen, ligatures spelled out, then Unicode NFKD decomposition with the
   combining marks dropped to strip accents (`e` becomes `e`). NFKD is used
   rather than a hand-written lookup table of accented letters because it is
   one standard-library call and cannot miss a case.
2. **Unwrapping.** The initial implementation split paragraphs on any blank
   line, which was WRONG and produced fragments. Measurement showed these
   files are double-spaced: one blank line is a wrap inside a paragraph
   (17,292 occurrences), three blank lines is a real paragraph break (3,510),
   five or nine is a story break. The split is therefore on three-or-more
   newlines. After this, `
` means only "new paragraph" - real narrative
   structure - instead of "the typesetter reached column 75".
3. **Front matter removal.** Title pages, contents lists and publisher
   addresses. Detected by three properties at once: over 200 characters, more
   than 90% lowercase letters, and ending in terminal punctuation. Length
   alone is insufficient - Aesop's contents list is 7,857 characters.

An explicit **allow-list** of permitted characters is used rather than a
deny-list of banned ones, so a character nobody anticipated is dropped rather
than silently becoming a vocabulary entry. This removed the Greek script
quoted by Aesop's translator and Gutenberg's `_italic_` markup.

Verified lossless: `decode(encode(text)) == text` over the entire 1.3M-character
corpus.

**Alternatives considered.**

- *Word-level.* Rejected: vocabulary would be roughly 25,000 words for this
  corpus, most seen a handful of times, and any word outside it is
  unrepresentable at generation time.
- *Byte Pair Encoding.* The correct choice at real scale, and rejected here
  only because of scale. BPE merges frequent adjacent pairs to build a
  subword vocabulary (GPT-2 uses 50,257 tokens). Its advantage is sequence
  length: one BPE token is roughly four characters, so a fixed context window
  covers about four times as much text, which matters because attention cost
  grows with the square of sequence length. Its cost here is the embedding
  table, which is proportional to vocabulary size:

  | Vocabulary | Embedding parameters at `n_embd=192` |
  |---:|---:|
  | 75 (chosen) | 14,400 |
  | 5,000 (small BPE) | 960,000 |
  | 50,257 (GPT-2) | 9,649,344 |

  The planned model is a few hundred thousand parameters in total, so a GPT-2
  vocabulary's embedding table alone would be roughly twenty times the entire
  model - training a lookup table with a transformer attached. A BPE
  vocabulary learned from only 1.3M characters would also contain thousands of
  merges seen a handful of times each, reproducing the Stage 1 rare-symbol
  problem in a form that cannot be inspected by eye.

**The compromise, stated plainly.** Character level makes the model spend
capacity learning spelling and word boundaries from scratch, and caps how much
context fits in a fixed window. It is the right choice *given a 1.3M-character
corpus and a CPU*, not in general. With 100x the data, BPE wins on every axis.

**Known limitation.** `encode` raises `KeyError` on any character outside the
vocabulary. This is deliberate for training data, which is cleaned first, but
Stage 6 accepts free-text prompts from a user and will need to handle it.

## 3. What are the embeddings?

Not yet built — Stage 3.

## 4. What does the neural network learn?

Not yet built — Stages 4 and 5.

## 5. What conditioning signals will you use?

Not yet built — Stage 7. Planned: a prepended genre control token, with the
three labels above. Per-genre files are kept separate from Stage 1 onward
specifically so this label exists without re-deriving it later.

## 6. How will you evaluate your model?

Not yet built — Stage 8. Planned: held-out validation loss and perplexity,
plus an honest account of what those metrics do not capture.

---

## Environment

| Item | Value | Note |
|---|---|---|
| Python | 3.13.14 | The spec said 3.12; it is not installed on this machine and nothing here needs a 3.12-only feature. 3.13 has full PyTorch wheel coverage. |
| PyTorch | 2.13.0+cpu | Installed from the CPU wheel index. The default PyPI wheel bundles ~2.5 GB of CUDA libraries that are dead weight without an NVIDIA GPU. |
| Hardware | CPU only, Windows | No GPU. This is the binding constraint on model size and training time. |
| Dependencies | PyTorch 2.13.0, NumPy 2.5.2 | Only two. NumPy is not used directly by any code here; it is installed because PyTorch emits a warning on every import without it, and a warning that is always present is a warning nobody reads. The corpus download uses `urllib` from the standard library rather than adding `requests`. |

## Stage log

- **Stage 0 — setup.** venv on Python 3.13, CPU PyTorch verified by a matmul.
X `download_data.py` and `inspect_data.py`. Corpus
  downloaded, cleaned, and measured; three problems found and recorded above
  for Stage 2 to act on.
- **Stage 2 — tokenization.** `tokenizer.py`: corpus cleaning plus
  `CharTokenizer`. Vocabulary 131 → 75 characters, newline share 4.08% →
  0.66%, round trip verified lossless on the full corpus.
