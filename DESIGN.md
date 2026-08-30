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

**Decision.** Two learned embedding tables, added together: a token embedding
of shape `vocab_size x n_embd` (75 x n_embd) and a positional embedding of
shape `block_size x n_embd`. Both are randomly initialised and trained by the
same gradient descent as the rest of the model. Nothing is pretrained; nothing
is downloaded.

**What an embedding table is.** A matrix with one row per vocabulary entry.
`nn.Embedding(75, 192)` is a 75 x 192 tensor of floats, and a "lookup" is
indexing a row. The integers produced by the tokenizer are names, not
quantities - id 47 is not one greater than id 46 in any meaningful sense,
because the ids come from alphabetical sorting - so feeding them to a network
directly would invite arithmetic on labels. The embedding table converts each
id into a vector the network can legitimately do arithmetic on.

The rows begin as random noise and are edited by backpropagation like any
other parameter. Any structure they end up with (for example, the row for `q`
drifting somewhere that makes `u` likely next) is learned because it reduced
the loss, not because it was programmed.

**Why a positional embedding is needed on top.** Self-attention computes each
position's output as a weighted sum over all positions, and a sum is
order-independent. Without positional information, "the dog bit the man" and
"the man bit the dog" are the same input to the attention mechanism - the same
multiset of vectors. This is not a shortage of training data; the information
is absent from the input. RNNs avoid the problem by construction because they
consume tokens sequentially. Transformers gave that up to gain parallelism,
and positional embeddings are the cost.

The two embeddings are **added** rather than concatenated. Concatenation would
also work but doubles the width for no gain; in a space of ~192 dimensions
there is room for the network to use different directions for "which
character" and "which position".

**Alternatives considered.**

- *Fixed sinusoidal positions* (the original transformer paper). Rejected in
  favour of learned positions, which are what GPT-2 uses, are simpler to
  explain, and work equally well at a fixed small context length. Sinusoidal
  encodings mainly buy extrapolation beyond the trained context length, which
  is irrelevant here.
- *One-hot vectors instead of an embedding table.* Mathematically identical to
  an embedding lookup (a one-hot vector times a matrix selects a row), but
  wasteful in memory and slower. The embedding table is the efficient form of
  the same operation.
- *Pretrained embeddings* (word2vec, GloVe, a sentence encoder). Rejected:
  they are word-level, and the assignment is to train a model rather than
  assemble one from pretrained parts.

**Data loading** (`dataset.py`). The cleaned corpus is encoded once into a
single `torch.int64` tensor. `get_batch` samples random offsets and returns
two tensors of shape `(batch_size, block_size)`, where `y` is `x` shifted one
position left. Because the model predicts at every position simultaneously, a
single sequence of length `block_size` supplies `block_size` distinct training
examples - one sequence of 128 characters is 128 examples.

The train/validation split is done **per genre and then concatenated**:

| Split | Characters |
|---|---:|
| Train (90% of each genre) | 1,180,303 |
| Validation (10% of each genre) | 131,147 |

Holding out the last 10% of the joined corpus instead would place the entire
validation set inside one genre, so validation loss would measure transfer
between genres rather than held-out performance. This is a small change in
code and a large change in what the number means.

There is deliberately no epoch counter and no shuffling: batches are sampled
with random offsets and replacement. At this corpus size the distinction does
not matter, and it removes a whole class of bookkeeping bugs.

## 4. What does the neural network learn?

**Objective.** Next-character prediction. Given a sequence of characters, the
model outputs a probability distribution over all 75 possible next characters,
and the loss is cross-entropy against the character that actually came next.
Nothing else is supervised - there are no labels, no human ratings, no
instruction data. Every property the model ends up with is a side effect of
being good at that one task.

**Architecture.** A decoder-only transformer (`model.py`), 819,275 parameters:

| Setting | Value |
|---|---|
| `n_layer` (blocks) | 4 |
| `n_head` (attention heads per block) | 4 |
| `n_embd` (model width) | 128 |
| `block_size` (context) | 64 characters |
| `dropout` | 0.1 |
| Vocabulary | 75 |

| Component | Parameters | Share |
|---|---:|---:|
| Token embedding (75 x 128) | 9,600 | 1.2% |
| Position embedding (64 x 128) | 8,192 | 1.0% |
| 4 transformer blocks | 791,552 | 96.6% |
| Output layer (128 x 75) | 9,675 | 1.2% |

Almost everything is in the blocks, which is the point: this is a transformer
with a small vocabulary attached, not an embedding table with a transformer
attached. The BPE comparison in question 2 would have inverted that ratio.

**How it was built, and what each piece is for.** The model was assembled in
five runnable steps rather than written at once, so each addition could be
justified against the previous one:

1. **Bigram baseline** (5,625 parameters). Predicts from the current character
   alone. Kept in the file as the number everything else must beat.
2. **Single-head self-attention.** Each position emits a query, a key and a
   value; affinity is the dot product of a query with a key; softmax over
   affinities gives weights; output is the weighted sum of values. The
   `1/sqrt(head_size)` scaling keeps softmax from saturating at
   initialisation.
3. **Multi-head attention.** Four heads of size 32 rather than one of size
   128 - the same parameter count, partitioned so each head can specialise in
   a different relationship.
4. **Feed-forward, residuals, layer norm.** Attention is communication; the
   per-position MLP (widening 4x and back) is computation. Residual
   connections (`x = x + f(x)`) give gradients an unobstructed path back
   through the stack and let each block start near the identity. LayerNorm is
   applied before each sublayer, following GPT-2 rather than the 2017 paper.
5. **The stack.** Four blocks, a final layer norm, and a linear layer to
   vocabulary-sized logits.

**The causal mask is the load-bearing constraint.** A position may attend to
earlier positions and itself, never to later ones. This is not a refinement:
targets are the inputs shifted by one position, so a model that could see
ahead would read its own answer, learn to copy rather than predict, and report
an excellent loss while generating nonsense. Implemented by setting affinities
above the diagonal to negative infinity before the softmax, which makes those
weights exactly zero.

**Sanity check used throughout.** An untrained model should score
`-ln(1/vocab_size) = 4.3175`. Measured 4.41 for the transformer and 4.89 for
the bigram; both sit slightly above because `nn.Embedding` initialises from a
normal distribution with standard deviation 1, so an untrained model has
random opinions rather than no opinions, and wrong confidence costs more than
indifference. Zeroing the bigram table reproduces 4.3175 exactly.

**Why `block_size` is 64 and not 128 - measured, not guessed.** Timing a
forward and backward pass on this machine (4 CPU threads):

| Configuration | Parameters | ms/step | 3000 steps |
|---|---:|---:|---:|
| L4 E128 block128 batch32 | 827,467 | 726 | 36.3 min |
| L6 E192 block128 batch32 | 2,719,563 | 1753 | 87.6 min |
| **L4 E128 block64 batch32** | **819,275** | **302** | **15.1 min** |
| L2 E128 block128 batch32 | 431,691 | 359 | 18.0 min |

The last row is the instructive one: half the parameters, but slower, because
attention cost grows with the square of sequence length. Parameter count is
not a proxy for compute. Halving the context bought a 2.4x speedup for a 1%
reduction in parameters, which is what brought training inside the 20-minute
budget.

**The compromise this creates, stated plainly.** With a 64-character context
the model can see roughly twelve words. It can learn spelling, word
boundaries, punctuation and local grammar. It cannot learn that a character
named at the start of a paragraph should reappear at the end. Any narrative
coherence in the output beyond about a sentence is coincidence, and the system
should not be described as understanding story structure.

**What it actually learned - measured after training.** 3000 steps, batch size
32, AdamW at learning rate 1e-3, 14.8 minutes on 4 CPU threads:

| Step | Train loss | Val loss | Gap |
|---:|---:|---:|---:|
| 0 | 4.5500 | 4.5558 | +0.006 |
| 250 | 2.1243 | 2.1606 | +0.036 |
| 1000 | 1.6527 | 1.6844 | +0.032 |
| 2000 | 1.4826 | 1.5426 | +0.060 |
| 3000 | **1.4151** | **1.4760** | +0.061 |

Reference points: 4.3175 is a model with no preference at all, and roughly
2.45 is what the bigram baseline reaches. Final validation perplexity is 4.38.

Three honest readings of that table:

1. The train/validation gap grows to about +0.06 and then stops growing
   (+0.070 at step 1750, +0.061 at step 3000). This is mild overfitting that
   is not running away; dropout at 0.1 is sufficient at this size.
2. **Validation loss was still falling when training stopped** (1.5183 ->
   1.4818 -> 1.4760). The run ended because of the 20-minute budget, not
   because the model converged. This model is under-trained, not
   over-trained, and more steps would still improve it.
3. Loss is not quality. What the number buys is described below.

**Concretely, the model learned English orthography from a single objective.**
Sampling 3000 characters and comparing against the corpus vocabulary:

- 574 words generated, **79.3% of them real words** from the corpus
- The remaining 20.7% are inventions - `weepsing`, `whistly`, `scroppe`,
  `kitter` - all of which obey English spelling and pronunciation rules
- Dialogue structure appears: quotation marks that open and close, speaker
  attributions outside them, paragraph breaks between speakers
- Capitalisation after full stops, commas before conjunctions

None of this was supervised. There is no dictionary, no grammar, no list of
valid words anywhere in the codebase. It is all a side effect of minimising
`-log P(next character)`.

What it did **not** learn, and cannot with a 64-character context: that a
character named at the start of a paragraph should reappear at the end.
Coherence beyond roughly one sentence is coincidence.

**Alternatives considered.**

- *An RNN or LSTM.* Would handle arbitrary context length and use less memory
  per step. Rejected because the assignment is about generative transformers,
  and because attention weights can be printed and inspected, which makes the
  mechanism teachable.
- *A fused QKV projection* (one `Linear(n_embd, 3 * n_embd)` instead of three
  separate ones per head). Meaningfully faster, and what production
  implementations do. Rejected here for readability: separate `key`, `query`
  and `value` layers in a loop over heads matches the explanation of the
  mechanism, and the speed measured above is already inside budget.
- *A deeper or wider model.* Measured at 2.7M parameters and 87.6 minutes -
  outside the CPU budget by a factor of four.

## 5. What conditioning signals will you use?

**Decision.** Genre, as one of three discrete labels (`fairytale`, `fable`,
`mystery`), supplied through a third learned embedding table added to every
position:

```python
x = token_embedding(idx) + position_embedding(pos) + genre_embedding(genre)
```

`nn.Embedding(3, 128)` - three rows, one learned vector per genre, broadcast
across all 64 positions of the sequence. **384 parameters**, taking the model
from 819,275 to 819,659.

It is the same mechanism as the positional embedding, used a third time: when
the model needs to know something, make it a learned vector and add it in. The
network works out what to do with it, because using it reduces the loss.

**Alternatives considered.**

- *A prepended control token* - the obvious choice, and what was originally
  planned. Rejected after measuring, not on taste. With a 64-character
  context, a token placed at the start of each paragraph appears in only
  12-29% of training windows (fairytale 12%, fable 22%, mystery 29%), so the
  large majority of training examples would carry no genre signal at all.
  Worse, that rate differs per genre, so the model could learn to identify
  mystery from how *often* tags appear rather than from the tag itself. And
  during generation the token scrolls out of the 64-character window, at
  which point conditioning silently stops. An embedding cannot be lost from
  the context because it is not in the context.
- *Cross-attention* - a separate encoder producing vectors that each block
  attends to. This is what conditioning on rich, variable-length input
  (an image, a retrieved document) requires. Rejected as disproportionate:
  roughly 400,000 extra parameters and a second input pipeline to represent a
  3-way categorical variable.
- *Classifier-free guidance* - train with and without the condition, then at
  generation time extrapolate away from the unconditioned prediction
  (`logits = uncond + w * (cond - uncond)`), giving a strength dial. This is
  how diffusion image models achieve prompt adherence. Rejected on cost: two
  forward passes per generated character, doubling generation time, for a
  refinement to conditioning that is already working.

**Batching detail that matters.** `get_batch` picks a genre per *sequence*,
not per batch. If all 32 sequences in a step came from one genre, each update
would be a pure single-genre signal and the model would oscillate between
three objectives. Mixed batches place fairy-tale and Sherlock examples side by
side, which is what forces the genre vector to carry information.

**Does it work? Measured, not asserted.** The test: take real held-out text
from each genre, and compute the model's loss on it while telling the model
each of the three genres in turn. If the conditioning is used, real text
should score its lowest loss under its own label.

Validation loss, rows = real held-out text, columns = the genre the model was
told:

| real text | fairytale | fable | mystery | |
|---|---:|---:|---:|---|
| fairytale | **1.3804** | 1.4626 | 1.4469 | correct |
| fable | 1.8799 | **1.6957** | 1.7963 | correct |
| mystery | 1.5550 | 1.5424 | **1.4727** | correct |

The diagonal is the minimum of every row: **3 of 3 genres correctly identified
by lowest loss.** Telling the model the wrong genre costs between 0.07 and
0.18 nats per character. The conditioning signal is used, not ignored.

A first attempt at this measurement used genre-distinctive marker words and
produced a meaningless table - the fable marker set dominated every row,
because fable is the smallest sub-corpus and the frequency-ratio scoring
promoted common words. It was discarded rather than reported. Held-out loss
needs no arbitrary word list and is the model's own objective.

**A number that looks like a regression and is not.** Training printed a final
validation loss of 1.5172, against 1.4760 for the unconditioned Stage 5 model.
Conditioning did not make the model worse; the two figures average over
different distributions. The old sampler drew from one concatenated tensor, so
each genre appeared in proportion to its size; the new one draws a genre per
sequence, so each appears 33% of the time. Fable is only 17.8% of the corpus
and is the hardest genre (1.6957 against 1.3804), so it is now over-sampled
and pulls the mean up. Re-weighting the per-genre losses by corpus size:

| | Loss |
|---|---:|
| Stage 5, unconditioned | 1.4760 |
| Stage 7, size-weighted | **1.4761** |
| Stage 7, as printed (uniform over genres) | 1.5163 |

Identical to four decimal places. Conditioning cost nothing in predictive
quality and bought genre control for 384 parameters.

**Limitation.** Conditioning biases the distribution; it does not switch
between three separate models. With the same prompt and seed, the three
genres produce similar opening characters and diverge in what populates the
text - `The Ass`, `the Horse` under `fable`; `Mr.`, `the death` under
`mystery`. At a 64-character context there is no mechanism for genre to shape
anything larger than local word choice and register.

## 6. How will you evaluate your model?

**Decision.** Held-out cross-entropy loss and perplexity, reported against two
baselines and broken down per genre, plus a separate experiment testing
whether the conditioning signal is used. Implemented in `evaluate.py`.

**What perplexity means.** Perplexity is `exp(loss)`, and its value is that it
has a concrete unit: the effective number of characters the model is choosing
between at each position. A perplexity of 75 means no idea at all; 1.0 means
certainty.

**Results.**

| | Loss | Perplexity |
|---|---:|---:|
| Uniform over 75 characters | 4.3175 | 75.00 |
| Best possible bigram (counted, not trained) | 2.3771 | 10.77 |
| **This model (validation)** | **1.5213** | **4.58** |
| This model (training) | 1.4141 | 4.11 |

The bigram row is what makes the result readable. It is not a trained model:
counting every character pair in the training text and normalising gives the
optimal bigram directly, so no amount of gradient descent can beat it at
predicting from the previous character alone. It reaches perplexity 10.77.
Attention over 64 characters of context more than halves that, to 4.58.

Per genre, on held-out text:

| Genre | Loss | Perplexity |
|---|---:|---:|
| fairytale | 1.3811 | 3.98 |
| mystery | 1.4783 | 4.39 |
| fable | 1.6944 | 5.44 |

Fable is consistently hardest. It is the smallest sub-corpus (233,553
characters, 17.8% of the total) and the most structurally varied - each fable
introduces a new cast with a title line and a closing moral, so there is less
repeated context to exploit. The corpus imbalance recorded in question 1, and
deliberately not corrected, appears here as a measurable 1.4-point perplexity
gap. That is the cost of that decision, stated rather than hidden.

**Conditioning is evaluated separately** because overall perplexity cannot
detect it - the full matrix is in question 5. Summary: 3 of 3 genres are
identified correctly by lowest loss, and telling the model the wrong genre
costs between 0.07 and 0.18 nats per character.

**Alternatives considered.**

- *BLEU or ROUGE.* Rejected as inapplicable. Both compare generated text to a
  reference, which presumes one correct output. Open-ended generation has no
  reference.
- *A trained classifier to score genre adherence.* Rejected as circular at
  this scale: a classifier trained on the same three books would mostly detect
  the same surface statistics the generator learned, so agreement would
  measure shared bias rather than quality.
- *Human evaluation.* Not rejected on merit - see below.

### What these metrics do not capture

This section matters more than the numbers above.

1. **Perplexity measures prediction, not generation.** Every figure here is
   computed on real held-out text: the model is shown genuine Grimm and asked
   for the next character. It is never evaluated on its own output. Prediction
   never compounds its own errors; generation always does.

2. **Locally plausible is not globally coherent.** 79.3% of generated words
   are real words, so one word in five is an invention like `twiversations`.
   Perplexity averages over positions, so a fifth of positions being wrong
   barely moves it, while a reader notices immediately. Nothing in the loss
   asks whether a character introduced in one sentence still exists two
   sentences later - and with a 64-character context the model cannot
   represent that question at all.

3. **Perplexity is blind to the sampling settings actually used.** Temperature
   0.5 produces "the" as one word in six (question 6a), which is degenerate
   text. Perplexity is *identical* across all temperature settings because it
   never samples. The failure most likely to spoil a demonstration is
   invisible to the headline metric.

4. **It is tokenizer- and test-set-dependent.** 4.58 here cannot be compared
   with GPT-2's roughly 20 on WebText: different units (characters versus
   subwords) and different text. Placing them in one table would be a
   category error.

5. **The train/validation gap is small (+0.10) but the model is
   under-trained, not over-trained.** Validation loss was still falling when
   the 20-minute budget ran out.

**What human evaluation would add.** Raters scoring samples for fluency,
genre-appropriateness and coherence - and, most directly relevant here,
being shown a sample and asked which genre it was conditioned on. The 3/3
loss result proves the model uses the signal; it does not prove a reader can
perceive the difference. Those are different claims and only the second one
matters to a user.

Doing it properly requires several raters and an inter-annotator agreement
statistic such as Cohen's kappa, because a single person's judgement is an
opinion rather than a measurement. That cost is why it was not done here. The
honest position is that this system has been measured on what it optimises
and not on what it produces.

## Appendix: generation and sampling

Not one of the six questions, but the stage that produced the numbers below.

**Decision.** Sample from the full distribution with a temperature parameter,
with top-k available and off by default.

**Temperature** divides the logits before softmax. Below 1.0 the gaps between
logits widen and the distribution sharpens; above 1.0 they shrink and it
flattens toward uniform. Measured on 2000 characters per setting, same prompt
and seed:

| Temperature | Real words | Unique words | Rate of "the" |
|---|---:|---:|---:|
| 0.5 | 96.3% | 35.9% | 16.5% |
| 0.8 | 86.5% | 53.1% | 9.3% |
| 1.0 | 80.2% | 58.7% | 4.2% |
| 1.5 | 53.4% | 76.2% | 0.3% |
| *corpus, same window size* | *100%* | *48.7%* | *3.3%* |

Temperature is a dial between two failure modes. At 0.5 almost every word is
real, but one word in six is "the" - five times the corpus rate - because low
temperature collapses the model onto its most confident guesses. At 1.5 output
is diverse but half the words are inventions. Around 0.8-1.0 is closest to the
corpus on diversity, which is the setting used elsewhere in this document.

**Top-k** restricts sampling to the k highest-scoring characters. Measured, it
makes almost no difference here: 86.5% real words without it, 87.1% with
`top_k=20`. The honest reason is that top-k exists to truncate a long tail of
implausible tokens, and a 75-character vocabulary has very little tail - a
trained model has already pushed implausible characters near zero. Top-k earns
its place with a 50,000-token BPE vocabulary, where the tail holds real
probability mass. This is the same small-vocabulary consequence discussed in
question 2, appearing again.

**Greedy decoding (argmax)** was rejected outright: it is deterministic and
collapses into repeated phrases immediately.

**Prompt handling.** Prompts are passed through the same `clean_text` used on
the corpus, so a character outside the 75-symbol vocabulary is normalised or
dropped rather than raising `KeyError` - the limitation recorded in question 2
is now closed.

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
- **Stage 3 — embeddings and batching.** `dataset.py`: corpus encoded to a
  single int64 tensor, per-genre 90/10 split (1,180,303 / 131,147 characters),
  and `get_batch` returning `(B, T)` input/target pairs.
- **Stage 4 — model.** `model.py`: bigram baseline, causal self-attention,
  multi-head, feed-forward with residuals and layer norm, and a 4-block GPT
  at 819,275 parameters. `inspect_data.py` retired; its output is covered by
  the `tokenizer.py` demo.
- **Stage 5 — training.** `train.py`: hand-written loop, AdamW, train/validation
  loss every 250 steps. 3000 steps in 14.8 minutes; loss 4.55 -> 1.4151 train,
  1.4760 validation. Checkpoint saved to `checkpoint.pt` (3.6 MB).
- **Stage 6 — generation.** `generate.py`: temperature, top-k, prompting,
  and a `--sweep` mode. Temperature measured against corpus statistics; top-k
  measured and found to make almost no difference at this vocabulary size.
- **Stage 7 — conditioning.** Genre embedding added to `model.py` (+384
  parameters); `dataset.py` now returns per-genre tensors and genre ids;
  `--genre` and `--compare` added to `generate.py`. Retrained in 12.7 minutes.
  Conditioning verified by held-out loss: 3/3 genres identified correctly.
- **Stage 8 — evaluation.** `evaluate.py`: perplexity against a uniform and
  an optimal-bigram baseline, per-genre breakdown, and the conditioning
  matrix. Validation perplexity 4.58 against 10.77 for the best bigram.

