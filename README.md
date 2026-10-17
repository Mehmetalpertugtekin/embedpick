# embedpick

Benchmark embedding models on **your own data**, not on someone else's leaderboard.

[Türkçe README](https://github.com/Mehmetalpertugtekin/embedpick/blob/main/README.tr.md) · [Try it online](https://huggingface.co/spaces/tugtekinalper/embedpick)

## Why

Picking an embedding model usually goes like this: open Hugging Face, sort by
downloads, take the top result. The scores on the model card were measured on
general-purpose English benchmarks. Your data is not that.

embedpick answers a narrower and more useful question: *given my documents and
my queries, which model actually finds the right thing, and what does it cost
me in time and memory?*

It reports quality, speed and memory side by side, and — unlike most benchmark
tooling — it puts a plain keyword-search baseline in the same table.

## The baseline is the point

Every run includes BM25, a classic keyword-matching algorithm with no neural
network involved. If an embedding model cannot beat it, running that model is
wasted compute.

Results on the included Turkish customer-support dataset (72 documents,
26 queries, k=5), with the baseline stemmed (`--stemmer simple`):

| Model | recall@5 | 95% CI | MRR@5 | docs/sec | dim | rel. speed |
|---|---|---|---|---|---|---|
| BM25 (baseline, simple stem) | 0.654 | [0.50, 0.81] | 0.622 | ~32,000 | — | — |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.615 | [0.46, 0.77] | 0.596 | 374 | 384 | 1.00x |
| multilingual-e5-small | 0.692 | [0.56, 0.81] | 0.708 | 322 | 384 | 0.86x |
| trmteb/turkish-embedding-model | **0.846** | [0.73, 0.94] | **0.865** | 119 | 768 | 0.32x |

Those intervals overlap, but that is not how to read them: both methods answer
the same 26 queries, so the uncertainty of the *difference* is smaller than the
uncertainty of either column. The paired comparison is the number to use:

| Model | difference vs baseline | 95% CI | verdict |
|---|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 | −0.038 | [−0.192, +0.115] | not shown |
| multilingual-e5-small | +0.038 | [−0.115, +0.212] | not shown |
| trmteb/turkish-embedding-model | **+0.192** | [+0.058, +0.346] | **significant** |

Read those rows carefully. Two of the three embedding models cannot be shown to
beat keyword search on this data at all, and MiniLM's point estimate is on the
wrong side of zero — while costing roughly 90x the indexing time. The
Turkish-specific model is the one real result here: its advantage survives both
a properly built baseline and a significance test.

"Not shown" means the dataset was too small to settle it, not that the methods
are equal. With 26 queries that is a low bar to clear, which is itself the
lesson.

So the lesson is not "embeddings are overrated." It is: **a badly chosen
embedding model is not better than keyword search, and nobody checks.**

### The baseline needs the same care as the model

An earlier version of this table reported BM25 at 0.564 and a +28 point gap,
because `tr_tokenize` only lowercased. Turkish is agglutinative, so `iade` and
`iadesi` were two unrelated terms and the baseline was being handicapped by its
own tokenizer. Adding stemming lifted it from 0.564 to 0.654 and cut the
headline gap to +19 points.

The conclusion survived; the claim did not. In a comparison, the weakest link
is the sloppiness on your own side, because nobody else is positioned to catch
it. `--stemmer none|simple|zemberek` is there so you can see what your baseline
is actually doing — and the improvement it buys is itself only +0.090 with a CI
of [−0.058, +0.231], so even fixing the baseline is not separable from noise at
this sample size.

## Findings from the sample dataset

**Documented configuration is not always the right configuration.** The e5
model card specifies `query: ` and `passage: ` prefixes on input text. Applying
them on this dataset *lowered* recall@5 from 0.750 to 0.692 and MRR from 0.792
to 0.708 — same model, same data, same hardware.

A plausible reason: these documents are short, five to eight words each, so an
English prefix takes up a large share of every one of them. Whatever the cause,
the recommended setting cost about six points here, and nothing short of
measuring on your own data would have shown it.

embedpick applies known prefixes by default, following the model authors'
instructions, and `--no-presets` turns them off so you can check. Across 26
queries that gap is worth roughly one and a half queries — well inside the
noise, so treat it as a question worth testing on your own data rather than a
finding. Run it with CIs before acting on it.

**BM25 and embeddings fail on different queries.** On `ödeme yaparken sorun`
("problem while paying") BM25 scored 1.00 and MiniLM scored 0.00. On
`sahte ürün şüphesi` ("suspected counterfeit") it was the reverse. They are
complementary, which is the empirical case for hybrid retrieval.

**Small corpora hide everything.** On a 16-document pilot all three models
scored 1.0 and looked identical. At 72 documents a 23-point spread appeared.
A benchmark that cannot separate models tells you nothing about them.

**Absolute timings are unreliable; ratios are not.** Across repeated runs on
the same machine, `docs/sec` varied by up to 4x while the ratio between models
stayed within about 20%. Report `rel_speed` when comparing across machines, and
note that the table rounds throughput to three significant digits on purpose —
the precision of a number is a claim about how much you trust it.

**A difference without an interval is a hypothesis, not a finding.** Every
quality column carries a bootstrap 95% CI, and every model is compared to the
baseline with a paired bootstrap. On 26 queries most differences do not clear
that bar. The honest move is to report it and add queries, not to drop the
column.

## Install

```bash
pip install embedpick
```

Or from source:

```bash
git clone https://github.com/Mehmetalpertugtekin/embedpick.git
cd embedpick
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -e .
```

Python 3.9+.

## Usage

```bash
# run with the bundled sample dataset
python -m embedpick

# your own data
python -m embedpick --corpus my_docs.csv --queries my_queries.csv

# pick models, evaluate top 10, show per-query detail
python -m embedpick --models intfloat/multilingual-e5-base -k 10 --verbose

# give the keyword baseline Turkish stemming before you trust the gap
python -m embedpick --stemmer simple

# save the table
python -m embedpick --out results.csv
```

`python -m embedpick --help` lists every flag.

### Turkish stemming for the baseline

`--stemmer` controls how the BM25 baseline tokenizes:

| value | what it does |
|---|---|
| `none` | lowercase only — the baseline will look worse than it is |
| `simple` | a dependency-free suffix stripper, written out in `embedpick/stemmers.py` |
| `zemberek` | real morphological analysis; `pip install embedpick[zemberek]`, needs a JVM |

`simple` over-strips at times (`fatura` → `fatur`) but does so consistently
(`faturamı` → `fatur` too), which is what retrieval needs. It is deliberately
readable rather than clever: you should be able to see what your baseline is
doing.

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Tests do not download any models, so they run in seconds. Make sure your
virtual environment is active first — a missing `rank_bm25` in the test output
usually means you installed into the system Python by mistake.

## Data format

**corpus.csv** — the documents to search over:

```csv
id,text
1,Kargom hala elime ulaşmadı
2,Sipariş takip numaram sistemde görünmüyor
```

**queries.csv** — queries and their correct answers:

```csv
query,relevant_ids
kargo gecikmesi,1;5
paketim nerede,1;2
```

Separate multiple correct answers with `;`. A query can have any number of
them. Files must be UTF-8.

Twenty to thirty queries is usually enough to see real differences. Include
queries that share no words with their correct answer — those are the ones
that test semantic matching rather than lucky keyword overlap.

## Metrics

**recall@k** — of the correct documents, what fraction appeared in the top k.
Use this when you show the user a list of results.

**MRR@k** — 1.0 if the first correct answer is at rank 1, 0.5 at rank 2, 0.33
at rank 3, averaged over queries. Use this when you show one answer, or feed
the top hit to an LLM.

These can disagree, and the disagreement is informative. In an earlier run
e5 had higher recall than MiniLM but lower MRR: it found more, but ranked
worse. Which model is "better" depends on what you are building.

**docs/sec** — indexing throughput. Machine-dependent, see the caveat above.

**dim** and **index_mb** — vector width and index size. A 768-dimensional
model needs twice the memory of a 384-dimensional one at the same corpus size.

## Web interface

A Gradio app is included. It shows the sample-dataset results immediately and
lets you upload your own CSVs to run a live benchmark.

```bash
pip install embedpick[ui]
python app.py
```

A hosted version runs at
[huggingface.co/spaces/tugtekinalper/embedpick](https://huggingface.co/spaces/tugtekinalper/embedpick).

## Label checking

Two things happen automatically:

- If `relevant_ids` points at a document id that is not in the corpus, the run
  stops and names the query. Otherwise that query silently scores zero forever
  and you blame the model.
- If *no* method — including BM25 — finds a query's answer, embedpick flags it.
  When everything fails on the same query, the label is usually wrong. This
  caught a mislabelled query during development.

## Sample dataset

`embedpick/sample_data/` contains 72 synthetic Turkish customer-support messages across nine
themes (shipping, returns, payment, account, product quality, promotions,
warranty, order management, support) and 26 labelled queries. It is written,
not scraped, so it carries no licensing or privacy constraints.

Several queries deliberately share no vocabulary with their answers — for
example `güvenlik ihlali şüphesi` ("suspected security breach") maps to
*"Hesabıma başkası girmiş olabilir"* ("someone else may have accessed my
account"). Keyword search cannot solve these; that is the point.

## Caveats

- Timing on a 72-document corpus is close to measurement noise, particularly
  for BM25. Treat the throughput column as indicative.
- BM25 and neural encoders scale differently. The ratio at 72 documents is not
  the ratio at 100,000.
- Results are from one machine, CPU only, no GPU.

## Roadmap

- Hybrid retrieval (BM25 + embedding score fusion) as a fourth row
- Hub detection: flag documents that surface for nearly every query
- Markdown report export
- Optional Gradio interface

## License

MIT
