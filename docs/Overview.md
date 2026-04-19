# Overview — the approach, in one page

The problem: given 598 messages across 300 threads from Slack, Gmail, and Outlook, discover the ~10–12 workplace topics hidden in there, link related discussions across apps (a Slack thread and a Gmail chain about the same project should land in the same cluster), and classify new incoming messages against what we've discovered — with a confidence score and a graceful "this looks like a new topic" verdict.

## The shape of the solution

Four layers, each one reading only from the layer above:

1. **Normalization** — each app's schema gets flattened into a single `NormalizedMessage` shape. Slack messages get a special LLM rewrite pass first, because most of them are too short to cluster on their own ("LGTM", "On it", "+1").
2. **Threading + clustering** — thread-level text goes through a sentence embedder, UMAP reduces it to 5 dimensions, HDBSCAN finds the clusters, and c-TF-IDF describes what makes each cluster distinct.
3. **Matching** — the same fitted model, used in reverse: embed a new message, ask BERTopic for the topic probabilities, apply two thresholds, and emit a verdict with confidence.
4. **FastAPI serving** — `/build`, `/match`, `/topics`, `/health` on top of it, with structured JSON logs, correlation IDs, and per-build token accounting.

## BERTopic does most of the heavy lifting

[BERTopic](https://maartengr.github.io/BERTopic/) is the topic-modeling library that owns stages 2 and 3. If you read the code, `cluster.py` is really just a wrapper around BERTopic: the UMAP → HDBSCAN → c-TF-IDF pipeline, the per-topic keyword extraction, the optional LLM representation model for labels, the persistence format, and the `.transform()` call used at inference — all of that is BERTopic out of the box.

What I'm actually contributing on top of BERTopic:

- Deciding to cluster threads, not messages, and building the per-app thread text accordingly.
- The Slack rewriter that turns filler into topic-bearing text before BERTopic ever sees it.
- Parameter choices (`min_topic_size=8`, UMAP `n_neighbors=15`, cosine metric, `random_state=42`) that are tuned to this dataset's size and shape.
- The serving layer — API design, the normalizer registry, the person directory, the matcher's threshold logic, the observability and token accounting.

The reasoning for every choice (UMAP vs PCA, HDBSCAN vs k-means, why LLM labels on top of c-TF-IDF instead of either alone) lives in [`Clustering.md`](Clustering.md). The full system layout is in the root [`README.md`](../README.md).

## Assumptions

A few PoC-scoped decisions worth calling out explicitly:

- **FastAPI as the minimal serving framework.** Just enough to expose the pipeline behind HTTP without dragging in a full service stack.
- **No vector DB, no relational DB.** Embeddings and messages live in the local filesystem or in memory for the duration of a build. This is deliberately a PoC-grade shortcut — a production deployment would front the embeddings with a vector store (pgvector, Qdrant) and the artifacts with object storage.
- **OpenAI as the primary LLM / embedding provider,** with `sentence-transformers` as the offline fallback. OpenAI powers the Slack rewriter, the OpenAI embedder, and the LLM topic-label path; the local sentence-transformer keeps the default build runnable without credentials.

## Where this goes next

Today, every `/build` rebuilds the topic model from scratch. That's fine for the demo but not for a system that should absorb new conversations continuously. Two natural extensions, both supported by BERTopic:

- **Incremental training with `partial_fit`.** BERTopic ships an online mode — `partial_fit` over incoming batches — so the model keeps learning as new messages arrive without ever needing a full rebuild. With the right backing clusterer (river / MiniBatchKMeans as BERTopic's example shows), this turns the service into a continuously-training model.
- **Merging models.** `BERTopic.merge_models` lets you train separate models on separate data windows (or separate tenants) and then reconcile them into one. Useful if you want to train a weekly model in the background and merge it into production without downtime.

In practice, the clean path to production is probably a combination: run `partial_fit` on a streaming intake for continuous updates, do a full rebuild on a weekly or monthly cadence as the source of truth, and use `merge_models` to reconcile the two. The code we have today — threads as the clustering unit, persisted artifacts, the Matcher that hot-reloads after any build — is deliberately shaped so none of those paths require architectural changes. They're parameter changes.

## Where to read next

- [`README.md`](../README.md) — setup, API reference, architecture, design decisions, scaling.
- [`Clustering.md`](Clustering.md) — the "why" behind every clustering choice.
