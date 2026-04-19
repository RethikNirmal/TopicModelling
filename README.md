# Cross-App Topic Clustering & Matching

A FastAPI service that ingests raw messages from **Slack, Gmail, and Outlook**, normalizes them into a single shape, clusters them into coherent workplace topics, links related discussions across apps, and classifies new incoming messages against the discovered topics.

Dataset: `data/conversations.json` — 598 messages across ~300 threads, covering roughly 10–12 distinct workplace topics (project launches, security patches, budget reviews, OKRs, offsite planning, etc.), with ~60–70% of topics spanning multiple apps.

If you want the short version first, [`docs/Overview.md`](docs/Overview.md) is a one-page summary of the approach, what BERTopic contributes, and where this extends next (`partial_fit` for online learning, `merge_models` for background rebuilds). The deep dive on clustering choices — UMAP vs PCA, HDBSCAN vs k-means, c-TF-IDF, LLM labels — is in [`docs/Clustering.md`](docs/Clustering.md).

---

## Table of contents

1. [Setup](#1-setup)
2. [Running the service](#2-running-the-service)
3. [How each part works](#3-how-each-part-works)
   1. [Normalization](#31-normalization)
   2. [Clustering](#32-clustering)
   3. [Cross-app linking](#33-cross-app-linking)
   4. [Topic matcher](#34-topic-matcher)
   5. [Quality validation](#35-quality-validation)
   6. [Observability](#36-observability)
4. [API reference](#4-api-reference)
5. [Artifacts on disk](#5-artifacts-on-disk)
6. [Design decisions (non-obvious)](#6-design-decisions-non-obvious)
7. [Tradeoffs and limitations](#7-tradeoffs-and-limitations)
8. [Scaling](#8-scaling)
9. [Repo layout](#9-repo-layout)

---

## 1. Setup

- Python 3.10 (virtualenv at `venv/`).
- An OpenAI API key is **required for a full-quality build**. Every `/build` runs the Slack rewrite pass once per Slack message; no on-disk cache yet, so each build pays the LLM cost afresh. OpenAI embeddings (`embedder=openai`) and LLM topic labels (`label_with=openai`) additionally need the key.

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt`: `bertopic`, `sentence-transformers`, `scikit-learn`, `hdbscan`, `umap-learn`, `numpy`, `openai`, `python-dotenv`, `fastapi`, `uvicorn[standard]`, `pydantic-settings`.

Create a `.env` at the project root (already gitignored):

```
OPENAI_API_KEY=sk-...
```

`.env` is loaded automatically wherever the pipeline talks to OpenAI. Without a key the Slack rewriter catches its own exceptions per-message and falls back to verbatim text, so the build technically completes — but clustering quality on short Slack messages ("LGTM", "On it") degrades substantially because the rewrite is where their topic signal lives.

---

## 2. Running the service

No CLI — everything is behind FastAPI.

```bash
source venv/bin/activate
uvicorn topic_cluster.api.app:app --port 8000
# Interactive docs: http://localhost:8000/docs
```

End-to-end walkthrough:

```bash
# 1. Build the model from the bundled conversations file.
#    Normalizes → rewrites Slack → threads → embeds → clusters → saves → hot-reloads matcher.
curl -s -X POST localhost:8000/build/from-path \
  -H 'content-type: application/json' \
  -d '{"path": "data/conversations.json", "embedder": "st", "label_with": "keywords"}'

# 2. Inspect the discovered topics.
curl -s localhost:8000/topics | jq '.topics[] | {topic_id, label, size}'

# 3. Classify a new message.
curl -s -X POST localhost:8000/match/single \
  -H 'content-type: application/json' \
  -d @data/samples/sample_match_alpha.json
```

For higher-quality topic labels, use the OpenAI representation model:

```bash
curl -s -X POST localhost:8000/build/from-path \
  -H 'content-type: application/json' \
  -d '{"path":"data/conversations.json","embedder":"openai","label_with":"openai","label_model":"gpt-4o-mini"}'
```

Logs are structured JSON on stdout (`LOG_LEVEL`, `LOG_FORMAT=json|text`). Every request carries an `X-Correlation-Id` header; the build response includes a `token_usage` breakdown per LLM call site. See §3.6.

---

## 3. How each part works

The pipeline is four layers — each layer consumes the previous layer's output, nothing reaches around:

```
raw messages (3 schemas)
   │  Layer 1: Normalization
   ▼
NormalizedMessage (single shape regardless of app)
   │  Layer 2: Threading + clustering
   ▼
Threads → embeddings → BERTopic → topic labels
   │  Layer 3: Matching (BERTopic model used in reverse)
   ▼
MatchResult for any new raw message
   │  Layer 4: FastAPI serving
   ▼
/build  /match  /topics  /health
```

### 3.1 Normalization

Each app ships a different schema. Downstream code shouldn't care. Every raw message flows through a per-app `Normalizer` that emits a single `NormalizedMessage` (`topic_cluster/schema.py`).

| File | Purpose |
|---|---|
| `schema.py` | `Person` and `NormalizedMessage` dataclasses — the common contract |
| `person.py` | `PersonDirectory` — canonical identity across apps, role-mailbox flagging, @mention resolution |
| `normalizers.py` | `SlackNormalizer`, `GmailNormalizer`, `OutlookNormalizer`, `NormalizerRegistry` |
| `slack_rewrite.py` | Context-aware LLM rewrite for short Slack messages |
| `api/services.py:normalize_in_memory` | The only orchestration entry point |

#### Unified schema

| Field | Type | Purpose |
|---|---|---|
| `message_id` | `str` | Native id, kept verbatim |
| `app` | `"SLACK" \| "GMAIL" \| "OUTLOOK"` | Source tag |
| `thread_key` | `str` | `"{app}:{native_thread_id}"` — defensive prefix |
| `parent_message_id` | `str \| None` | Reply parent; `None` for Outlook |
| `timestamp` | `datetime` | UTC ISO |
| `sender_id` | `str` | Lowercased email |
| `participants` | `list[str]` | Sender + to + cc for email; sender only for Slack |
| `subject` | `str \| None` | Email subject or Slack channel |
| `canonical_subject` | `str \| None` | `Re:/Fwd:` stripped — embedding input only, never for grouping |
| `original_content` | `str` | Verbatim body; audit trail for Slack where `text` is rewritten |
| `text` | `str` | The cleaned text we embed and cluster on |
| `mentions` | `list[str]` | Resolved `@mention` person_ids |
| `raw_mentions` | `list[str]` | Unresolved tokens, kept for debugging |
| `raw` | `dict` | Original payload, preserved |

#### Per-app differences

- **Slack.** `thread_key = "SLACK:{thread_id or message_id}"`. `parent_message_id` populated only when `is_thread_reply=True`. `participants = [sender_email]` — channels don't expose recipient lists, don't synthesize one. `text` is the LLM rewrite when available, else `[#channel] content`. Channel stored as `subject` / `canonical_subject`.
- **Gmail.** `thread_key = "GMAIL:{thread_id or message_id}"`. `parent_message_id = in_reply_to`. `participants = from + to + cc` (lowercased, deduped). `text = canonical_subject + body`.
- **Outlook.** Same as Gmail but uses `conversation_id`, `recipients`, `body_full`. `parent_message_id = None` — reconstructing reply chains from timestamps/subjects is fragile and out of scope.

#### Threading

Preserved two ways:

1. **`thread_key`** is the same for every message in a thread — `groupby(thread_key)` gives you the thread.
2. **`parent_message_id`** populated for Slack and Gmail (Outlook skips by design).

Canonical subject (`^(?:(?:re|fwd|fw)\s*:\s*)+` stripped) is **not** used for grouping. The same base subject legitimately spans multiple distinct threads (e.g. four separate chains titled `Alpha Launch Checklist`); collapsing them by subject would destroy the cross-app linking signal.

#### Identity / mentions

`PersonDirectory` is built once from the raw corpus, keyed by `person_id == lowercased email`. For each address it picks the most-frequently-seen display name and infers a first name from the display name or the email local part.

- **Role mailbox flagging** via an explicit `ROLE_LOCAL_PARTS` set (`hr`, `cto`, `tech.lead`, …) — no heuristics.
- **Mention resolution** (`@bob` → `bob.martinez@company.com`): first-name match first, then email-local-part fallback. Both require a unique match; ambiguous mentions stay in `raw_mentions`.
- External addresses flagged via `domain != COMPANY_DOMAIN` (hardcoded to `"company.com"`).

#### Slack rewriter — why Slack needs special treatment

Gmail and Outlook messages are long-form (subject + multi-sentence body + quoted history) — plenty of surface area for an embedding model. Slack is the opposite:

- **Short.** 34 of 176 Slack messages are under 20 characters.
- **Mostly standalone.** 71 of 105 Slack threads are singletons; there's no sibling context.
- **Dense with workplace vernacular** — "LGTM", "On it", "WFH today", "+1", "LMK", "PTAL". To a sentence embedder these are near-identical filler regardless of what project they're about, so every "LGTM" lands near every other "LGTM" and nowhere near its actual topic.

Embed them as-is and they either clump into a giant filler cluster or get marked noise, and the cross-app link ("Slack ack of the Alpha launch" ↔ "Gmail Alpha chain") is lost.

`SlackRewriter` calls `gpt-4o-mini` (temp 0.0, ≤200 chars) to rewrite each Slack message into a **topic-bearing standalone sentence** before embedding. The LLM sees the target plus focused context and turns "LGTM" into e.g. "Agrees the alpha launch checklist is ready" — same intent, now dense with topic-specific nouns.

Context given to the model:

- All siblings in the same thread.
- Up to 20 prior messages in the same channel within a 48-hour window.

Prompt has five rules (ignore unrelated context, strip filler, don't invent facts, fall back to original if no topic is extractable, return only the rewritten text).

Behaviors:

- **No on-disk cache yet.** Every build calls the LLM once per Slack message (~176 calls on the bundled dataset). Per-`message_id` disk cache is a planned improvement.
- On any LLM exception the rewriter logs and falls back to the verbatim message text — no opt-out, only graceful per-message degradation.
- `original_content` always preserves the verbatim body for audit.

This is the only place an LLM touches *input data*. Topic labeling LLM calls are separate, output-side, and optional.

### 3.2 Clustering

BERTopic (embedding + UMAP + HDBSCAN + c-TF-IDF) clustering **threads, not messages**.

For the full reasoning — why threads, why sentence embeddings over TF-IDF, why UMAP over PCA, why HDBSCAN over k-means, how c-TF-IDF is used, why LLM labels on top of c-TF-IDF — see [`docs/Clustering.md`](docs/Clustering.md).

TL;DR:

- `ThreadBuilder` concatenates each thread's messages per-app into a single `text` before embedding.
- Default embedder: `all-MiniLM-L6-v2` (384 dims, local). Swap via `embedder=openai` → `text-embedding-3-small` (1536 dims).
- `MAX_TEXT_CHARS = 1500` exists on `ThreadBuilder` but the truncation is currently commented out (`thread.py:51-52`); sentence-transformers internally truncates to its token window anyway.
- BERTopic config (all `ClassVar` on `TopicModel`): `MIN_TOPIC_SIZE=8`, `RANDOM_STATE=42`, `UMAP_NEIGHBORS=15`, `UMAP_COMPONENTS=5`, `calculate_probabilities=True`.
- HDBSCAN picks the number of clusters automatically; `-1` is noise, which feeds directly into the matcher's new-topic detection.
- On the bundled dataset: ~12–13 topics, ~11–13 noise threads, silhouette ≈ 0.35.

Topic label modes:

- **`label_with="keywords"`** (default). Top 3 c-TF-IDF keywords joined with ` / `, e.g. `"security / deployment / vulnerability"`. Deterministic, no LLM call.
- **`label_with="openai"`**. BERTopic's `OpenAI` representation model with a tight prompt → 2–6 word Title Case phrase. ~13 LLM calls per build, fraction of a cent. Labels become `"Security Patch Deployment"`.

### 3.3 Cross-app linking

The core assignment requirement: detect when different apps discuss the same topic (e.g. `"Project Alpha"` Slack thread ↔ `"Alpha Initiative"` Gmail chain ↔ `"alpha launch checklist"` Outlook invite).

Cross-app linking is a **property** of the clustering output, not a separate stage. Because we:

1. Cluster **threads** (already normalized),
2. Use a **shared semantic embedding space** (same embedder for every app's thread text),
3. Group by `topic_id` after fitting,

…threads from different apps land in the same cluster when they're about the same topic. Linking is implicit: **threads sharing a `topic_id` across apps are linked.**

Surfaced in two places:

- `artifacts/topics.json` — each topic has `representative_thread_keys` with 3 highest-probability members, deliberately picked to span apps where possible.
- `artifacts/assignments.json` — `{thread_key: {topic_id, probability, app}}`. Grouping by `topic_id` and collecting distinct `app` values is the link set.

Quick derivation:

```python
import json, collections
assignments = json.load(open("artifacts/assignments.json"))
by_topic = collections.defaultdict(lambda: collections.defaultdict(list))
for thread_key, v in assignments.items():
    by_topic[v["topic_id"]][v["app"]].append(thread_key)
# by_topic[7] → {"SLACK": [...], "GMAIL": [...], "OUTLOOK": [...]}
```

#### Signals used

- **Semantic similarity** (primary). Sentence embeddings handle the vocabulary drift between apps — `"Project Alpha"` / `"Alpha Initiative"` / `"the alpha launch"` all land near each other because the embedder captures paraphrase.
- **Participant overlap** (secondary, available on `NormalizedMessage.participants` and `Thread.participants`). Not folded into the embedding because Slack participants = `[sender_email]` only — the signal is asymmetrically weak across apps and would bias clustering. Kept for explainability and post-hoc reranking.
- **Temporal proximity** (secondary, via `Thread.start_ts` / `end_ts`). Same reason — kept on the `Thread` object, not in the embedding.

#### Handling vocabulary differences

- Sentence embeddings (trained on paraphrase/NLI) cluster near-synonymous phrases together.
- Canonical subjects strip `Re:/Fwd:` chains so the embedder sees the topic phrase cleanly.
- Slack LLM rewrites turn filler ("LGTM") into topic-bearing text — this is what lets Slack threads land in the same cluster as the email chain they're echoing.

#### Known failure cases

- **Singleton Slack threads** on short-lived topics can land in noise (`-1`) even when semantically related — HDBSCAN requires local density.
- **Generic check-ins** ("can we sync on this?") embed closer to each other than to their actual topic; one-liners without specific nouns tend to get flagged ambiguous or drift to the wrong cluster.
- **Cross-topic spillover in Slack channels.** A busy `#general` channel gives the rewriter weaker context than a focused project channel, so rewrites in `#general` are lower fidelity.
- **No explicit `links.json` output.** Callers derive links by grouping `assignments.json` on `topic_id`. One-function extension, not shipped.

### 3.4 Topic matcher

Classifies a new raw message against the existing topic model (`topic_cluster/matcher.py`).

Flow:

```
raw message
   │ 1. NormalizerRegistry.for_app(app).normalize(raw)   → NormalizedMessage
   │ 2. embedder.encode([msg.text])                      → (1, d) vector
   │ 3. bertopic.transform([text], embeddings=vec)       → (topics[0], probs[0])
   │ 4. Map BERTopic's [noise, t0, t1, …] column order   → actual topic_id
   │ 5. Apply two thresholds                             → is_new_topic / is_ambiguous
   ▼
MatchResult
```

#### The off-by-one that was a bug first

BERTopic's `transform()` with `calculate_probabilities=True` returns a probability row shaped `[noise, topic_0, topic_1, ...]`. Column 0 is noise; column `i` (for `i ≥ 1`) maps to `topic_id == i - 1`. `matcher.py` handles this explicitly (`_col_to_topic`). Worth flagging — on the first cut every test message was off by one.

#### Confidence scoring and thresholds

Two class constants, easy to tune:

```
NEW_TOPIC_PROB_THRESHOLD = 0.35   # top prob below this (or top == -1) → new topic
AMBIGUOUS_GAP_THRESHOLD  = 0.15   # top - second below this → ambiguous
```

Outputs:

- **`is_new_topic=True`** when the top topic is noise or the top probability is below 0.35. Emits a `suggested_new_topic_label` — the 3 most frequent non-stopword tokens joined with ` / `.
- **`is_ambiguous=True`** when the assignment stands but top − second < 0.15. Message still goes to the top topic, but the caller is told confidence is tight.
- Otherwise confident assignment.

Thresholds were picked on the bundled dataset by eyeballing the held-out probability distribution. They're class constants so tuning on your own data is one line.

#### Example output

```json
{
  "message_id": "test-002",
  "app": "GMAIL",
  "top_topic_id": 10,
  "top_topic_label": "Project Alpha Launch Status Update",
  "top_probability": 0.81,
  "second_topic_id": 3,
  "second_probability": 0.49,
  "is_new_topic": false,
  "is_ambiguous": false,
  "suggested_new_topic_label": null,
  "reason": "top probability 0.814 >= 0.35"
}
```

### 3.5 Quality validation

Tracked at three points:

| Where | Metric | What it catches |
|---|---|---|
| Build time | Silhouette (cosine, non-noise only) | Whether clustering as a whole is meaningful. ~0.35 means real; ~0.05 means random. Surfaced in `topics.json`, `/topics`, `/health` |
| Build time | Noise count (`-1` threads) | If this spikes after a dataset or config change, embedding quality dropped or a rogue topic was injected |
| Match time | Top probability + top-second gap | Per-message confidence; drives `is_new_topic` and `is_ambiguous` |

Every `MatchResult` carries a `reason` field explaining the verdict in one of four forms:

- `"top topic is noise (-1)"`
- `"top probability 0.xx < 0.35"`
- `"top - second = 0.xx < 0.15"`
- `"top probability 0.xx >= 0.35"`

Combined with `top_topic_label`, `top_probability`, and the second-best fields, a downstream consumer has everything needed to render "why was this classified here?".

In production I'd add: per-topic coherence (c_v / c_npmi) alongside silhouette, drift monitoring on the match-time `is_new_topic` rate, and human-in-the-loop labels for ambiguous cases feeding a supervised reranker.

### 3.6 Observability

Structured JSON logging, correlation IDs, timing, and token accounting are wired through the whole pipeline via `topic_cluster/obs.py`.

- **Structured JSON logs on stdout.** Every log line has `ts`, `level`, `logger`, `msg`, `correlation_id`, plus any stage-specific extras (`elapsed_ms`, `n_threads`, `prompt_tokens`, `silhouette`, `top_probability`, …). Controlled by `LOG_LEVEL` and `LOG_FORMAT=json|text`.
- **Correlation IDs.** HTTP middleware assigns one per request (or honors an inbound `X-Correlation-Id`), propagates it through the request via `ContextVar`, and echoes it back in the response header. Every log line inside the request carries the same id.
- **Stage timing.** `timed_stage(logger, "stage.name", **ctx)` wraps each pipeline step (normalize, slack_rewrite, embed, bertopic_fit, save, matcher.embed, matcher.transform, …) and emits paired `start` / `ok|error` events with `elapsed_ms`.
- **Token accounting.** Every OpenAI call (Slack rewrite, OpenAI embeddings, LLM topic labels) records `prompt_tokens` and `completion_tokens` into a request-scoped tally. The `BuildResponse.token_usage` field breaks it down per call site (e.g. `slack_rewrite:gpt-4o-mini`, `embed:text-embedding-3-small`).
- **Cluster quality as a log event.** `cluster.quality` is emitted at end of fit with `n_topics`, `n_noise`, `silhouette`, `embedder` — so drift across rebuilds is observable without cracking open the artifacts.
- **Matcher verdict logging.** `matcher.verdict` logs every classification with `top_topic_id`, `top_probability`, `second_topic_id`, `second_probability`, `is_new_topic`, `is_ambiguous`.

---

## 4. API reference

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health` | `{status, model_loaded, embedder, n_topics, artifacts_dir}` |
| `GET`  | `/topics` | Full topic list with silhouette score |
| `POST` | `/build` | Rebuild model from inline raw payload |
| `POST` | `/build/from-path` | Rebuild model from a path on disk |
| `POST` | `/match` | Batch classify N raw messages |
| `POST` | `/match/single` | Classify one raw message (sugar over `/match`) |

Normalization is not a standalone endpoint — it runs as the first stage of `/build` and `/match`. The normalized payload is written to `artifacts/normalized_messages.json` on every build for inspection. Interactive docs at `http://localhost:8000/docs`.

### Build options (`BuildOptions` in `api/schemas.py`)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `embedder` | `"st" \| "openai"` | `"st"` | Embedding backend |
| `embed_model` | `str \| None` | `None` | Override model name for the backend |
| `label_with` | `"keywords" \| "openai"` | `"keywords"` | How to label discovered topics |
| `label_model` | `str` | `"gpt-4o-mini"` | LLM used when `label_with="openai"` |

`BuildResponse` additionally includes `token_usage` — a dict of `{label: {calls, prompt_tokens, completion_tokens, total_tokens}}` so you can see what the build cost.

### Lifespan and concurrency

- `lifespan` (`api/app.py`) calls `reload_matcher(state)` at startup, so the Matcher is loaded once and reused.
- `AppState` carries a `threading.Lock`. BERTopic's `transform()` isn't safe under concurrent calls on the same instance — the lock is held during each match and during the Matcher swap after a build.
- After any successful `/build` the in-process Matcher is hot-reloaded — no restart needed.

### Error mapping

- `ValueError` (unknown app, bad path) → **400**.
- `RuntimeError` (missing `OPENAI_API_KEY` where needed) → **503**.
- Matcher not loaded → **503** with a `detail` telling you to call `/build` first.
- Anything else → FastAPI default 500 (we don't swallow unknown exceptions).

### Sample payloads

Sample inputs live under `data/samples/` (they're inputs, not build outputs):

- `data/samples/sample_match_obvious.json` — Slack security CVE; expected to hit the security topic with high confidence.
- `data/samples/sample_match_alpha.json` — Gmail alpha launch status; expected to hit the alpha launch topic.
- `data/samples/sample_match_new.json` — Outlook cafeteria menu; expected to trip `is_new_topic=true`.

For a larger test set spanning all three apps and both matched + new-topic cases, see `data/new_conversation.json` (10 messages — 4 Slack, 3 Gmail, 3 Outlook). POST the whole array to `/match`:

```bash
curl -s -X POST localhost:8000/match \
  -H 'content-type: application/json' \
  -d "{\"messages\": $(cat data/new_conversation.json)}"
```

---

## 5. Artifacts on disk

After a successful build:

| File | Written by | Purpose |
|---|---|---|
| `normalized_messages.json` | normalize step | Canonical form of every raw message |
| `person_directory.json` | normalize step | Discovered people + role-mailbox flags |
| `bertopic_model/` | `TopicModel.save` | Full BERTopic model (safetensors + JSON config) |
| `topics.json` | `TopicModel.save` | Topic list: labels, keywords, sizes, representative threads, silhouette |
| `assignments.json` | `TopicModel.save` | `{thread_key: {topic_id, probability, app}}` — cross-app link data |

Test-input payloads live under `data/samples/` and `data/new_conversation.json`, not under `artifacts/`.

---

## 6. Design decisions (non-obvious)

Read before "fixing" anything that looks weird:

- **Never merge threads by canonical subject.** `Alpha Launch Checklist` legitimately spans 4+ distinct email threads; collapsing by subject destroys the cross-app linking signal. Canonical subject is embedding input only.
- **`thread_key = "{app}:{native_id}"`** even though native ids don't collide today. Defensive, costs nothing.
- **`parent_message_id = None` for Outlook.** Reconstructing reply chains from timestamps/subjects is fragile and out of scope.
- **Slack `participants = [sender_email]` only.** No recipient list exists for channels; don't synthesize one.
- **Slack `text` is LLM-rewritten; `original_content` is the verbatim audit trail.** Both fields exist on purpose. Fixed count/time windows don't work — the LLM picks what's topically relevant from a 48h / 20-message context window.
- **Cluster threads, not messages.** ~300 threads matches `min_topic_size=8`; messages alone over-fragment.
- **`min_topic_size=8`**, not BERTopic's default of 5. Tuned for the expected 10–12 topics in this corpus. See `docs/Clustering.md` §6.
- **BERTopic prob columns are `[noise, topic_0, topic_1, …]`**, not `[topic_0, …]`. `matcher.py` handles the off-by-one mapping.
- **No embedding cache today.** `TopicModel.fit` re-embeds every thread on each build. A content-addressed cache (SHA over `(thread_keys, texts, embedder_name)`) is planned.
- **Role mailboxes flagged via an explicit `ROLE_LOCAL_PARTS` set**, not heuristics.

---

## 7. Tradeoffs and limitations

### Known limitations

- **Thresholds tuned by eyeballing** the probability distribution on the bundled dataset. Production should tune against a labeled held-out set, per topic class.
- **`PersonDirectory` company domain is hardcoded** (`company.com`). Multi-tenant needs a per-tenant allowlist.
- **No Slack rewrite cache.** Every build pays for ~176 LLM calls.
- **No explicit `links.json`** — cross-app links are derived by grouping `assignments.json` on `topic_id`.

### What I'd do with more time

1. Explicit cross-app linker: emit `links.json` with `{topic_id, label, apps, thread_keys_by_app, participant_overlap, time_window}`, surfaced via `GET /links`.
2. Focused test suite: `test_normalizers.py`, `test_matcher_thresholds.py`, `test_person_directory.py`, plus an integration test hitting `/build/from-path` + `/match/single` end-to-end.
3. Per-topic coherence (c_v) alongside silhouette.
4. Upgrade `_suggest_label` for new topics to a small LLM call — it's currently a stopword-filtered token counter.
5. Supervised reranker trained on human labels for ambiguous assignments.

### Failure cases to be aware of

- Short Slack messages in noisy high-traffic channels — rewriter's 48h/20-message context window can get dominated by an unrelated topic, rewrite drifts, thread lands in the wrong cluster.
- Topics that only appear in one app — cluster is real, but cross-app linking degrades to single-app by construction.
- Topics with fewer than 8 threads — rolled into noise or a larger neighbor by `min_topic_size=8`. Surfacing them needs a lower threshold or a secondary pass.

---

## 8. Scaling

The current design is demo-quality. To go to production:

### 100K messages

- **Embeddings.** `sentence-transformers` on CPU runs ~1K sentences/sec; 100K is ~2 min, batchable. OpenAI scales horizontally via `asyncio` and the existing `BATCH_SIZE=100`; API cost dominates.
- **Clustering.** BERTopic handles 100K points on a modern laptop (UMAP is the bottleneck). Turn on `low_memory=True` at this size.
- **Slack rewrites.** ~$10–$20 at `gpt-4o-mini` rates and a few hours wall clock at the free-tier rate limit. First step is the missing per-`message_id` disk cache; then parallelize via `asyncio` + OpenAI async client.
- **Artifacts.** Replace JSON dumps with Parquet for the bigger files.

### 1M messages

- **Embeddings.** GPU worker or a managed embedding service. Persist in a vector DB (pgvector, Qdrant) keyed by thread so re-embedding on config changes is opt-in per thread.
- **Clustering.** BERTopic's river-based online mode — ingest incrementally instead of rebuilding. At this scale "rebuild the world" is not feasible.
- **Matcher.** Shared model server (one model, many stateless FastAPI workers via RPC). Current `threading.Lock` bottlenecks at one in-flight match per process.
- **Rewriter.** Background job queue (Celery / RQ / cloud queue), not synchronous at ingest.

### General production hardening

- **Observability.** Already in place — structured JSON logs, correlation IDs, token usage, stage timing. Next step: Prometheus metrics, distributed tracing, log shipping to an aggregator.
- **Caching.** (a) Slack rewrite cache keyed by `message_id`; (b) a request-level cache for `/match` keyed by `(app, message_id)` so replayed batches are free.
- **Batching.** `/match` is already batch; `Embedder.encode([...])` is much cheaper on 32 texts than on 32 separate calls. Expose to clients explicitly.
- **Auth and abuse.** API key auth, rate limiting per key, input size caps on `/build` and `/match`.
- **Path-traversal hardening.** `/build/from-path` accepts arbitrary filesystem paths. Fine locally; for anything exposed, restrict to a known `data/` prefix.
- **Secrets.** `OPENAI_API_KEY` is env-only and never logged. Use a secret manager in prod.

---

## 9. Repo layout

```
Th_cluster/
├── README.md
├── requirements.txt
├── .env                         # gitignored, holds OPENAI_API_KEY
├── data/
│   ├── conversations.json       # 598 messages / 300 threads — the build corpus
│   ├── new_conversation.json    # 10-message test set for /match (8 known topics + 2 new-topic cases)
│   └── samples/
│       ├── sample_match_obvious.json
│       ├── sample_match_alpha.json
│       └── sample_match_new.json
├── docs/
│   ├── Overview.md              # one-page summary — approach, BERTopic's role, extensibility
│   └── Clustering.md            # deep dive on clustering choices (§3.2 points here)
├── artifacts/                   # build outputs only — see §5
│   ├── bertopic_model/
│   ├── normalized_messages.json
│   ├── person_directory.json
│   ├── topics.json
│   └── assignments.json
└── topic_cluster/
    ├── schema.py                # Person, NormalizedMessage dataclasses
    ├── person.py                # PersonDirectory
    ├── normalizers.py           # SlackNormalizer, GmailNormalizer, OutlookNormalizer
    ├── slack_rewrite.py         # SlackRewriter
    ├── thread.py                # ThreadBuilder
    ├── embedders.py             # SentenceTransformerEmbedder, OpenAIEmbedder
    ├── cluster.py               # TopicModel (BERTopic wrapper)
    ├── matcher.py               # Matcher
    ├── obs.py                   # structured logging, correlation IDs, timing, token tally
    └── api/
        ├── app.py               # FastAPI() + lifespan + correlation middleware
        ├── config.py            # Settings
        ├── state.py             # AppState
        ├── dependencies.py      # DI
        ├── schemas.py           # Pydantic request/response models
        ├── services.py          # normalize_in_memory, run_build, reload_matcher
        └── routes/
            ├── health.py        # GET /health, GET /topics
            ├── build.py         # POST /build, POST /build/from-path
            └── match.py         # POST /match, POST /match/single
```
