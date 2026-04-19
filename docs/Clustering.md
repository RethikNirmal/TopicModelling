# Clustering — design notes

The "why" doc for the clustering pipeline. README covers how to run it; this one covers why the pieces look the way they do, and what I weighed against before committing.

Main files: `topic_cluster/cluster.py`, `topic_cluster/embedders.py`, `topic_cluster/thread.py`.

---

## 1. What the pipeline does

End-to-end for 300 threads:

```
Thread.text                                (str per thread)
   │  sentence-transformers or OpenAI embeddings
   ▼
embeddings ∈ ℝ^(300, d)          d = 384 (ST) or 1536 (OpenAI)
   │  UMAP(n_neighbors=15, n_components=5, metric=cosine)
   ▼
reduced ∈ ℝ^(300, 5)
   │  HDBSCAN(min_cluster_size=min_topic_size=8)   (inside BERTopic)
   ▼
cluster labels ∈ {-1, 0, 1, …, K-1}     (K discovered)
   │  c-TF-IDF across K super-documents
   ▼
per-cluster keywords
   │  (optional) LLM representation on keywords + representative docs
   ▼
human-readable label
```

`TopicModel.fit(threads)` in `cluster.py` runs it all. BERTopic holds the UMAP → HDBSCAN → c-TF-IDF chain; embeddings are computed outside and handed in, so the embedder is swappable without touching BERTopic's config.

Outputs:
- `artifacts/topics.json` — discovered topics (label, keywords, size, 3 representative threads each).
- `artifacts/assignments.json` — `{thread_key: {topic_id, probability, app}}`. This is the substrate for cross-app linking.
- `artifacts/bertopic_model/` — full BERTopic state, reloaded by `Matcher` at serve time.

---

## 2. Threads, not messages

I cluster threads, not individual messages. Messages alone are too noisy:

- Slack messages are short ("LGTM", "On it", "+1"); 34/176 are under 20 chars and 71/105 threads are singletons. Embed them directly and filler collapses into one giant cluster.
- Email replies inherit the thread's subject but often say nothing more than "Thanks" or "Approved". The topic lives in the chain, not in any one message.
- ~600 messages is a lot of near-duplicates; ~300 threads is the same content at ~5× the density per document.

`ThreadBuilder.build()` concatenates the messages in a thread (chronologically, per-app format) before embedding. Every downstream parameter (`min_topic_size`, UMAP `n_neighbors`) was picked assuming N ≈ 300 cohesive documents.

---

## 3. Sentence embeddings, not TF-IDF

Default: `all-MiniLM-L6-v2` (384 dims, local). Swap to OpenAI `text-embedding-3-small` (1536 dims).

I didn't use TF-IDF as the clustering input because it's keyword-based: two documents are close only if they share literal tokens. The assignment is built around cross-app vocabulary drift — "Project Alpha" in Slack, "Alpha Initiative" in email subjects, "the alpha project" in a DM. Under TF-IDF those are three different nouns and the cross-app similarity disappears on step one. Sentence embeddings are trained on paraphrase/NLI, so they place the three phrasings within a few cosine degrees of each other.

TF-IDF does show up later (§6), but for describing clusters, not discovering them.

On the two backends: the local model is the default because no API key is needed and 384 dims is plenty for 300 short docs. OpenAI embeddings were marginally tighter in A/B runs, not enough to make an API key mandatory. The embedder name (`st:…` or `openai:…`) is persisted into `topics.json` so the Matcher reloads the same vector space at inference time — otherwise the probabilities are noise.

---

## 4. Why reduce dimensions at all?

You can't cluster density in 384 dimensions. In high dims:

- Pairwise distances concentrate — nearest and farthest points are almost equidistant.
- k-NN queries (HDBSCAN relies on them) stop being meaningful.
- Local density approaches zero for any fixed radius, because volume scales as rᵈ.

So we reduce. The question is which projection.

---

## 5. UMAP, not PCA (or t-SNE)

UMAP with `n_neighbors=15`, `n_components=5`, `metric="cosine"`, `random_state=42`.

**Vs PCA.** PCA is linear; embedding manifolds aren't. PCA preserves global structure (the big directions of variance) but flattens local structure — and local structure is exactly what HDBSCAN needs. Near-duplicates within a topic end up further apart after the projection than they were originally, and density clustering on that output performs badly. Take 10 Alpha threads and 10 AWS-cost threads: PCA separates their means cleanly but smears the inside of each blob, so HDBSCAN sees stringy fragments, not two tight clusters.

UMAP's objective explicitly preserves "these points were neighbors in the input, they should still be neighbors in the output" — exactly what HDBSCAN consumes next. Fast on 300 points, deterministic with a fixed seed.

Parameter notes:
- `n_neighbors=15` — standard default. Small → fine structure but noisy; large → clusters smear together. 15 is the middle for N ≈ 300 with 10-ish expected topics.
- `n_components=5` — BERTopic's recommended target. 2 loses separability past a handful of clusters; 50+ puts the curse of dimensionality back.
- `metric="cosine"` — sentence embeddings are trained cosine. Euclidean drifts off-distribution.
- `random_state=42` — UMAP is stochastic by default; you don't want cluster ids jittering across rebuilds.

**Vs t-SNE.** Three strikes: doesn't preserve density (HDBSCAN needs it), non-reproducible even with a seed (KL divergence has many equivalent minima), O(N²) without Barnes-Hut so it doesn't scale.

---

## 6. HDBSCAN, not k-means

k-means is the first thing you reach for. For this problem it's wrong in four ways:

1. **Needs pre-specified k.** The number of topics is supposed to be *discovered*. If the real answer is 11 and you pick 10, two genuine topics fuse silently.
2. **No noise class.** Every point is forced into some cluster. A one-off "coffee machine broken" gets assigned to whichever centroid is least wrong. That's measurement error dressed up as data — and it kills new-topic detection at serve time, because the matcher never sees "this doesn't belong anywhere" during training.
3. **Assumes spherical equal-size clusters.** "Q2 OKR Planning" has 29 threads; "Acme Incident Postmortem" has 13. k-means with k=12 pulls centroids toward big clusters and fragments the small ones.
4. **Euclidean distance.** Sentence embeddings want cosine.

HDBSCAN is the opposite on every axis:

| Property | k-means | HDBSCAN |
|---|---|---|
| Picks k for you | ❌ pre-specified | ✅ emerges from density |
| Handles noise | ❌ everything assigned | ✅ `-1` label |
| Cluster shapes | ❌ spherical only | ✅ density-connected, any shape |
| Varying cluster sizes | ❌ biases to equal sizes | ✅ free parameter |
| Works with cosine / UMAP output | ✅ on any metric | ✅ yes |

The one knob that encodes domain knowledge is `min_cluster_size` (BERTopic's `min_topic_size`). I set it to **8**:

- Dataset has ~10–12 topics across ~300 threads → 25–30 threads per topic on average. 8 is well below, so real topics don't get rejected as noise.
- BERTopic's default (5) over-fragments — real topics get split into lookalike sub-clusters.
- Above ~12 starts merging smaller legitimate topics (e.g. Acme postmortem at 13 threads) into their neighbors.
- 8 is where validation stabilized across both embedder backends: silhouette ≈ 0.35, 12–13 topics, 11–13 noise threads.

**The `-1` noise label isn't a bug; it's the point.** Threads that land in noise are ones the model couldn't confidently place — exactly what the matcher flags as `is_new_topic=true` at inference time. Without it, new-topic detection has no basis.

**Vs DBSCAN.** DBSCAN needs a single `eps` and assumes uniform density across clusters. A tightly-worded email chain is denser in embedding space than a sprawling Slack channel on the same topic. HDBSCAN builds the hierarchy across all `eps` values and extracts the most stable clusters automatically — strict upgrade. Also, HDBSCAN emits soft probabilities that feed the matcher's confidence scoring.

**Vs GMM / agglomerative.** GMMs have the same pre-specified-k and Gaussian-cluster problems as k-means with a fancier likelihood. Agglomerative works but needs a cut threshold that plays the same role as `eps` — same hand-tuning problem.

---

## 7. c-TF-IDF — what it does here

TF-IDF earns its keep in this pipeline, but as a **describer** of clusters, not the clustering input.

Class-based TF-IDF treats each cluster as one super-document (concatenate every member's text) and asks which words are over-represented relative to the other super-documents. Words common inside cluster `c` but rare outside bubble up; generic words like "team" or "meeting" appear everywhere and get penalized. Roughly: `tf(w, c) * idf(w)` with `idf(w) = log(1 + A / f(w))`, where `A` is the average words per cluster and `f(w)` is `w`'s total frequency across clusters.

Used in two places:

1. **`topics.json.keywords`** — `TopicModel._build_topics_payload` pulls BERTopic's per-topic top-10 and writes them out. Served by `GET /topics`. This is the audit trail for why each cluster was identified the way it was.
2. **Default topic label** (`label_with="keywords"`) — `_label_for` joins the top 3 keywords with ` / `, giving things like `"security / deployment / vulnerability"`. Deterministic, no LLM call.

What c-TF-IDF is **not** used for: discovering clusters (sentence embeddings + UMAP + HDBSCAN do that) or matching new messages (`.transform()` on the same embedding space). Clustering wants semantic similarity; keyword extraction wants statistical distinctiveness. Different jobs.

---

## 8. LLM topic labels vs BERTopic's default

BERTopic's default label format is `{topic_id}_{top_four_keywords}`, e.g. `"6_security_deployment_vulnerability_patch"`. Fine for a demo; falls apart for anything readable:

- **No syntax.** Keywords are a bag of words. `"security / deployment / vulnerability"` tells you what the topic's *about* but not what's *happening* — is it shipping a patch or discovering a vulnerability in the deployment pipeline? Different topics, same keywords.
- **Stopwords leak.** Even with c-TF-IDF's weighting, sometimes a generic word survives. One of my early runs had `"may / offsite / to"` — the "to" is pure noise.
- **Can't disambiguate siblings.** If two clusters both have `"review"` and `"feedback"` in the top terms (performance review vs OKR review), the keyword labels look nearly identical. Representative docs disambiguate them; bag-of-words doesn't.

Hand-written labels solve all of this but don't survive a rebuild — cluster ids shift every time the dataset changes or parameters move. The maintenance burden is real.

LLM labels (`label_with="openai"`) use BERTopic's `OpenAI` representation model with a tight prompt:

```
I have a topic that contains the following representative documents:
[DOCUMENTS]

The topic is described by these keywords: [KEYWORDS]

Based on the documents and keywords, give a short topic label
(2-6 words, Title Case, no quotes, no trailing punctuation) that
captures what this topic is about. Reply with the label only.
```

BERTopic fills `[DOCUMENTS]` with 4 representative threads (up to 300 chars each) and `[KEYWORDS]` with the c-TF-IDF top-10. Sample output vs the keyword form:

| Keyword label | LLM label |
|---|---|
| `security / deployment / vulnerability` | `Security Patch Deployment and Vulnerability Remediation` |
| `okr / objectives / objective` | `Q2 OKR Planning and Alignment` |
| `alpha / launch / checklist` | `Project Alpha Launch Status Update` |
| `review / performance / next` | `H1 Performance Review Process` |

Cost: ~1 `gpt-4o-mini` call per topic, 12–13 per build, fraction of a cent. Much cheaper than the Slack rewriter's 176 calls per build.

**Why layer the LLM on top of c-TF-IDF instead of asking it from scratch:**
- Keywords are a strong prior. Without them, labels drift toward generic summaries ("A team discussion about projects").
- The statistical layer is a sanity check. If the LLM's label doesn't reference any of the top-10 keywords, either the cluster is incoherent or the model hallucinated. Worth surfacing either way.
- Graceful degradation. The default path (`label_with="keywords"`) runs without network access or credentials. LLM mode is an upgrade, not a dependency.

`_label_for` has a fallback chain: LLM representation if configured and present, else the keyword join, else `topic_{tid}`. Labels are never null.

---

## 9. Quality — silhouette score

Silhouette per point `i` is roughly `(b - a) / max(a, b)`, where `a` is mean intra-cluster distance and `b` is mean nearest-other-cluster distance. Runs from -1 (wrong cluster) through 0 (boundary) to +1 (tight, well-separated). On this dataset it sits around 0.35 with 12 topics and ~11 noise threads, which says the topics are real and separation is meaningful. Near 0 would mean random-looking groupings.

Three implementation choices:

- **Computed on the full embedding, not the UMAP output.** UMAP isn't isometric; silhouette on the 5d output would measure UMAP's faithfulness as much as actual cluster quality. Raw embedding is the honest ground truth.
- **Cosine metric.** Same reasoning as UMAP.
- **Non-noise only.** `-1` isn't a cluster. Including it would penalize the silhouette for noise points, which is double-counting — noise rate is already a separate metric.

Surfaced in `/topics`, `/health`, and the `cluster.quality` structured log event, so drift across rebuilds is visible without cracking open the artifacts.

---

## 10. Rejected alternatives, summarized

| Choice | Rejected | Why |
|---|---|---|
| Cluster threads | Cluster messages | Too noisy; short Slack msgs and empty email replies collapse into filler |
| Sentence embeddings | TF-IDF as input | Can't handle cross-app vocabulary drift — the exact signal tested |
| UMAP → 5d | PCA | Linear; destroys local structure; HDBSCAN breaks downstream |
| UMAP → 5d | t-SNE | Doesn't preserve density; non-reproducible; doesn't scale |
| HDBSCAN | k-means | Pre-specified k; no noise class; spherical equal-size assumption |
| HDBSCAN | DBSCAN | Single `eps`; uniform density; hard assignments |
| HDBSCAN | GMM / agglomerative | Gaussian assumption / manual cut threshold |
| c-TF-IDF for keywords only | TF-IDF for clustering | Great at describing, terrible at discovering |
| LLM labels (optional) | BERTopic default `k_w1_w2_w3_w4` | Unreadable; stopwords leak; can't disambiguate siblings |
| LLM on top of c-TF-IDF | Raw LLM labels | Keyword prior prevents generic phrasing; doubles as a sanity check |
| Silhouette on full embedding | Silhouette on UMAP output | Would measure UMAP faithfulness, not cluster quality |
| `min_topic_size=8` | BERTopic default (5) | Over-fragments real topics |
| `min_topic_size=8` | 12+ | Merges legitimately smaller topics (e.g. Acme at 13 threads) |

---

## 11. At inference time

`Matcher.match(raw)` — what `POST /match` calls — runs the pipeline in reverse:

1. Normalize through the same `NormalizerRegistry` used at fit time.
2. Embed with the same embedder (`embedder.encode([text])`).
3. `model.transform([text], embeddings=…)` — BERTopic projects through the fit-time UMAP, asks HDBSCAN for membership via `approximate_predict`, returns a probability row shaped `[noise, topic_0, topic_1, …]`.
4. Apply thresholds: top prob < 0.35 → new topic; top − second < 0.15 → ambiguous.

Fit-time consistency matters because the inference path literally re-executes the training-time UMAP and HDBSCAN. Any mismatch (wrong embedder reloaded, different seed, different metric) silently shifts the probability mapping and misclassifies everything. The matcher reads the embedder name from `topics.json` for exactly this reason — it doesn't trust its own default.

---

## 12. Guardrails if you change things

- **Don't change `random_state`.** Reproducibility across builds matters for comparing silhouette over time.
- **Keep the UMAP metric and the silhouette metric the same.** If UMAP is cosine, silhouette has to be cosine too. Otherwise you're measuring inconsistent geometries.
- **Swap the embedder → cluster ids change.** They're emergent, not stable. The Matcher reloads the embedder name from `topics.json` to stay consistent.
- **Lower `min_topic_size` → more topics, slightly lower silhouette.** Not necessarily worse — depends on whether the new sub-clusters are genuine. Read the labels.
- **Silhouette collapses toward 0?** Clusters are no longer well-separated. Inspect c-TF-IDF keywords per cluster — two fused topics usually leave an obvious footprint.
