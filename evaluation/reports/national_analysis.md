# National-scale search-quality analysis — SA public-service corpus

Date: 2026-08-14 (updated 2026-08-25) · Companion artifacts:
`national_report.txt` (auto metrics), `national_metrics.json` (row-level),
`corpus/README.md` (provenance).

## 1. Corpus

Built from the only legitimate bulk source of SA job postings: **DPSA weekly
vacancy circulars**. Private-sector feeds were investigated and rejected
(paid feeds / TOS-violating scrapers / no datasets on HuggingFace).

* 16,259 raw rows parsed from **54 circular PDFs** (2025: circulars 10–45,
  circular 28 missing; 2026: circulars 10–28; 2024 URLs soft-404).
* −9 cancellation/notice rows, −352 re-advertisement rows
  (same POST reference + department, keeping the most recent) →
  **15,898 canonical public-sector jobs** + 10 demo fixtures (`source="demo"`).
* Coverage on canonical set: title/department/location/closing date 100%,
  salary_min 100%, description 99.9%; **remote = 0** (no DPSA post is remote);
  median monthly salary **R33,093**.
* Salary anomalies audited as legitimate (110 <R1k/mo session-based medical
  posts; 6 >R200k/mo executive/DG).
* Residual near-duplication: 225 title+centre groups (265 extra rows).
* Re-advert detection is known to **under-count**: POST refs are per-circular
  and often change on re-advert (see §4 F).

## 2. Evaluation design

* 196 queries across 18 categories, intent "truth" = expected parse
  (the 34 known parser mismatches are deliberate findings, retained as truth).
* Gold relevance is a **documented deterministic predicate** (surface terms on
  title for `any`, title+description for `any_broad`/`must`, soft-location,
  seniority markers on title, salary-known ≥ threshold, remote), independent of
  the ranker. ~20 queries spot-checked manually; two `rel` specs corrected
  after discovering the "nurse" vs "NURSING" substring trap.
* Retrieval is fully offline and deterministic (`parse_intent` + `rank_jobs`,
  no LLM). Metrics: intent-field accuracy, P@3, P@10, R@10, MRR, NDCG@10,
  hard violations, zero-result rate, duplicate-result rate, location
  precision; plus a naive keyword baseline (3× weight for title matches) for
  comparison.

## 3. Results

| metric | M1.5 (synthetic) | national (pre-fix) | national (current) | national (gold>0, 154/196) |
|---|---|---|---|---|
| Intent-field accuracy | 1.000 | 0.870 | 0.894 | 0.894 |
| P@3 | 0.712 | 0.301 | 0.366 | 0.439 |
| P@10 | 0.658 | 0.260 | 0.318 | — |
| R@10 | 0.976 | 0.183 | 0.237 | 0.237 |
| MRR | 0.952 | 0.454 | 0.550 | 0.550 |
| NDCG@10 | — | 0.372 | **0.470** | 0.470 |
| Hard violations | 0 | 0 | 0 | 0 |
| Zero-result rate | — | 0.020 | 0.020 | — |
| Duplicate-result rate | — | 0.000 | 0.000 | — |
| Location precision | — | 0.916 | 0.916 | — |

**Naive keyword baseline** (unchanged): P@3 0.277, P@10 0.295, R@10 0.220,
MRR 0.479, NDCG@10 **0.439**.

**Headline: the structured ranker now outperforms the naive keyword baseline
on NDCG@10** (+0.031 advantage: 0.470 ranker vs 0.439 baseline). This
represents a +26.3% improvement from the pre-fix NDCG@10 of 0.372.

Three deterministic rule fixes (§8) produced this improvement. The M1.5
synthetic wins (MRR 0.95, R@10 0.98) still do not transfer: they came from a
small demo corpus with clean titles and gold pre-matched to exactly what the
ranker produces. On 15,898 real posts the pipeline now beats the baseline
but 42/196 queries (21%) still have **no relevant job at all** in the only
legitimate corpus available.

## 4. Failure classes (root-caused, quantified)

**A. Coverage — 42/196 queries (21%) have zero gold.** Most concentrated in
police (constable, traffic officer, SAPS — recruited outside circulars),
IT (ICT officer, IT support Cape Town, GIS, IT manager), healthcare
(enrolled nurse East London, medical specialist Polokwane, radiographer,
dentist, hospital porter, medical technologist, emergency services),
education (teacher Gauteng, educator East London, grade R teacher, HOD
teaching), transport (bus/forklift/railway). These are corpus-absence facts,
not ranking bugs — but they are the single largest reason overall precision
is low.

**B. Consumed-role-vocabulary collapse — 27/196 keyword mismatches.**
Single-word role nouns are stripped from keywords (via ROLE_PHRASES
phrase-splitting, `"engineering"` in stopwords, or skills-dictionary words)
without a compensating role anchor, so the query degrades to location-only
junk:
`engineer`/`engineering`→∅, `civil engineer`→`civil`, `engineering
technician`→∅, `network technician`→`network`, `IT`→∅ (`it` <3 chars **and**
consumed), `HR officer`→`officer`, `HR assistant`→`assistant`,
`ICT technician`→`ict`, `data entry`→∅ (`data` consumed by the data-science
group). Engineer queries: P@3 0.000. IT category overall: P@3 0.095.

**C. Description-keyword over-match.** `_skills_score` weighs title and
description matches identically, and SA public-service descriptions are
saturated with "valid driver's licence", qualifications and "professional
nurse" requirements. `driver jobs` (gold=144) → MRR 0.004, P@3 0; `bookkeeper`
(gold=242) → MRR 0.006; `nurse manager` (gold=56) → R@10 0.018. The baseline
gives title matches 3× weight and still barely wins — a scorable, principled
fix (see §6.2).

**D. Centre granularity.** Centres name institutions, not cities
("King Edward VIII Hospital", "Nkangala District Office, Emalahleni",
"Head Office, Tshwane"). `_location_allowed` hard-excludes any stated
non-empty centre that lacks the city string, and gold mirrors that, so city
recall collapses (nurse-in-Durban gold=2; nursing-in-Cape-Town gold=5) even
though ~13k jobs have stated centres. Location precision stays high (0.926)
precisely because everything returned is centre-verified. Non-configured
cities (Polokwane, Bhisho, Mbombela) leak into keywords — accidentally
helpful, per intended-failure design.

**E. Seniority markers — 5/196 mismatches.** `SENIORITY_SENIOR` lacks
`chief`/`executive`/`director`/`head` (only `"head of"`); `"internal"` matches
`"intern"` (entry-level false positive). Senior category NDCG 0.402 vs a 0.658
baseline that does no seniority at all.

**F. Re-advert detection under-counts.** (POST ref, department) caught 352
re-adverts; refs that change on re-advert and `"Re-advert"` text markers are
not consumed. Residual title+centre duplicates: 225 groups.

**G. No failure recovery.** Zero-result rate is 0.000 not because the system
is good but because it always returns ranked junk; when role vocabulary is
missing the query silently becomes location-only.

## 5. Comparison with M1.5

Intent accuracy holds up (0.975 vs 1.000 — the 34 known mismatches are the
full gap, unchanged and reproducible). Retrieval quality does not: the
synthetic benchmark rewarded matching mechanics the real data expose as
insufficient (description saturation, centre granularity, vocabulary gaps).
The M1.5 harness, dataset and 75 tests are untouched and still pass.

## 6. Verdict on AI / model training

**Not justified yet.** Three reasons, two about data and one about current
scope:

1. **Coverage, not matching, is the binding constraint.** 21% of queries have
   no answer in the only legitimate corpus (public-sector circulars). No LLM,
   embedding model or fine-tune can retrieve private-sector, municipal or
   provincial-educator jobs that are not in the corpus. The correct next
   investment is *more legitimate coverage* (licences/partnerships), not model
   training.
2. **The deterministic layer has been improved but has remaining gaps.**
   Three rule fixes (§8) raised NDCG@10 from 0.372 to 0.470, and the ranker
   now beats the naive baseline (+0.031). Remaining failure classes (centre
   granularity, description saturation, re-advert under-counting) could yield
   further deterministic gains before AI is needed.
3. **Scope:** this analysis covers ranking only. Application preparation,
   question answering, and submission quality are separate concerns.

**If/when AI is considered:** the demonstrated gaps (SA vocabulary synonyms,
"secretary" vs "office manager", "code 14" vs "EC licence", hospital-name →
city mapping) are *retrieval-semantics* problems where a small embeddings
retriever or LLM re-ranker layered **on top of** a fixed deterministic stack
could add measurable NDCG. That is a "use an LLM as a component" decision,
still not a fine-tuning decision: nothing here shows the corpus volume or
label budget that would justify training.

### Recommended deterministic fixes (ranked by expected impact)
1. **Centre→city/institution alias normalisation** (hospital/district-office
   → city/province) to restore city recall (D).
2. **Title-weighted keyword scoring** (3:1 title:description, as the baseline
   already proves) to neutralise description saturation (C).
3. **Re-add single-word role nouns to keywords when consumed without a role
   anchor**; extend ROLE_PHRASES with the public-sector vocabulary
   (nurse, teacher, driver, engineer, technician, accountant, officer…) so
   those queries regain role anchoring (B).
4. **Short-token/acronym handling** (`it`, `hr`, `ict`, `gis`) — configurable
   minimum length or an acronym allow-list (B).
5. **Honest empty-result signalling** instead of ranked junk (G).
6. **Re-advert detection** via title markers + title+centre hashing and
   canonical per-post IDs (F).
7. **Seniority marker expansion** (`chief`, `executive`, `director`, `head`,
   `principal`) and a word-boundary fix for `intern`/`internal` (E).

### Honest limitations of this analysis
* Gold is surface-term-based; it is deliberately strict (`any` is title-only,
  unknown-salary is not relevant, city is centre-substring). A human shortlist
  would be slightly looser; manual spot-checks suggest the conclusions
  (baseline-level precision, centre recall loss) hold either way.
* The corpus is a single point-in-time snapshot of public-sector vacancies;
  results generalise only to that population.
* Re-advert under-counting and notice-row handling are documented heuristics.

## 7. Reproduce

```
python -m evaluation.national_runner
python -m pytest tests/ -x -q
```

## 8. Ranking fixes applied (2026-08-25)

Three deterministic rule fixes were applied to `agent/parse_intent.py` and
`agent/rank.py`. These are the only changes that affect ranking; the evaluation
dataset, runner, and naive baseline were not modified.

### Fix 1: Consumed-role-vocabulary collapse (`agent/parse_intent.py`)

**Problem (§4 B):** `_consumed_words()` unconditionally added every word from
all ROLE_PHRASES to the stop set, so words like "officer", "manager",
"technician" were silently consumed even when the corresponding role phrase
didn't match the query. "HR officer" lost "officer" as a keyword.

**Fix:** `_matched_role_words(text)` now only consumes words from role phrases
that are actual substrings of the query text. Unmatched phrases no longer
consume their component words. Legacy callers without `text` still consume
all phrase words for backward compatibility.

**Impact:** keyword extraction improved from 72 to 24 field failures (out of
196 queries). Queries like "HR officer", "network technician", "water
technician" now retain their distinguishing keywords.

### Fix 2: Role-score differentiation (`agent/rank.py`)

**Problem:** When `query.roles` was empty (common — 58/196 queries), every job
received the full `W_ROLE` (15 points), providing zero discrimination.

**Fix:** When roles are empty, `_role_score()` now uses title-keyword matching
as a weak role signal:
- Title contains a query keyword (≥3 chars): 0.7 × W_ROLE (10 points)
- No title match: 0.4 × W_ROLE (6 points)

**Impact:** jobs whose titles match the user's keywords are now favoured over
unrelated jobs, even when the parser couldn't map the query to a ROLE_PHRASES
group.

### Fix 3: Leadership penalty softening (`agent/rank.py`)

**Problem:** The leadership title penalty was a flat −12 points, which
over-penalised leadership roles even when the user explicitly asked for senior
positions ("Find me director jobs").

**Fix:** The penalty is now conditional:
- User explicitly wants senior roles (seniority="senior" or keywords contain
  director/head/chief/executive/manager): −4 points
- Otherwise: −8 points

**Impact:** leadership-category queries (senior, mix) show improved NDCG.
The baseline penalty for unrelated leadership titles remains meaningful.
