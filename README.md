# Task Master

AI Job Application Agent for South African job seekers. Upload your CV, describe what you want in natural language, and the agent finds, ranks, matches, prepares, and helps you submit applications.

## Status

Now a full AI agent, not just a job-search API:

- **Agent orchestrator** — state-machine controller managing the full workflow from CV parsing through application submission
- **CV upload/parsing** — extract structured profile from PDF CVs (deterministic + LLM)
- **Natural-language intent** — understand search, apply, review, approve, cancel commands
- **Job search** — bundled demo dataset + live DPSA Public Service Vacancy Circular
- **Candidate-job matching** — skill, experience, education, location, certification matching
- **Application preparation** — tailored cover letters, document generation, question resolution
- **Question engine** — answers known questions from profile, identifies unknowns, never fabricates
- **Application tracking** — persistent history, duplicate prevention, status management
- **Human approval** — no submissions without explicit user approval
- **Web UI** — agent-style interface with progress, matches, applications, and controls
- **CLI** — natural-language agent mode with search/apply/approve/cancel commands

## Setup

```
pip install -r requirements.txt
python cli.py
```

No accounts, no API keys, no paid services required.

### Optional: enable a local LLM (Ollama)

The pipeline works fully offline with built-in rule-based intent parsing. For better understanding, run a local model:

1. Install Ollama: https://ollama.com/download
2. Pull a model, e.g. `ollama pull qwen2.5:7b` (7B needs ~4-6 GB RAM; `qwen2.5:3b` is snappier on modest machines)
3. Run `python cli.py` — Task Master detects Ollama automatically.

If Ollama isn't running, you'll see a note and Task Master falls back to rules. Set `TASK_MASTER_LLM_OFFLINE=1` to force the fallback.

## Example

```
> Here is my CV. Find me suitable entry-level software development jobs
  in South Africa and apply to the best 5.

CV processed.

Your profile contains:
  Diploma/qualification
  Software development experience/projects
  Python, JavaScript, SQL, etc.

I found 42 jobs.

18 are strong matches.

I recommend these 5 applications:

  1. Junior Software Developer - Company A - 92%
  2. Graduate Developer - Company B - 89%
  3. Web Developer - Company C - 87%
  4. Software Developer Intern - Company D - 85%
  5. Junior Full Stack Developer - Company E - 83%

Preparing applications...

All applications are ready.

Review and approve submission.
> approve

Submitting...

5 applications submitted.
```

## Architecture

```
agent/
  orchestrator.py     Agent state machine — central controller
  parse_intent.py     NL intent parsing (search, apply, show, approve, cancel)
  rank.py             Job ranking with human-readable reasons
  search.py           Source registry + deduplication

application/
  models.py           Application, ApplicationStatus, MissingInfo
  question_engine.py  Answer resolution from profile + answer store
  cover_letter.py     Tailored cover letter generation
  form_filler.py      Application form adapter abstraction
  documents.py        Document generation (CV check, cover letter, summary)
  tracker.py          Persistent application tracking + duplicate prevention
  scoring.py          Application priority scoring (job + match + readiness)

candidate/
  profile.py          CandidateProfile with knowledge tracking (known vs unknown)
  cv_parser.py        PDF/text to CandidateProfile (deterministic + LLM)
  matching.py         Candidate-job matching + readiness assessment
  storage.py          Profile persistence

sources/
  base.py             Job dataclass + JobSource ABC
  demo.py             Bundled demo jobs (offline)
  dpsa_circular.py    DPSA Public Service Vacancy Circular parser
  schemaorg.py        Schema.org JobPosting JSON-LD parser

config/
  regions/za.json     Region facts: currency, locations, skills, sources

app.py                Flask web UI + agent API
cli.py                CLI agent interface
llm.py                Ollama abstraction
config.py             Configuration and paths
```

## Agent States

The agent progresses through these states:

```
RECEIVED -> UNDERSTANDING_REQUEST -> SEARCHING -> RANKING -> MATCHING
         -> SELECTING -> PREPARING_APPLICATION
         -> NEEDS_INFORMATION (if questions unanswered)
         -> AWAITING_APPROVAL
         -> SUBMITTING -> SUBMITTED/FAILED
         -> COMPLETED
```

## Agent Principles

1. **Never fabricate** candidate information
2. **Never submit** without explicit approval
3. **Never silently skip** required questions
4. **Always explain** why a job was selected
5. **Always preserve** application history
6. **Always prevent** duplicate applications
7. **Always distinguish** unknown from false information
8. **Always show** the agent's current state
9. **Prefer deterministic** code for business logic
10. **Keep components** modular and testable

## Web UI

```
python app.py
```

The web interface provides:
- CV upload and profile extraction
- Natural-language command box
- Agent progress and activity
- Job search results with match scores
- Application readiness assessment
- Application preview and approval
- Application status and history

## CLI

```
python cli.py
```

The CLI now supports natural-language agent commands:
- `"Find me developer jobs in Durban"`
- `"Apply to the best 5 jobs"`
- `"Apply to jobs where I have at least 80% match"`
- `"Show my applications"`
- `"Show applications that need my attention"`
- `"Approve all applications"`
- `"Cancel all applications"`

## Configuration

- `config/regions/za.json` — region facts (currency, locations + aliases, skills dictionary, enabled sources)
- `config.py` — paths and env vars: `OLLAMA_HOST`, `TASK_MASTER_MODEL`, `TASK_MASTER_REGION`, `TASK_MASTER_LLM_OFFLINE`

## Live job sources

Each source is behind the same `JobSource` interface (`sources/base.py`) with its own tests.

### dpsa_circular (enabled)

The DPSA Public Service Vacancy Circular is currently **enabled** for South Africa. Update the `url` in `config/regions/za.json` when a new circular is published.

### schemaorg (disabled)

Set `search_url` to a job-board search page that embeds schema.org JobPosting JSON-LD, then set `enabled: true`.

## Tests

```
python -m pytest
```

All 714 tests are offline — no network, no model required. Tests cover:
- Candidate profile, CV parsing, matching
- Intent parsing, job ranking, search
- DPSA circular parsing, schema.org parsing
- Application models, question engine, cover letters
- Application tracking, scoring, form filling
- Agent state transitions, CLI output
- Evaluation quality metrics

## Ranking quality

Measured against a 15,898-job South African public-sector corpus (DPSA vacancy
circulars) with 196 evaluation queries:

| Metric | Value |
|--------|-------|
| NDCG@10 | **0.470** |
| Naive keyword baseline NDCG@10 | 0.439 |
| Ranker advantage over baseline | +0.031 |
| Precision@3 | 0.366 |
| MRR | 0.550 |
| Intent-field accuracy | 0.894 |

Evaluation corpus: 196 queries, 15,898 jobs, 154 gold-labelled queries.
Full details in `evaluation/reports/national_analysis.md`.

## Design principles

- Nothing hard-coded: region facts live in config; user prompts parsed into structured queries at runtime.
- Explanations are deterministic and visible — never a black-box match score.
- Pipeline is small testable steps, so new capabilities add steps without rewriting.
- Any application submission requires explicit human approval.
- Candidate information is never invented — unknown fields are identified and requested.
