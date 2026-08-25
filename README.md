# Task Master

Local-first job-search agent for South African job seekers. Type a natural-language request, get a ranked, explained shortlist of jobs. Milestone 0.

## Status

Milestone 0 scope, built per the approved design:

- CLI chat loop
- Natural-language request -> structured intent (Ollama + rule fallbacks)
- SA job search: bundled demo dataset (always available offline) plus the live DPSA Public Service Vacancy Circular
- Deterministic filtering + ranking with human-readable reasons for every match

The offline demo pipeline is always available, and `dpsa_circular` is currently **enabled** for South Africa (schema.org remains disabled). See below for how to point at a new circular.

Known limitation: the rule-based intent parser matches known roles and skills; arbitrary domain words such as "aerospace" are not automatically treated as keyword/skill filters unless the user edits the query.

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
> I'm a recent computer science graduate in Durban. Find me entry-level software
  engineering jobs, preferably remote or in Durban, and show me the best matches.

I understood:
  roles:         software engineer, software developer
  seniority:     entry-level
  locations:     Durban (50 km)
  remote:        preferred
  salary:        (not specified)
  skills:        computer science

[search / edit / quit]> search
```

Each result shows its match score, a line-by-line explanation, and the source link. Type a number to keep a job (saved to `data/kept_jobs.json`).

## Configuration

- `config/regions/za.json` — the only place with region facts (currency, locations + aliases, skills dictionary, enabled sources). A second country = a new JSON file, no code changes.
- `config.py` — paths and env vars: `OLLAMA_HOST`, `TASK_MASTER_MODEL`, `TASK_MASTER_REGION`, `TASK_MASTER_LLM_OFFLINE`.

## Live job sources

Each source is behind the same `JobSource` interface (`sources/base.py`) with its own tests, so a board changing its markup only breaks that one adapter.

### dpsa_circular (enabled)

The DPSA Public Service Vacancy Circular is currently **enabled** for South Africa and points to **Circular 27 of 2026**:

```
https://www.dpsa.gov.za/dpsa2g/documents/vacancies/2026/PSV%20CIRCULAR%2027%20of%202026.pdf
```

When a new circular is published, update `config/regions/za.json`:

- replace the `url` value with the new circular's PDF link,
- keep `"enabled": true`,
- preserve `"default_company": "DPSA / Government"`.

Robustness:

- Downloads use a **20-second HTTP timeout**; HTTP 4xx/5xx and network failures surface as per-source errors.
- A download/HTTP/PDF failure is **isolated to that source** and does not abort the rest of the search (`agent/search.py` catches `JobSourceError` per source).
- Tests use local fixtures and mocks and **never fetch the live DPSA PDF**.
- Salary handling is conservative: annual salaries are converted to monthly where appropriate; hourly or missing salaries are left unset rather than invented; multi-grade salary extraction keeps the entry (lowest) grade.

### schemaorg (disabled)

Not yet enabled. Set `search_url` to a job-board search page that embeds schema.org JobPosting JSON-LD (e.g. a za.indeed.com search URL), then set `enabled: true`.

## Tests

```
python -m pytest
```

All tests are offline — no network, no model required.

## Design principles

- Nothing is hard-coded: region facts live in config; the user's prompt is parsed into a structured query at runtime.
- Explanations are deterministic and visible — never a black-box match score.
- The pipeline is a set of small testable steps (`agent/`), so later milestones (CV intelligence, then submission automation) add steps without rewriting.
- Any future application submission will require explicit human approval before anything leaves the machine.

## Roadmap

- Milestone 1: CV upload/parsing, JD analysis, CV-to-job matching, tailored CV + cover letter drafts.
- Milestone 2: more sources, geocoding/radius, application watchlist.
- Milestone 3: web UI, auth, browser automation gated behind human approval, region generalisation.
