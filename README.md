# Task Master

Local-first job-search agent for South African job seekers. Type a natural-language request, get a ranked, explained shortlist of jobs. Milestone 0.

## Status

Milestone 0 scope, built per the approved design:

- CLI chat loop
- Natural-language request -> structured intent (Ollama + rule fallbacks)
- SA job search (bundled demo dataset now; DPSA + schema.org sources ready but disabled)
- Deterministic filtering + ranking with human-readable reasons for every match

Live job sources are intentionally **disabled by default**. The intent -> search -> rank -> explain pipeline is proven against the offline demo dataset first. Enable live sources only when you're ready (see below).

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

## Enabling live job sources (next milestone step)

When the offline pipeline passes its tests to your satisfaction, enable sources one at a time in `config/regions/za.json`:

1. `dpsa_circular` — set `url` to the current week's DPSA Public Service Vacancy Circular PDF link, set `enabled: true`.
2. `schemaorg` — set `search_url` to a job-board search page that embeds schema.org JobPosting JSON-LD (e.g. a za.indeed.com search URL), set `enabled: true`.

Each source is behind the same `JobSource` interface (`sources/base.py`) with its own tests, so a board changing its markup only breaks that one adapter.

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
