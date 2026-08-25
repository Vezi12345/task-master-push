from __future__ import annotations

import json
import sys

import config
import llm as llm_module
from agent import orchestrator
from agent.rank import RankedJob

BANNER = """
Task Master  ·  local-first job search for South African job seekers
type "help" for tips, "quit" to exit
"""

HELP = """
Tips:
  Say what you want naturally, e.g.:
    "I'm a recent computer science graduate in Durban. Find me entry-level
     software engineering jobs, preferably remote or in Durban, and show me
     the best matches."
  Add filters: "paying at least R25k", "within 50 km", "fully remote", "onsite".
  After I show matches, type a job number to keep it (saved to data/kept_jobs.json).
Commands:
  search   run the search and ranking
  edit     correct what I understood (add your fix to the request)
  more     show the ranked list again
  help     show this help
  quit     exit
"""


def _configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def main() -> None:
    _configure_console()
    print(BANNER)
    config.ensure_data_dir()
    region = config.load_region()
    llm = llm_module if not llm_module.LLM_OFFLINE else None

    prompt = _prompt()
    while prompt not in ("quit", "exit", "q"):
        if prompt in ("help", "?"):
            print(HELP)
            prompt = _prompt()
            continue

        extra = ""
        while True:
            if extra:
                combined = f"{prompt} {extra}".strip()
            else:
                combined = prompt
            result = orchestrator.run_pipeline(combined, region, llm)
            _print_notes(result.notes)
            _print_understood(result.query)
            command = input("[search / edit / quit]> ").strip().lower()
            if command in ("search", "s", "go", "run"):
                _run_search_and_rank(result, region, llm)
                break
            if command in ("edit", "e"):
                extra = input("Corrections (e.g. 'salary at least R25k', 'Durban and Cape Town'): ").strip()
                continue
            if command in ("quit", "exit", "q"):
                print("Bye.")
                return
            print("Unknown command. Try 'search', 'edit', or 'quit'.")

        prompt = _prompt()
    print("Bye.")


def _prompt() -> str:
    try:
        return input("\n> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        sys.exit(0)


def _print_notes(notes: list[str]) -> None:
    for note in notes:
        print(f"  note: {note}")


def _print_understood(query) -> None:
    print("\nI understood:")
    print(f"  roles:         {', '.join(query.roles) if query.roles else '(not specified)'}")
    print(f"  seniority:     {query.seniority or '(not specified)'}")
    if query.locations:
        print(f"  locations:     {', '.join(f'{loc.city} ({loc.radius_km} km)' for loc in query.locations)}")
    else:
        print("  locations:     (not specified)")
    remote_map = {"any": "any", "preferred": "preferred", "required": "required", "no": "on-site only"}
    print(f"  remote:        {remote_map.get(query.remote, query.remote)}")
    if query.min_salary:
        print(f"  salary:        at least {query.currency} {query.min_salary:,}")
    else:
        print("  salary:        (not specified)")
    print(f"  skills:        {', '.join(query.skills) if query.skills else '(not specified)'}")


def _run_search_and_rank(result, region, llm) -> None:
    jobs, messages = orchestrator.search_jobs(result.query, region)
    ranked = orchestrator.rank_jobs(jobs, result.query, llm)
    for message in messages:
        print(f"  searching: {message}")
    print(f"  found {len(jobs)} jobs -> {len(ranked)} after filtering.\n")
    if not ranked:
        print("  No jobs matched. Try 'edit' to loosen your requirements.")
        return
    _print_ranked(ranked)
    _keep_loop(ranked)


def _print_ranked(ranked: list[RankedJob]) -> None:
    print("Best matches:\n")
    for idx, item in enumerate(ranked, start=1):
        job = item.job
        location = job.location or "location not stated"
        remote_note = "Remote" if job.remote else "On-site"
        print(f" #{idx}  {job.title} — {job.company} · {location} ({remote_note})  {item.score}%")
        for reason in item.reasons:
            print(f"     {reason}")
        print(f"     → {job.url}" if job.url else "     → (no link available)")
        print()


def _keep_loop(ranked: list[RankedJob]) -> None:
    while True:
        choice = input("[number to keep · 'more' · 'edit' · 'quit']> ").strip().lower()
        if choice in ("quit", "exit", "q"):
            return
        if choice in ("more", "m", ""):
            _print_ranked(ranked)
            continue
        if choice == "edit":
            return
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(ranked):
                _keep_job(ranked[index].job)
            else:
                print(f"  no job #{choice}")
            continue
        print("  type a job number, 'more', or 'quit'")


def _keep_job(job) -> None:
    payload = {
        "title": job.title,
        "company": job.company,
        "location": job.location,
        "remote": job.remote,
        "salary_text": job.salary_text,
        "url": job.url,
        "source": job.source,
    }
    records = _load_kept()
    records.append(payload)
    config.KEPT_JOBS_FILE.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  kept '{job.title} at {job.company}' -> {config.KEPT_JOBS_FILE}")


def _load_kept() -> list[dict]:
    if not config.KEPT_JOBS_FILE.exists():
        return []
    try:
        return json.loads(config.KEPT_JOBS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


if __name__ == "__main__":
    main()
