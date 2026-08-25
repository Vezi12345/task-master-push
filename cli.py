from __future__ import annotations

import json
import sys

import config
import llm as llm_module
from agent import orchestrator
from agent.rank import RankedJob

BANNER = """
Task Master  ·  AI Job Application Agent for South Africa
type "help" for tips, "quit" to exit
"""

HELP = """
Natural language commands:
  "Find me entry-level software developer jobs in Durban"
  "Apply to the best 5 jobs"
  "Apply to jobs where I have at least 80% match"
  "Show my applications"
  "Show applications that need my attention"
  "Approve application abc123"
  "Cancel application abc123"
  "Approve all applications"
  "Cancel all applications"

Search shortcuts:
  search   run the search and ranking (legacy mode)
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

    from agent.orchestrator import JobApplicationAgent

    agent = JobApplicationAgent(region, llm)

    prompt = _prompt()
    while prompt not in ("quit", "exit", "q"):
        if prompt in ("help", "?"):
            print(HELP)
            prompt = _prompt()
            continue

        if prompt in ("search", "s", "go", "run"):
            _legacy_search(region, llm)
            prompt = _prompt()
            continue

        result = agent.process_input(prompt)

        for msg in result.messages:
            if msg.role == "agent":
                print(f"\n{msg.content}")

        if result.error:
            print(f"\n  Error: {result.error}")

        if result.ranked:
            _print_ranked(result.ranked)

        if result.matched_jobs:
            _print_matched(result.matched_jobs)

        if result.applications:
            _print_applications(result.applications)

        if result.missing_information:
            print("\n  Please provide the missing information above.")

        if result.state and result.state.value == "awaiting_approval":
            print("\n  Reply 'approve' to submit, or 'cancel' to abort.")

        prompt = _prompt()
    print("Bye.")


def _prompt() -> str:
    try:
        return input("\n> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        sys.exit(0)


def _legacy_search(region, llm) -> None:
    prompt = input("Enter search query: ").strip()
    if not prompt:
        return
    result = orchestrator.run_pipeline(prompt, region, llm)
    for note in result.notes:
        print(f"  note: {note}")
    _print_understood(result.query)
    print()
    command = input("[search / edit / quit]> ").strip().lower()
    if command in ("search", "s", "go", "run"):
        jobs, messages = orchestrator.search_jobs(result.query, region)
        ranked = orchestrator.rank_jobs(jobs, result.query, llm)
        for message in messages:
            print(f"  searching: {message}")
        print(f"  found {len(jobs)} jobs -> {len(ranked)} after filtering.\n")
        if ranked:
            _print_ranked(ranked)
    elif command in ("quit", "exit", "q"):
        print("Bye.")
        sys.exit(0)


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


def _print_ranked(ranked: list) -> None:
    print("\nBest matches:\n")
    for idx, item in enumerate(ranked, start=1):
        job = item.job
        location = job.location or "location not stated"
        remote_note = "Remote" if job.remote else "On-site"
        print(f" #{idx}  {job.title} - {job.company} | {location} ({remote_note})  {item.score}%")
        for reason in item.reasons:
            print(f"     {reason}")
        print(f"     -> {job.url}" if job.url else "     -> (no link available)")
        print()


def _print_matched(matched: list) -> None:
    print("\nCandidate matches:\n")
    for idx, item in enumerate(matched, start=1):
        job = item["job"]
        rank = item["rank"]
        match = item["candidate_match"]
        readiness = item["readiness"]
        location = job.location or "location not stated"
        print(f" #{idx}  {job.title} - {job.company} | {location}")
        print(f"     Job preference: {rank.score}%  |  Candidate match: {match.score}%  |  Readiness: {readiness.score}%")
        if match.matched_skills:
            print(f"     Matched: {', '.join(match.matched_skills[:5])}")
        if match.missing_skills:
            print(f"     Missing: {', '.join(match.missing_skills[:5])}")
        if match.strengths:
            for s in match.strengths:
                print(f"     + {s}")
        if match.concerns:
            for c in match.concerns:
                print(f"     - {c}")
        print(f"     -> {job.url}" if job.url else "     -> (no link available)")
        print()


def _print_applications(applications: list) -> None:
    print("\nApplications:\n")
    for idx, app in enumerate(applications, 1):
        preview = app.to_preview() if hasattr(app, "to_preview") else app
        status = preview.get("status", "unknown")
        print(f" #{idx}  {preview.get('role', 'N/A')} - {preview.get('company', 'N/A')} | {preview.get('location', 'N/A')}")
        print(f"     Status: {status}  |  Priority: {preview.get('application_priority', 0)}%")
        if preview.get("warnings"):
            for w in preview["warnings"]:
                print(f"     Warning: {w}")
        if preview.get("errors"):
            for e in preview["errors"]:
                print(f"     Error: {e}")
        if preview.get("documents", {}).get("cover_letter_ready"):
            print("     Document: Cover letter ready")
        if preview.get("documents", {}).get("cv_ready"):
            print("     Document: CV ready")
        print()


if __name__ == "__main__":
    main()
