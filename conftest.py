from __future__ import annotations

from sources.base import Job


def make_valid_job(**overrides) -> Job:
    """A minimal job record that passes sources.validation checks.

    Used to stub *real* sources in tests: the live pipeline has no demo
    fallback, so tests must supply genuine-format records themselves.
    """
    data = dict(
        title="Graduate Software Developer",
        company="DVT",
        location="Remote (South Africa)",
        remote=True,
        description=(
            "Graduate development programme training C# and .NET for client "
            "projects. Fully remote within South Africa. Recent graduates "
            "encouraged to apply."
        ),
        url="https://www.dvt.co.za/opportunities/graduate-software-developer",
        source="dpsa_circular",
    )
    data.update(overrides)
    return Job(**data)
