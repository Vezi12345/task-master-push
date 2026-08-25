from __future__ import annotations

"""Offline search-quality evaluation dataset.

Each entry describes a realistic South African job-search query plus the
intent a correct system should extract and the jobs (by ``(source, title)``
reference into the evaluation corpus) a reasonable human would shortlist.

Fields of ``expected``:
  roles       - expected role group(s), a subset of what the parser emits
  seniority   - expected seniority string ("" if unspecified)
  locations   - expected city names (subset of what the parser emits)
  remote      - expected remote preference ("any"|"preferred"|"required"|"no")
  min_salary  - expected minimum monthly salary, or None
  skills      - expected skills from the region skills_dictionary
  keywords    - expected meaningful free-text domain keywords

``hard`` (optional) declares constraints that must NEVER be violated in the
top results: ``remote="required"`` and/or ``min_salary=<int>``. Location is
treated as a soft preference by design (remote and location-less jobs are
admitted), so it is not a hard constraint.

``gold`` references resolve against the corpus built by
:func:`evaluation.runner.build_corpus` via ``(source, title)`` tuples.
"""

D = lambda title: ("demo", title)
C = lambda title: ("dpsa_circular", title)

QUERIES: list[dict] = [
    # ---------------------------------------------------------------- software
    {
        "category": "software",
        "query": "Find me software engineering jobs.",
        "expected": {
            "roles": ["software engineer", "software developer"],
            "seniority": "",
            "locations": [],
            "remote": "any",
            "min_salary": None,
            "skills": [],
            "keywords": [],
        },
        "gold": [
            D("Junior Software Developer"),
            D("Software Engineer (Graduate Programme)"),
            D("Graduate Software Developer"),
            D("Software Developer"),
            D("Junior Web Developer"),
            D("Senior Software Engineer"),
        ],
    },
    {
        "category": "software",
        "query": "Find me software engineering jobs in Johannesburg.",
        "expected": {
            "roles": ["software engineer", "software developer"],
            "seniority": "",
            "locations": ["Johannesburg"],
            "remote": "any",
            "min_salary": None,
            "skills": [],
            "keywords": [],
        },
        "gold": [
            D("Senior Software Engineer"),
            D("Junior Web Developer"),
        ],
    },
    {
        "category": "software",
        "query": "Find me remote software engineering jobs, preferably in Cape Town.",
        "expected": {
            "roles": ["software engineer", "software developer"],
            "seniority": "",
            "locations": ["Cape Town"],
            "remote": "preferred",
            "min_salary": None,
            "skills": [],
            "keywords": [],
        },
        "gold": [
            D("Junior Software Developer"),
            D("Graduate Software Developer"),
            D("Software Engineer (Graduate Programme)"),
        ],
    },
    {
        "category": "software",
        "query": "Find me software engineering jobs using Python in South Africa.",
        "expected": {
            "roles": ["software engineer", "software developer"],
            "seniority": "",
            "locations": [],
            "remote": "any",
            "min_salary": None,
            "skills": ["python"],
            "keywords": [],
        },
        "gold": [
            D("Junior Software Developer"),
            D("Software Developer"),
            D("Junior Data Analyst"),
            D("Senior Software Engineer"),
        ],
    },
    {
        "category": "software",
        "query": "Find me C# developer jobs in South Africa.",
        "expected": {
            "roles": ["software engineer", "software developer"],
            "seniority": "",
            "locations": [],
            "remote": "any",
            "min_salary": None,
            "skills": ["csharp"],
            "keywords": [],
        },
        "gold": [
            D("Graduate Software Developer"),
        ],
    },
    {
        "category": "software",
        "query": "Find me Java developer jobs in South Africa.",
        "expected": {
            "roles": ["software engineer", "software developer"],
            "seniority": "",
            "locations": [],
            "remote": "any",
            "min_salary": None,
            "skills": ["java"],
            "keywords": [],
        },
        "gold": [
            D("Software Engineer (Graduate Programme)"),
        ],
    },
    # ------------------------------------------------------------------- data
    {
        "category": "data",
        "query": "Find me junior data analyst jobs in Cape Town.",
        "expected": {
            "roles": ["data analyst", "data scientist"],
            "seniority": "entry-level",
            "locations": ["Cape Town"],
            "remote": "any",
            "min_salary": None,
            "skills": [],
            "keywords": [],
        },
        "gold": [
            D("Junior Data Analyst"),
        ],
    },
    {
        "category": "data",
        "query": "Find me data science jobs with Python in South Africa.",
        "expected": {
            "roles": ["data analyst", "data scientist"],
            "seniority": "",
            "locations": [],
            "remote": "any",
            "min_salary": None,
            "skills": ["data science", "python"],
            "keywords": [],
        },
        "gold": [
            D("Junior Data Analyst"),
        ],
    },
    {
        "category": "data",
        "query": "Find me data analyst jobs in Johannesburg.",
        "expected": {
            "roles": ["data analyst", "data scientist"],
            "seniority": "",
            "locations": ["Johannesburg"],
            "remote": "any",
            "min_salary": None,
            "skills": [],
            "keywords": [],
        },
        "gold": [],
    },
    # --------------------------------------------------------------- graduate
    {
        "category": "graduate",
        "query": "I'm looking for graduate software engineering positions in Johannesburg.",
        "expected": {
            "roles": ["software engineer", "software developer"],
            "seniority": "entry-level",
            "locations": ["Johannesburg"],
            "remote": "any",
            "min_salary": None,
            "skills": [],
            "keywords": [],
        },
        "gold": [
            D("Junior Web Developer"),
            D("Junior Software Developer"),
            D("Graduate Software Developer"),
        ],
    },
    {
        "category": "graduate",
        "query": "Find me graduate software developer jobs, remote preferred.",
        "expected": {
            "roles": ["software engineer", "software developer"],
            "seniority": "entry-level",
            "locations": [],
            "remote": "preferred",
            "min_salary": None,
            "skills": [],
            "keywords": [],
        },
        "gold": [
            D("Graduate Software Developer"),
            D("Junior Software Developer"),
            D("Software Engineer (Graduate Programme)"),
            D("Junior Web Developer"),
        ],
    },
    {
        "category": "graduate",
        "query": "Find me software engineering internships in Johannesburg.",
        "expected": {
            "roles": ["software engineer", "software developer"],
            "seniority": "entry-level",
            "locations": ["Johannesburg"],
            "remote": "any",
            "min_salary": None,
            "skills": [],
            "keywords": [],
        },
        "gold": [
            D("Junior Web Developer"),
        ],
    },
    # ----------------------------------------------------------------- remote
    {
        "category": "remote",
        "query": "Find me fully remote software engineering jobs in South Africa.",
        "expected": {
            "roles": ["software engineer", "software developer"],
            "seniority": "",
            "locations": [],
            "remote": "required",
            "min_salary": None,
            "skills": [],
            "keywords": [],
        },
        "gold": [
            D("Junior Software Developer"),
            D("Graduate Software Developer"),
        ],
        "hard": {"remote": "required"},
    },
    {
        "category": "remote",
        "query": "Find remote Python developer jobs.",
        "expected": {
            "roles": ["software engineer", "software developer"],
            "seniority": "",
            "locations": [],
            "remote": "preferred",
            "min_salary": None,
            "skills": ["python"],
            "keywords": [],
        },
        "gold": [
            D("Junior Software Developer"),
            D("Software Developer"),
        ],
    },
    {
        "category": "remote",
        "query": "Find fully remote Python developer jobs in South Africa.",
        "expected": {
            "roles": ["software engineer", "software developer"],
            "seniority": "",
            "locations": [],
            "remote": "required",
            "min_salary": None,
            "skills": ["python"],
            "keywords": [],
        },
        "gold": [
            D("Junior Software Developer"),
        ],
        "hard": {"remote": "required"},
    },
    {
        "category": "remote",
        "query": "Find on-site software developer jobs in Durban.",
        "expected": {
            "roles": ["software engineer", "software developer"],
            "seniority": "",
            "locations": ["Durban"],
            "remote": "no",
            "min_salary": None,
            "skills": [],
            "keywords": [],
        },
        "gold": [
            D("Software Developer"),
        ],
    },
    {
        "category": "remote",
        "query": "Find software engineering jobs in Johannesburg, no remote.",
        "expected": {
            "roles": ["software engineer", "software developer"],
            "seniority": "",
            "locations": ["Johannesburg"],
            "remote": "no",
            "min_salary": None,
            "skills": [],
            "keywords": [],
        },
        "gold": [
            D("Senior Software Engineer"),
            D("Junior Web Developer"),
        ],
    },
    # ----------------------------------------------------------------- salary
    {
        "category": "salary",
        "query": "Find me software developer jobs in Cape Town paying at least R20,000.",
        "expected": {
            "roles": ["software engineer", "software developer"],
            "seniority": "",
            "locations": ["Cape Town"],
            "remote": "any",
            "min_salary": 20000,
            "skills": [],
            "keywords": [],
        },
        "gold": [
            D("Software Engineer (Graduate Programme)"),
            D("Junior Software Developer"),
            D("Graduate Software Developer"),
        ],
        "hard": {"min_salary": 20000},
    },
    {
        "category": "salary",
        "query": "Find me software engineering jobs in Johannesburg with a minimum salary of R50,000.",
        "expected": {
            "roles": ["software engineer", "software developer"],
            "seniority": "",
            "locations": ["Johannesburg"],
            "remote": "any",
            "min_salary": 50000,
            "skills": [],
            "keywords": [],
        },
        "gold": [
            D("Senior Software Engineer"),
        ],
        "hard": {"min_salary": 50000},
    },
    {
        "category": "salary",
        "query": "Find me entry-level software engineering jobs in Durban paying at least R25,000.",
        "expected": {
            "roles": ["software engineer", "software developer"],
            "seniority": "entry-level",
            "locations": ["Durban"],
            "remote": "any",
            "min_salary": 25000,
            "skills": [],
            "keywords": [],
        },
        "gold": [
            D("Software Developer"),
            D("Junior Software Developer"),
        ],
        "hard": {"min_salary": 25000},
    },
    # --------------------------------------------------------------- location
    {
        "category": "location",
        "query": "Find me entry-level software engineering jobs in Durban.",
        "expected": {
            "roles": ["software engineer", "software developer"],
            "seniority": "entry-level",
            "locations": ["Durban"],
            "remote": "any",
            "min_salary": None,
            "skills": [],
            "keywords": [],
        },
        "gold": [
            D("Software Developer"),
            D("Graduate Software Developer"),
            D("Junior Software Developer"),
        ],
    },
    {
        "category": "location",
        "query": "Find me admin clerk jobs in Durban.",
        "expected": {
            "roles": ["administrator / clerk"],
            "seniority": "",
            "locations": ["Durban"],
            "remote": "any",
            "min_salary": None,
            "skills": [],
            "keywords": [],
        },
        "gold": [
            D("Administration Clerk"),
            C("POST 14/05/02 : ADMIN CLERK: FINANCE"),
        ],
    },
    {
        "category": "location",
        "query": "Find cleaning jobs in Cape Town.",
        "expected": {
            "roles": [],
            "seniority": "",
            "locations": ["Cape Town"],
            "remote": "any",
            "min_salary": None,
            "skills": [],
            "keywords": ["cleaning"],
        },
        "gold": [
            C("POST 14/05/03 : CLEANER"),
        ],
    },
    {
        "category": "location",
        "query": "Find senior software engineering jobs in Johannesburg.",
        "expected": {
            "roles": ["software engineer", "software developer"],
            "seniority": "senior",
            "locations": ["Johannesburg"],
            "remote": "any",
            "min_salary": None,
            "skills": [],
            "keywords": [],
        },
        "gold": [
            D("Senior Software Engineer"),
        ],
    },
    # ----------------------------------------------------------------- domain
    {
        "category": "domain",
        "query": "Find fintech graduate developer jobs in South Africa.",
        "expected": {
            "roles": ["software engineer", "software developer"],
            "seniority": "entry-level",
            "locations": [],
            "remote": "any",
            "min_salary": None,
            "skills": [],
            "keywords": ["fintech"],
        },
        "gold": [
            D("Junior Software Developer"),
            D("Software Engineer (Graduate Programme)"),
            D("Graduate Software Developer"),
        ],
    },
    {
        "category": "domain",
        "query": "Find remote fintech developer jobs in South Africa.",
        "expected": {
            "roles": ["software engineer", "software developer"],
            "seniority": "",
            "locations": [],
            "remote": "preferred",
            "min_salary": None,
            "skills": [],
            "keywords": ["fintech"],
        },
        "gold": [
            D("Junior Software Developer"),
            D("Software Engineer (Graduate Programme)"),
        ],
    },
    {
        "category": "domain",
        "query": "Find me entry-level aerospace software engineering jobs in Durban.",
        "expected": {
            "roles": ["software engineer", "software developer"],
            "seniority": "entry-level",
            "locations": ["Durban"],
            "remote": "any",
            "min_salary": None,
            "skills": [],
            "keywords": ["aerospace"],
        },
        "gold": [],
    },
    {
        "category": "domain",
        "query": "Find professional nurse jobs in Durban.",
        "expected": {
            "roles": [],
            "seniority": "",
            "locations": ["Durban"],
            "remote": "any",
            "min_salary": None,
            "skills": [],
            "keywords": ["nurse"],
        },
        "gold": [
            C("POST 14/04/01 : PROFESSIONAL NURSE (SPECIALTY: PRIMARY HEALTH CARE)"),
        ],
    },
    {
        "category": "domain",
        "query": "Find me government admin clerk jobs in Durban.",
        "expected": {
            "roles": ["administrator / clerk"],
            "seniority": "",
            "locations": ["Durban"],
            "remote": "any",
            "min_salary": None,
            "skills": [],
            "keywords": ["government"],
        },
        "gold": [
            D("Administration Clerk"),
            C("POST 14/05/02 : ADMIN CLERK: FINANCE"),
        ],
    },
    {
        "category": "domain",
        "query": "Find me banking jobs in Johannesburg.",
        "expected": {
            "roles": ["finance"],
            "seniority": "",
            "locations": ["Johannesburg"],
            "remote": "any",
            "min_salary": None,
            "skills": ["finance"],
            "keywords": [],
        },
        "gold": [
            D("Finance Graduate"),
        ],
    },
    # -------------------------------------------------------------------- mix
    {
        "category": "mix",
        "query": "Find me junior remote software developer jobs paying at least R20,000 in South Africa.",
        "expected": {
            "roles": ["software engineer", "software developer"],
            "seniority": "entry-level",
            "locations": [],
            "remote": "preferred",
            "min_salary": 20000,
            "skills": [],
            "keywords": [],
        },
        "gold": [
            D("Junior Software Developer"),
            D("Graduate Software Developer"),
            D("Software Developer"),
        ],
        "hard": {"min_salary": 20000},
    },
    {
        "category": "mix",
        "query": "Find me graduate data analyst jobs in Cape Town paying at least R25,000.",
        "expected": {
            "roles": ["data analyst", "data scientist"],
            "seniority": "entry-level",
            "locations": ["Cape Town"],
            "remote": "any",
            "min_salary": 25000,
            "skills": [],
            "keywords": [],
        },
        "gold": [
            D("Junior Data Analyst"),
        ],
        "hard": {"min_salary": 25000},
    },
    {
        "category": "mix",
        "query": "Find me Python developer jobs in Cape Town.",
        "expected": {
            "roles": ["software engineer", "software developer"],
            "seniority": "",
            "locations": ["Cape Town"],
            "remote": "any",
            "min_salary": None,
            "skills": ["python"],
            "keywords": [],
        },
        "gold": [
            D("Junior Software Developer"),
            D("Junior Data Analyst"),
        ],
    },
]
