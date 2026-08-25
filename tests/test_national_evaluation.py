from __future__ import annotations

from evaluation import national_dataset
from evaluation import national_runner


def test_national_dataset_size():
    assert 150 <= len(national_dataset.QUERIES) <= 300


def test_national_dataset_entries_are_well_formed():
    expected_fields = {
        "roles", "seniority", "locations", "remote", "min_salary", "skills", "keywords"
    }
    for entry in national_dataset.QUERIES:
        assert {"query", "category", "expected", "rel"}.issubset(entry)
        assert expected_fields.issubset(entry["expected"])
        assert entry["query"].strip()
        assert isinstance(entry["rel"], dict)
        assert entry["category"]


def test_national_dataset_categories_are_reasonable():
    cats = {e["category"] for e in national_dataset.QUERIES}
    assert len(cats) >= 10


def test_corpus_loads_and_is_large():
    raw = national_runner.load_dpsa_jobs()
    assert len(raw) >= 5000
    assert all(j.title and j.company for j in raw)


def test_dedup_keeps_unique_post_department_pairs():
    corpus = national_runner.build_national_corpus()
    dpsa = [j for j in corpus if j.source != "demo"]
    keys = [(j.id, j.company) for j in dpsa if j.id]
    assert len(keys) == len(set(keys))
    assert len(dpsa) < len(national_runner.load_dpsa_jobs())


def test_demo_fixtures_appended_last():
    corpus = national_runner.build_national_corpus()
    assert corpus[-1].source == "demo"
    assert sum(1 for j in corpus if j.source == "demo") >= 10


def test_notice_rows_are_removed():
    raw = national_runner.load_dpsa_jobs()
    corpus = national_runner.build_national_corpus()
    dpsa = [j for j in corpus if j.source != "demo"]
    assert all(not national_runner._is_notice(j) for j in dpsa)
    assert len(raw) - len(dpsa) >= national_runner.corpus_stats()["notice_rows"]


def test_is_relevant_substring_semantics():
    from sources.base import Job
    job = Job(title="PROFESSIONAL NURSE GRADE 1", company="Health",
              location="King Edward VIII Hospital, Durban", description="x",
              salary_min=25000, remote=False)
    assert national_runner.is_relevant(job, {"any": ["nurse"], "locations": ["Durban"]})
    # "nurse" is a substring of "NURSING"; titles spelling NURSING only match "nursing".
    job2 = Job(title="OPERATIONAL MANAGER NURSING", company="Health",
               location="Durban", description="x")
    assert not national_runner.is_relevant(job2, {"any": ["nurse"]})
    assert national_runner.is_relevant(job2, {"any": ["nursing"]})


def test_is_relevant_salary_unknown_is_not_relevant():
    from sources.base import Job
    job = Job(title="CLERK", company="Health", location="Pretoria", description="x",
              salary_min=None)
    assert not national_runner.is_relevant(job, {"min_salary": 30000})
    assert national_runner.is_relevant(job, {"min_salary": None})


def test_is_relevant_soft_location_passes_empty():
    from sources.base import Job
    job = Job(title="ADMIN CLERK", company="Health", location="", description="x")
    assert national_runner.is_relevant(job, {"any": ["clerk"], "locations": ["Durban"]})


def test_is_relevant_remote_required():
    from sources.base import Job
    on_site = Job(title="CLERK", company="Health", location="", description="x", remote=False)
    remote = Job(title="CLERK", company="Health", location="", description="x", remote=True)
    assert not national_runner.is_relevant(on_site, {"remote": "required"})
    assert national_runner.is_relevant(remote, {"remote": "required"})


def test_term_in_regex_versus_substring():
    assert national_runner.term_in("nurse", "PROFESSIONAL NURSE")
    assert not national_runner.term_in("nurse", "OPERATIONAL MANAGER NURSING")
    assert national_runner.term_in(r"\bhr\b", "HR OFFICER")
    assert not national_runner.term_in(r"\bhr\b", "CHAIRMAN")


def test_ndcg_metrics_are_bounded():
    from sources.base import Job
    from types import SimpleNamespace
    jobs = [Job(title=f"POST {i} : CLERK", company="C", location="Durban",
                description="d") for i in range(10)]
    ranked = [SimpleNamespace(job=j) for j in jobs]
    gold = {id(j) for j in jobs[:3]}
    assert abs(national_runner._ndcg10(ranked, gold) - 1.0) < 1e-6
    assert abs(national_runner._precision(ranked, gold, 3) - 1.0) < 1e-6
    assert abs(national_runner._recall(ranked, gold) - 1.0) < 1e-6


def test_subset_evaluation_runs_offline_with_sane_metrics():
    metrics = national_runner.run_national_evaluation()
    assert metrics["num_queries"] == len(national_dataset.QUERIES)
    assert metrics["corpus"]["canonical_jobs"] >= 5000
    for key in ("intent_field_accuracy", "precision_3", "mrr", "ndcg_10"):
        assert 0.0 <= metrics[key] <= 1.0, key
    assert metrics["violations"] >= 0
    assert metrics["zero_result_rate"] >= 0.0
    assert metrics["dup_result_rate"] >= 0.0
    assert metrics["queries_with_gold"] > metrics["num_queries"] // 2
