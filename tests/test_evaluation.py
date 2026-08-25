from __future__ import annotations

from evaluation import dataset
from evaluation import runner


def test_dataset_size():
    assert 28 <= len(dataset.QUERIES) <= 35


def test_dataset_entries_are_well_formed():
    required_fields = {"query", "category", "expected", "gold"}
    expected_fields = {
        "roles", "seniority", "locations", "remote", "min_salary", "skills", "keywords"
    }
    for entry in dataset.QUERIES:
        assert required_fields.issubset(entry)
        assert expected_fields.issubset(entry["expected"])
        assert entry["query"].strip()
        assert isinstance(entry["gold"], list)


def test_all_gold_references_resolve_to_corpus():
    corpus = runner.build_corpus()
    index = runner.build_index(corpus)
    unresolved = []
    for entry in dataset.QUERIES:
        for key in entry["gold"]:
            if key not in index:
                unresolved.append((entry["query"], key))
    assert not unresolved, unresolved


def test_hard_constraints_only_use_remote_and_min_salary():
    for entry in dataset.QUERIES:
        hard = entry.get("hard", {})
        assert set(hard) <= {"remote", "min_salary"}


def test_evaluation_runs_offline_with_sane_metrics():
    metrics = runner.run_evaluation()
    assert metrics["corpus_size"] >= 20
    assert metrics["num_queries"] == len(dataset.QUERIES)
    for key in ("intent_accuracy", "intent_field_accuracy", "precision_3",
                "precision_10", "recall_10", "mrr"):
        assert 0.0 <= metrics[key] <= 1.0, key
    assert metrics["violations"] >= 0
    assert metrics["categories"] and set(metrics["categories"]) >= {
        "software", "data", "graduate", "remote", "salary", "location", "domain", "mix"
    }
