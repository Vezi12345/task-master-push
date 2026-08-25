from sources.schemaorg import _iter_postings, _to_job

SAMPLE = {
    "@context": "https://schema.org/",
    "@graph": [
        {
            "@type": "JobPosting",
            "title": "Junior Python Developer",
            "hiringOrganization": {"@type": "Organization", "name": "Acme ZA"},
            "jobLocation": {
                "@type": "Place",
                "address": {
                    "@type": "PostalAddress",
                    "addressLocality": "Cape Town",
                    "addressRegion": "Western Cape",
                    "addressCountry": "ZA",
                },
            },
            "description": "Remote-friendly role. Python and SQL.",
            "baseSalary": {"@type": "MonetaryAmount", "value": {"@type": "QuantitativeValue", "value": 28000}, "currency": "ZAR"},
            "datePosted": "2026-08-10",
            "url": "https://example.com/jobs/1",
        }
    ],
}


def test_iter_postings_finds_graph_entries():
    postings = list(_iter_postings(SAMPLE))
    assert len(postings) == 1
    assert postings[0]["title"] == "Junior Python Developer"


def test_to_job_maps_fields():
    posting = list(_iter_postings(SAMPLE))[0]
    job = _to_job(posting)
    assert job.title == "Junior Python Developer"
    assert job.company == "Acme ZA"
    assert "Cape Town" in job.location
    assert job.salary_min == 28000
    assert job.salary_text == "ZAR 28000"
    assert job.posted_date == "2026-08-10"


def test_to_job_ignores_non_posting_nodes():
    posting = list(_iter_postings(SAMPLE))[0]
    job = _to_job(posting)
    assert job is not None
    assert job.url == "https://example.com/jobs/1"
