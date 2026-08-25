# National evaluation corpus (DPSA public-service circulars)

This directory holds the large, offline, legitimate corpus used by the
national-scale evaluation (`../national_runner.py`).

## Provenance

* Source: the Department of Public Service and Administration (DPSA) weekly
  vacancy circulars for South Africa — the only *legitimate* bulk source of SA
  job postings found during this milestone's due-diligence sweep.
* Private-sector feeds were ruled out: the HuggingFace datasets API returns no
  SA job-posting datasets; Techmap/AWS's feed is paid and scrapes boards;
  CareerJunction / PNet / LinkedIn scrapers violate their terms of service;
  StatsSA QLFS is a household survey, not job ads.
* Download URL pattern:
  `https://www.dpsa.gov.za/dpsa2g/documents/vacancies/<year>/PSV%20CIRCULAR%20<N>%20of%20<year>.pdf`
* 54 circular PDFs were probed, downloaded and text-extracted:
  * 2025: circulars 10–45 (36 documents; circular 28 unavailable),
  * 2026: circulars 10–28 (19 documents).
  * 2024 URLs all return a soft-404 HTML "Abridged BP programme" page, so no
    2024 circulars are included. Circulars 1–9 of each year are not published
    on this site for the years covered.
* Download date: corpus collected in one batch; exact timestamps per document
  are recorded in `dpsa/manifest.json`.

## Layout

```
dpsa/
  manifest.json      # year, circular number, URL, bytes, pages, chars, downloaded_at
  per_circular.json  # parsed job counts per circular
  dpsa_jobs.json     # 16,259 parsed jobs (raw extraction, ~34 MB)
  raw/*.pdf          # the downloaded PDFs (~67 MB)
  text/*.txt         # extracted plain text (~30 MB)
README.md
```

## Parsing

Each circular PDF is parsed with `sources.dpsa_circular.parse_circular`; jobs
are serialised as the `sources.base.Job` dataclass. Raw extraction produced
16,259 jobs with near-complete field coverage: title/company 100%, location
~99%, salary_min ~99%, closing date ~92%, description ~99.9%.

## Data quality notes

* 352 of the 16,259 raw rows are *re-advertisements* of the same POST reference
  by the same department across weekly circulars. The evaluation de-duplicates
  by `(POST reference, department)` keeping the most recent advertisement,
  yielding **15,898 canonical public-service jobs**.
* POST references are per-circular (each circular restarts at "POST 10/01"),
  so same-number-different-department rows are distinct posts and are kept.
  Re-advertisements whose reference number changes on re-advert (common) are
  *not* detected by this key; the title+centre near-duplicate analysis below
  partially captures them. This is a documented limitation of re-advert
  detection, not a claim that only 352 re-adverts exist.
* 9 rows are cancellation / "please note" notices; they are removed as notice
  rows by `national_runner._is_notice`.
* 225 title+centre groups appear more than once (different POST references,
  same title and work centre, e.g. multiple "POST 12/77" numbered adverts are
  separate, but some title+centre pairs genuinely repeat); 265 such extra rows
  exist. This is residual near-duplication, not a bug.
* Salary anomalies were audited and are legitimate: 110 rows under R1,000/month
  are session/hourly-based medical posts; 6 rows above R200,000/month are
  executive / director-general roles.
* Many centres name institutions rather than cities ("King Edward VIII
  Hospital", "Head Office, Tshwane", "Nkangala District Office, Emalahleni").
  City-level substring filtering therefore under-matches on recall; this is an
  architecture gap, not a parsing error.

## Coverage gap (honesty note)

The corpus is **public-sector only**. Municipalities (e.g. SAPS constable
recruitment), provincial educator bulk-adverts and the entire private sector do
not appear in these circulars. Queries targeting those (a deliberate subset of
`../national_dataset.py`, category `private_gap`) legitimately have small or
empty gold sets. The ten in-repo demo jobs (`sources/demo.py`, labelled
`source="demo"`) are the only private-sector records and are excluded from all
corpus statistics and counts above.
