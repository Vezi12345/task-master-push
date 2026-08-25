from __future__ import annotations

"""National-scale search-quality evaluation dataset (South Africa).

This dataset is designed to evaluate the search pipeline against a large,
real, public-sector-dominated corpus built from DPSA public service vacancy
circulars (see :mod:`evaluation.national_runner`). It is intentionally corpus-
independent: relevance is defined by documented surface-term rules over the
normalized job fields, NOT by pointing at specific ``(source, title)`` rows.

Per entry:

``expected``  - the intent a correct parser should extract (same semantics and
                fields as :mod:`evaluation.dataset`).
``rel``       - deterministic relevance spec used to compute gold sets at
                runtime. Keys:
                  any          - surface terms; job.title must contain >= 1
                                 (strict: titles only, catches over-matching).
                  any_broad    - terms matched against title OR description.
                  must         - all of these must appear in title+description.
                  locations    - soft-location cities: a job with a stated
                                 location must contain the city substring;
                                 location-less jobs stay eligible (mirrors the
                                 ranker's documented soft-location semantics).
                  seniority    - "" | "entry-level" | "senior" (title markers).
                  remote       - "any" | "required" | "no" (preferred has no
                                 hard relevance effect, matching the system).
                  min_salary   - known salary_min must be >= this (unknown
                                 salary is NOT relevant to a salary query).
``hard``      - optional hard constraints (same as :mod:`evaluation.dataset`).

Intent expectations are the *truth* for the query wording; where the current
rule-based parser cannot express them (e.g. non-config locations such as
Bloemfontein, roles without a ROLE_PHRASES group such as nurse/accountant, or
consumed vocabulary such as ``technician``/``engineer``), the intent check
fails and the report classifies the gap. Relevance specs are chosen to be
precise but not identical to the ranker's internals so the two are measured
independently.
"""

# ---------------------------------------------------------------------------
# helper builders for compact entries
# ---------------------------------------------------------------------------

def Q(category, query, expected, rel=None, hard=None):
    entry = {
        "category": category,
        "query": query,
        "expected": expected,
    }
    if rel:
        entry["rel"] = rel
    if hard:
        entry["hard"] = hard
    return entry


def E(roles=(), seniority="", locations=(), remote="any", min_salary=None,
      skills=(), keywords=()):
    return {
        "roles": list(roles),
        "seniority": seniority,
        "locations": list(locations),
        "remote": remote,
        "min_salary": min_salary,
        "skills": list(skills),
        "keywords": list(keywords),
    }


# common role / keyword term sets
NURSE = {"any": ["nurse"]}
DOCTOR = {"any": ["medical officer"]}
MED_SPEC = {"any": ["medical specialist"]}
PHARMACIST = {"any": ["pharmacist"]}
RADIOGRAPHER = {"any": ["radiographer"]}
SOCIAL_WORKER = {"any": ["social worker"]}
PSYCHOLOGIST = {"any": ["psychologist"]}
PHYSIO = {"any": ["physiotherapist"]}
DENTIST = {"any": ["dentist"]}
LAB = {"any": ["medical technologist", "laboratory", "laboratory technician"]}
PORTER = {"any": ["porter"]}
CLEANER = {"any": ["cleaner"]}
EMERGENCY = {"any": ["emergency"]}
TEACHER = {"any": ["teacher", "educator"]}
LECTURER = {"any": ["lecturer"]}
PRINCIPAL = {"any": ["principal"]}
ECD = {"any": ["ecd", "early childhood"]}
EDU_ASSISTANT = {"any": ["education assistant", "assistant"]}
CLERK = {"any": ["clerk"]}
ADMIN = {"any": ["admin", "administration"]}
REGISTRAR = {"any": ["registrar"]}
PROSECUTOR = {"any": ["prosecutor"]}
MAGISTRATE = {"any": ["magistrate"]}
LEGAL = {"any": ["legal"]}
COURT = {"any": ["court"]}
CORRECTIONAL = {"any": ["correctional"]}
POLICE = {"any": ["police", "saps", "constable"]}
SECURITY = {"any": ["security"]}
DETECTIVE = {"any": ["detective"]}
SECRETARY = {"any": ["secretary"]}
DATA_CAPTURER = {"any": ["data capturer"]}
RECEPTIONIST = {"any": ["receptionist"]}
TYPIST = {"any": ["typist"]}
ACCOUNTANT = {"any": ["accountant"]}
FINANCE = {"any": ["finance", "financial"]}
AUDITOR = {"any": ["auditor"]}
TREASURY = {"any": ["treasury"]}
BUDGET = {"any": ["budget"]}
HR = {"any": ["\bhr\b", "personnel", "human resource"]}
TRAINING = {"any": ["training"]}
CIVIL_ENG = {"any": ["civil engineer"]}
ENGINEER = {"any": ["engineer"]}
TECHNICIAN = {"any": ["technician"]}
SURVEYOR = {"any": ["surveyor"]}
ELECTRICIAN = {"any": ["electrician"]}
ARTISAN = {"any": ["artisan"]}
PLUMBER = {"any": ["plumber"]}
PAINTER = {"any": ["painter"]}
BUILDER = {"any": ["builder"]}
DRIVER = {"any": ["driver"]}
BUS_DRIVER = {"any": ["bus driver"]}
TRAFFIC = {"any": ["traffic"]}
ICT = {"any": ["\bict\b"]}
IT = {"any": ["\bit\b", "information technology"]}
DEVELOPER = {"any": ["developer", "programmer"]}
NETWORK = {"any": ["network"]}
GIS = {"any": ["\bgis\b", "geographic information"]}
ANALYST = {"any": ["analyst"]}
AGRICULTURE = {"any": ["agricultur"]}
FOREST = {"any": ["forest"]}
ENVIRONMENTAL = {"any": ["environmental"]}
VETERINARY = {"any": ["veterinary"]}
INTERN = {"any": ["\bintern\b", "internship"]}
GRADUATE = {"any": ["graduate"]}
STUDENT = {"any": ["student"]}
YOUTH = {"any": ["youth"]}
LEARNERSHIP = {"any": ["learnership"]}
CADET = {"any": ["cadet"]}
DIRECTOR = {"any": ["\bdirector\b"]}
DEPUTY_DIRECTOR = {"any": ["deputy director"]}
CHIEF = {"any": ["\bchief\b"]}
MANAGER = {"any": ["manager"]}
SUPERVISOR = {"any": ["supervisor"]}
CATERING = {"any": ["catering"]}
CHEF = {"any": ["chef"]}
CALL_CENTRE = {"any": ["call centre"]}
SALES = {"any": ["sales"]}
MARKETING = {"any": ["marketing"]}
RETAIL = {"any": ["retail"]}
INSPECTOR = {"any": ["inspector"]}
ADJUDICATOR = {"any": ["adjudicator"]}
SOFTWARE = {"any": ["software", "developer", "programmer"]}
DATA_SCI = {"any": ["data", "analyst", "analytics"]}

# standard location refs
DBN = ["Durban"]
JHB = ["Johannesburg"]
CPT = ["Cape Town"]
PTA = ["Pretoria"]
ELS = ["East London"]
BFN = ["Bloemfontein"]
PLK = ["Polokwane"]
BHO = ["Bhisho"]
MBO = ["Mbombela"]


QUERIES: list[dict] = [
    # ============================================================== healthcare
    Q("healthcare", "Find me professional nurse jobs in Durban.",
      E(keywords=["professional", "nurse"], locations=DBN), dict(NURSE, locations=DBN)),
    Q("healthcare", "Find nurse jobs in Gauteng.",
      E(keywords=["nurse"], locations=["Johannesburg"]), dict(NURSE, locations=["Johannesburg"])),
    Q("healthcare", "I want nursing jobs in Cape Town.",
      E(keywords=["nursing"], locations=CPT), dict({"any": ["nursing", "nurse"]}, locations=CPT)),
    Q("healthcare", "Find me enrolled nurse jobs in East London.",
      E(keywords=["enrolled", "nurse"], locations=ELS), dict({"any": ["enrolled nurse"]}, locations=ELS)),
    Q("healthcare", "Find me senior professional nurse jobs in Pretoria.",
      E(seniority="senior", keywords=["professional", "nurse"], locations=PTA),
      dict(NURSE, seniority="senior", locations=PTA)),
    Q("healthcare", "Find me medical officer jobs in Bloemfontein.",
      E(keywords=["medical", "officer"], locations=[]), dict(DOCTOR, locations=BFN)),
    Q("healthcare", "Find medical specialist jobs in South Africa.",
      E(keywords=["medical", "specialist"]), MED_SPEC),
    Q("healthcare", "I need a medical specialist in Polokwane.",
      E(keywords=["medical", "specialist"], locations=[]), dict(MED_SPEC, locations=PLK)),
    Q("healthcare", "Find me pharmacist jobs in Johannesburg.",
      E(keywords=["pharmacist"], locations=JHB), dict(PHARMACIST, locations=JHB)),
    Q("healthcare", "Find radiographer jobs in Pretoria.",
      E(keywords=["radiographer"], locations=PTA), dict(RADIOGRAPHER, locations=PTA)),
    Q("healthcare", "Find me social worker jobs in Mbombela.",
      E(keywords=["social", "worker"], locations=[]), dict(SOCIAL_WORKER, locations=MBO)),
    Q("healthcare", "Find psychologist jobs in Cape Town.",
      E(keywords=["psychologist"], locations=CPT), dict(PSYCHOLOGIST, locations=CPT)),
    Q("healthcare", "Find physiotherapist jobs in South Africa.",
      E(keywords=["physiotherapist"]), PHYSIO),
    Q("healthcare", "Find dentist jobs in Durban.",
      E(keywords=["dentist"], locations=DBN), dict(DENTIST, locations=DBN)),
    Q("healthcare", "Find medical technologist jobs in Gauteng.",
      E(keywords=["medical", "technologist"], locations=["Johannesburg"]),
      dict({"any": ["medical technologist", "laboratory", "laboratory technician"]}, locations=["Johannesburg"])),
    Q("healthcare", "Find me hospital porter jobs in Johannesburg.",
      E(keywords=["hospital", "porter"], locations=JHB), dict(PORTER, locations=JHB)),
    Q("healthcare", "Find cleaner jobs in Cape Town.",
      E(keywords=["cleaner"], locations=CPT), dict(CLEANER, locations=CPT)),
    Q("healthcare", "Find emergency services jobs in Pretoria.",
      E(keywords=["emergency", "services"], locations=PTA), dict(EMERGENCY, locations=PTA)),
    Q("healthcare", "Find me staff nurse jobs paying at least R25,000 in Gauteng.",
      E(seniority="", keywords=["staff", "nurse"], locations=["Johannesburg"], min_salary=25000),
      dict(NURSE, locations=["Johannesburg"], min_salary=25000),
      hard={"min_salary": 25000}),
    Q("healthcare", "Find me nurse manager jobs in South Africa.",
      E(keywords=["nurse", "manager"]), dict(NURSE, must=["manager"])),
    Q("healthcare", "Find me community health worker jobs in Bhisho.",
      E(keywords=["community", "health", "worker"], locations=[]),
      dict({"any": ["health worker", "community"]}, locations=BHO)),
    Q("healthcare", "Find me nursing assistant jobs in East London.",
      E(keywords=["nursing", "assistant"], locations=ELS),
      dict({"any": ["nursing", "nurse"]}, must=["assistant"], locations=ELS)),
    # =============================================================== education
    Q("education", "Find me teacher jobs in Gauteng.",
      E(keywords=["teacher"], locations=["Johannesburg"]), dict(TEACHER, locations=["Johannesburg"])),
    Q("education", "Find lecturer jobs in South Africa.",
      E(keywords=["lecturer"]), LECTURER),
    Q("education", "Find me school principal jobs in Pretoria.",
      E(seniority="senior", keywords=["school"], locations=PTA), dict(PRINCIPAL, seniority="senior", locations=PTA)),
    Q("education", "Find me education assistant jobs in Cape Town.",
      E(keywords=["education", "assistant"], locations=CPT), dict(EDU_ASSISTANT, locations=CPT)),
    Q("education", "Find ECD practitioner jobs in Johannesburg.",
      E(keywords=["ecd"], locations=JHB), dict(ECD, locations=JHB)),
    Q("education", "Find me educator jobs in East London.",
      E(keywords=["educator"], locations=ELS), dict(TEACHER, locations=ELS)),
    Q("education", "Find me head of department teaching jobs in South Africa.",
      E(seniority="senior", keywords=["department", "teaching"]),
      dict(TEACHER, must=["head"])),
    Q("education", "Find me school admin clerk jobs in Durban.",
      E(roles=["administrator / clerk"], locations=DBN, keywords=["school"]),
      dict(CLERK, locations=DBN)),
    Q("education", "Find me grade R teacher jobs in Polokwane.",
      E(keywords=["grade", "teacher"], locations=[]), dict(TEACHER, locations=PLK)),
    Q("education", "Find me curriculum developer jobs in South Africa.",
      E(keywords=["curriculum"]), dict({"any": ["curriculum"]})),
    # ============================================================ legal/justice
    Q("legal", "Find me court clerk jobs in Pretoria.",
      E(roles=["administrator / clerk"], locations=PTA, keywords=["court"]), dict(COURT, locations=PTA)),
    Q("legal", "Find me prosecutor jobs in South Africa.",
      E(keywords=["prosecutor"]), PROSECUTOR),
    Q("legal", "Find magistrate jobs in South Africa.",
      E(keywords=["magistrate"]), MAGISTRATE),
    Q("legal", "Find me legal advisor jobs in Pretoria.",
      E(keywords=["legal", "advisor"], locations=PTA), dict(LEGAL, locations=PTA)),
    Q("legal", "Find court interpreter jobs in Johannesburg.",
      E(keywords=["court", "interpreter"], locations=JHB), dict({"any": ["interpreter"]}, locations=JHB)),
    Q("legal", "Find me correctional officer jobs in South Africa.",
      E(keywords=["correctional", "officer"]), CORRECTIONAL),
    Q("legal", "Find me registrar of deeds jobs in Pretoria.",
      E(keywords=["registrar", "deeds"], locations=PTA), dict(REGISTRAR, locations=PTA)),
    Q("legal", "Find me sheriff jobs in South Africa.",
      E(keywords=["sheriff"]), {"any": ["sheriff"]}),
    Q("legal", "Find me law researcher jobs in Cape Town.",
      E(keywords=["law", "researcher"], locations=CPT), dict(LEGAL, locations=CPT)),
    Q("legal", "Find me admin clerk jobs at the courts in Durban.",
      E(roles=["administrator / clerk"], locations=DBN, keywords=["courts"]), dict(CLERK, locations=DBN)),
    # ========================================================== police/security
    Q("police", "Find me police jobs in South Africa.",
      E(keywords=["police"]), POLICE),
    Q("police", "Find me security officer jobs in Johannesburg.",
      E(keywords=["security", "officer"], locations=JHB), dict(SECURITY, locations=JHB)),
    Q("police", "Find me detective jobs in South Africa.",
      E(keywords=["detective"]), DETECTIVE),
    Q("police", "Find me constable jobs in Gauteng.",
      E(keywords=["constable"], locations=["Johannesburg"]), dict({"any": ["constable", "police", "saps"]}, locations=["Johannesburg"])),
    Q("police", "Find me traffic officer jobs in Pretoria.",
      E(keywords=["traffic", "officer"], locations=PTA), dict(TRAFFIC, locations=PTA)),
    Q("police", "Find me SAPS jobs in Durban.",
      E(keywords=["saps"], locations=DBN), dict(POLICE, locations=DBN)),
    # ================================================================== admin
    Q("admin", "Find me admin clerk jobs in Durban.",
      E(roles=["administrator / clerk"], locations=DBN, keywords=[]), dict(CLERK, locations=DBN)),
    Q("admin", "Find me administration clerk jobs in Pretoria.",
      E(roles=["administrator / clerk"], locations=PTA, keywords=[]), dict(ADMIN, locations=PTA)),
    Q("admin", "Find me office administrator jobs in Cape Town.",
      E(roles=["administrator / clerk"], locations=CPT, keywords=[]), dict(ADMIN, locations=CPT)),
    Q("admin", "Find me administrative assistant jobs in Johannesburg.",
      E(roles=["administrator / clerk"], locations=JHB, keywords=[]),
      dict({"any": ["administrative", "admin", "administration"]}, locations=JHB)),
    Q("admin", "Find me secretary jobs in Pretoria.",
      E(keywords=["secretary"], locations=PTA), dict(SECRETARY, locations=PTA)),
    Q("admin", "Find me personal assistant jobs in Cape Town.",
      E(keywords=["personal", "assistant"], locations=CPT),
      dict({"any": ["personal assistant", "secretary"]}, locations=CPT)),
    Q("admin", "Find me data capturer jobs in South Africa.",
      E(keywords=["capturer"]), DATA_CAPTURER),
    Q("admin", "Find me receptionist jobs in Johannesburg.",
      E(keywords=["receptionist"], locations=JHB), dict(RECEPTIONIST, locations=JHB)),
    Q("admin", "Find me typing jobs in Pretoria.",
      E(keywords=["typing"], locations=PTA), dict(TYPIST, locations=PTA)),
    Q("admin", "Find me registry clerk jobs in South Africa.",
      E(roles=["administrator / clerk"], keywords=["registry"]), dict(CLERK, must=["registry"])),
    Q("admin", "Find me junior admin clerk jobs in Durban.",
      E(roles=["administrator / clerk"], seniority="entry-level", locations=DBN, keywords=[]),
      dict(CLERK, seniority="entry-level", locations=DBN)),
    Q("admin", "Find me senior secretary jobs in Pretoria.",
      E(seniority="senior", keywords=["secretary"], locations=PTA), dict(SECRETARY, seniority="senior", locations=PTA)),
    Q("admin", "Find me general assistant jobs in Cape Town.",
      E(keywords=["general", "assistant"], locations=CPT), dict({"any": ["assistant"]}, locations=CPT)),
    Q("admin", "Find me records clerk jobs in Johannesburg.",
      E(roles=["administrator / clerk"], locations=JHB, keywords=["records"]), dict(CLERK, locations=JHB)),
    # ================================================================= finance
    Q("finance", "Find me accountant jobs in Pretoria.",
      E(keywords=["accountant"], locations=PTA), dict(ACCOUNTANT, locations=PTA)),
    Q("finance", "Find me accounting jobs in Johannesburg.",
      E(roles=["finance"], locations=JHB, skills=["finance"], keywords=[]), dict(FINANCE, locations=JHB)),
    Q("finance", "Find me finance clerk jobs in Cape Town.",
      E(roles=["finance"], locations=CPT, skills=["finance"], keywords=[]), dict(CLERK, must=["finance"], locations=CPT)),
    Q("finance", "Find me bookkeeper jobs in South Africa.",
      E(keywords=["bookkeeper"]), {"any": ["bookkeeper", "accountant"]}),
    Q("finance", "Find me internal auditor jobs in Pretoria.",
      E(keywords=["internal", "auditor"], locations=PTA), dict(AUDITOR, locations=PTA)),
    Q("finance", "Find me treasury jobs in South Africa.",
      E(keywords=["treasury"]), TREASURY),
    Q("finance", "Find me budget analyst jobs in Gauteng.",
      E(keywords=["budget"], locations=["Johannesburg"]), dict(BUDGET, locations=["Johannesburg"])),
    Q("finance", "Find me finance graduate jobs in South Africa.",
      E(roles=["finance"], seniority="entry-level", skills=["finance"], keywords=[]),
      dict(FINANCE, seniority="entry-level")),
    Q("finance", "Find me senior accountant jobs in Johannesburg.",
      E(seniority="senior", keywords=["accountant"], locations=JHB), dict(ACCOUNTANT, seniority="senior", locations=JHB)),
    Q("finance", "Find me banking jobs in South Africa.",
      E(roles=["finance"], skills=["finance"], keywords=[]), dict({"any": ["finance", "financial", "banking"]})),
    Q("finance", "Find me payroll administrator jobs in Pretoria.",
      E(roles=["administrator / clerk"], locations=PTA, keywords=["payroll"]), dict({"any": ["payroll"]}, locations=PTA)),
    Q("finance", "Find me costing clerk jobs in South Africa.",
      E(roles=["administrator / clerk"], keywords=["costing"]), dict(CLERK, must=["costing"])),
    # ==================================================================== hr
    Q("hr", "Find me HR officer jobs in Pretoria.",
      E(keywords=["hr"], locations=PTA), dict(HR, locations=PTA)),
    Q("hr", "Find me human resources manager jobs in Johannesburg.",
      E(keywords=["human", "resources", "manager"], locations=JHB),
      dict(HR, must=["manager"], locations=JHB)),
    Q("hr", "Find me personnel officer jobs in South Africa.",
      E(keywords=["personnel", "officer"]), dict(HR, locations=[])),
    Q("hr", "Find me training officer jobs in Durban.",
      E(keywords=["training", "officer"], locations=DBN), dict(TRAINING, locations=DBN)),
    Q("hr", "Find me labour relations officer jobs in Cape Town.",
      E(keywords=["labour", "relations", "officer"], locations=CPT), dict({"any": ["labour"]}, locations=CPT)),
    Q("hr", "Find me HR assistant jobs in Gauteng.",
      E(keywords=["hr"], locations=["Johannesburg"]), dict(HR, locations=["Johannesburg"])),
    Q("hr", "Find me recruitment officer jobs in South Africa.",
      E(keywords=["recruitment", "officer"]), {"any": ["recruitment", "recruiting"]}),
    Q("hr", "Find me skills development facilitator jobs in Pretoria.",
      E(keywords=["skills", "development", "facilitator"], locations=PTA),
      dict({"any": ["development"]}, locations=PTA)),
    # ============================================================== engineering
    Q("engineering", "Find me civil engineer jobs in South Africa.",
      E(keywords=["civil", "engineer"]), CIVIL_ENG),
    Q("engineering", "Find me civil engineering technician jobs in Pretoria.",
      E(keywords=["civil", "engineering", "technician"], locations=PTA), dict(TECHNICIAN, must=["civil"], locations=PTA)),
    Q("engineering", "Find me engineer jobs in Johannesburg.",
      E(keywords=["engineer"], locations=JHB), dict(ENGINEER, locations=JHB)),
    Q("engineering", "Find me engineering technician jobs in Cape Town.",
      E(keywords=["engineering", "technician"], locations=CPT), dict(TECHNICIAN, locations=CPT)),
    Q("engineering", "Find me surveyor jobs in South Africa.",
      E(keywords=["surveyor"]), SURVEYOR),
    Q("engineering", "Find me electrician jobs in Pretoria.",
      E(keywords=["electrician"], locations=PTA), dict(ELECTRICIAN, locations=PTA)),
    Q("engineering", "Find me artisan jobs in South Africa.",
      E(keywords=["artisan"]), ARTISAN),
    Q("engineering", "Find me plumber jobs in Johannesburg.",
      E(keywords=["plumber"], locations=JHB), dict(PLUMBER, locations=JHB)),
    Q("engineering", "Find me painter jobs in Cape Town.",
      E(keywords=["painter"], locations=CPT), dict(PAINTER, locations=CPT)),
    Q("engineering", "Find me bricklayer jobs in South Africa.",
      E(keywords=["bricklayer"]), {"any": ["bricklayer", "builder"]}),
    Q("engineering", "Find me mechanical engineer jobs in South Africa.",
      E(keywords=["mechanical", "engineer"]), dict(ENGINEER, must=["mechanical"])),
    Q("engineering", "Find me electrical engineer jobs in Pretoria.",
      E(keywords=["electrical", "engineer"], locations=PTA), dict(ENGINEER, must=["electrical"], locations=PTA)),
    Q("engineering", "Find me boilermaker jobs in South Africa.",
      E(keywords=["boilermaker"]), {"any": ["boilermaker", "artisan"]}),
    Q("engineering", "Find me diesel mechanic jobs in South Africa.",
      E(keywords=["diesel", "mechanic"]), {"any": ["mechanic"]}),
    # ==================================================================== it
    Q("it", "Find me IT jobs in Pretoria.",
      E(keywords=["it"], locations=PTA), dict(IT, locations=PTA)),
    Q("it", "Find me ICT officer jobs in South Africa.",
      E(keywords=["ict", "officer"]), dict(ICT, locations=[])),
    Q("it", "Find me ICT technician jobs in Johannesburg.",
      E(keywords=["ict", "technician"], locations=JHB), dict(ICT, locations=JHB)),
    Q("it", "Find me IT support jobs in Cape Town.",
      E(roles=["it support"], locations=CPT, keywords=[]),
      dict({"any": ["it support", "support technician", "help desk"]}, locations=CPT)),
    Q("it", "Find me system developer jobs in South Africa.",
      E(roles=["software engineer", "software developer"], keywords=["system"]), dict(DEVELOPER, must=["system"])),
    Q("it", "Find me network technician jobs in Pretoria.",
      E(keywords=["network", "technician"], locations=PTA), dict(NETWORK, locations=PTA)),
    Q("it", "Find me GIS jobs in South Africa.",
      E(keywords=["gis"]), GIS),
    Q("it", "Find me geographic information systems jobs in Gauteng.",
      E(keywords=["geographic", "information", "systems"], locations=["Johannesburg"]), dict(GIS, locations=["Johannesburg"])),
    Q("it", "Find me IT manager jobs in Johannesburg.",
      E(keywords=["it", "manager"], locations=JHB), dict(IT, must=["manager"], locations=JHB)),
    Q("it", "Find me web developer jobs in South Africa.",
      E(roles=["software engineer", "software developer"], keywords=[]), SOFTWARE),
    Q("it", "Find me software developer jobs in Pretoria.",
      E(roles=["software engineer", "software developer"], locations=PTA, keywords=[]),
      dict(SOFTWARE, locations=PTA)),
    Q("it", "Find me data analyst jobs in South Africa.",
      E(roles=["data analyst", "data scientist"], keywords=[]), dict({"any": ["analyst"]})),
    Q("it", "Find me IT security jobs in Pretoria.",
      E(keywords=["it", "security"], locations=PTA), dict(IT, must=["security"], locations=PTA)),
    Q("it", "Find me database administrator jobs in South Africa.",
      E(skills=["sql"], keywords=[]), {"any": ["database"]}),
    # =============================================================== transport
    Q("transport", "Find me driver jobs in South Africa.",
      E(keywords=["driver"]), DRIVER),
    Q("transport", "Find me bus driver jobs in Gauteng.",
      E(keywords=["bus", "driver"], locations=["Johannesburg"]), dict(BUS_DRIVER, locations=["Johannesburg"])),
    Q("transport", "Find me truck driver jobs in South Africa.",
      E(keywords=["truck", "driver"]), dict(DRIVER, must=["truck"])),
    Q("transport", "Find me code 14 driver jobs in Durban.",
      E(keywords=["code", "driver"], locations=DBN), dict(DRIVER, locations=DBN)),
    Q("transport", "Find me fleet manager jobs in Pretoria.",
      E(keywords=["fleet", "manager"], locations=PTA), dict(DRIVER, must=["fleet"], locations=PTA)),
    Q("transport", "Find me driving instructor jobs in South Africa.",
      E(keywords=["driving", "instructor"]), {"any": ["driving"]}),
    Q("transport", "Find me forklift driver jobs in South Africa.",
      E(keywords=["forklift", "driver"]), dict(DRIVER, must=["forklift"])),
    Q("transport", "Find me railway jobs in South Africa.",
      E(keywords=["railway"]), {"any": ["railway"]}),
    # ========================================================= agric & environment
    Q("agric", "Find me agricultural technician jobs in South Africa.",
      E(keywords=["agricultural", "technician"]), dict(AGRICULTURE, must=["technician"])),
    Q("agric", "Find me agricultural economist jobs in Pretoria.",
      E(keywords=["agricultural", "economist"], locations=PTA), dict(AGRICULTURE, must=["economist"], locations=PTA)),
    Q("agric", "Find me forest officer jobs in South Africa.",
      E(keywords=["forest", "officer"]), dict(FOREST, locations=[]) ),
    Q("agric", "Find me environmental officer jobs in Cape Town.",
      E(keywords=["environmental", "officer"], locations=CPT), dict(ENVIRONMENTAL, locations=CPT)),
    Q("agric", "Find me veterinary technician jobs in South Africa.",
      E(keywords=["veterinary", "technician"]), dict(VETERINARY, must=["technician"])),
    Q("agric", "Find me marine biologist jobs in Cape Town.",
      E(keywords=["marine", "biologist"], locations=CPT), {"any": ["marine", "biology"]}),
    Q("agric", "Find me water technician jobs in South Africa.",
      E(keywords=["water", "technician"]), dict(TECHNICIAN, must=["water"])),
    Q("agric", "Find me conservation officer jobs in South Africa.",
      E(keywords=["conservation", "officer"]), {"any": ["conservation"]}),
    # =============================================================== graduate
    Q("graduate", "Find me graduate jobs in South Africa.",
      E(seniority="entry-level", keywords=[]), GRADUATE),
    Q("graduate", "Find me internship jobs in Johannesburg.",
      E(seniority="entry-level", locations=JHB, keywords=[]), dict(INTERN, locations=JHB)),
    Q("graduate", "Find me graduate trainee jobs in Pretoria.",
      E(seniority="entry-level", locations=PTA, keywords=[]),
      dict(GRADUATE, locations=PTA)),
    Q("graduate", "Find me student jobs in Cape Town.",
      E(keywords=["student"], locations=CPT), dict(STUDENT, locations=CPT)),
    Q("graduate", "Find me youth jobs in South Africa.",
      E(keywords=["youth"]), YOUTH),
    Q("graduate", "Find me learnership jobs in Durban.",
      E(seniority="entry-level", keywords=["learnership"], locations=DBN), dict(LEARNERSHIP, locations=DBN)),
    Q("graduate", "Find me TVET graduate jobs in South Africa.",
      E(seniority="entry-level", keywords=["tvet"]), dict(GRADUATE, must=["tvet"])),
    Q("graduate", "Find me cadet jobs in South Africa.",
      E(keywords=["cadet"]), CADET),
    Q("graduate", "Find me graduate nurse jobs in South Africa.",
      E(seniority="entry-level", keywords=["nurse"]), dict(NURSE, seniority="entry-level")),
    Q("graduate", "Find me junior clerk jobs in Pretoria.",
      E(roles=["administrator / clerk"], seniority="entry-level", locations=PTA, keywords=[]),
      dict(CLERK, seniority="entry-level", locations=PTA)),
    Q("graduate", "Find me recent graduate accounting jobs in South Africa.",
      E(roles=["finance"], seniority="entry-level", skills=["finance"], keywords=[]),
      dict(FINANCE, seniority="entry-level")),
    Q("graduate", "Find me entry-level IT jobs in Gauteng.",
      E(seniority="entry-level", keywords=["it"], locations=["Johannesburg"]),
      dict(IT, seniority="entry-level", locations=["Johannesburg"])),
    Q("graduate", "Find me graduate programmes in South Africa.",
      E(seniority="entry-level", keywords=["programmes"]), GRADUATE),
    Q("graduate", "Find me internship jobs for graduates in Cape Town.",
      E(seniority="entry-level", locations=CPT, keywords=["graduates"]), dict(INTERN, locations=CPT)),
    # ================================================================== senior
    Q("senior", "Find me director jobs in South Africa.",
      E(keywords=["director"]), DIRECTOR),
    Q("senior", "Find me deputy director jobs in Pretoria.",
      E(keywords=["deputy", "director"], locations=PTA), dict(DEPUTY_DIRECTOR, locations=PTA)),
    Q("senior", "Find me chief director jobs in South Africa.",
      E(seniority="senior", keywords=["chief", "director"]), dict(DIRECTOR, must=["chief"])),
    Q("senior", "Find me executive manager jobs in Johannesburg.",
      E(seniority="senior", keywords=["executive", "manager"], locations=JHB),
      dict(MANAGER, must=["executive"], locations=JHB)),
    Q("senior", "Find me senior manager jobs in South Africa.",
      E(seniority="senior", keywords=["manager"]), dict(MANAGER, seniority="senior")),
    Q("senior", "Find me department head jobs in Pretoria.",
      E(seniority="senior", keywords=["department"], locations=PTA),
      dict({"any": ["head"]}, locations=PTA)),
    Q("senior", "Find me director-general jobs in South Africa.",
      E(keywords=["director", "general"]), dict(DIRECTOR, must=["general"])),
    Q("senior", "Find me chief executive officer jobs in South Africa.",
      E(seniority="senior", keywords=["chief", "executive", "officer"]),
      dict({"any": ["chief executive"]})),
    # ================================================================= salary
    Q("salary", "Find me jobs in Pretoria paying at least R20,000.",
      E(locations=PTA, min_salary=20000, keywords=[]),
      dict({"any": []}, locations=PTA, min_salary=20000), hard={"min_salary": 20000}),
    Q("salary", "Find me nurse jobs paying at least R30,000.",
      E(keywords=["nurse"], min_salary=30000), dict(NURSE, min_salary=30000), hard={"min_salary": 30000}),
    Q("salary", "Find me clerk jobs paying at least R15,000 in Cape Town.",
      E(roles=["administrator / clerk"], locations=CPT, min_salary=15000, keywords=[]),
      dict(CLERK, locations=CPT, min_salary=15000), hard={"min_salary": 15000}),
    Q("salary", "Find me engineer jobs paying at least R50,000.",
      E(keywords=["engineer"], min_salary=50000), dict(ENGINEER, min_salary=50000), hard={"min_salary": 50000}),
    Q("salary", "Find me medical officer jobs paying at least R60,000.",
      E(keywords=["medical", "officer"], min_salary=60000), dict(DOCTOR, min_salary=60000), hard={"min_salary": 60000}),
    Q("salary", "Find me jobs in Johannesburg paying at least R40,000.",
      E(locations=JHB, min_salary=40000, keywords=[]),
      dict({"any": []}, locations=JHB, min_salary=40000), hard={"min_salary": 40000}),
    Q("salary", "Find me accountant jobs paying at least R25,000.",
      E(keywords=["accountant"], min_salary=25000), dict(ACCOUNTANT, min_salary=25000), hard={"min_salary": 25000}),
    Q("salary", "Find me IT jobs paying at least R35,000 in Pretoria.",
      E(keywords=["it"], locations=PTA, min_salary=35000), dict(IT, locations=PTA, min_salary=35000), hard={"min_salary": 35000}),
    Q("salary", "Find me director jobs paying at least R80,000.",
      E(keywords=["director"], min_salary=80000), dict(DIRECTOR, min_salary=80000), hard={"min_salary": 80000}),
    Q("salary", "Find me driver jobs paying at least R12,000.",
      E(keywords=["driver"], min_salary=12000), dict(DRIVER, min_salary=12000), hard={"min_salary": 12000}),
    # ================================================================ location
    Q("location", "Find me admin jobs in Durban.",
      E(roles=["administrator / clerk"], locations=DBN, keywords=[]), dict(ADMIN, locations=DBN)),
    Q("location", "Find me jobs in Pretoria.",
      E(locations=PTA, keywords=[]), dict({"any": []}, locations=PTA)),
    Q("location", "Find me jobs in Johannesburg.",
      E(locations=JHB, keywords=[]), dict({"any": []}, locations=JHB)),
    Q("location", "Find me jobs in Cape Town.",
      E(locations=CPT, keywords=[]), dict({"any": []}, locations=CPT)),
    Q("location", "Find me jobs in Sandton.",
      E(locations=["Johannesburg"], keywords=[]), dict({"any": []}, locations=["Johannesburg"])),
    Q("location", "Find me jobs in East London.",
      E(locations=ELS, keywords=[]), dict({"any": []}, locations=ELS)),
    Q("location", "Find me nurse jobs in Bloemfontein.",
      E(keywords=["nurse"], locations=[]), dict(NURSE, locations=BFN)),
    Q("location", "Find me clerk jobs in Polokwane.",
      E(roles=["administrator / clerk"], locations=[], keywords=[]), dict(CLERK, locations=PLK)),
    Q("location", "Find me jobs in Bhisho.",
      E(locations=[], keywords=[]), dict({"any": []}, locations=BHO)),
    Q("location", "Find me jobs in Mbombela.",
      E(locations=[], keywords=[]), dict({"any": []}, locations=MBO)),
    # ================================================================== remote
    Q("remote", "Find me fully remote jobs in South Africa.",
      E(remote="required", keywords=[]), dict({"any": []}, remote="required"), hard={"remote": "required"}),
    Q("remote", "Find me remote data entry jobs in South Africa.",
      E(remote="preferred", keywords=["data", "entry"]), dict({"any": ["data"]}, remote="preferred")),
    Q("remote", "Find me on-site nurse jobs in Durban.",
      E(remote="no", keywords=["nurse"], locations=DBN), dict(NURSE, remote="no", locations=DBN)),
    Q("remote", "Find me remote IT support jobs in South Africa.",
      E(roles=["it support"], remote="preferred", keywords=[]), dict(IT, remote="preferred")),
    # ============================================================ private gap
    Q("private_gap", "Find me software engineering jobs in South Africa.",
      E(roles=["software engineer", "software developer"], keywords=[]), SOFTWARE),
    Q("private_gap", "Find me software developer jobs in Cape Town.",
      E(roles=["software engineer", "software developer"], locations=CPT, keywords=[]),
      dict(SOFTWARE, locations=CPT)),
    Q("private_gap", "Find me data scientist jobs in South Africa.",
      E(roles=["data analyst", "data scientist"], keywords=[]), dict({"any": ["data", "scientist"]})),
    Q("private_gap", "Find me Python developer jobs in Johannesburg.",
      E(roles=["software engineer", "software developer"], locations=JHB, skills=["python"], keywords=[]),
      dict(SOFTWARE, locations=JHB)),
    Q("private_gap", "Find me call centre agent jobs in Durban.",
      E(keywords=["call", "centre"], locations=DBN), dict(CALL_CENTRE, locations=DBN)),
    Q("private_gap", "Find me sales representative jobs in South Africa.",
      E(keywords=["sales", "representative"]), dict(SALES, must=["representative"])),
    Q("private_gap", "Find me marketing jobs in Johannesburg.",
      E(keywords=["marketing"], locations=JHB), dict(MARKETING, locations=JHB)),
    Q("private_gap", "Find me retail jobs in South Africa.",
      E(keywords=["retail"]), RETAIL),
    Q("private_gap", "Find me bank teller jobs in South Africa.",
      E(keywords=["bank", "teller"]), {"any": ["teller"]}),
    Q("private_gap", "Find me waiter jobs in Cape Town.",
      E(keywords=["waiter"], locations=CPT), {"any": ["waiter"]}),
    Q("private_gap", "Find me chef jobs in South Africa.",
      E(keywords=["chef"]), CHEF),
    Q("private_gap", "Find me fintech jobs in South Africa.",
      E(keywords=["fintech"]), {"any": ["fintech", "financial", "finance"]}),
    # ==================================================================== mix
    Q("mix", "Find me senior nurse jobs in Durban paying at least R30,000.",
      E(seniority="senior", keywords=["nurse"], locations=DBN, min_salary=30000),
      dict(NURSE, seniority="senior", locations=DBN, min_salary=30000), hard={"min_salary": 30000}),
    Q("mix", "Find me graduate clerk jobs in Pretoria paying at least R15,000.",
      E(roles=["administrator / clerk"], seniority="entry-level", locations=PTA, min_salary=15000, keywords=[]),
      dict(CLERK, seniority="entry-level", locations=PTA, min_salary=15000), hard={"min_salary": 15000}),
    Q("mix", "Find me accountant jobs in Cape Town paying at least R30,000.",
      E(keywords=["accountant"], locations=CPT, min_salary=30000),
      dict(ACCOUNTANT, locations=CPT, min_salary=30000), hard={"min_salary": 30000}),
    Q("mix", "Find me IT technician jobs in Johannesburg.",
      E(keywords=["it", "technician"], locations=JHB), dict(IT, locations=JHB)),
    Q("mix", "Find me senior social worker jobs in South Africa.",
      E(seniority="senior", keywords=["social", "worker"]), dict(SOCIAL_WORKER, seniority="senior")),
    Q("mix", "Find me pharmacist jobs in Pretoria paying at least R40,000.",
      E(keywords=["pharmacist"], locations=PTA, min_salary=40000),
      dict(PHARMACIST, locations=PTA, min_salary=40000), hard={"min_salary": 40000}),
    Q("mix", "Find me environmental officer jobs in South Africa.",
      E(keywords=["environmental", "officer"]), dict(ENVIRONMENTAL, locations=[])),
    Q("mix", "Find me police officer jobs in Johannesburg.",
      E(keywords=["police", "officer"], locations=JHB), dict(POLICE, locations=JHB)),
    Q("mix", "Find me medical specialist jobs in Cape Town paying at least R80,000.",
      E(keywords=["medical", "specialist"], locations=CPT, min_salary=80000),
      dict(MED_SPEC, locations=CPT, min_salary=80000), hard={"min_salary": 80000}),
    Q("mix", "Find me graduate IT jobs in Johannesburg.",
      E(seniority="entry-level", keywords=["it"], locations=JHB), dict(IT, seniority="entry-level", locations=JHB)),
    Q("mix", "Find me HR jobs in Durban.",
      E(keywords=["hr"], locations=DBN), dict(HR, locations=DBN)),
    Q("mix", "Find me senior auditor jobs in Pretoria.",
      E(seniority="senior", keywords=["auditor"], locations=PTA), dict(AUDITOR, seniority="senior", locations=PTA)),
]

assert 150 <= len(QUERIES) <= 300, len(QUERIES)
