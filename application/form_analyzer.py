from __future__ import annotations

"""Application-form analysis.

Inspects the HTML of a real application page and discovers every field,
question, option and gate — including questions the system has never seen
before. Nothing here is a hard-coded question list: labels, legends,
aria-attributes and placeholders are read from the live form and each item
is semantically categorised so the evidence-grounded answer engine can pick
the right evidence source.
"""

import re
from dataclasses import dataclass, field as dc_field
from html.parser import HTMLParser
from typing import Optional

from application.browser import UserActionRequired, detect_challenge


# ---------------------------------------------------------------------------
# semantic categories (used to route questions to admissible evidence)
# ---------------------------------------------------------------------------

_CATEGORY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"race|equity|population group|bbbee|broad-based black|ethnicit", "demographic"),
    (r"\bgender\b|\bmale\b|\bfemale\b", "demographic"),
    (r"disabilit|handicap|impairment", "demographic"),
    (r"date of birth|birth date|how old|\bage\b", "demographic"),
    (r"citizenship|nationality|work permit|work visa|authoris(ed|ed) to work|authorized to work|right to work|eligible to work", "work_authorisation"),
    (r"salary|remuneration|compensation|pay expectation", "salary"),
    (r"years? of (professional |relevant )?experience|how many years", "experience"),
    (r"highest qualification|degree|diploma|education|field of study|tertiary", "education"),
    (r"\bcv\b|\bresume\b|cover letter|upload|attach|portfolio|transcript|id document", "documents"),
    (r"terms (and|&) conditions|terms of use|terms of service|privacy policy", "terms"),
    (r"consent|agree to be contacted|i agree|permission to|data processing|privacy notice|popia", "consent"),
    (r"availab(le|ility)|notice period|start date|when can you start|earliest start", "availability"),
    (r"relocat|willing to (move|travel)|travel", "preference"),
    (r"e-?mail", "contact"),
    (r"phone|mobile|cellphone|contact number|tel\b", "contact"),
    (r"first name|last name|surname|full name|given name", "identity"),
    # web links are NOT identity data — filling them with a person's name
    # was a real-world misfill observed on Greenhouse (Website field)
    (r"linkedin|portfolio url|website|github|personal site", "web_link"),
    (r"city|location|address|suburb|province", "location"),
)


def classify_question_label(text: str) -> str:
    lowered = (text or "").lower()
    for pattern, category in _CATEGORY_PATTERNS:
        if re.search(pattern, lowered):
            return category
    return "other"


def humanise_name(name: str) -> str:
    return re.sub(r"[_\-\s]+", " ", (name or "").strip()).strip().capitalize()


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------

@dataclass
class AnalyzedField:
    """One discovered form control (radio groups are one logical field)."""

    selector: str = ""
    name: str = ""
    label: str = ""
    question: str = ""
    field_type: str = ""          # text|email|tel|select|radio|checkbox|date|file|textarea|number|url
    required: bool = False
    options: list[str] = dc_field(default_factory=list)
    category: str = "other"
    is_consent: bool = False
    is_terms: bool = False
    is_demographic: bool = False

    @property
    def display_question(self) -> str:
        return self.question or self.label or humanise_name(self.name)


@dataclass
class SubmitButton:
    selector: str = ""
    text: str = ""


@dataclass
class FormAnalysis:
    page_url: str = ""
    platform: str = ""
    has_form: bool = False
    form_selector: str = ""
    fields: list[AnalyzedField] = dc_field(default_factory=list)
    submit_button: Optional[SubmitButton] = None
    challenge: Optional[UserActionRequired] = None
    notes: list[str] = dc_field(default_factory=list)
    # employer forbids AI-generated content — generated drafts must never be
    # auto-submitted and the user must be told to write answers themselves
    own_words_required: bool = False

    @property
    def required_fields(self) -> list[AnalyzedField]:
        return [f for f in self.fields if f.required]

    @property
    def optional_fields(self) -> list[AnalyzedField]:
        return [f for f in self.fields if not f.required]

    @property
    def unanswered_required_questions(self) -> list[str]:
        return [f.display_question for f in self.required_fields]

    def summary(self) -> dict:
        return {
            "page_url": self.page_url,
            "platform": self.platform,
            "has_form": self.has_form,
            "field_count": len(self.fields),
            "required_count": len(self.required_fields),
            "optional_count": len(self.optional_fields),
            "categories": sorted({f.category for f in self.fields}),
            "submit_button": self.submit_button.text if self.submit_button else "",
            "challenge": self.challenge.kind if self.challenge else None,
            "own_words_required": self.own_words_required,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

_FILLABLE_TAGS = {"input", "select", "textarea"}
_SKIP_INPUT_TYPES = {"hidden", "submit", "button", "image", "reset"}

_TYPE_ALIASES = {
    "email": "email",
    "tel": "tel",
    "phone": "tel",
    "date": "date",
    "month": "date",
    "file": "file",
    "checkbox": "checkbox",
    "radio": "radio",
    "textarea": "textarea",
    "select-one": "select",
    "number": "number",
    "url": "url",
}


def _css_for(tag: str, attrs: dict, dom_index: int = -1) -> str:
    """Unique CSS/Playwright selector for a control.

    Real ATS forms (e.g. Greenhouse education blocks) contain inputs with
    NO id/name — a bare ``input`` selector would collide with the first
    text input on the page (observed live: it overwrote First Name).
    Anchor on unique attributes first, else fall back to the control's
    document-order position among its tag."""
    if attrs.get("id"):
        return f"#{attrs['id']}"
    if attrs.get("name"):
        safe = attrs["name"].replace('"', '\\"')
        return f'{tag}[name="{safe}"]'
    for attr in ("aria-label", "placeholder", "title"):
        val = attrs.get(attr)
        if val:
            safe = str(val).replace('"', '\\"')
            return f'{tag}[{attr}="{safe}"]'
    if dom_index >= 0:
        return f"{tag} >> nth={dom_index}"
    return tag or "input"


def _infer_type(tag: str, itype_raw: str, name: str, label: str) -> str:
    if tag == "textarea":
        return "textarea"
    if tag == "select":
        return "select"
    mapped = _TYPE_ALIASES.get(itype_raw)
    if mapped and mapped != "text":
        return mapped
    # generic text inputs are often semantically typed by their label/name
    context = f"{label} {name}".lower()
    if re.search(r"e-?mail", context):
        return "email"
    if re.search(r"\bphone\b|\bmobile\b|cellphone|contact number|\btel\b", context):
        return "tel"
    if re.search(r"\bdate\b|birth", context):
        return "date"
    return "text"


class _FormParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.fields: list[AnalyzedField] = []
        self.buttons: list[SubmitButton] = []
        self.has_form = False
        # label bookkeeping
        self._labels_by_for: dict[str, str] = {}
        self._in_label = False
        self._current_label_for: Optional[str] = None
        self._current_label_text: list[str] = []
        self._last_label_text = ""
        self._fields_in_label: list[int] = []
        # group bookkeeping (radios / checkboxes sharing a name)
        self._group_index: dict[str, int] = {}
        # legend context
        self._in_legend = False
        self._current_legend: list[str] = []
        self._legend_stack: list[str] = []
        # select-option collection
        self._select_field_idx: Optional[int] = None
        self._pending_option_value: Optional[str] = None
        self._pending_option_text: list[str] = []
        # document-order counters per tag — used for positional fallback
        # selectors so anonymous controls stay unique
        self._tag_seen = {"input": 0, "select": 0, "textarea": 0}

    # -- helpers -----------------------------------------------------------
    def _flush_label(self) -> None:
        text = " ".join(self._current_label_text).strip()
        if text:
            if self._current_label_for:
                self._labels_by_for[self._current_label_for] = text
            # remember for the common '<label>Question</label><input>' pattern
            self._last_label_text = text
            # backfill fields added inside this label BEFORE the text
            # (e.g. '<label><input/> I agree to...</label>') — otherwise
            # consent/term checkboxes lose their question entirely
            for idx in self._fields_in_label:
                field_obj = self.fields[idx]
                if not field_obj.label:
                    field_obj.label = text
                    field_obj.question = field_obj.question or text
                    field_obj.category = classify_question_label(
                        f"{text} {field_obj.name}")
                    field_obj.is_consent = field_obj.category == "consent" or (
                        field_obj.field_type == "checkbox"
                        and bool(re.search(r"consent|agree|permission", text, re.I))
                    )
                    field_obj.is_terms = field_obj.category == "terms"
                    field_obj.is_demographic = field_obj.category == "demographic"
        self._fields_in_label = []
        self._in_label = False
        self._current_label_for = None
        self._current_label_text = []

    def _wrapped_label_text(self) -> str:
        return " ".join(self._current_label_text).strip() if self._in_label else ""

    # -- parser events -----------------------------------------------------
    def handle_starttag(self, tag, attrs):
        attrs_d = {k.lower(): (v or "") for k, v in attrs}
        # count EVERY fillable tag occurrence (including hidden/submit and
        # collapsed radio members) so positional selectors match the DOM
        dom_index = -1
        if tag in self._tag_seen:
            dom_index = self._tag_seen[tag]
            self._tag_seen[tag] += 1
        if tag == "form":
            self.has_form = True
        elif tag == "label":
            self._flush_label()
            self._in_label = True
            self._current_label_for = attrs_d.get("for") or None
            self._current_label_text = []
        elif tag == "legend":
            self._in_legend = True
            self._current_legend = []
        elif tag == "select":
            idx = self._add_field(tag, attrs_d, dom_index)
            self._select_field_idx = idx
        elif tag == "option":
            self._pending_option_value = attrs_d.get("value", "")
            self._pending_option_text = []
        elif tag == "input":
            itype = (attrs_d.get("type") or "").lower()
            if itype in _SKIP_INPUT_TYPES:
                if itype == "submit":
                    self.buttons.append(SubmitButton(
                        selector=_css_for("input", attrs_d, dom_index),
                        text=attrs_d.get("value") or "Submit",
                    ))
                return
            self._add_field(tag, attrs_d, dom_index)
        elif tag == "textarea":
            self._add_field(tag, attrs_d, dom_index)

    def handle_endtag(self, tag):
        if tag == "label":
            self._flush_label()
        elif tag == "legend":
            self._in_legend = False
            if self._current_legend:
                self._legend_stack.append(" ".join(self._current_legend).strip())
        elif tag == "option" and self._pending_option_value is not None:
            text = " ".join(self._pending_option_text).strip()
            if (
                self._select_field_idx is not None
                and self._select_field_idx < len(self.fields)
            ):
                value = text or self._pending_option_value
                group = self.fields[self._select_field_idx]
                if value and value not in group.options:
                    group.options.append(value)
            self._pending_option_value = None
            self._pending_option_text = []
        elif tag == "select":
            self._select_field_idx = None

    def handle_data(self, data):
        if self._in_label:
            self._current_label_text.append(data)
        if self._in_legend:
            self._current_legend.append(data)
        if self._pending_option_value is not None:
            self._pending_option_text.append(data)

    # -- field construction --------------------------------------------------
    def _add_field(self, tag: str, attrs: dict, dom_index: int = -1) -> int:
        name = attrs.get("name", "")
        itype_raw = (attrs.get("type") or "").lower()
        dom_id = attrs.get("id", "")
        label = self._labels_by_for.get(dom_id, "") if dom_id else ""
        label = label or self._wrapped_label_text()
        label = label or attrs.get("aria-label", "")
        label = label or attrs.get("placeholder", "")
        ftype = _infer_type(tag, itype_raw, name, label)
        # fall back to the most recently seen free-standing label
        proximity_ok = not (attrs.get("type") == "hidden")
        if not label and self._last_label_text and proximity_ok:
            label = self._last_label_text

        required = (
            attrs.get("required") is not None
            or attrs.get("aria-required", "").lower() == "true"
            or "required" in (attrs.get("class") or "").lower()
            or label.strip().endswith("*")
        )

        question = label.strip().rstrip("*").strip() or humanise_name(name)
        if not question and self._legend_stack:
            question = self._legend_stack[-1]

        field_obj = AnalyzedField(
            selector=_css_for(tag, attrs, dom_index),
            name=name,
            label=label.strip(),
            question=question,
            field_type=ftype,
            required=required,
        )
        field_obj.category = classify_question_label(f"{label} {name}")

        # Refine: "start date month"/"start date year" are date-picker
        # sub-fields, not availability text fields.  The "availability"
        # category produces free-text answers like "AVAILABLE IMMEDIATELY"
        # which are wrong for month selects and year number inputs.
        if field_obj.category == "availability" and re.search(
            r"year|month|day", f"{label} {name}", re.I
        ):
            field_obj.category = "other"

        field_obj.is_consent = field_obj.category == "consent" or (
            ftype == "checkbox"
            and bool(re.search(r"consent|agree|permission", question, re.I))
        )
        field_obj.is_terms = field_obj.category == "terms"
        field_obj.is_demographic = field_obj.category == "demographic"

        key = f"{tag}:{name}" if name else f"{tag}:{dom_id}"
        if ftype in ("radio", "checkbox") and name:
            existing_idx = self._group_index.get(key)
            if existing_idx is not None:
                group = self.fields[existing_idx]
                value = attrs.get("value", "")
                if value and value not in group.options:
                    group.options.append(value)
                group.required = group.required or required
                return existing_idx
            self._group_index[key] = len(self.fields)
            if attrs.get("value"):
                field_obj.options.append(attrs["value"])

        self.fields.append(field_obj)
        if self._in_label:
            self._fields_in_label.append(len(self.fields) - 1)
        return len(self.fields) - 1


_BUTTON_TEXT_RE = re.compile(r"submit|send application|apply now|confirm", re.IGNORECASE)

# employer explicitly prohibits AI-generated / non-original content
_OWN_WORDS_RE = re.compile(
    r"your own words"
    r"|without (?:the use of )?ai"
    r"|ai[- ]generated (?:content|answers|responses) (?:are |is )?not"
    r"|do not use ai"
    r"|may not use ai"
    r"|prohibit\w* (?:the use of )?ai",
    re.IGNORECASE,
)


def analyze_application_page(html: str, page_url: str = "", platform: str = "") -> FormAnalysis:
    """Analyse a REAL application page's HTML into structured form data."""
    analysis = FormAnalysis(page_url=page_url, platform=platform)

    if _OWN_WORDS_RE.search(html):
        analysis.own_words_required = True
        analysis.notes.append(
            "Employer requires answers in your own words — AI-generated "
            "drafts will not be auto-submitted"
        )

    challenge = detect_challenge(html, page_url)
    if challenge is not None:
        analysis.challenge = challenge
        analysis.notes.append(
            f"Page is gated by {challenge.kind}; analysis stopped to avoid bypassing it"
        )
        return analysis

    parser = _FormParser()
    try:
        parser.feed(html)
    except Exception as exc:
        analysis.notes.append(f"HTML parse warning: {exc}")

    analysis.has_form = parser.has_form or bool(parser.fields)
    analysis.form_selector = "form" if parser.has_form else ""
    analysis.fields = parser.fields

    if parser.buttons:
        preferred = next(
            (b for b in parser.buttons if _BUTTON_TEXT_RE.search(b.text)),
            parser.buttons[-1],
        )
        analysis.submit_button = preferred
    else:
        match = re.search(
            r'<button[^>]*type=["\']?submit["\']?[^>]*>(.*?)</button>',
            html, re.IGNORECASE | re.DOTALL,
        )
        if match:
            text = re.sub(r"<[^>]+>", " ", match.group(1))
            analysis.submit_button = SubmitButton(selector="button[type='submit']", text=text.strip()[:80])

    if not analysis.has_form:
        analysis.notes.append("No application form was found on this page")
    return analysis
