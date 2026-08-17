from extraction import FIELD_NAMES
from reconciliation import possible_split


def F(**over):
    """All-empty extraction fields dict with overrides: F(patient_name='X')."""
    f = {n: {"value": "", "confidence": "high"} for n in FIELD_NAMES}
    for k, v in over.items():
        f[k] = {"value": v, "confidence": "high"}
    return f


def patient(seq, source_strips, treatments=None, tab_nos=None, **over):
    """A minimal already-merged patient dict, shaped like merge_patient's
    output plus the seq/record_no reconcile_page adds."""
    fields = F(**over)
    fields["treatments"] = treatments or []
    fields["tab_nos"] = tab_nos or []
    return {"fields": fields, "source_strips": source_strips, "seq": seq,
            "record_no": over.get("record_no", ""), "warnings": [],
            "review": False}


def test_possible_split_recno_edit_distance_one():
    # "307" vs "3067" — one dropped/extra digit (edit distance 1). Live
    # validation 2026-08-17 round 1/2: true patient 304's own two strips
    # split into record_no "304" and "3067".
    a = patient(1, [1], record_no="307", patient_name="Aciro Rose",
               treatments=[{"value": "T1", "confidence": "high"}])
    b = patient(2, [2], record_no="3067", patient_name="Aciro Rose",
               treatments=[{"value": "T2", "confidence": "high"}])
    assert possible_split(a, b)


def test_possible_split_recno_substring():
    # "31" vs "314" — clipped trailing digits (containment, not edit
    # distance <= 1 given the length gap).
    a = patient(1, [1], record_no="313", patient_name="Aber Susan",
               treatments=[{"value": "T1", "confidence": "high"}])
    b = patient(2, [2], record_no="31", patient_name="Okwir Moses",
               treatments=[{"value": "T2", "confidence": "high"}])
    assert possible_split(a, b)


def test_possible_split_clip_shaped_single_strip_fragment():
    # patient_b is a single-strip fragment with continuation-shaped
    # content (a treatment line), the whole checkbox cluster empty, and
    # at most 2 CLIP_FIELDS populated (just patient_name here) — the
    # clip-signature shape of a boundary fragment misclassified as its
    # own patient, even though record_no values themselves aren't close.
    a = patient(1, [1], record_no="304", patient_name="Acire Rosemary",
               sex="M", first_time_odh="N", hh_owns_phone="Y",
               hh_owns_toilet="Y", result_pn="P", diagnosis="PID",
               full_cost="26000",
               treatments=[{"value": "T1", "confidence": "high"}])
    b = patient(2, [2], record_no="9999", patient_name="Adong Florence",
               treatments=[{"value": "T2", "confidence": "high"}])
    assert possible_split(a, b)


def test_possible_split_negative_two_distinct_patients():
    a = patient(1, [1], record_no="304", patient_name="Aciro Rose",
               sex="M", first_time_odh="N", hh_owns_phone="Y",
               hh_owns_toilet="Y", result_pn="P", diagnosis="PID",
               full_cost="26000",
               treatments=[{"value": "T1", "confidence": "high"}])
    b = patient(2, [2, 3], record_no="305", patient_name="Okello Sam",
               sex="M", first_time_odh="N", hh_owns_phone="Y",
               hh_owns_toilet="N", result_pn="N", diagnosis="PUD",
               full_cost="28000",
               treatments=[{"value": "T2", "confidence": "high"}])
    assert not possible_split(a, b)
