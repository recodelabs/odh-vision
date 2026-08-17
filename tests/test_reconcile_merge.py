import pytest

from extraction import FIELD_NAMES
from reconciliation import (classify_strips, is_continuation, merge_patient,
                            continuation_conflict, VALIDATORS)


def F(**over):
    """All-empty extraction fields dict with overrides: F(patient_name='X')."""
    f = {n: {"value": "", "confidence": "high"} for n in FIELD_NAMES}
    for k, v in over.items():
        f[k] = {"value": v, "confidence": "high"}
    return f


PRIMARY = dict(record_no="304", patient_name="Aciro Rose", sex="M",
               age_yrs="26", village="Katuru", day="15", month="3",
               diagnosis="PID", treatment_line1="T. O cef 2g stat",
               treatment_line2="T. O centa stat", tab_no="6",
               full_cost="26000", balance="21000")


def test_is_continuation_true_for_treatment_overflow():
    prim = F(**PRIMARY)
    cont = F(treatment_line1="T. Nitro 100mg bd/7",
             treatment_line2="T. Ibuprofen 400mg", tab_no="10")
    assert is_continuation(prim, cont)


def test_is_continuation_false_on_new_identity():
    prim = F(**PRIMARY)
    other = F(record_no="305", patient_name="Namono Grace",
              treatment_line1="T. Cipro")
    assert not is_continuation(prim, other)


def test_continuation_conflict_same_recno_different_name():
    prim = F(**PRIMARY)
    odd = F(record_no="304", patient_name="Someone Else",
            treatment_line1="T. X")
    assert continuation_conflict(prim, odd) == "ambiguous-continuation"


def test_is_continuation_true_on_name_match_despite_recno_mismatch():
    # Margin record_no digits are proven unreliable; a fuzzy name match
    # (ratio >= 0.75) must classify as a continuation even though the
    # record_no values differ and are both present.
    prim = F(**{**PRIMARY, "record_no": "307"})
    cont = F(record_no="3067", patient_name="Aciro Rose",
             treatment_line1="T. X")
    assert is_continuation(prim, cont)


def test_continuation_conflict_none_when_name_matches_despite_recno_match():
    # Same record_no AND matching name is a normal continuation, not a
    # conflict.
    prim = F(**PRIMARY)
    same = F(record_no="304", patient_name="Aciro Rose",
             treatment_line1="T. X")
    assert continuation_conflict(prim, same) is None


def test_classify_strips_recno_mismatch_name_match_merges_with_warning():
    records = {
        "1": {"fields": F(record_no="307", patient_name="Aciro Rose",
                          sex="M", diagnosis="PID", full_cost="26000")},
        "2": {"fields": F(record_no="3067", patient_name="Aciro Rose",
                          treatment_line1="T. X")},
    }
    kinds = classify_strips(records)
    assert kinds == [
        (1, "primary", None),
        (2, "continuation", "recno-mismatch-name-match"),
    ]


def test_classify_strips_primary_cont_empty():
    records = {
        "1": {"fields": F(**PRIMARY)},
        "2": {"fields": F(treatment_line1="T. Nitro", tab_no="10")},
        "3": {"fields": F(record_no="305", patient_name="Namono Grace",
                          diagnosis="PUD", full_cost="28000")},
        "4": {"fields": F()},                       # blank block
    }
    kinds = classify_strips(records)
    assert [(k, kind) for k, kind, _ in kinds] == [
        (1, "primary"), (2, "continuation"), (3, "primary"), (4, "empty")]


def test_merge_concats_treatments_and_fills_empties():
    prim = F(**PRIMARY)
    cont = F(treatment_line1="T. Nitro 100mg", treatment_line3="T. Pcm",
             tab_no="10", cost_after_discount="20000")
    m = merge_patient([(1, prim), (2, cont)])
    tvals = [t["value"] for t in m["fields"]["treatments"]]
    assert tvals == ["T. O cef 2g stat", "T. O centa stat",
                     "T. Nitro 100mg", "T. Pcm"]
    assert [t["value"] for t in m["fields"]["tab_nos"]] == ["6", "10"]
    assert m["fields"]["cost_after_discount"]["value"] == "20000"
    assert "cost_after_discount" in m["filled_from_continuation"]
    assert m["source_strips"] == [1, 2] and m["merged_from"] == [2]
    assert m["review"] is False


def test_merge_resolves_invalid_day_by_validator():
    prim = F(**{**PRIMARY, "day": "86"})            # ink blot
    cont = F(day="16", treatment_line1="T. X")
    m = merge_patient([(1, prim), (2, cont)])
    assert m["fields"]["day"]["value"] == "16"
    assert m["resolved_conflicts"][0]["field"] == "day"
    assert m["resolved_conflicts"][0]["rejected"] == "86"
    assert m["review"] is False


def test_merge_flags_unresolvable_conflict():
    prim = F(**PRIMARY)                              # full_cost 26000
    cont = F(full_cost="99000", treatment_line1="T. X")   # both valid, differ
    m = merge_patient([(1, prim), (2, cont)])
    assert m["fields"]["full_cost"]["value"] == "26000"   # primary kept
    assert m["review"] is True
    assert m["warnings"][0]["field"] == "full_cost"


def test_merge_keeps_primary_recno_silently_on_mismatch():
    # record_no is known-unreliable; a differing continuation record_no
    # must not be flagged as a merge conflict or force review — that
    # signal is surfaced separately by classify_strips/reconcile_page.
    prim = F(**{**PRIMARY, "record_no": "307"})
    cont = F(record_no="3067", patient_name="Aciro Rose",
             treatment_line1="T. X")
    m = merge_patient([(1, prim), (2, cont)])
    assert m["fields"]["record_no"]["value"] == "307"
    assert m["review"] is False
    assert not any(w.get("field") == "record_no" for w in m["warnings"])


def test_validators_shape():
    assert VALIDATORS["day"]("16") and not VALIDATORS["day"]("86")
    assert VALIDATORS["month"]("3") and not VALIDATORS["month"]("48")
    assert VALIDATORS["sex"]("F") and not VALIDATORS["sex"]("X")
    assert VALIDATORS["full_cost"]("26000") and not VALIDATORS["full_cost"]("26k")
