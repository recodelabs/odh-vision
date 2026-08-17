import csv
import importlib
import json

cmp_mod = None


def setup_module(module):
    global cmp_mod
    cmp_mod = importlib.import_module("1d_compare_models")


def _write_extraction(base, model, stem, values):
    """values: {record_key: {field: (value, conf)}}"""
    d = base / model
    d.mkdir(parents=True, exist_ok=True)
    from extraction import FIELD_NAMES
    records = {}
    for rk, fields in values.items():
        f = {name: {"value": "", "confidence": "high"} for name in FIELD_NAMES}
        for name, (v, c) in fields.items():
            f[name] = {"value": v, "confidence": c}
        records[rk] = {"fields": f, "usage": {"input_tokens": 1,
                                              "output_tokens": 1,
                                              "latency_s": 0.1}}
    (d / f"{stem}.json").write_text(json.dumps(
        {"stem": stem, "model": model, "prompt_version": "1",
         "records": records, "totals": {}}))


def test_norm():
    assert cmp_mod.norm("  Aromo   KETTY") == "aromo ketty"


def test_compare_and_csv(tmp_path):
    _write_extraction(tmp_path, "m1", "p1", {
        "1": {"patient_name": ("Aromo Ketty", "high"),
              "diagnosis": ("Malaria", "high")},
        "2": {"patient_name": ("Akello", "medium")},
    })
    _write_extraction(tmp_path, "m2", "p1", {
        "1": {"patient_name": ("aromo  ketty", "medium"),      # agrees (norm)
              "diagnosis": ("PID", "low")},                   # disagrees
        "2": {"patient_name": ("Akello", "high")},            # agrees
    })
    result = cmp_mod.compare_extractions(str(tmp_path / "m1"),
                                         str(tmp_path / "m2"))
    assert result["n_records"] == 2
    assert len(result["disagreements"]) == 1
    d = result["disagreements"][0]
    assert (d["field"], d["a_value"], d["b_value"]) == ("diagnosis",
                                                        "Malaria", "PID")
    assert result["per_field"]["diagnosis"]["rate"] == 0.0
    assert result["per_field"]["patient_name"]["rate"] == 1.0
    assert result["both_empty"] > 0

    out_csv = tmp_path / "cmp.csv"
    cmp_mod.write_csv(result["disagreements"], str(out_csv))
    rows = list(csv.DictReader(open(out_csv)))
    assert len(rows) == 1 and rows[0]["field"] == "diagnosis"
