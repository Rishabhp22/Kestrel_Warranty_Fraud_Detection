"""
tests/test_api.py -- Comprehensive test suite for Kestrel Fraud Detection API.
"""

import sys
import json
import io
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "api"))

from app import app


def test_api():
    print("=" * 60)
    print("  RUNNING API VERIFICATION TEST SUITE")
    print("=" * 60)

    client = app.test_client()

    # 1. Test GET / (Dashboard UI)
    print("\n1. Testing GET / (Dashboard UI)...")
    res = client.get("/")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert "Kestrel Home Appliances" in res.data.decode("utf-8")
    assert "Warranty Fraud Risk Assessment Platform" in res.data.decode("utf-8")
    print("  [PASS] GET / returned 200 and rendered dashboard HTML.")

    # 2. Test POST /predict (Valid Request)
    print("\n2. Testing POST /predict (Valid payload)...")
    valid_payload = {
        "claim_id": "WC711348",
        "submitted_at": "2026-07-01 00:41",
        "partner_id": "SP3219",
        "sku": "KH-MG-03",
        "product_serial": "KH395147108",
        "days_since_purchase": 577,
        "claim_amount_inr": 3730.0,
        "photo_attached": "Y",
        "partner_inspected": "Y",
        "claim_description": "power button not working",
        "inspector_note": "Minor fault, part swapped",
        "customer_prior_claims": 1,
        "source": "crm"
    }

    res = client.post("/predict", json=valid_payload)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.data}"
    data = res.get_json()

    assert data["claim_id"] == "WC711348"
    assert "fraud_score" in data
    assert isinstance(data["fraud_score"], float)
    assert 0.0 <= data["fraud_score"] <= 1.0
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert "reasons" in data
    assert isinstance(data["reasons"], list)
    assert 3 <= len(data["reasons"]) <= 5

    print(f"  [PASS] Valid prediction response received:")
    print(f"         Claim ID   : {data['claim_id']}")
    print(f"         Fraud Score: {data['fraud_score']}")
    print(f"         Risk Level : {data['risk_level']}")
    print(f"         Reasons ({len(data['reasons'])}):")
    for r in data["reasons"]:
        print(f"           - {r}")

    # 3. Test POST /predict (Missing Fields & Invalid JSON)
    print("\n3. Testing POST /predict error handling (Missing fields & Invalid types)...")
    # Missing partner_id
    bad_payload = valid_payload.copy()
    del bad_payload["partner_id"]
    res = client.post("/predict", json=bad_payload)
    assert res.status_code == 400, f"Expected 400, got {res.status_code}"
    err_json = res.get_json()
    assert "missing_fields" in err_json
    print("  [PASS] Properly caught missing field with HTTP 400 and clear message.")

    # Non-numeric claim_amount_inr
    bad_payload2 = valid_payload.copy()
    bad_payload2["claim_amount_inr"] = "INVALID_AMOUNT"
    res = client.post("/predict", json=bad_payload2)
    assert res.status_code == 400
    print("  [PASS] Properly caught invalid numeric value with HTTP 400.")

    # Non-JSON content type
    res = client.post("/predict", data="plain text body")
    assert res.status_code == 400
    print("  [PASS] Properly rejected non-JSON body with HTTP 400.")

    # 4. Test POST /predict_bulk (CSV Upload)
    print("\n4. Testing POST /predict_bulk with test_unlabelled.csv slice...")
    test_slice = pd.read_csv(ROOT / "test_unlabelled.csv", nrows=10)
    csv_buf = io.BytesIO()
    test_slice.to_csv(csv_buf, index=False)
    csv_buf.seek(0)

    # Test direct CSV response
    res = client.post(
        "/predict_bulk",
        data={"file": (csv_buf, "test_slice.csv")},
        content_type="multipart/form-data"
    )
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.data}"
    assert "text/csv" in res.headers["Content-Type"]
    scored_csv_text = res.data.decode("utf-8")
    df_scored = pd.read_csv(io.StringIO(scored_csv_text))
    assert len(df_scored) == 10
    assert "fraud_score" in df_scored.columns
    assert "risk_level" in df_scored.columns
    print(f"  [PASS] Direct CSV return verified ({len(df_scored)} rows, columns: {list(df_scored.columns)[-2:]})")

    # Test JSON format for AJAX dashboard
    csv_buf2 = io.BytesIO()
    test_slice.to_csv(csv_buf2, index=False)
    csv_buf2.seek(0)
    res_json = client.post(
        "/predict_bulk?format=json",
        data={"file": (csv_buf2, "test_slice.csv")},
        content_type="multipart/form-data"
    )
    assert res_json.status_code == 200
    json_data = res_json.get_json()
    assert json_data["total_rows"] == 10
    assert "preview" in json_data
    assert "download_url" in json_data
    print(f"  [PASS] AJAX JSON format verified (avg score: {json_data['average_score']}, preview rows: {len(json_data['preview'])})")

    # Test empty file error handling
    res_bad = client.post(
        "/predict_bulk",
        data={"file": (io.BytesIO(b""), "empty.csv")},
        content_type="multipart/form-data"
    )
    assert res_bad.status_code == 400
    print("  [PASS] Properly rejected empty CSV upload with HTTP 400.")

    # 5. Test GET /analytics
    print("\n5. Testing GET /analytics...")
    res = client.get("/analytics")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    ana_data = res.get_json()
    assert ana_data["total_claims"] == 2252
    assert "average_fraud_score" in ana_data
    assert "highest_fraud_score" in ana_data
    assert "risk_distribution" in ana_data
    assert "HIGH" in ana_data["risk_distribution"]
    assert "MEDIUM" in ana_data["risk_distribution"]
    assert "LOW" in ana_data["risk_distribution"]
    assert "top_risk_partners" in ana_data
    assert len(ana_data["top_risk_partners"]) > 0

    print("  [PASS] Analytics returned valid population statistics:")
    print(f"         Total Claims     : {ana_data['total_claims']}")
    print(f"         Mean Fraud Score : {ana_data['average_fraud_score']:.4f}")
    print(f"         Max Fraud Score  : {ana_data['highest_fraud_score']:.4f} (Claim: {ana_data.get('highest_claim_id')})")
    print(f"         Risk Distribution: High={ana_data['risk_distribution']['HIGH']['count']}, Med={ana_data['risk_distribution']['MEDIUM']['count']}, Low={ana_data['risk_distribution']['LOW']['count']}")
    print(f"         Top Risk Partners: {len(ana_data['top_risk_partners'])} partner summaries loaded.")

    print("\n" + "=" * 60)
    print("  ALL API AND PIPELINE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    test_api()
