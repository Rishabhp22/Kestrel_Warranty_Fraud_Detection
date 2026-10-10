"""
tests/test_upload_workflow.py -- Automated verification for upload-first workflow.
Tests all endpoints, schema validation, demo dataset pre-scored path,
bulk scoring path, and predictions.csv preservation.
"""

import io
import requests
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:5000"


def test_upload_workflow():
    print("=" * 65)
    print("  TESTING UPLOAD-FIRST WORKFLOW AND ROUTE PIPELINE")
    print("=" * 65)

    # 1. Test GET / (Landing Page)
    print("\n1. Testing GET / (Fresh Session Landing Page)...")
    res = requests.get(f"{BASE}/")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    html = res.text

    assert "Kestrel Warranty Intelligence" in html, "Missing required heading"
    assert "Upload your warranty claims or explore the demo dataset to begin risk analysis." in html, "Missing required subtitle"
    assert "Upload Claims" in html, "Missing Upload Claims action"
    assert "Analyze Demo" in html, "Missing Analyze Demo action"
    assert "view-landing" in html, "Missing view-landing element"
    assert "view-select" in html, "Missing view-select element"
    assert "selection-preview-box" in html, "Missing preview box"
    assert "btn-trigger-analyze" in html, "Missing explicit Analyze Data button"
    print("  [PASS] Fresh session starts with clean landing page and zero stale counts.")

    # 2. Test GET /api/demo_info
    print("\n2. Testing GET /api/demo_info (Demo Dataset Metadata)...")
    res = requests.get(f"{BASE}/api/demo_info")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    demo_info = res.json()
    assert demo_info["filename"] == "test_unlabelled.csv"
    assert demo_info["label"] == "Demo Dataset"
    assert demo_info["total_rows"] == 2252
    assert len(demo_info["preview"]) == 5
    assert len(demo_info["required_fields"]) == 12
    print(f"  [PASS] Demo dataset inspected: {demo_info['filename']} with {demo_info['total_rows']} rows.")

    # 3. Test POST /validate_csv (Valid & Invalid CSV files)
    print("\n3. Testing POST /validate_csv (Schema Validation without scoring)...")
    test_slice = pd.read_csv(ROOT / "test_unlabelled.csv", nrows=10)
    valid_buf = io.BytesIO()
    test_slice.to_csv(valid_buf, index=False)
    valid_buf.seek(0)

    # Valid CSV
    res_val = requests.post(f"{BASE}/validate_csv", files={"file": ("my_claims.csv", valid_buf, "text/csv")})
    assert res_val.status_code == 200, f"Expected 200, got {res_val.status_code}"
    val_data = res_val.json()
    assert val_data["valid"] is True
    assert val_data["filename"] == "my_claims.csv"
    assert val_data["total_rows"] == 10
    assert len(val_data["preview"]) == 5
    print("  [PASS] Valid CSV parsed and verified without executing model scoring.")

    # Invalid CSV (missing columns)
    bad_buf = io.BytesIO(b"claim_id,product_serial\nWC111,KH999")
    res_bad = requests.post(f"{BASE}/validate_csv", files={"file": ("invalid.csv", bad_buf, "text/csv")})
    assert res_bad.status_code == 400, f"Expected 400, got {res_bad.status_code}"
    bad_data = res_bad.json()
    assert bad_data["valid"] is False
    assert "missing_columns" in bad_data
    assert len(bad_data["missing_columns"]) == 10
    print(f"  [PASS] Invalid CSV rejected with clear missing columns explanation ({len(bad_data['missing_columns'])} missing).")

    # 4. Test GET /api/demo_claims (Analyze Demo flow)
    print("\n4. Testing GET /api/demo_claims (Pre-scored Demo Loading Flow)...")
    res_demo = requests.get(f"{BASE}/api/demo_claims")
    assert res_demo.status_code == 200, f"Expected 200, got {res_demo.status_code}"
    demo_claims = res_demo.json()

    assert demo_claims["filename"] == "test_unlabelled.csv"
    assert demo_claims["total_claims"] == 2252
    assert demo_claims["high_risk_count"] == 11
    assert demo_claims["medium_risk_count"] == 70
    assert demo_claims["low_risk_count"] == 2171
    assert demo_claims["high_risk_pct"] == 0.49
    assert demo_claims["medium_risk_pct"] == 3.11
    assert demo_claims["low_risk_pct"] == 96.4
    assert demo_claims["highest_fraud_score"] > 0.60
    assert "score_distribution" in demo_claims
    assert len(demo_claims["claims"]) == 2252
    print(f"  [PASS] Demo claims verified: 2,252 rows (High: 11, Medium: 70, Low: 2,171) loaded from pre-scored path.")

    # 5. Test POST /predict_bulk?format=json (Upload Scoring Flow)
    print("\n5. Testing POST /predict_bulk?format=json (Custom Batch Scoring Flow)...")
    valid_buf.seek(0)
    res_bulk = requests.post(f"{BASE}/predict_bulk?format=json", files={"file": ("batch_sample.csv", valid_buf, "text/csv")})
    assert res_bulk.status_code == 200, f"Expected 200, got {res_bulk.status_code}"
    bulk_data = res_bulk.json()
    assert bulk_data["total_rows"] == 10
    assert bulk_data["total_claims"] == 10
    assert "average_score" in bulk_data
    assert "high_risk_count" in bulk_data
    assert "medium_risk_count" in bulk_data
    assert "low_risk_count" in bulk_data
    assert len(bulk_data["claims"]) == 10
    assert "score_distribution" in bulk_data
    print(f"  [PASS] Custom upload scored successfully: {bulk_data['total_rows']} claims.")

    # 6. Verify integrity of outputs/predictions.csv
    print("\n6. Verifying outputs/predictions.csv integrity...")
    pred_path = ROOT / "outputs" / "predictions.csv"
    assert pred_path.exists(), "outputs/predictions.csv not found!"
    df_pred = pd.read_csv(pred_path)
    assert len(df_pred) == 2252, f"Expected 2252 rows, found {len(df_pred)}"
    assert list(df_pred.columns) == ["claim_id", "score"], f"Expected ['claim_id', 'score'], found {list(df_pred.columns)}"
    print("  [PASS] outputs/predictions.csv strictly preserved (2,252 rows, columns: ['claim_id', 'score']).")

    print("\n" + "=" * 65)
    print("  ALL UPLOAD-FIRST WORKFLOW VERIFICATIONS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    test_upload_workflow()
