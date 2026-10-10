"""
tests/test_alerts.py -- Test suite for Zapier Catch Hook integration.

Verifies:
1. HIGH-risk claims trigger one webhook request.
2. LOW- and MEDIUM-risk claims do not trigger requests.
3. A missing webhook URL does not break prediction (NOT_CONFIGURED).
4. A webhook timeout or non-success response does not break prediction (failsafe).
5. Repeated requests for the same claim do not generate duplicate successful alerts (idempotency).
6. Failed delivery can be retried safely.
7. Existing API routes, dashboard and bulk download functionality still work.
"""

import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "api"))

from app import app
from alerts import (
    send_high_risk_alert,
    build_webhook_payload,
    is_alert_already_sent,
    init_alert_store,
    mask_url
)

# Test payloads
HIGH_RISK_CLAIM = {
    "claim_id": "WC712506",
    "submitted_at": "2026-08-16 12:29",
    "partner_id": "SP3260",
    "sku": "KH-MG-01",
    "product_serial": "KH450246940",
    "days_since_purchase": 19,
    "claim_amount_inr": 2131.0,
    "photo_attached": "Y",
    "partner_inspected": "Y",
    "claim_description": "water leaking",
    "customer_prior_claims": 2,
    "source": "crm",
    "inspector_note": "Confidential investigation note that must never leak"
}

MEDIUM_RISK_CLAIM = {
    "claim_id": "WC712530",
    "submitted_at": "2026-08-18 00:08",
    "partner_id": "SP3096",
    "sku": "KH-IC-03",
    "product_serial": "kh814519714",
    "days_since_purchase": 97,
    "claim_amount_inr": 3188.0,
    "photo_attached": "Y",
    "partner_inspected": "Y",
    "claim_description": "blade jammed",
    "customer_prior_claims": 1,
    "source": "crm"
}

LOW_RISK_CLAIM = {
    "claim_id": "WC711748",
    "submitted_at": "2026-07-15 19:09",
    "partner_id": "SP3214",
    "sku": "KH-RH-01",
    "product_serial": "KH800724084",
    "days_since_purchase": 138,
    "claim_amount_inr": 681.0,
    "photo_attached": "Y",
    "partner_inspected": "N",
    "claim_description": "loud noise while running",
    "customer_prior_claims": 0,
    "source": "crm"
}


def run_all_alert_tests():
    print("=" * 65)
    print("  RUNNING ZAPIER INTEGRATION & RELIABILITY TEST SUITE")
    print("=" * 65)

    client = app.test_client()

    # 1. HIGH-risk claims trigger one webhook request & check privacy payload
    print("\n1. Testing HIGH-risk claims trigger exactly one webhook request...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_db = Path(tmp_dir) / "test_alerts.db"
        fake_url = "https://hooks.zapier.com/hooks/catch/123456/abcdef/"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"status": "success", "id": "zap123"}'

        with patch.dict(os.environ, {"ZAPIER_WEBHOOK_URL": fake_url}), \
             patch("alerts.DEFAULT_DB_PATH", test_db), \
             patch("requests.post", return_value=mock_resp) as mock_post:

            res = client.post("/predict", json=HIGH_RISK_CLAIM)
            assert res.status_code == 200, f"Expected 200, got {res.status_code}"
            data = res.get_json()

            assert data["risk_level"] == "HIGH"
            assert data["alert"]["triggered"] is True
            assert data["alert"]["status"] == "SENT"
            assert data["alert"]["http_status"] == 200

            # Verify mock called exactly once
            assert mock_post.call_count == 1
            call_args, call_kwargs = mock_post.call_args
            assert call_args[0] == fake_url
            sent_payload = call_kwargs["json"]

            # Validate payload schema
            assert sent_payload["event"] == "high_risk_claim"
            assert sent_payload["claim_id"] == "WC712506"
            assert sent_payload["risk_level"] == "HIGH"
            assert isinstance(sent_payload["fraud_score"], float)
            assert isinstance(sent_payload["reasons"], list)
            assert len(sent_payload["reasons"]) >= 3
            assert sent_payload["claim_amount_inr"] == 2131.0
            assert sent_payload["submitted_at"] == "2026-08-16 12:29"
            assert "alert_created_at" in sent_payload

            # CRITICAL SECURITY CHECK: inspector_note is NEVER sent
            assert "inspector_note" not in sent_payload
            assert "Confidential" not in str(sent_payload)

            print("  [PASS] HIGH-risk claim triggered 1 webhook with safe payload and reasons.")

    # 2. LOW- and MEDIUM-risk claims do not trigger webhook
    print("\n2. Testing LOW- and MEDIUM-risk claims do NOT trigger webhooks...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_db = Path(tmp_dir) / "test_alerts.db"
        fake_url = "https://hooks.zapier.com/hooks/catch/123456/abcdef/"

        with patch.dict(os.environ, {"ZAPIER_WEBHOOK_URL": fake_url}), \
             patch("alerts.DEFAULT_DB_PATH", test_db), \
             patch("requests.post") as mock_post:

            # Test Low Risk
            res_low = client.post("/predict", json=LOW_RISK_CLAIM)
            assert res_low.status_code == 200
            data_low = res_low.get_json()
            assert data_low["risk_level"] == "LOW"
            assert data_low["alert"]["triggered"] is False
            assert data_low["alert"]["status"] == "NOT_TRIGGERED"

            # Test Medium Risk
            res_med = client.post("/predict", json=MEDIUM_RISK_CLAIM)
            assert res_med.status_code == 200
            data_med = res_med.get_json()
            assert data_med["risk_level"] == "MEDIUM"
            assert data_med["alert"]["triggered"] is False
            assert data_med["alert"]["status"] == "NOT_TRIGGERED"

            mock_post.assert_not_called()
            print("  [PASS] Neither LOW nor MEDIUM claims triggered any outbound requests.")

    # 3. Missing webhook URL does not break prediction
    print("\n3. Testing missing webhook URL does not break prediction...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_db = Path(tmp_dir) / "test_alerts.db"

        # Explicitly empty webhook URL
        with patch.dict(os.environ, {"ZAPIER_WEBHOOK_URL": ""}), \
             patch("alerts.DEFAULT_DB_PATH", test_db), \
             patch("requests.post") as mock_post:

            res = client.post("/predict", json=HIGH_RISK_CLAIM)
            assert res.status_code == 200, f"Expected 200, got {res.status_code}"
            data = res.get_json()

            assert data["risk_level"] == "HIGH"
            assert data["fraud_score"] >= 0.30
            assert data["alert"]["triggered"] is True
            assert data["alert"]["status"] == "NOT_CONFIGURED"
            mock_post.assert_not_called()
            print("  [PASS] Prediction succeeded normally with alert.status == 'NOT_CONFIGURED'.")

    # 4. Webhook timeout or non-success response does not break prediction
    print("\n4. Testing webhook timeout and 500 error handling (failsafe resilience)...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_db = Path(tmp_dir) / "test_alerts.db"
        fake_url = "https://hooks.zapier.com/hooks/catch/123456/abcdef/"

        # 4a. Timeout exception
        with patch.dict(os.environ, {"ZAPIER_WEBHOOK_URL": fake_url}), \
             patch("alerts.DEFAULT_DB_PATH", test_db), \
             patch("requests.post", side_effect=requests.exceptions.Timeout("Connection timed out")):

            res = client.post("/predict", json=HIGH_RISK_CLAIM)
            assert res.status_code == 200, "Timeout must not crash prediction"
            data = res.get_json()
            assert data["claim_id"] == "WC712506"
            assert data["risk_level"] == "HIGH"
            assert data["alert"]["status"] == "FAILED"
            assert "timed out" in data["alert"]["error"].lower()

        # 4b. HTTP 500 Server Error response
        mock_500 = MagicMock()
        mock_500.status_code = 500
        mock_500.text = "Internal Server Error at Zapier"

        with patch.dict(os.environ, {"ZAPIER_WEBHOOK_URL": fake_url}), \
             patch("alerts.DEFAULT_DB_PATH", test_db), \
             patch("requests.post", return_value=mock_500):

            claim_retry = dict(HIGH_RISK_CLAIM)
            claim_retry["claim_id"] = "WC712999"
            res = client.post("/predict", json=claim_retry)
            assert res.status_code == 200, "HTTP 500 must not crash prediction"
            data = res.get_json()
            assert data["risk_level"] == "HIGH"
            assert data["alert"]["status"] == "FAILED"
            assert data["alert"]["http_status"] == 500

            print("  [PASS] Both network timeouts and HTTP 500 errors handled gracefully without breaking predictions.")

    # 5. Repeated requests for same claim do not generate duplicate successful alerts (idempotency)
    print("\n5. Testing idempotency: duplicate predictions do not re-send alerts...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_db = Path(tmp_dir) / "test_alerts.db"
        fake_url = "https://hooks.zapier.com/hooks/catch/123456/abcdef/"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"status": "success"}'

        with patch.dict(os.environ, {"ZAPIER_WEBHOOK_URL": fake_url}), \
             patch("alerts.DEFAULT_DB_PATH", test_db), \
             patch("requests.post", return_value=mock_resp) as mock_post:

            # 1st call: Should send
            res1 = client.post("/predict", json=HIGH_RISK_CLAIM)
            assert res1.get_json()["alert"]["status"] == "SENT"
            assert mock_post.call_count == 1

            # 2nd call: Same claim, should NOT send again
            res2 = client.post("/predict", json=HIGH_RISK_CLAIM)
            assert res2.get_json()["alert"]["status"] == "ALREADY_SENT"
            assert mock_post.call_count == 1  # Still 1!

            # 3rd call: Same claim, still blocked
            res3 = client.post("/predict", json=HIGH_RISK_CLAIM)
            assert res3.get_json()["alert"]["status"] == "ALREADY_SENT"
            assert mock_post.call_count == 1  # Still 1!

            print("  [PASS] Idempotency store blocked duplicate delivery (call count remained 1).")

    # 6. Failed delivery can be retried safely
    print("\n6. Testing failed delivery can be retried safely...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_db = Path(tmp_dir) / "test_alerts.db"
        fake_url = "https://hooks.zapier.com/hooks/catch/123456/abcdef/"

        # Attempt 1: Network failure
        with patch.dict(os.environ, {"ZAPIER_WEBHOOK_URL": fake_url}), \
             patch("alerts.DEFAULT_DB_PATH", test_db), \
             patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")):

            claim_retry = dict(HIGH_RISK_CLAIM)
            claim_retry["claim_id"] = "WC718888"
            res1 = client.post("/predict", json=claim_retry)
            assert res1.get_json()["alert"]["status"] == "FAILED"

        # Attempt 2: Network recovered, should retry and succeed!
        mock_success = MagicMock()
        mock_success.status_code = 200
        mock_success.text = '{"status": "success"}'

        with patch.dict(os.environ, {"ZAPIER_WEBHOOK_URL": fake_url}), \
             patch("alerts.DEFAULT_DB_PATH", test_db), \
             patch("requests.post", return_value=mock_success) as mock_post_retry:

            res2 = client.post("/predict", json=claim_retry)
            assert res2.get_json()["alert"]["status"] == "SENT"
            assert mock_post_retry.call_count == 1

        # Attempt 3: Now marked SENT, should not resend
        with patch.dict(os.environ, {"ZAPIER_WEBHOOK_URL": fake_url}), \
             patch("alerts.DEFAULT_DB_PATH", test_db), \
             patch("requests.post") as mock_post_blocked:

            res3 = client.post("/predict", json=claim_retry)
            assert res3.get_json()["alert"]["status"] == "ALREADY_SENT"
            mock_post_blocked.assert_not_called()

            print("  [PASS] Retry successfully executed after failure, then became idempotent once SENT.")

    # 7. Existing routes, dashboard and bulk download functionality still work
    print("\n7. Testing existing routes: dashboard, bulk scoring, and analytics...")
    # Dashboard GET /
    res_dash = client.get("/")
    assert res_dash.status_code == 200
    assert "Warranty Fraud Risk Assessment Platform" in res_dash.data.decode("utf-8")

    # Analytics GET /analytics
    res_ana = client.get("/analytics")
    assert res_ana.status_code == 200
    ana_data = res_ana.get_json()
    assert ana_data["total_claims"] == 2252
    assert "risk_distribution" in ana_data

    # Bulk scoring POST /predict_bulk
    csv_sample = "claim_id,submitted_at,partner_id,sku,product_serial,days_since_purchase,claim_amount_inr,photo_attached,partner_inspected,claim_description,customer_prior_claims,source\nWC711748,2026-07-15 19:09,SP3214,KH-RH-01,KH800724084,138,681.0,Y,N,loud noise,0,crm\n"
    res_bulk = client.post("/predict_bulk", data={"file": (io.BytesIO(csv_sample.encode("utf-8")), "batch.csv")})
    assert res_bulk.status_code == 200
    assert "fraud_score,risk_level" in res_bulk.data.decode("utf-8")

    print("  [PASS] All existing routes (dashboard, analytics, bulk scoring) functioning perfectly.")

    print("\n" + "=" * 65)
    print("  ALL 7 ZAPIER INTEGRATION AND RELIABILITY TESTS PASSED!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    import io
    run_all_alert_tests()
