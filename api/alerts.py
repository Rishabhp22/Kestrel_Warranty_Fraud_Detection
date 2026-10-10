"""
api/alerts.py -- Reliable Zapier Webhook Email Alerting for HIGH Risk Claims.

Handles:
1. Environment variable loading for ZAPIER_WEBHOOK_URL.
2. Idempotent alert dispatch using a lightweight SQLite event store.
3. Payload construction adhering strictly to data privacy & leakage rules (no inspector_note).
4. Graceful error handling and URL masking to protect secrets and ensure prediction resilience.
"""

import os
import sys
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import requests

logger = logging.getLogger("kestrel.alerts")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = ROOT / "outputs" / "alerts.db"


def load_env(env_path: Optional[Path] = None) -> None:
    """
    Minimal .env loader that populates os.environ without overriding existing keys.
    Does not require external dependencies like python-dotenv.
    """
    if env_path is None:
        env_path = ROOT / ".env"
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception as e:
            logger.warning(f"Error reading .env file: {e}")


def get_webhook_url() -> Optional[str]:
    """
    Retrieve Zapier Catch Hook URL from environment or .env file.
    Returns None if unset or empty string.
    """
    load_env()
    url = os.environ.get("ZAPIER_WEBHOOK_URL", "").strip()
    return url if url else None


def mask_url(url: Optional[str]) -> str:
    """
    Safely mask a webhook URL for logging without exposing path secrets or tokens.
    """
    if not url:
        return "[UNCONFIGURED]"
    if len(url) <= 30:
        return url[:10] + "***"
    return url[:28] + "...***"


def init_alert_store(db_path: Optional[Path] = None) -> Path:
    """
    Initialize SQLite store for tracking alert delivery events and enforcing idempotency.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS alert_events (
                idempotency_key TEXT PRIMARY KEY,
                claim_id TEXT NOT NULL,
                event TEXT NOT NULL,
                status TEXT NOT NULL,
                http_status INTEGER,
                created_at TEXT NOT NULL,
                sent_at TEXT,
                error_message TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def is_alert_already_sent(
    claim_id: str,
    event: str = "high_risk_claim",
    db_path: Optional[Path] = None
) -> bool:
    """
    Check if a successful alert was already dispatched for this claim and event.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    if not db_path.exists():
        return False

    idempotency_key = f"{claim_id}:{event}"
    conn = None
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status FROM alert_events WHERE idempotency_key = ?",
            (idempotency_key,)
        )
        row = cursor.fetchone()
        if row and row[0] == "SENT":
            return True
    except Exception as e:
        logger.warning(f"Error checking alert idempotency: {e}")
    finally:
        if conn:
            conn.close()
    return False


def build_webhook_payload(claim_data: Dict[str, Any], prediction_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build privacy-safe triage webhook payload.
    Strictly excludes free-text inspector_note and sensitive raw customer IDs.
    """
    amt = claim_data.get("claim_amount_inr")
    if amt is not None:
        try:
            amt = float(amt)
        except (ValueError, TypeError):
            amt = None

    sub_time = claim_data.get("submitted_at")
    if sub_time is not None:
        sub_time = str(sub_time).strip()

    return {
        "event": "high_risk_claim",
        "claim_id": str(claim_data.get("claim_id", "")).strip(),
        "fraud_score": float(prediction_result.get("fraud_score", 0.0)),
        "risk_level": "HIGH",
        "reasons": list(prediction_result.get("reasons", [])),
        "claim_amount_inr": amt,
        "submitted_at": sub_time,
        "alert_created_at": datetime.now(timezone.utc).isoformat()
    }


def send_high_risk_alert(
    claim_data: Dict[str, Any],
    prediction_result: Dict[str, Any],
    db_path: Optional[Path] = None,
    timeout: float = 4.0
) -> Dict[str, Any]:
    """
    Dispatch an alert to Zapier Catch Hook if claim is HIGH risk.
    
    Guarantees:
    - Never raises unhandled exceptions to caller.
    - Idempotent: will not resend if already marked SENT in SQLite.
    - Allows retry if prior attempt FAILED.
    - Masked logging prevents secret leakage.
    - Only triggered when risk_level == 'HIGH'.
    """
    risk_level = prediction_result.get("risk_level")
    if risk_level != "HIGH":
        return {
            "triggered": False,
            "status": "NOT_TRIGGERED",
            "message": f"Claim risk level is {risk_level}; alerts only fire for HIGH risk."
        }

    claim_id = str(claim_data.get("claim_id", "")).strip()
    event_type = "high_risk_claim"
    idempotency_key = f"{claim_id}:{event_type}"

    # Verify webhook URL configuration
    webhook_url = get_webhook_url()
    if not webhook_url:
        return {
            "triggered": True,
            "status": "NOT_CONFIGURED",
            "message": "Zapier webhook URL is not configured (ZAPIER_WEBHOOK_URL is unset)."
        }

    # Initialize store if not present
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    init_alert_store(db_path)

    # Check idempotency: avoid duplicate successful alerts
    if is_alert_already_sent(claim_id, event=event_type, db_path=db_path):
        return {
            "triggered": True,
            "status": "ALREADY_SENT",
            "message": f"Alert already successfully delivered to Zapier for claim {claim_id}."
        }

    payload = build_webhook_payload(claim_data, prediction_result)
    created_at = datetime.now(timezone.utc).isoformat()

    try:
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json", "User-Agent": "Kestrel-Fraud-Detection/1.0"},
            timeout=timeout
        )

        if 200 <= response.status_code < 300:
            sent_at = datetime.now(timezone.utc).isoformat()
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO alert_events
                    (idempotency_key, claim_id, event, status, http_status, created_at, sent_at, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (idempotency_key, claim_id, event_type, "SENT", response.status_code, created_at, sent_at, None)
                )
                conn.commit()
            finally:
                conn.close()

            logger.info(f"High risk alert delivered to Zapier for claim {claim_id} (HTTP {response.status_code})")
            return {
                "triggered": True,
                "status": "SENT",
                "http_status": response.status_code,
                "message": "High-risk email alert dispatched successfully to Zapier."
            }
        else:
            err_msg = f"Zapier HTTP {response.status_code}: {response.text[:120]}"
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO alert_events
                    (idempotency_key, claim_id, event, status, http_status, created_at, sent_at, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (idempotency_key, claim_id, event_type, "FAILED", response.status_code, created_at, None, err_msg)
                )
                conn.commit()
            finally:
                conn.close()

            logger.warning(f"Zapier delivery failed for claim {claim_id} via {mask_url(webhook_url)}: {err_msg}")
            return {
                "triggered": True,
                "status": "FAILED",
                "http_status": response.status_code,
                "message": f"Zapier returned non-success response (HTTP {response.status_code})."
            }

    except Exception as e:
        err_msg = f"{type(e).__name__}: {str(e)[:120]}"
        try:
            conn = sqlite3.connect(str(db_path))
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO alert_events
                    (idempotency_key, claim_id, event, status, http_status, created_at, sent_at, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (idempotency_key, claim_id, event_type, "FAILED", None, created_at, None, err_msg)
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as db_err:
            logger.error(f"Failed to record failed alert in SQLite: {db_err}")

        logger.warning(f"Zapier webhook request failed for claim {claim_id} via {mask_url(webhook_url)}: {err_msg}")
        return {
            "triggered": True,
            "status": "FAILED",
            "error": err_msg,
            "message": "Outbound webhook request failed or timed out."
        }
