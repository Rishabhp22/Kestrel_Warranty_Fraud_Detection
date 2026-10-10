"""
api/app.py -- Kestrel Warranty Fraud Detection REST API and Dashboard.

Endpoints:
    GET  /              -- Operational triage UI dashboard
    POST /predict       -- Single claim prediction & TreeSHAP explanation
    POST /predict_bulk  -- Batch claim CSV upload and scoring
    GET  /analytics     -- Operational test batch analytics and partner summaries
    GET  /download_scored -- Download latest bulk scored CSV
"""

import os
import sys
import json
import pickle
import io
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template, Response, send_file

# Windows console UTF-8 safety
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root and src/ are importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "api"))

from features import FraudFeatureEngineer
from explainer import get_risk_level, generate_human_reasons
from alerts import send_high_risk_alert, get_webhook_url
from ai_assistant import analyze_claim_with_groq

MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder="templates")
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Global in-memory model and pipeline objects (loaded once at startup)
print("[API] Loading model artifacts at startup...")
engineer_path = MODELS_DIR / "engineer.pkl"
model_path = MODELS_DIR / "selected_model.pkl"

if not engineer_path.exists() or not model_path.exists():
    raise FileNotFoundError(
        f"Model artifacts not found in {MODELS_DIR}. Run 'python src/train.py' first."
    )

engineer: FraudFeatureEngineer = FraudFeatureEngineer.load(engineer_path)
with open(model_path, "rb") as f:
    model = pickle.load(f)

feat_cols = engineer.get_feature_cols()
print(f"[API] Initialized successfully. Model: {type(model).__name__}, Features: {len(feat_cols)}")

# Load partners and products reference for analytics
partners_ref = pd.read_csv(ROOT / "partners.csv") if (ROOT / "partners.csv").exists() else pd.DataFrame()
products_ref = pd.read_csv(ROOT / "products.csv") if (ROOT / "products.csv").exists() else pd.DataFrame()

# Required claim fields for single prediction
REQUIRED_FIELDS = [
    "claim_id",
    "submitted_at",
    "partner_id",
    "sku",
    "product_serial",
    "days_since_purchase",
    "claim_amount_inr",
    "photo_attached",
    "partner_inspected",
    "claim_description",
    "customer_prior_claims",
    "source",
]


def sanitize_records(records):
    """
    Sanitize list of dictionary records to ensure no NaN, inf, or pd.NA values
    are passed to JSON serializer, which would otherwise produce invalid JSON (e.g. 'NaN').
    """
    clean_list = []
    for record in records:
        clean_row = {}
        for k, v in record.items():
            try:
                if pd.isna(v) or (isinstance(v, (float, np.floating)) and (np.isnan(v) or np.isinf(v))):
                    clean_row[k] = None
                else:
                    clean_row[k] = v
            except Exception:
                clean_row[k] = None
        clean_list.append(clean_row)
    return clean_list



@app.route("/", methods=["GET"])
def index():
    """Render dashboard UI."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict -- Single claim risk scoring and explanation.
    """
    if not request.is_json:
        return jsonify({
            "error": "Request content type must be application/json"
        }), 400

    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({
            "error": "Invalid JSON body provided. Expected JSON object with claim fields."
        }), 400

    # Validate presence of required fields
    missing = [f for f in REQUIRED_FIELDS if f not in data or data[f] is None or str(data[f]).strip() == ""]
    if missing:
        return jsonify({
            "error": f"Missing required fields: {', '.join(missing)}",
            "missing_fields": missing,
            "required_fields": REQUIRED_FIELDS
        }), 400

    # Validate numeric types
    try:
        dsp = float(data["days_since_purchase"])
        amt = float(data["claim_amount_inr"])
        priors = int(data["customer_prior_claims"])
    except (ValueError, TypeError) as e:
        return jsonify({
            "error": f"Invalid numeric field format: {str(e)}"
        }), 400

    claim_id = str(data["claim_id"]).strip()

    try:
        # Featurize claim using pre-fitted pipeline
        claim_df = pd.DataFrame([{
            "claim_id": claim_id,
            "submitted_at": str(data["submitted_at"]).strip(),
            "partner_id": str(data["partner_id"]).strip(),
            "sku": str(data["sku"]).strip(),
            "product_serial": str(data["product_serial"]).strip(),
            "days_since_purchase": dsp,
            "claim_amount_inr": amt,
            "photo_attached": str(data["photo_attached"]).strip(),
            "partner_inspected": str(data["partner_inspected"]).strip(),
            "claim_description": str(data["claim_description"]).strip(),
            "customer_prior_claims": priors,
            "source": str(data["source"]).strip(),
        }])

        feats_df = engineer.transform(claim_df)
        X = feats_df[feat_cols].values.astype(float)

        # Raw model probability prediction
        raw_prob = float(model.predict_proba(X)[0, 1])
        score = round(raw_prob, 4)
        risk_level = get_risk_level(score)

        # Generate 3-5 plain English reasons using TreeSHAP
        reasons = generate_human_reasons(data, feats_df, model, feat_cols)

        pred_response = {
            "claim_id": claim_id,
            "fraud_score": score,
            "risk_level": risk_level,
            "reasons": reasons
        }

        # Dispatch Zapier email alert if claim is HIGH risk (idempotent, failsafe)
        alert_outcome = send_high_risk_alert(data, pred_response)
        pred_response["alert"] = alert_outcome

        return jsonify(pred_response), 200

    except Exception as e:
        return jsonify({
            "error": f"Inference processing error: {str(e)}"
        }), 500


@app.route("/predict_bulk", methods=["POST"])
def predict_bulk():
    """
    POST /predict_bulk -- Batch CSV upload and scoring.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Expected 'file' multipart form field."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename provided."}), 400

    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Invalid file type. File must have a .csv extension."}), 400

    try:
        content = file.read().decode("utf-8", errors="replace")
        if not content.strip():
            return jsonify({"error": "Uploaded CSV file is completely empty."}), 400

        df = pd.read_csv(io.StringIO(content))
        if len(df) == 0:
            return jsonify({"error": "Uploaded CSV file contains no data rows."}), 400

        # Validate required columns
        missing_cols = [col for col in REQUIRED_FIELDS if col not in df.columns]
        if missing_cols:
            return jsonify({
                "error": f"CSV missing required columns: {', '.join(missing_cols)}",
                "missing_columns": missing_cols
            }), 400

        # Vectorized batch transformation and scoring
        feats_df = engineer.transform(df)
        X = feats_df[feat_cols].values.astype(float)
        scores = model.predict_proba(X)[:, 1]

        # Add predictions to output dataframe
        scored_df = df.copy()
        scored_df["fraud_score"] = np.round(scores, 4)
        scored_df["risk_level"] = [get_risk_level(s) for s in scored_df["fraud_score"]]

        # Save to disk for download route
        latest_path = OUTPUTS_DIR / "bulk_scored_latest.csv"
        scored_df.to_csv(latest_path, index=False)

        # If AJAX JSON format requested by dashboard
        if request.args.get("format") == "json":
            total_n = len(scored_df)
            high_n = int((scored_df["fraud_score"] >= 0.30).sum())
            med_n = int(((scored_df["fraud_score"] >= 0.10) & (scored_df["fraud_score"] < 0.30)).sum())
            low_n = int((scored_df["fraud_score"] < 0.10).sum())
            avg_score = float(scored_df["fraud_score"].mean())
            max_score = float(scored_df["fraud_score"].max()) if total_n > 0 else 0.0
            max_idx = int(scored_df["fraud_score"].idxmax()) if total_n > 0 else 0
            max_claim_id = str(scored_df.iloc[max_idx]["claim_id"]) if total_n > 0 else "N/A"

            # Score distribution histogram bins
            bin_0_5 = int(((scored_df["fraud_score"] >= 0.0) & (scored_df["fraud_score"] < 0.05)).sum())
            bin_5_10 = int(((scored_df["fraud_score"] >= 0.05) & (scored_df["fraud_score"] < 0.10)).sum())
            bin_10_20 = int(((scored_df["fraud_score"] >= 0.10) & (scored_df["fraud_score"] < 0.20)).sum())
            bin_20_30 = int(((scored_df["fraud_score"] >= 0.20) & (scored_df["fraud_score"] < 0.30)).sum())
            bin_30_plus = int((scored_df["fraud_score"] >= 0.30).sum())

            preview_records = sanitize_records(scored_df[[
                "claim_id", "submitted_at", "claim_amount_inr", "partner_id", "fraud_score", "risk_level"
            ]].head(25).to_dict(orient="records"))

            table_cols = [c for c in [
                "claim_id", "submitted_at", "partner_id", "sku", "product_serial",
                "days_since_purchase", "claim_amount_inr", "photo_attached",
                "partner_inspected", "claim_description", "customer_prior_claims",
                "source", "fraud_score", "risk_level"
            ] if c in scored_df.columns]
            all_records = sanitize_records(scored_df[table_cols].head(5000).to_dict(orient="records"))

            return jsonify({
                "message": f"Successfully scored {total_n} claims.",
                "filename": file.filename,
                "dataset_name": f"Uploaded ({file.filename})",
                "is_demo": False,
                "total_rows": total_n,
                "total_claims": total_n,
                "average_score": round(avg_score, 4),
                "average_fraud_score": round(avg_score, 4),
                "highest_fraud_score": round(max_score, 4),
                "highest_claim_id": max_claim_id,
                "high_risk_count": high_n,
                "high_risk_pct": round(high_n / total_n * 100, 2) if total_n > 0 else 0.0,
                "medium_risk_count": med_n,
                "medium_risk_pct": round(med_n / total_n * 100, 2) if total_n > 0 else 0.0,
                "low_risk_count": low_n,
                "low_risk_pct": round(low_n / total_n * 100, 2) if total_n > 0 else 0.0,
                "score_distribution": {
                    "< 5%": bin_0_5,
                    "5% - 10%": bin_5_10,
                    "10% - 20%": bin_10_20,
                    "20% - 30%": bin_20_30,
                    "≥ 30%": bin_30_plus
                },
                "download_url": "/download_scored",
                "preview": preview_records,
                "claims": all_records
            }), 200

        # Direct CSV response
        output_buffer = io.StringIO()
        scored_df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)

        return Response(
            output_buffer.getvalue(),
            mimetype="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=scored_{file.filename}"
            }
        )

    except Exception as e:
        return jsonify({"error": f"Failed to process CSV file: {str(e)}"}), 500


@app.route("/validate_csv", methods=["POST"])
def validate_csv():
    """
    POST /validate_csv -- Validate schema and inspect CSV file without scoring.
    """
    if "file" not in request.files:
        return jsonify({"valid": False, "error": "No file uploaded. Expected 'file' multipart field."}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"valid": False, "error": "Empty filename provided."}), 400
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"valid": False, "error": "Invalid format. Only .csv files are supported."}), 400

    try:
        content = file.read().decode("utf-8", errors="replace")
        if not content.strip():
            return jsonify({"valid": False, "error": "Uploaded CSV file is completely empty."}), 400
        df = pd.read_csv(io.StringIO(content))
        if len(df) == 0:
            return jsonify({"valid": False, "error": "Uploaded CSV file contains no data rows."}), 400

        missing_cols = [col for col in REQUIRED_FIELDS if col not in df.columns]
        if missing_cols:
            return jsonify({
                "valid": False,
                "error": f"CSV missing required columns: {', '.join(missing_cols)}",
                "missing_columns": missing_cols,
                "required_fields": REQUIRED_FIELDS
            }), 400

        preview_records = sanitize_records(df.head(5).to_dict(orient="records"))
        return jsonify({
            "valid": True,
            "filename": file.filename,
            "total_rows": len(df),
            "columns": list(df.columns),
            "preview": preview_records
        }), 200
    except Exception as e:
        return jsonify({"valid": False, "error": f"Failed to parse CSV: {str(e)}"}), 400


@app.route("/api/demo_info", methods=["GET"])
def demo_info():
    """
    GET /api/demo_info -- Metadata and preview for pre-configured project Demo Dataset.
    """
    test_path = ROOT / "test_unlabelled.csv"
    pred_path = OUTPUTS_DIR / "predictions.csv"
    if not test_path.exists():
        return jsonify({"error": "test_unlabelled.csv not found in project root"}), 404

    try:
        df = pd.read_csv(test_path)
        preview_records = sanitize_records(df.head(5).to_dict(orient="records"))
        return jsonify({
            "filename": "test_unlabelled.csv",
            "label": "Demo Dataset",
            "dataset_type": "demo",
            "total_rows": len(df),
            "status": "Verified Pre-scored Available",
            "has_prescored": pred_path.exists(),
            "description": "Q2 2026 operational test claims (April–August 2026)",
            "columns": list(df.columns),
            "preview": preview_records,
            "required_fields": REQUIRED_FIELDS
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to inspect demo dataset: {str(e)}"}), 500


@app.route("/api/demo_claims", methods=["GET"])
def demo_claims():
    """
    GET /api/demo_claims -- Load verified pre-scored Demo Dataset without re-scoring.
    """
    test_path = ROOT / "test_unlabelled.csv"
    pred_path = OUTPUTS_DIR / "predictions.csv"

    if not test_path.exists() or not pred_path.exists():
        return jsonify({"error": "Demo or prediction artifacts missing on server."}), 404

    try:
        test_df = pd.read_csv(test_path)
        pred_df = pd.read_csv(pred_path)
        merged = test_df.merge(pred_df, on="claim_id", how="left")

        scores = merged["score"].values
        total_n = len(scores)
        avg_score = float(np.mean(scores))
        max_idx = int(np.argmax(scores))
        max_score = float(scores[max_idx])
        max_claim_id = str(merged.iloc[max_idx]["claim_id"])

        high_n = int((scores >= 0.30).sum())
        med_n = int(((scores >= 0.10) & (scores < 0.30)).sum())
        low_n = int((scores < 0.10).sum())

        # Histogram bins
        bin_0_5 = int(((scores >= 0.0) & (scores < 0.05)).sum())
        bin_5_10 = int(((scores >= 0.05) & (scores < 0.10)).sum())
        bin_10_20 = int(((scores >= 0.10) & (scores < 0.20)).sum())
        bin_20_30 = int(((scores >= 0.20) & (scores < 0.30)).sum())
        bin_30_plus = int((scores >= 0.30).sum())

        merged["fraud_score"] = np.round(merged["score"], 4)
        merged["risk_level"] = [get_risk_level(s) for s in merged["fraud_score"]]

        table_cols = [c for c in [
            "claim_id", "submitted_at", "partner_id", "sku", "product_serial",
            "days_since_purchase", "claim_amount_inr", "photo_attached",
            "partner_inspected", "claim_description", "customer_prior_claims",
            "source", "fraud_score", "risk_level"
        ] if c in merged.columns]
        all_records = sanitize_records(merged[table_cols].to_dict(orient="records"))

        return jsonify({
            "filename": "test_unlabelled.csv",
            "dataset_name": "Demo Dataset (test_unlabelled.csv)",
            "is_demo": True,
            "is_prescored": True,
            "total_claims": total_n,
            "total_rows": total_n,
            "average_fraud_score": round(avg_score, 4),
            "highest_fraud_score": round(max_score, 4),
            "highest_claim_id": max_claim_id,
            "high_risk_count": high_n,
            "high_risk_pct": round(high_n / total_n * 100, 2),
            "medium_risk_count": med_n,
            "medium_risk_pct": round(med_n / total_n * 100, 2),
            "low_risk_count": low_n,
            "low_risk_pct": round(low_n / total_n * 100, 2),
            "score_distribution": {
                "< 5%": bin_0_5,
                "5% - 10%": bin_5_10,
                "10% - 20%": bin_10_20,
                "20% - 30%": bin_20_30,
                "≥ 30%": bin_30_plus
            },
            "claims": all_records
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to load demo claims: {str(e)}"}), 500


@app.route("/download_scored", methods=["GET"])
def download_scored():
    """Download the latest scored batch CSV."""
    latest_path = OUTPUTS_DIR / "bulk_scored_latest.csv"
    if not latest_path.exists():
        # Fallback to predictions.csv if bulk was not yet uploaded
        pred_path = OUTPUTS_DIR / "predictions.csv"
        if pred_path.exists():
            return send_file(pred_path, as_attachment=True, download_name="predictions.csv")
        return jsonify({"error": "No scored file available for download."}), 404

    return send_file(latest_path, as_attachment=True, download_name="scored_claims.csv")


@app.route("/analytics", methods=["GET"])
def analytics():
    """
    GET /analytics -- Aggregate risk analytics across test set population.
    """
    pred_path = OUTPUTS_DIR / "predictions.csv"
    test_path = ROOT / "test_unlabelled.csv"

    if not pred_path.exists() or not test_path.exists():
        return jsonify({"error": "Predictions or test data not found. Run predictions first."}), 404

    try:
        preds = pd.read_csv(pred_path)
        test = pd.read_csv(test_path)
        merged = test.merge(preds, on="claim_id", how="left")

        scores = merged["score"].values
        total_claims = int(len(scores))
        avg_score = float(np.mean(scores))
        max_idx = int(np.argmax(scores))
        max_score = float(scores[max_idx])
        max_claim_id = str(merged.iloc[max_idx]["claim_id"])

        high_count = int((scores >= 0.30).sum())
        med_count = int(((scores >= 0.10) & (scores < 0.30)).sum())
        low_count = int((scores < 0.10).sum())

        # Partner-level risk aggregation
        merged_p = merged.merge(partners_ref, on="partner_id", how="left")
        p_group = merged_p.groupby(["partner_id", "city", "partner_type"]).agg(
            total_claims=("claim_id", "count"),
            average_score=("score", "mean"),
            high_risk_claims=("score", lambda s: int((s >= 0.30).sum())),
            medium_risk_claims=("score", lambda s: int(((s >= 0.10) & (s < 0.30)).sum())),
        ).reset_index()

        # Sort partners by count of elevated risk claims, then average score
        p_group["elevated_risk"] = p_group["high_risk_claims"] + p_group["medium_risk_claims"]
        p_sorted = p_group.sort_values(by=["elevated_risk", "average_score"], ascending=False).head(15)

        partner_summary = []
        for _, row in p_sorted.iterrows():
            partner_summary.append({
                "partner_id": str(row["partner_id"]),
                "city": str(row["city"]) if pd.notnull(row["city"]) else "Unknown",
                "partner_type": str(row["partner_type"]) if pd.notnull(row["partner_type"]) else "Unknown",
                "total_claims": int(row["total_claims"]),
                "average_score": round(float(row["average_score"]), 4),
                "high_risk_claims": int(row["high_risk_claims"]),
                "medium_risk_claims": int(row["medium_risk_claims"]),
            })

        # Product family aggregation
        merged_pr = merged.merge(products_ref, on="sku", how="left")
        fam_group = merged_pr.groupby("family").agg(
            total_claims=("claim_id", "count"),
            average_score=("score", "mean"),
            high_risk_claims=("score", lambda s: int((s >= 0.30).sum())),
        ).reset_index().sort_values(by="average_score", ascending=False)

        family_summary = []
        for _, row in fam_group.iterrows():
            family_summary.append({
                "family": str(row["family"]),
                "total_claims": int(row["total_claims"]),
                "average_score": round(float(row["average_score"]), 4),
                "high_risk_claims": int(row["high_risk_claims"]),
            })

        # Validation reference metadata
        meta_path = MODELS_DIR / "training_meta.json"
        val_meta = {}
        if meta_path.exists():
            with open(meta_path, "r") as f:
                val_meta = json.load(f)

        return jsonify({
            "total_claims": total_claims,
            "average_fraud_score": round(avg_score, 4),
            "highest_fraud_score": round(max_score, 4),
            "highest_claim_id": max_claim_id,
            "risk_distribution": {
                "HIGH": {
                    "count": high_count,
                    "percentage": round(high_count / total_claims * 100, 2)
                },
                "MEDIUM": {
                    "count": med_count,
                    "percentage": round(med_count / total_claims * 100, 2)
                },
                "LOW": {
                    "count": low_count,
                    "percentage": round(low_count / total_claims * 100, 2)
                }
            },
            "top_risk_partners": partner_summary,
            "product_family_summary": family_summary,
            "validation_performance": val_meta
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to compute analytics: {str(e)}"}), 500


@app.route("/ai/investigate", methods=["POST"])
def ai_investigate():
    """
    POST /ai/investigate -- Grounded AI Fraud Investigation Assistant powered by Groq.
    
    1. Validates claim payload.
    2. Runs backend ML scoring & TreeSHAP explanations.
    3. Calls Groq with grounded prompts & prompt-injection defenses.
    4. Returns structured 5-part investigation brief.
    """
    if not request.is_json:
        return jsonify({
            "error": "Request content type must be application/json"
        }), 400

    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({
            "error": "Invalid JSON body provided. Expected JSON object with claim fields."
        }), 400

    # Validate presence of required fields
    missing = [f for f in REQUIRED_FIELDS if f not in data or data[f] is None or str(data[f]).strip() == ""]
    if missing:
        return jsonify({
            "error": f"Missing required fields: {', '.join(missing)}",
            "missing_fields": missing,
            "required_fields": REQUIRED_FIELDS
        }), 400

    # Validate numeric types
    try:
        dsp = float(data["days_since_purchase"])
        amt = float(data["claim_amount_inr"])
        priors = int(data["customer_prior_claims"])
    except (ValueError, TypeError) as e:
        return jsonify({
            "error": f"Invalid numeric field format: {str(e)}"
        }), 400

    claim_id = str(data["claim_id"]).strip()

    try:
        # Featurize claim using pre-fitted pipeline
        claim_df = pd.DataFrame([{
            "claim_id": claim_id,
            "submitted_at": str(data["submitted_at"]).strip(),
            "partner_id": str(data["partner_id"]).strip(),
            "sku": str(data["sku"]).strip(),
            "product_serial": str(data["product_serial"]).strip(),
            "days_since_purchase": dsp,
            "claim_amount_inr": amt,
            "photo_attached": str(data["photo_attached"]).strip(),
            "partner_inspected": str(data["partner_inspected"]).strip(),
            "claim_description": str(data["claim_description"]).strip(),
            "customer_prior_claims": priors,
            "source": str(data["source"]).strip(),
        }])

        feats_df = engineer.transform(claim_df)
        X = feats_df[feat_cols].values.astype(float)

        # Raw model probability prediction & risk band (Source of Truth)
        raw_prob = float(model.predict_proba(X)[0, 1])
        score = round(raw_prob, 4)
        risk_level = get_risk_level(score)

        # Generate 3-5 plain English reasons using TreeSHAP
        reasons = generate_human_reasons(data, feats_df, model, feat_cols)

        # Query Groq AI Assistant with grounded evidence
        ai_result = analyze_claim_with_groq(data, score, risk_level, reasons)

        return jsonify({
            "claim_id": claim_id,
            "fraud_score": score,
            "risk_level": risk_level,
            "reasons": reasons,
            "ai_analysis": ai_result
        }), 200

    except Exception as e:
        return jsonify({
            "error": f"Inference or AI processing error: {str(e)}"
        }), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[SERVER] Starting Kestrel Fraud Detection Service on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
