"""
Prediction History and Audit Logging Module.
Stores applicant assessments during the session with privacy protection and export capability.
"""
import os
import json
import uuid
from datetime import datetime
import pandas as pd
from pathlib import Path

HISTORY_FILE = Path(__file__).resolve().parent.parent / "models" / "prediction_history.json"

class HistoryStore:
    def __init__(self):
        self.history_file = HISTORY_FILE
        self._ensure_file()

    def _ensure_file(self):
        if not self.history_file.exists():
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, "w") as f:
                json.dump([], f)

    def load_history(self) -> list[dict]:
        try:
            with open(self.history_file, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def log_assessment(self, applicant_data: dict, risk_tier: str, default_prob: float, top_factors: list[str]) -> dict:
        """
        Logs an assessment with anonymized Applicant ID and privacy protection.
        """
        history = self.load_history()
        
        applicant_id = f"APP-{uuid.uuid4().hex[:6].upper()}"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        entry = {
            "applicant_id": applicant_id,
            "timestamp": timestamp,
            "loan_amount": applicant_data.get("credit_amount"),
            "duration": applicant_data.get("duration"),
            "checking_status": applicant_data.get("checking_status"),
            "credit_history": applicant_data.get("credit_history"),
            "risk_tier": risk_tier,
            "default_probability": round(float(default_prob), 4),
            "key_drivers": ", ".join(top_factors[:3]) if top_factors else "None"
        }

        history.insert(0, entry)  # latest first
        if len(history) > 100:
            history = history[:100]

        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=2)

        return entry

    def get_history_dataframe(self) -> pd.DataFrame:
        history = self.load_history()
        if not history:
            return pd.DataFrame()
        return pd.DataFrame(history)

    def clear_history(self):
        with open(self.history_file, "w") as f:
            json.dump([], f)
