"""
Responsible AI & Fairness Evaluation Module.
Assesses demographic parity, disparate impact ratios, and sensitive attribute disparities
across gender, age, and foreign worker status.
"""
import pandas as pd
import numpy as np

class FairnessEvaluator:
    def __init__(self, df: pd.DataFrame, y_pred: np.ndarray, y_true: np.ndarray):
        self.df = df.copy().reset_index(drop=True)
        # Approval means predicted Low/Good Risk (prediction = 0)
        self.approval = (y_pred == 0).astype(int)
        self.y_true = y_true.values if hasattr(y_true, "values") else np.array(y_true)

    def evaluate_gender_fairness(self) -> dict:
        """Evaluates model approval parity across male vs female applicant records."""
        is_female = self.df["personal_status_sex"].str.startswith("female")
        approval_female = float(self.approval[is_female].mean()) if is_female.sum() > 0 else 0.0
        approval_male = float(self.approval[~is_female].mean()) if (~is_female).sum() > 0 else 0.0

        disparate_impact = approval_female / approval_male if approval_male > 0 else 0.0
        parity_diff = abs(approval_male - approval_female)

        return {
            "attribute": "Gender (Female vs Male)",
            "female_count": int(is_female.sum()),
            "male_count": int((~is_female).sum()),
            "female_approval_rate": round(approval_female, 3),
            "male_approval_rate": round(approval_male, 3),
            "disparate_impact_ratio": round(disparate_impact, 3),
            "demographic_parity_diff": round(parity_diff, 3),
            "status": "Acceptable (> 0.80)" if disparate_impact >= 0.80 else "Potential Disparity (< 0.80)"
        }

    def evaluate_age_fairness(self, age_cutoff: int = 25) -> dict:
        """Evaluates model approval parity across young (< 25) vs mature (>= 25) applicants."""
        is_young = self.df["age"] < age_cutoff
        approval_young = float(self.approval[is_young].mean()) if is_young.sum() > 0 else 0.0
        approval_mature = float(self.approval[~is_young].mean()) if (~is_young).sum() > 0 else 0.0

        disparate_impact = approval_young / approval_mature if approval_mature > 0 else 0.0
        parity_diff = abs(approval_mature - approval_young)

        return {
            "attribute": f"Age (< {age_cutoff} vs >= {age_cutoff})",
            "young_count": int(is_young.sum()),
            "mature_count": int((~is_young).sum()),
            "young_approval_rate": round(approval_young, 3),
            "mature_approval_rate": round(approval_mature, 3),
            "disparate_impact_ratio": round(disparate_impact, 3),
            "demographic_parity_diff": round(parity_diff, 3),
            "status": "Acceptable (> 0.80)" if disparate_impact >= 0.80 else "Potential Disparity (< 0.80)"
        }

    def evaluate_foreign_worker_fairness(self) -> dict:
        """Evaluates model approval parity across foreign workers vs domestic residents."""
        is_foreign = self.df["foreign_worker"].str.lower() == "yes"
        approval_foreign = float(self.approval[is_foreign].mean()) if is_foreign.sum() > 0 else 0.0
        approval_domestic = float(self.approval[~is_foreign].mean()) if (~is_foreign).sum() > 0 else 0.0

        disparate_impact = approval_foreign / approval_domestic if approval_domestic > 0 else 0.0
        parity_diff = abs(approval_domestic - approval_foreign)

        return {
            "attribute": "Foreign Worker Status (Foreign vs Domestic)",
            "foreign_count": int(is_foreign.sum()),
            "domestic_count": int((~is_foreign).sum()),
            "foreign_approval_rate": round(approval_foreign, 3),
            "domestic_approval_rate": round(approval_domestic, 3),
            "disparate_impact_ratio": round(disparate_impact, 3),
            "demographic_parity_diff": round(parity_diff, 3),
            "status": "Acceptable (> 0.80)" if disparate_impact >= 0.80 else "Potential Disparity (< 0.80)"
        }
