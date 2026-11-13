"""
Historical Data Analysis and Nearest-Neighbor Cohort Matching.
Retrieves actual historical applicant records with similar financial profiles,
computes observed repayment outcomes, and prevents target leakage.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer

class HistoricalMatcher:
    def __init__(self, X_train_raw: pd.DataFrame, y_train_raw: pd.Series, X_train_proc: np.ndarray, preprocessor: ColumnTransformer):
        self.X_train_raw = X_train_raw.reset_index(drop=True)
        self.y_train_raw = y_train_raw.reset_index(drop=True)
        self.X_train_proc = X_train_proc
        self.preprocessor = preprocessor

        # Fit NearestNeighbors on historical training feature space (strictly features, zero target leakage)
        self.nn = NearestNeighbors(n_neighbors=20, metric="euclidean")
        self.nn.fit(X_train_proc)

    def find_similar_applicants(self, applicant_proc: np.ndarray, k: int = 5) -> dict:
        """
        Finds the top k most similar historical applicants in the feature space.
        Returns cohort summary statistics and comparative dataframe.
        """
        k = max(1, min(k, len(self.X_train_raw)))
        distances, indices = self.nn.kneighbors(applicant_proc, n_neighbors=k)
        
        neighbor_idx = indices[0]
        neighbor_dist = distances[0]

        matched_records = self.X_train_raw.iloc[neighbor_idx].copy()
        matched_outcomes = self.y_train_raw.iloc[neighbor_idx].values

        # Build clean comparison columns
        matched_records["Observed Outcome"] = [
            "Repaid (Good Credit)" if y == 0 else "Defaulted (Delinquent)"
            for y in matched_outcomes
        ]
        matched_records["Feature Distance"] = [round(float(d), 3) for d in neighbor_dist]

        # Cohort summary statistics
        total_cohort = len(matched_outcomes)
        default_count = int(np.sum(matched_outcomes == 1))
        repaid_count = int(np.sum(matched_outcomes == 0))
        cohort_default_rate = default_count / total_cohort if total_cohort > 0 else 0.0

        avg_credit_amount = float(matched_records["credit_amount"].mean())
        avg_duration = float(matched_records["duration"].mean())
        avg_age = float(matched_records["age"].mean())

        key_columns = [
            "credit_amount",
            "duration",
            "checking_status",
            "savings_status",
            "credit_history",
            "employment",
            "installment_rate",
            "Observed Outcome",
            "Feature Distance"
        ]
        comparison_table = matched_records[key_columns].rename(columns={
            "credit_amount": "Loan Amount (DM)",
            "duration": "Duration (Mo)",
            "checking_status": "Checking Status",
            "savings_status": "Savings Status",
            "credit_history": "Credit History",
            "employment": "Employment",
            "installment_rate": "Installment Rate (%)"
        })

        return {
            "cohort_size": total_cohort,
            "default_count": default_count,
            "repaid_count": repaid_count,
            "cohort_default_rate": cohort_default_rate,
            "avg_credit_amount": avg_credit_amount,
            "avg_duration": avg_duration,
            "avg_age": avg_age,
            "comparison_table": comparison_table,
            "matched_records_raw": matched_records
        }
