"""
Automated Comprehensive Test Suite for Credit Risk Prediction System.
Tests data preprocessing, model inference, XAI, historical analysis,
What-If logic, and fairness metrics.
"""
import unittest
import numpy as np
import pandas as pd
from src.config import (
    CSV_DATA_PATH,
    TARGET_COL,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    classify_risk_tier
)
from src.data_processor import (
    load_data,
    get_dataset_profile,
    prepare_train_test_split,
    transform_single_applicant
)
from src.model_trainer import load_trained_artifacts
from src.xai_engine import XAIEngine
from src.historical_matcher import HistoricalMatcher
from src.fairness_evaluator import FairnessEvaluator
from src.history_store import HistoryStore

class TestCreditRiskSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_data()
        cls.artifacts = load_trained_artifacts()
        cls.all_models = cls.artifacts["all_models"]
        cls.best_model = cls.artifacts["best_model"]
        cls.preprocessor = cls.artifacts["preprocessor"]
        cls.train_bundle = cls.artifacts["train_data"]
        cls.feature_names = cls.train_bundle["feature_names"]
        cls.X_train_proc = cls.train_bundle["X_train_proc"]

        cls.sample_applicant = {
            "duration": 24,
            "credit_amount": 3500,
            "installment_rate": 3,
            "residence_since": 2,
            "age": 35,
            "existing_credits": 1,
            "num_dependents": 1,
            "checking_status": "0 to < 200 DM",
            "credit_history": "existing credits paid duly till now",
            "purpose": "car (new)",
            "savings_status": "100 to < 500 DM",
            "employment": "1 to < 4 years",
            "personal_status_sex": "male : single",
            "other_parties": "none",
            "property_magnitude": "real estate",
            "other_payment_plans": "none",
            "housing": "own",
            "job": "skilled employee/official",
            "telephone": "yes",
            "foreign_worker": "no"
        }

    def test_01_dataset_integrity(self):
        self.assertEqual(len(self.df), 1000)
        self.assertIn(TARGET_COL, self.df.columns)
        self.assertEqual(self.df[TARGET_COL].isnull().sum(), 0)
        profile = get_dataset_profile(self.df)
        self.assertEqual(profile["missing_values"], 0)
        self.assertEqual(profile["total_records"], 1000)

    def test_02_preprocessing_pipeline(self):
        applicant_proc = transform_single_applicant(self.sample_applicant, self.preprocessor)
        self.assertEqual(applicant_proc.shape[0], 1)
        self.assertEqual(applicant_proc.shape[1], len(self.feature_names))

    def test_03_all_four_models_predict(self):
        applicant_proc = transform_single_applicant(self.sample_applicant, self.preprocessor)
        for name, model in self.all_models.items():
            prob = model.predict_proba(applicant_proc)[:, 1][0]
            self.assertTrue(0.0 <= prob <= 1.0, f"{name} probability out of bounds: {prob}")
            pred = model.predict(applicant_proc)[0]
            self.assertIn(pred, [0, 1])

    def test_04_risk_tiering(self):
        tier_low, color_low, _ = classify_risk_tier(0.15)
        self.assertEqual(tier_low, "Low Risk")

        tier_med, color_med, _ = classify_risk_tier(0.40)
        self.assertEqual(tier_med, "Medium Risk")

        tier_high, color_high, _ = classify_risk_tier(0.75)
        self.assertEqual(tier_high, "High Risk")

    def test_05_xai_shap_and_narrative(self):
        xai = XAIEngine(
            model=self.best_model,
            feature_names=self.feature_names,
            X_train_proc=self.X_train_proc,
            model_name="Random Forest"
        )
        applicant_proc = transform_single_applicant(self.sample_applicant, self.preprocessor)
        exp = xai.explain_instance_shap(applicant_proc, top_n=4)

        self.assertIn("shap_values", exp)
        self.assertIn("top_increasing", exp)
        self.assertIn("top_decreasing", exp)
        self.assertIn("explanation_text", exp)
        self.assertTrue(len(exp["explanation_text"]) > 50)

    def test_06_xai_lime(self):
        xai = XAIEngine(
            model=self.best_model,
            feature_names=self.feature_names,
            X_train_proc=self.X_train_proc,
            model_name="Random Forest"
        )
        applicant_proc = transform_single_applicant(self.sample_applicant, self.preprocessor)
        lime_weights = xai.explain_instance_lime(applicant_proc, num_features=5)
        self.assertTrue(len(lime_weights) > 0)

    def test_07_historical_matcher(self):
        matcher = HistoricalMatcher(
            X_train_raw=self.train_bundle["X_train"],
            y_train_raw=self.train_bundle["y_train"],
            X_train_proc=self.X_train_proc,
            preprocessor=self.preprocessor
        )
        applicant_proc = transform_single_applicant(self.sample_applicant, self.preprocessor)
        cohort = matcher.find_similar_applicants(applicant_proc, k=5)

        self.assertEqual(cohort["cohort_size"], 5)
        self.assertEqual(cohort["default_count"] + cohort["repaid_count"], 5)
        self.assertTrue(0.0 <= cohort["cohort_default_rate"] <= 1.0)
        self.assertEqual(len(cohort["comparison_table"]), 5)

    def test_08_fairness_evaluator(self):
        y_pred = self.best_model.predict(self.train_bundle["X_test_proc"])
        evaluator = FairnessEvaluator(
            df=self.train_bundle["X_test"],
            y_pred=y_pred,
            y_true=self.train_bundle["y_test"]
        )
        res_gender = evaluator.evaluate_gender_fairness()
        self.assertIn("disparate_impact_ratio", res_gender)
        self.assertTrue(res_gender["disparate_impact_ratio"] >= 0.0)

    def test_09_history_store(self):
        store = HistoryStore()
        entry = store.log_assessment(self.sample_applicant, "Low Risk", 0.18, ["checking_status", "duration"])
        self.assertTrue(entry["applicant_id"].startswith("APP-"))
        history_df = store.get_history_dataframe()
        self.assertFalse(history_df.empty)

if __name__ == "__main__":
    unittest.main()
