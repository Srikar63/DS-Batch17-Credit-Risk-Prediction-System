"""
Explainable AI (XAI) Engine using SHAP and LIME.
Generates local and global explanations, risk-driving factors, charts,
and dynamic natural-language credit explanations.
"""
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from src.config import TARGET_LABELS

class XAIEngine:
    def __init__(self, model, feature_names: list[str], X_train_proc: np.ndarray, model_name: str = "Random Forest"):
        self.model = model
        self.feature_names = feature_names
        self.X_train_proc = X_train_proc
        self.model_name = model_name
        
        # Initialize SHAP TreeExplainer (or Linear/Kernel if other model)
        if hasattr(model, "estimators_"):  # Random Forest / Decision Tree
            self.shap_explainer = shap.TreeExplainer(model)
            self.explainer_type = "tree"
        elif hasattr(model, "coef_"):
            self.shap_explainer = shap.LinearExplainer(model, X_train_proc)
            self.explainer_type = "linear"
        else:
            # Fallback to KernelExplainer with sampled background
            background = shap.kmeans(X_train_proc, min(25, len(X_train_proc)))
            self.shap_explainer = shap.KernelExplainer(model.predict_proba, background)
            self.explainer_type = "kernel"

        # Initialize LIME Explainer
        self.lime_explainer = LimeTabularExplainer(
            training_data=X_train_proc,
            feature_names=feature_names,
            class_names=["Repaid (Low Risk)", "Default (High Risk)"],
            mode="classification",
            random_state=42
        )

    def explain_instance_shap(self, applicant_proc: np.ndarray, top_n: int = 5) -> dict:
        """
        Computes SHAP values for a single applicant instance.
        Returns:
            - base_value: expected default probability/log-odds
            - shap_values: array of contributions for Class 1 (Default)
            - top_increasing: list of (feature_name, shap_value)
            - top_decreasing: list of (feature_name, shap_value)
            - explanation_text: dynamic human-readable explanation
        """
        if self.explainer_type == "tree":
            raw_shap = self.shap_explainer.shap_values(applicant_proc)
            if isinstance(raw_shap, list):
                shap_vals = raw_shap[1][0]
                base_val = self.shap_explainer.expected_value[1]
            elif isinstance(raw_shap, np.ndarray):
                if len(raw_shap.shape) == 3:
                    shap_vals = raw_shap[0, :, 1]
                    expected = self.shap_explainer.expected_value
                    base_val = expected[1] if hasattr(expected, '__len__') else expected
                else:
                    shap_vals = raw_shap[0]
                    base_val = self.shap_explainer.expected_value
            else:
                shap_vals = raw_shap.values[0, :, 1]
                expected = self.shap_explainer.expected_value
                base_val = expected[1] if hasattr(expected, '__len__') else expected
        elif self.explainer_type == "linear":
            raw_shap = self.shap_explainer.shap_values(applicant_proc)
            shap_vals = raw_shap[0] if isinstance(raw_shap, np.ndarray) else raw_shap
            base_val = self.shap_explainer.expected_value
        else:
            raw_shap = self.shap_explainer.shap_values(applicant_proc)
            shap_vals = raw_shap[1][0] if isinstance(raw_shap, list) else raw_shap[0]
            base_val = 0.30

        # Sort contributions
        sorted_indices = np.argsort(shap_vals)
        # Top positive (increasing default risk)
        top_inc_idx = [i for i in sorted_indices[::-1] if shap_vals[i] > 0][:top_n]
        # Top negative (decreasing default risk / credit protective)
        top_dec_idx = [i for i in sorted_indices if shap_vals[i] < 0][:top_n]

        top_increasing = [(self.feature_names[i], float(shap_vals[i])) for i in top_inc_idx]
        top_decreasing = [(self.feature_names[i], float(shap_vals[i])) for i in top_dec_idx]

        explanation_text = self._generate_dynamic_narrative(top_increasing, top_decreasing, shap_vals, base_val)

        return {
            "shap_values": shap_vals,
            "base_value": float(base_val),
            "top_increasing": top_increasing,
            "top_decreasing": top_decreasing,
            "explanation_text": explanation_text
        }

    def explain_instance_lime(self, applicant_proc: np.ndarray, num_features: int = 6) -> list[tuple[str, float]]:
        """
        Computes LIME local tabular explanation for cross-validation.
        """
        if applicant_proc.ndim > 1:
            applicant_proc = applicant_proc.flatten()

        exp = self.lime_explainer.explain_instance(
            data_row=applicant_proc,
            predict_fn=self.model.predict_proba,
            num_features=num_features,
            labels=[1]
        )
        return exp.as_list(label=1)

    def _generate_dynamic_narrative(
        self,
        top_increasing: list[tuple[str, float]],
        top_decreasing: list[tuple[str, float]],
        shap_vals: np.ndarray,
        base_val: float
    ) -> str:
        """
        Generates dynamic plain-language narrative explaining the credit decision drivers.
        """
        narrative_parts = []
        narrative_parts.append(
            f"The model's baseline population expected risk is approximately {base_val:.1%}. "
            "For this applicant, several specific financial characteristics altered the risk assessment."
        )

        if top_increasing:
            inc_descriptions = []
            for feat, val in top_increasing[:3]:
                readable_feat = self._humanize_feature_name(feat)
                inc_descriptions.append(f"**{readable_feat}** (risk increase impact: +{val:.3f})")
            narrative_parts.append(
                "Primary risk-elevating factors include: " + "; ".join(inc_descriptions) + ". "
                "In historical data, these characteristics are associated with a higher likelihood of repayment delinquency."
            )

        if top_decreasing:
            dec_descriptions = []
            for feat, val in top_decreasing[:3]:
                readable_feat = self._humanize_feature_name(feat)
                dec_descriptions.append(f"**{readable_feat}** (protective credit impact: {val:.3f})")
            narrative_parts.append(
                "Key credit-protective factors mitigating risk include: " + "; ".join(dec_descriptions) + ". "
                "These attributes historically indicate stronger financial resilience and reliable debt servicing."
            )

        narrative_parts.append(
            "*Responsible Lending Note: This explanation reflects statistical model attributions (SHAP feature values) "
            "and does not constitute proof of real-world causation. A human credit officer should review the complete profile.*"
        )

        return "\n\n".join(narrative_parts)

    def _humanize_feature_name(self, feat_name: str) -> str:
        """Translates encoded feature column name into professional banking terminology."""
        if feat_name.startswith("checking_status_"):
            val = feat_name.replace("checking_status_", "")
            return f"Checking Account Balance ({val})"
        elif feat_name.startswith("credit_history_"):
            val = feat_name.replace("credit_history_", "")
            return f"Credit History Pattern ({val})"
        elif feat_name.startswith("savings_status_"):
            val = feat_name.replace("savings_status_", "")
            return f"Savings / Liquidity Level ({val})"
        elif feat_name.startswith("employment_"):
            val = feat_name.replace("employment_", "")
            return f"Length of Current Employment ({val})"
        elif feat_name.startswith("purpose_"):
            val = feat_name.replace("purpose_", "")
            return f"Loan Purpose ({val})"
        elif feat_name.startswith("property_magnitude_"):
            val = feat_name.replace("property_magnitude_", "")
            return f"Property Asset Ownership ({val})"
        elif feat_name.startswith("other_payment_plans_"):
            val = feat_name.replace("other_payment_plans_", "")
            return f"Other Active Installment Plans ({val})"
        elif feat_name.startswith("housing_"):
            val = feat_name.replace("housing_", "")
            return f"Housing Tenure ({val})"
        elif feat_name == "duration":
            return "Loan Term Duration"
        elif feat_name == "credit_amount":
            return "Requested Credit Principal"
        elif feat_name == "installment_rate":
            return "Installment to Income Ratio"
        elif feat_name == "age":
            return "Applicant Age"
        elif feat_name == "existing_credits":
            return "Number of Open Credits at Bank"
        return feat_name.replace("_", " ").title()

    def plot_local_contributions(self, top_increasing: list, top_decreasing: list):
        """
        Creates a clean horizontal bar chart displaying top risk-increasing and risk-decreasing drivers.
        """
        items = []
        for feat, val in top_decreasing:
            items.append((self._humanize_feature_name(feat), val, "#10b981"))  # emerald
        for feat, val in reversed(top_increasing):
            items.append((self._humanize_feature_name(feat), val, "#f43f5e"))  # rose/red

        if not items:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No significant feature deviations detected", ha='center', va='center')
            return fig

        labels = [x[0] for x in items]
        values = [x[1] for x in items]
        colors = [x[2] for x in items]

        fig, ax = plt.subplots(figsize=(9, max(4, len(items) * 0.45)), dpi=120)
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values, color=colors, height=0.6, edgecolor='none', alpha=0.9)

        ax.axvline(0, color='#64748b', linewidth=1, linestyle='--')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10, fontweight='500')
        ax.set_xlabel("SHAP Impact on Default Risk (Δ Log-Odds / Prob)", fontsize=10, labelpad=8)
        ax.set_title("Local Feature Contribution Breakdown (Applicant Level)", fontsize=12, fontweight='bold', pad=12)

        # Annotate bar values
        for bar in bars:
            width = bar.get_width()
            offset = 0.003 if width >= 0 else -0.003
            ha = 'left' if width >= 0 else 'right'
            ax.text(width + offset, bar.get_y() + bar.get_height()/2, f"{width:+.3f}",
                    va='center', ha=ha, fontsize=9, color='#1e293b', fontweight='600')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        ax.grid(axis='x', linestyle=':', alpha=0.5)
        plt.tight_layout()
        return fig

    def plot_global_importance(self, X_sample_proc: np.ndarray, max_display: int = 15):
        """
        Creates a clean global feature importance plot using mean absolute SHAP values.
        """
        if self.explainer_type == "tree":
            sample_shap = self.shap_explainer.shap_values(X_sample_proc)
            if isinstance(sample_shap, list):
                shap_arr = np.array(sample_shap[1])
            elif len(sample_shap.shape) == 3:
                shap_arr = sample_shap[:, :, 1]
            else:
                shap_arr = sample_shap
        else:
            sample_shap = self.shap_explainer.shap_values(X_sample_proc)
            shap_arr = sample_shap[1] if isinstance(sample_shap, list) else sample_shap

        mean_abs_shap = np.mean(np.abs(shap_arr), axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:max_display]

        labels = [self._humanize_feature_name(self.feature_names[i]) for i in reversed(top_indices)]
        values = [mean_abs_shap[i] for i in reversed(top_indices)]

        fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values, color='#3b82f6', height=0.6, alpha=0.85)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Mean |SHAP Value| (Average Impact on Model Risk Output)", fontsize=10, labelpad=8)
        ax.set_title("Global Feature Importance Ranking (Credit Risk Model)", fontsize=12, fontweight='bold', pad=12)

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.001, bar.get_y() + bar.get_height()/2, f"{w:.3f}",
                    va='center', ha='left', fontsize=9, color='#334155')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        ax.grid(axis='x', linestyle=':', alpha=0.5)
        plt.tight_layout()
        return fig
