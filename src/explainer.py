"""
Explainability Module for FraudLens AI.
Computes local and global feature impact using Tree SHAP and feature contribution analysis,
providing human-interpretable risk reasoning for fraud auditors.
"""

import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class FraudExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        if HAS_SHAP:
            try:
                self.explainer = shap.TreeExplainer(model)
            except Exception as e:
                print(f"[FraudLens AI] Warning: SHAP initialization fallback: {e}")
                self.explainer = None

    def explain_instance(self, input_vector: pd.DataFrame, top_k: int = 6):
        """
        Computes the top factors driving the fraud risk score for a single transaction.
        Returns a list of dicts: [{'feature': name, 'impact': float, 'value': float, 'direction': 'risk_increasing'|'risk_reducing'}]
        """
        # If SHAP is available
        if self.explainer is not None:
            try:
                shap_vals = self.explainer.shap_values(input_vector)
                # For binary classification, shap_values can be 1D, 2D, or list of arrays
                if isinstance(shap_vals, list):
                    vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
                elif len(shap_vals.shape) == 2:
                    vals = shap_vals[0]
                else:
                    vals = shap_vals[0][:, 1] if shap_vals.ndim == 3 else shap_vals[0]

                contributions = []
                for feat, val, inp_val in zip(self.feature_names, vals, input_vector.iloc[0]):
                    contributions.append({
                        "feature": feat,
                        "impact": float(val),
                        "abs_impact": abs(float(val)),
                        "value": float(inp_val),
                        "direction": "Risk Increasing ⬆" if val > 0 else "Risk Decreasing ⬇"
                    })

                contributions.sort(key=lambda x: x["abs_impact"], reverse=True)
                return contributions[:top_k]
            except Exception as e:
                print(f"[FraudLens AI] SHAP calculation error, using fallback: {e}")

        # Fallback using model feature importances and standardized input anomaly
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            row = input_vector.iloc[0].values
            impacts = (row - np.mean(row)) * importances
            contributions = []
            for feat, imp, val in zip(self.feature_names, impacts, row):
                contributions.append({
                    "feature": feat,
                    "impact": float(imp),
                    "abs_impact": abs(float(imp)),
                    "value": float(val),
                    "direction": "Risk Increasing ⬆" if imp > 0 else "Risk Decreasing ⬇"
                })
            contributions.sort(key=lambda x: x["abs_impact"], reverse=True)
            return contributions[:top_k]

        return []
