import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class RelapseLSCModel:

    def __init__(self, feature_names=None):
        # default feature set; adjust as needed
        self.feature_names = feature_names or [
            "frac_Primitive",
            "frac_LSC_high",
            "prim_LSC_density",
            "entropy_hierarchy",
            # optional bulk / delta features when available:
            # "delta_prim_post_chemo",
            # "delta_prim_post_allo",
            # "delta_prim_rel",
        ]
        self.model = None

    def _select_features(self, df):
        missing = [f for f in self.feature_names if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features in input: {missing}")
        X = df[self.feature_names].values.astype(float)
        return X

    def fit(self, feature_df):
        if "outcome" not in feature_df.columns:
            raise ValueError("feature_df must contain 'outcome' for training.")

        df = feature_df.dropna(subset=self.feature_names + ["outcome"]).copy()
        y = (df["outcome"] == "relapse").astype(int).values
        X = self._select_features(df)

        clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=5000,
        )
        clf.fit(X, y)
        self.model = clf
        return self

    def predict_proba(self, feature_row):
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        if isinstance(feature_row, dict):
            feature_row = pd.Series(feature_row)

        X = np.array([feature_row[f] for f in self.feature_names], dtype=float).reshape(1, -1)
        proba = self.model.predict_proba(X)[0, 1]
        return float(proba)

    def explain(self):
        if self.model is None:
            raise ValueError("Model not trained.")
        coefs = self.model.coef_[0]
        return pd.DataFrame({
            "feature": self.feature_names,
            "weight": coefs,
        }).sort_values("weight", ascending=False)
