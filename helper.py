import random
import math
import copy
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif

DEFAULT_SIGMA = 1.0
BASE_SIGMA_MIN = 1e-6

class SimpleScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.fitted = False
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        self.fitted = True
    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Scaler not fitted")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class ReturnSurface:
    def __init__(self, feature_dim, sigma=None, decay=1.0, max_samples=1000000, scaler=None):
        self.feature_dim = feature_dim
        self.sigma_fixed = sigma
        self.decay = float(decay)
        self.max_samples = int(max_samples)
        self.scaler = scaler if scaler is not None else SimpleScaler()
        self.feature_weights = np.ones(feature_dim, dtype=float)  # <-- MI weights
        self._init_storage()

    def _init_storage(self):
        self.features = np.zeros((0, self.feature_dim), dtype=float)
        self.returns = np.zeros((0,), dtype=float)
        self.weights = np.zeros((0,), dtype=float)
        self._scaler_fitted = False

    def compute_feature_weights(self):
        """Compute Mutual Information weights between features and profit sign."""
        if len(self.features) < 10:
            print("⚠ Not enough samples to compute MI weights yet.")
            return
        y = (self.returns > 0).astype(int)
        try:
            mi = mutual_info_classif(self.features, y, discrete_features=False)
            mi = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)
            mi += 1e-6  # avoid zeros
            mi = mi / np.mean(mi)  # normalize around 1.0
            self.feature_weights = mi
            print(f"[MI] Updated feature weights (mean normalized): {np.round(mi, 3)}")
        except Exception as e:
            print("MI computation failed:", e)

    def _normalize(self, X):
        return self.scaler.transform(X)

    def _estimate_base_sigma(self):
        if len(self.features) < 2:
            return DEFAULT_SIGMA
        n = len(self.features)
        m = min(400, n)
        idx = np.random.choice(n, m, replace=False)
        samples = self.features[idx]
        diffs = samples[:, None, :] - samples[None, :, :]
        d2 = np.sum(diffs * diffs, axis=2).ravel()
        med = np.median(d2) if d2.size > 0 else 0.0
        base = math.sqrt(max(med, 1e-12))
        return max(base, BASE_SIGMA_MIN)

    def _estimate_sigma(self):
        if self.sigma_fixed is not None:
            return self.sigma_fixed
        n = len(self.features)
        if n < 2:
            return DEFAULT_SIGMA
        base = self._estimate_base_sigma()
        scale = max(1.0, math.sqrt(n))
        sigma = base / scale
        return max(sigma, BASE_SIGMA_MIN)

    def _rbf_weights(self, Xq, sigma):
        if len(self.features) == 0:
            return np.zeros((Xq.shape[0], 0))
        # --- MI-weighted kernel modification ---
        diffs = self.features[None, :, :] - Xq[:, None, :]
        wdiffs = diffs ** 2 * self.feature_weights[None, None, :]
        d2 = np.sum(wdiffs, axis=2)
        denom = 2.0 * (sigma ** 2)
        w = np.exp(-d2 / denom)
        return w

    def predict(self, feature_vec, metric='prob'):
        Xq = np.atleast_2d(feature_vec).astype(float)
        if (not self._scaler_fitted) or (len(self.features) == 0):
            return 0.0
        Xqn = self._normalize(Xq)
        sigma = self._estimate_sigma()
        w = self._rbf_weights(Xqn, sigma)
        weighted = w * self.weights[None, :]
        denom = np.sum(weighted, axis=1) + 1e-12
        if metric == 'prob':
            indicator = (self.returns > 0).astype(float)
            numer = np.dot(weighted, indicator)
            return float(numer[0] / denom[0])
        elif metric == 'expectation':
            numer = np.dot(weighted, self.returns)
            return float(numer[0] / denom[0])
        else:
            raise ValueError("metric must be 'prob' or 'expectation'")

    def update(self, feature_vec, realized_return, add_weight=1.0):
        X = np.atleast_2d(feature_vec).astype(float)
        if not self._scaler_fitted and len(self.features) == 0:
            self.scaler.fit(X)
            self._scaler_fitted = True
        Xn = self._normalize(X)
        if len(self.weights) > 0:
            self.weights *= self.decay
            self.weights[self.weights < 1e-6] = 0.0
        self.features = np.vstack([self.features, Xn])
        self.returns = np.concatenate([self.returns, np.array([float(realized_return)])])
        self.weights = np.concatenate([self.weights, np.array([float(add_weight)])])
        if len(self.weights) > self.max_samples:
            idx = np.argsort(self.weights)
            to_drop = len(self.weights) - self.max_samples
            keep_mask = np.ones(len(self.weights), dtype=bool)
            keep_mask[idx[:to_drop]] = False
            self.features = self.features[keep_mask]
            self.returns = self.returns[keep_mask]
            self.weights = self.weights[keep_mask]

    def n_samples(self):
        return len(self.weights)
    def stats(self):
        if self.n_samples() == 0:
            return {'n':0, 'mean':0.0, 'std':0.0, 'sigma': float(self._estimate_sigma())}
        return {'n': self.n_samples(), 'mean': float(np.mean(self.returns)), 'std': float(np.std(self.returns)), 'sigma': float(self._estimate_sigma())}
    
    def save_to_csv(self, filename_prefix="surface"):
        """Save all stored data (features, returns, weights) to CSV."""
        if len(self.features) == 0:
            print(f"⚠ No samples to save for {filename_prefix}.")
            return
        feature_cols = [f"f{i}" for i in range(self.features.shape[1])]
        df = pd.DataFrame(self.features, columns=feature_cols)
        df["return"] = self.returns
        df["weight"] = self.weights
        df.to_csv(f"{filename_prefix}_data.csv", index=False)

        fw_df = pd.DataFrame([self.feature_weights], columns=feature_cols)
        fw_df.to_csv(f"{filename_prefix}_feature_weights.csv", index=False)


class KernelReturnFilter:
    def __init__(self, feature_dim, sigma=None, decay=1.0, max_samples=8000):
        self.long = ReturnSurface(feature_dim, sigma=sigma, decay=decay, max_samples=max_samples)
        self.short = ReturnSurface(feature_dim, sigma=sigma, decay=decay, max_samples=max_samples)
        self.feature_dim = feature_dim

    def predict_prob(self, feature_vec, side):
        surf = self.long if side == 'long' else self.short
        return surf.predict(feature_vec, metric='prob')

    def predict_expec(self, feature_vec, side):
        surf = self.long if side == 'long' else self.short
        return surf.predict(feature_vec, metric='expectation')

    def update(self, feature_vec, realized_return, side, add_weight=1.0):
        surf = self.long if side == 'long' else self.short
        surf.update(feature_vec, realized_return, add_weight=add_weight)

    def stats(self):
        s_long = self.long.stats()
        s_short = self.short.stats()
        return {
            'n_long': s_long['n'], 'mean_long': s_long['mean'], 'std_long': s_long['std'], 'sigma_long': s_long['sigma'],
            'n_short': s_short['n'], 'mean_short': s_short['mean'], 'std_short': s_short['std'], 'sigma_short': s_short['sigma']
        }
