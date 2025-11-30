"""
Linear regression model using Gradient Descent (Task 4)

Implements full-batch gradient descent with optional early stopping.
Input features are standardized; the target is centered/scaled as well.
"""

import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error


class GradientDescentLinearModel:
    def __init__(self, learning_rate=0.005, epochs=1000, early_stopping=True, patience=20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.w = None
        self.b = 0.0
        self.train_curve = []
        self.val_curve = []
        self.mean_ = None
        self.std_ = None
        self.y_mean_ = 0.0
        self.y_std_ = 1.0

    def _scale(self, X):
        return (X - self.mean_) / self.std_

    def _predict_raw(self, X):
        return X @ self.w + self.b

    def _step(self, Xs, ys):
        pred = self._predict_raw(Xs)
        errors = pred - ys
        m = Xs.shape[0]
        grad_w = (2.0 / m) * (Xs.T @ errors)
        grad_b = 2.0 * errors.mean()
        grad_w = np.clip(grad_w, -10.0, 10.0)
        grad_b = np.clip(grad_b, -10.0, 10.0)
        self.w -= self.learning_rate * grad_w
        self.b -= self.learning_rate * grad_b
        return float(root_mean_squared_error(ys, pred))

    def fit(self, X, y, X_val=None, y_val=None):
        self.train_curve = []
        self.val_curve = []
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        self.y_mean_ = y.mean()
        self.y_std_ = y.std() + 1e-8

        Xs = self._scale(X)
        ys = (y - self.y_mean_) / self.y_std_
        if X_val is not None:
            Xs_val = self._scale(X_val)
            ys_val = (y_val - self.y_mean_) / self.y_std_

        n, d = Xs.shape
        self.w = np.zeros(d)
        self.b = 0.0

        best_val = None
        wait = 0
        for _ in range(self.epochs):
            tr_rmse = self._step(Xs, ys)
            self.train_curve.append(tr_rmse)

            if X_val is not None:
                val_pred = self._predict_raw(Xs_val)
                val_rmse = float(root_mean_squared_error(ys_val, val_pred))
                self.val_curve.append(val_rmse)
                if self.early_stopping:
                    if best_val is None or val_rmse < best_val - 1e-6:
                        best_val = val_rmse
                        wait = 0
                    else:
                        wait += 1
                        if wait >= self.patience:
                            break

        return self

    def continue_training(self, X, y, X_val=None, y_val=None, steps=500):
        Xs = self._scale(X)
        ys = (y - self.y_mean_) / self.y_std_
        if X_val is not None:
            Xs_val = self._scale(X_val)
            ys_val = (y_val - self.y_mean_) / self.y_std_

        best_val = None
        wait = 0
        for _ in range(steps):
            tr_rmse = self._step(Xs, ys)
            self.train_curve.append(tr_rmse)

            if X_val is not None:
                val_pred = self._predict_raw(Xs_val)
                val_rmse = float(root_mean_squared_error(ys_val, val_pred))
                self.val_curve.append(val_rmse)
                if self.early_stopping:
                    if best_val is None or val_rmse < best_val - 1e-6:
                        best_val = val_rmse
                        wait = 0
                    else:
                        wait += 1
                        if wait >= self.patience:
                            break

        return self

    def transfer_training(self,
                                X_pre_train, y_pre_train, X_pre_val, y_pre_val,
                                X_ft_train, y_ft_train, X_ft_val, y_ft_val,
                                fine_tune_steps=300,
                                prefix_pre="pre",
                                prefix_ft="ft"):
        res = {}
        self.fit(X_pre_train, y_pre_train, X_pre_val, y_pre_val)
        res[f"before_{prefix_pre}_{prefix_pre}"] = self.evaluate(X_pre_train, y_pre_train, X_pre_val, y_pre_val)
        res[f"before_{prefix_pre}_{prefix_ft}"] = self.evaluate(X_pre_train, y_pre_train, X_ft_val, y_ft_val)
        X_comb = np.vstack([X_pre_train, X_ft_train])
        y_comb = np.concatenate([y_pre_train, y_ft_train])
        self.fit(X_comb, y_comb, X_ft_val, y_ft_val)
        res[f"after_{prefix_ft}_{prefix_pre}"] = self.evaluate(X_pre_train, y_pre_train, X_pre_val, y_pre_val)
        res[f"after_{prefix_ft}_{prefix_ft}"] = self.evaluate(X_ft_train, y_ft_train, X_ft_val, y_ft_val)
        return res

    def predict(self, X):
        Xs = self._scale(X)
        return self._predict_raw(Xs) * self.y_std_ + self.y_mean_

    def evaluate(self, X_train, y_train, X_val=None, y_val=None):
        y_pred_train = self.predict(X_train)
        result = {
            "r2_train": float(r2_score(y_train, y_pred_train)),
            "rmse_train": float(root_mean_squared_error(y_train, y_pred_train))
        }
        if X_val is not None:
            y_pred_val = self.predict(X_val)
            result["r2_val"] = float(r2_score(y_val, y_pred_val))
            result["rmse_val"] = float(root_mean_squared_error(y_val, y_pred_val))
        return result
