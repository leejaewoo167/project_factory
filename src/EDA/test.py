import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.bounds = []

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.bounds = []
        for i in range(X.shape[1]):
            q1 = np.percentile(X[:, i], 25)
            q3 = np.percentile(X[:, i], 75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self.bounds.append((lower, upper))
        return self

    def transform(self, X):
        X = np.asarray(X)
        one_d = False
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            one_d = True

        X_clipped = X.copy()
        for i, (lower, upper) in enumerate(self.bounds):
            X_clipped[:, i] = np.clip(X[:, i], lower, upper)

        if one_d:
            return X_clipped.ravel()
        return X_clipped

# 테스트용 데이터
df = pd.DataFrame({'col1': [1, 2, 3, 100, 5]})

# 적용
clipper = IQRClipper()
df['col1_clipped'] = clipper.fit_transform(df[['col1']])