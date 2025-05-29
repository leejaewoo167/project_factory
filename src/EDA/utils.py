import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, QuantileTransformer

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
    
    
def is_multimodal_series(series, 
                         max_components=5, 
                         threshold=2):
    
    se = series.dropna().values.astype(float)
    
    if len(se) < 10:
        return False
    
    se = se.reshape(-1, 1)
    lower_bound = np.inf
    best_n = 1

    for n in range(1, max_components + 1):
        try:
            gmm = GaussianMixture(n_components=n, 
                                  random_state=600)
            gmm.fit(se)
            bic = gmm.bic(se)
            if bic < lower_bound - threshold:
                lower_bound = bic
                best_n = n
        except:
            continue

    return best_n > 1

def standardize(series):
    """
    다봉 여부에 따라 Series를 표준화하여 반환
    """
    data = series.values.reshape(-1, 1).astype(float)
    
    if is_multimodal_series(series):
        transformer = QuantileTransformer(output_distribution='normal', random_state=42)
    else:
        transformer = StandardScaler()

    transformed = transformer.fit_transform(data).flatten()
    return pd.Series(transformed, index=series.index, name=series.name)