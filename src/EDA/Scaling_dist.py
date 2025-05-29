import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, QuantileTransformer

np.random.seed(600)

def cal_z_score(x: pd.Series):
    return (x - np.mean(x)) / np.std(x)

def del_missing_outlier(x:pd.DataFrame):
    
    # step 1. remove missing value
    x = x.dropna().reset_index(drop=True)
    
    # step 2. remove outlier 
    Q1 = x.quantile(0.25)
    Q3 = x.quantile(0.75)
    IQR = Q3 - Q1

    x = x[~((x < (Q1 - 1.5 * IQR))|(x > (Q3 + 1.5 * IQR)))]
    return x.reset_index(drop=True)

def plotting(r_nums: int,
             c_nums: int,
             df: pd.DataFrame):
    
    num_df = list(df.columns)
    fig, axs = plt.subplots(r_nums, 
                            c_nums, 
                            sharey=True, 
                            tight_layout=True)

    r, c = 0, 0
    for i, col in enumerate(num_df):
        axs[r][c].hist(corrects[col])
        axs[r][c].set_xlabel(col)
        
        if (i+1) % 5 == 0 and i != 0:
            r+= 1
            c = 0
        else:
            c += 1
    plt.show()

df = pd.read_csv('./data/train.csv')

num_df = df.select_dtypes('number')
num_df = num_df.drop(columns=['id', 'count', 'mold_code', 'EMS_operation_time', 'upper_mold_temp3', 'lower_mold_temp3'])
corrects = num_df.loc[num_df['passorfail'] == 0,].reset_index(drop=True)
errors = num_df.loc[num_df['passorfail'] == 1,].reset_index(drop=True)

plotting(4,5,corrects)

scaled_df = pd.DataFrame()
for i, col in enumerate(corrects.columns):
    colle_col = del_missing_outlier(corrects[col])
    # colle_col = standardize(colle_col)
    scaled_df[col] = colle_col
    
    # print(f"z-score: {cal_z_score(colle_col)}")
plotting(4,5,scaled_df)

























































































# def is_multimodal_series(series, 
#                          max_components=5, 
#                          threshold=2):
    
#     se = series.dropna().values.astype(float)
    
#     if len(se) < 10:
#         return False
    
#     se = se.reshape(-1, 1)
#     lower_bound = np.inf
#     best_n = 1

#     for n in range(1, max_components + 1):
#         try:
#             gmm = GaussianMixture(n_components=n, 
#                                   random_state=600)
#             gmm.fit(se)
#             bic = gmm.bic(se)
#             if bic < lower_bound - threshold:
#                 lower_bound = bic
#                 best_n = n
#         except:
#             continue

#     return best_n > 1

# def standardize(series):
#     """
#     다봉 여부에 따라 Series를 표준화하여 반환
#     """
#     data = series.values.reshape(-1, 1).astype(float)
    
#     if is_multimodal_series(series):
#         transformer = QuantileTransformer(output_distribution='normal', random_state=42)
#     else:
#         transformer = StandardScaler()

#     transformed = transformer.fit_transform(data).flatten()
#     return pd.Series(transformed, index=series.index, name=series.name)