import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind

from src.EDA.utils import IQRClipper
from sklearn.impute import SimpleImputer

def plotting(correct:pd.DataFrame, 
             errors: pd.DataFrame):

    return None
    
def find_threshold(errors: pd.DataFrame):
    return errors.mean()

def normality(t_df: pd.DataFrame):
    _, p_val = shapiro(t_df)
    
    if p_val > 0.05:
        print('It does follow the normality')
        return True
    else:
        print('It does not follow the normality')
        return False

def test_parametric(corrects: pd.DataFrame, 
                    errors: pd.DataFrame):
    _, p_val = ttest_ind(corrects, errors)
    
    if p_val > 0.05:
        print('They maybe same between')
        return True
    else:
        print('They do not same between')
        return False

def test_nonparametric(corrects: pd.DataFrame,
                       errors: pd.DataFrame):
    _, p_val = mannwhitneyu(corrects, errors, method='auto')
    
    if p_val > 0.05:
        print('They maybe same between')
        return True
    else:
        print('They do not same between')
        return False

df = pd.read_csv('./data/train.csv')
f_li = ['molten_temp', 'cast_pressure', 'biscuit_thickness', 'passorfail']

# Step 1: process the missing value and outlier
df_fil = df[f_li]
df_fill_cols = list(df_fil.iloc[:, :3].columns)

# missing value
f_dict = dict()
for c in df_fill_cols:
    df_m_fil = df_fil.loc[:, [c, 'passorfail']]
    missing_rate = (sum(df_m_fil[c].isna()) / len(df))*100
    if missing_rate < 30:
        f_dict[c] = df_m_fil.dropna().reset_index(drop=True)
    else:
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        df_m_fil[c] = imp.fit_transform(df_m_fil[c])
        f_dict[c] = df_m_fil
        
# outlier
for k,v in f_dict.items():
    q1 = np.percentile(v[k], 25)
    q3 = np.percentile(v[k], 75)
    
    iqr = q3 - q1
    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr
    
    outs = v[k].loc[(v[k] < lower_bound) | (v[k] > upper_bound), ]
    n_out = len(outs)
    tmp = len(v)
    
    print(f'the # of outliers: {n_out}')
    if n_out > int(round(len(v[k])*0.2, 0)):
        clipper = IQRClipper()
        v[k] = clipper.fit_transform(v[k])
        print('CLIP')
    else:
        out_idx = list(outs.index)
        v = v.drop(out_idx).reset_index(drop=True)
        print('DROP')
    print(f'the # of {k} frame is change {tmp} -> {len(v)}')

# Step 3: check the difference distribution between correctness group and incorrectness group
# Step 4: if it is correct, find threshold in incorrectness group

thresholds = dict()
for c in df_fill_cols:
    print(c)
    pt = f_dict[c]
    pt_c = pt.loc[pt['passorfail'] == 0, ][c]
    pt_e = pt.loc[pt['passorfail'] == 1, ][c]

    if normality(pt):
        if test_parametric(corrects=pt_c,
                        errors=pt_e):        
            thresholds[c] = find_threshold(errors=pt_e)
        else:
            print('drop')
            pass
    else:
        if test_nonparametric(corrects=pt_c,
                            errors=pt_e):
            thresholds[c] = find_threshold(errors=pt_e)
        else:
            print('drop')
            pass





















































































# # 0. missing value processing
# tr_col = 'passorfail'
# idx_col = 'mold_code'
# nd_cols = ['upper_mold_temp1', 'upper_mold_temp2', 
#            'lower_mold_temp1', 'lower_mold_temp2',
#            'sleeve_temperature', 'Coolant_temperature', 
#            'cast_pressure']

# df[nd_cols] = df[nd_cols].replace(1449, np.nan)
# nd_cols.extend([tr_col, idx_col])

# n_df = df[nd_cols]
# cols = list(n_df.columns)

# ################  prototype ################
# p_li = ['upper_mold_temp1', 'passorfail', 'mold_code']
# n_df = n_df[p_li]
# n_df = n_df.dropna()
# # 1. outlier processing -> IQR

# clipper = IQRClipper()
# n_df['upper_mold_temp1'] = clipper.fit_transform(n_df['upper_mold_temp1'])
# # del fitted_col

# # 2. standardization to each feature


# 3. find the threshold by 3-sigma level


# 4. inverse scaling to real value


# 5. save the threshold(lower and upper bound)