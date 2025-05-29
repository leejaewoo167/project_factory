import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

np.random.seed(600)

def plotting(df: pd.DataFrame,
             col_x: str,
             col_y: str,
             x_label: str,
             y_label: str,
             plot_type: str):
    
    fig, ax = plt.subplots()
    
    if plot_type == 'line':
        ax.plot(df[col_x], df[col_y])
    else:
        ax.bar(df[col_x], df[col_y])
    ax.set(xlabel = x_label, ylabel = y_label)
    ax.grid()
    plt.xticks(rotation=45)
    plt.show()

df = pd.read_csv('./data/train.csv')
df.info()
len(df['cast_pressure'].loc[df['cast_pressure']==1449, ])
df[['upper_mold_temp3','lower_mold_temp3']] = df[['upper_mold_temp3','lower_mold_temp3']].replace(1449, np.nan) # -> Nan 

# 1. # of errors in each hour
df_time = df.drop(columns=['id'])

df_time['hour'] = pd.to_datetime(df_time['date']).dt.hour
hour_passfail = df_time.groupby('hour')['passorfail'].value_counts()
hour_passfail = hour_passfail.to_frame().reset_index()

error_count = hour_passfail.loc[hour_passfail['passorfail'] == 1,]
plotting(error_count, 
         'hour', 
         'count', 
         'hour',
         '# of errors',
         'line')

# 2. # of errors in each mold code
codes =  list(df_time['mold_code'].unique())

pf_name_li = ['corrects', 'errors']
pf_li = list(df_time['passorfail'].unique())

mold_code_df = pd.DataFrame()
mold_code_df['mold_code'] = pd.Series(codes)
for k, v in zip(pf_name_li, pf_li):
    pf = df_time.loc[df_time['passorfail'] == v, ]
    
    mold_code = pf.groupby('mold_code')['passorfail'].count()
    mold_code = mold_code.to_frame().reset_index()
    mold_code_df[k] = mold_code['passorfail']

for c in pf_name_li:
    mold_code_df[f'{c}_rate'] = mold_code_df[c] / (mold_code_df['corrects'] + mold_code_df['errors'])  
mold_code_df = mold_code_df.drop(columns=pf_name_li, axis=1)
mold_code_df = mold_code_df.set_index('mold_code')

mold_code_df.plot.bar(stacked=True, figsize=(10,10))
plt.xticks(rotation = 45, fontsize = 20)
plt.yticks(fontsize = 20)
plt.legend(fontsize=15) 