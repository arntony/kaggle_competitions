import pandas as pd

df_trn=pd.read_csv('../Data/train.csv')
num_cols=df_trn.dtypes[(df_trn.dtypes=='int64') | (df_trn.dtypes=='float64')].index.values.tolist()
non_num_cols=df_trn.dtypes[(df_trn.dtypes!='int64') & (df_trn.dtypes!='float64')].index.values.tolist()
df_trn[num_cols].fillna(0,inplace=True)
df_trn[non_num_cols].fillna('',inplace=True)
print(df_trn)
