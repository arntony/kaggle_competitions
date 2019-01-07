
import pandas as pd

class prepare_data:
    
    def prep(self):
        df_trn=pd.read_csv('/home/arnab/Projects/kaggle_competitions/Titanic/Data/train.csv')
        self.df_trn_new=self.preprocess(df_trn)
        self.col_max=self.df_trn_new.max(axis=0)
        self.df_norm=self.df_trn_new/self.col_max
    
    def preprocess(self, df_trn):
        num_cols=df_trn.dtypes[(df_trn.dtypes=='int64') | (df_trn.dtypes=='float64')].index.values.tolist()
        non_num_cols=df_trn.dtypes[(df_trn.dtypes!='int64') & (df_trn.dtypes!='float64')].index.values.tolist()
        df_trn[num_cols]=df_trn[num_cols].fillna(0)
        df_trn[non_num_cols]=df_trn[non_num_cols].fillna('')
        sex={'male':1,'female':0}
        df_trn.Sex=df_trn.Sex.apply(lambda x: sex[x])
        cabin_type=df_trn.Cabin.apply(lambda x: '' if len(x)==0 else x[0])
        cabin_dict={'cabin_type_'+c: cabin_type.apply(lambda x: 1 if x==c else 0).tolist() for c in cabin_type.unique()}
        df_cabin=pd.DataFrame(cabin_dict)
        df_trn_new=df_trn.drop(['Embarked', 'PassengerId', 'Name', 'Ticket','Cabin'],axis=1)
        return pd.concat([df_trn_new,df_cabin],axis=1)