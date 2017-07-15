'''
Clubbing values in the categorical variables X0, X1, X2
aa, ab, ac, ...., az ->a
ba, bb, bc, ...., bz -> b
'''

import pandas as pd

path='C:/JERU/Kaggle/Mercedes/Data/'
data = pd.read_csv(path+'train.csv')
data_test = pd.read_csv(path+"test.csv")


#---only columns X0, X1 and X2 contain variables like aa, ab, ac, bb,......
var_names = ["X0","X1","X2"]

def club_alp(var):
    if len(var)==2:
        return var[0]
    else:
        return var
    
def create_club_feat(df,col_names):
    for z in col_names:
        df[z+"_club"]=df[z].apply(lambda x: club_alp(x))
        
create_club_feat(data, var_names)
create_club_feat(data_test, var_names)