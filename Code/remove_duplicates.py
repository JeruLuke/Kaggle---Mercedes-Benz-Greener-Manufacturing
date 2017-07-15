# -*- coding: utf-8 -*-
"""
Code to remove duplicate columns
"""
import pandas as pd

def duplicate_columns(frame):
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            iv = vs.iloc[:,i].tolist()
            for j in range(i+1, lcs):
                jv = vs.iloc[:,j].tolist()
                if iv == jv:
                    dups.append(cs[i])
                    break

    return dups

path='D:/BACKUP/Kaggle/Mercedes Benz/Data/Copy Data/'
data=pd.read_csv(path+'train.csv')

#---Separating binary columns from dataframe
binary_columns = data.dtypes[data.dtypes!="object"].index[1:]
data_b=data[binary_columns]

#---Function to find duplicate columns 
cols= duplicate_columns(data_b)

#---remove duplicate columns
data_b = data_b.loc[:,~data_b.columns.isin(cols)]