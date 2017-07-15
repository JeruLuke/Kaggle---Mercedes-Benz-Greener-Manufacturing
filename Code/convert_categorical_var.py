# -*- coding: utf-8 -*-
"""
Convert categorical variable to numerical variables using impact encoding

DF          : the dataframe of the train set
column_name : list of column names after removing one that are correlated and duplicated
labels      : dataframe containng target variables

"""

def mean_median(DF,column_name,labels):
    df=pd.DataFrame(DF[column_name])
    df["y"]=labels
    u=df.groupby(column_name).mean()
    u_dict=u.to_dict()["y"]
    cat_dict[column_name]=u_dict
    DF[column_name]=DF[column_name].map(u_dict)
    






