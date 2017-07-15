# -*- coding: utf-8 -*-
"""
Replace least frequently occuring variables in a column with a single variable
"""

def replace_lf(DF,column_name):
    feat=DF[column_name]
    df=feat.value_counts()
    o=df[df<thresh].index.tolist()
    replace_dict[column_name]=dict(df[df>=thresh])
    feat[feat.isin(o)]="A9"
    DF[column_name]=feat

