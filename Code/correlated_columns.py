'''
Script to find columns that are correlated 

Note: keep in mind columns are correlated with themselves, so the main diagonal 
elemenst of the correlation matrix must be replaced
'''
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


#train_path = 'D:/BACKUP/Kaggle/Mercedes Benz/Data/Copy Data/train.csv'
train_path = 'C:/JERU/Kaggle/Mercedes/train.csv'
train_df = pd.read_csv(train_path)
print("Train shape : ", train_df.shape)

#---copy of dataframe
copy_df = train_df.copy()

#--- Finding correlation of dataframe
corr = train_df.corr()

#--- fill diagonal elements with any number other than 1.0
np.fill_diagonal(corr.values, -2)

#--- Get list of all correlated columns
already_in = set()
result = []
for col in corr:
    perfect_corr = corr[col][corr[col] == 1].index.tolist()
    if perfect_corr and col not in already_in:
        already_in.update(set(perfect_corr))
        perfect_corr.append(col)
        result.append(perfect_corr)

#--- remove correlated columns but retaining one
count = 0
for i in range(0, len(result)):
    for j in range(0, len(result[i])): 
        if j != 0:
            copy_df = copy_df.drop(result[i][j], 1)
            count = count + 1
print count   #----Number of columns dropped          

    