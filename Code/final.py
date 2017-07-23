import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.metrics import r2_score
from scipy.stats import boxcox
import pickle
path='D:/BACKUP/Kaggle/Mercedes Benz/Data/Copy Data/'
data=pd.read_csv(path+'train.csv')
#data["id2"]=data["ID"]
#labels=data["y"].copy()
#data.ix[data["y"]>=120,"y"]=120+np.random.rand((data["y"]>=120).sum())
#data=data.ix[data["y"]<200,:]
data.reset_index(inplace=True,drop=True)
labels=data["y"].copy()
thresh=5

#load test data
data_test = pd.read_csv(path+"test.csv")
test_id = data_test["ID"]
#data_test["id2"]=data_test["ID"]

##############################################################################
'''
Clubbing values in the categorical variables X0, X1, X2
aa, ab, ac, ...., az, ->a
ba, bb, bc, ...., bz -> b
'''
var_names = ["X0","X1","X2"]
def club_alp(var):
    if len(var)==2:
        return var[0]
    else:
        return var
    
def create_club_feat(df,col_names):
    for z in col_names:
        df[z+"_club"]=df[z].apply(lambda x: club_alp(x))
        
create_club_feat(data,var_names)
create_club_feat(data_test,var_names)
##############################################################################


#box-cox the response variable
def box_cox_fn(variable):
    box_trans=boxcox(variable)
    variable=(((variable)**box_trans[1])-1)/box_trans[1]
    return variable,box_trans[1]

def drop_cols(df,col_names):
    df.drop(col_names,axis=1,inplace=True)
    
def replace_lf(DF,column_name):
    feat=DF[column_name]
    df=feat.value_counts()
    o=df[df<thresh].index.tolist()
    replace_dict[column_name]=dict(df[df>=thresh])
    feat[feat.isin(o)]="A9"
    DF[column_name]=feat

#--Impact encoding
def mean_median(DF,column_name,labels):
    df=pd.DataFrame(DF[column_name])
    df["y"]=labels
    u=df.groupby(column_name).mean()
    u_dict=u.to_dict()["y"]
    cat_dict[column_name]=u_dict
    DF[column_name]=DF[column_name].map(u_dict)

def replace_lf_test(DF,column_name):
    feat=DF[column_name]
    dict_n=replace_dict[column_name]
    dict_n=list(dict_n.keys())
    feat[~feat.isin(dict_n)]="A9"
    DF[column_name]=feat
        
def mean_median_test(DF,column_name):
    DF[column_name]=DF[column_name].map(cat_dict[column_name])


def categorical_convert(DF,cat_columns,train_=True):
    #convert categorical variables to numerical ones
    if train_==True:
       for kk in cat_columns:
           replace_lf(DF,kk)
       for kk in cat_columns:
           mean_median(DF,kk,labels_train)
    else:
        for kk in cat_columns:
           replace_lf_test(DF,kk)
        for kk in cat_columns:
           mean_median_test(DF,kk)

def inv_boxcox(data,lambda_):
    r=lambda_*data
    g=1/lambda_
    return((1+r)**g)

##shamelessly plucked from stackoverflow(https://stackoverflow.com/questions/14984119/python-pandas-remove-duplicate-columns)
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

def second_order(df,c_names):
    names_col=[]
    pp=0
    for ii in c_names[:c_names.size-1]:
        for jj in c_names[pp:c_names.size]:
            if ii!=jj:
               names_col.append(ii+jj)
               df[ii+jj]=df[ii] +"_"+df[jj]
        pp+=1
    return df,names_col

def correct_cat_dict(dict_,value,replace_value):
    for g in dict_.keys():
        try:
            yu=dict_[g][value]
            dict_[g][value]=replace_value
        except:
            pass
    return dict_
           

            
#drop unwanted cols in train and test set
drop_cols(data,["X4"])
drop_cols(data_test,["X4"])

cat_columns = data.dtypes[data.dtypes=="object"].index
binary_columns = data.dtypes[data.dtypes!="object"].index[1:]

'''
#create high order features
data,col_names_train=second_order(data,cat_columns)
data_test,col_names_test=second_order(data_test,cat_columns)
'''

#filter out the binary features
data_b=data[binary_columns]

#remove duplicate columns
#data_b = data_b.loc[:,~data_b.columns.duplicated()]
cols= duplicate_columns(data_b)

#remove duplicate columns
data_b = data_b.loc[:,~data_b.columns.isin(cols)]

#find all zero/nearly zero columns and remove it
all_z = data_b.sum(axis=0)
all_z = all_z[all_z>=thresh]

#final all ones and nearly ones columns and remove it
sample_size=data.shape[0]
all_z=all_z[all_z<=sample_size-thresh]

correlated_cols=['X263', 'X279', 'X232','X120','X76','X136','X130','X158','X194','X205','X157']

identity_cols = list(set(all_z.index.tolist()).difference(correlated_cols))

#final_columns = ["y"]+list(cat_columns)+col_names_train+identity_cols
final_columns = ["y"]+list(cat_columns)+identity_cols+["ID"]
data_set = data[final_columns]

#subset columns in test set
data_test=data_test[final_columns[1:]]



########################################

#perform k-fold cross validation
max_estimators_options=[7]   
#build randomForest classifier
##build predictive models
print("lets start cross validation")
#cross validation to find optimal regularization parameter

test_results=[]

cv = KFold(sample_size, n_folds=5,shuffle=True, random_state=123)
result_r2=np.empty([cv.n_folds,len(max_estimators_options)],dtype=float)
CV_stacked=[]
##
count=0
for alp in max_estimators_options:
    params = {'n_estimators':400, 'max_depth': 7, 'min_samples_split':50,"max_features":75,
          'random_state':0,"verbose":1, 'n_jobs' : -1 }
    model=RandomForestRegressor(**params)
    result=[]
    actual=[]
    pred_CV=[]

    for traincv, testcv in cv:
        X_train = data_set.ix[traincv,final_columns[1:]]
        #labels_train = data_set.ix[traincv,"y"]
        labels_train = labels[traincv]
        #box-cox the labels
        #labels_train,lambda_ = box_cox_fn(labels_train)
        mean_train_lab = labels_train.mean()
        
        X_CV = data_set.ix[testcv,final_columns[1:]]
        #labels_test= data_set.ix[testcv,"y"]
        labels_test= labels[testcv]

        cat_dict={}
        replace_dict={}
        categorical_convert(X_train,cat_columns,train_=True)
        cat_dict = correct_cat_dict(cat_dict,"A9",mean_train_lab)
        categorical_convert(X_CV,cat_columns,train_=False)
        test_data = data_test.copy()
        categorical_convert(test_data,cat_columns,train_=False)

        final_pred = model.fit(X_train,labels_train).predict(X_CV)
        #val_score =r2_score(labels_test,inv_boxcox(final_pred,lambda_))
        #test_pred = inv_boxcox(model.predict(test_data),lambda_)
        val_score =r2_score(labels_test,final_pred)
        actual+=labels_test.tolist()
        pred_CV+=final_pred.tolist()
        test_pred = model.predict(test_data)
        test_results.append(test_pred)
        #val_score =r2_score(data_set.ix[testcv,"y"],final_pred)
        #print(val_score)
        result.append(val_score)
    result_r2[:,count]=result
    stacked_CV = r2_score(actual,pred_CV)
    CV_stacked.append(stacked_CV)
    print(count)
    print(result)
    count=count+1

mean =result_r2.mean(axis=0)
std=result_r2.std(axis=0)
###fine best estimator size
print(mean)
print(std)
print(CV_stacked)

############################
#getting important features---

imp = model.feature_importances_

#-- taking columns of train set in a list:
feat_cols = data_set.columns[1:].get_values()
feat_cols.tolist()

#-- collecting columns having imp > 0.01
imp_col = []
ccc = 0
for i in range(0, len(imp)):
    if imp[i] > 0.0005 :
        print feat_cols[i]
        imp_col.append(feat_cols[i])
        ccc = ccc + 1
print ccc

np.save(path + 'imp_feat', imp_col)
'''
test_results = np.array(test_results)

submit = pd.DataFrame()
submit["ID"]=test_id
submit["y"]=test_results.sum(axis=0)/(cv.n_folds*len(max_estimators_options))
      
submit.to_csv(path+"submit_model_21.csv",index=False)  
'''
        



