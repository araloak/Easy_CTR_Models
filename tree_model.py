from data_helper import *
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings
import time
import copy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction import DictVectorizer         #
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler       #归一化
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold #去掉特征大部分为某值的方法
import lightgbm as lgb
#from feature_selector import FeatureSelector
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')
sns.set(style="white", context="notebook")

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
#pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
#pd.set_option('max_colwidth',100)

def transStrFeature(data,test,featureList):
    for i in featureList:
        map_dict = {i:j+1 for j,i in enumerate(data[i].unique())}
        data[i] = data[i].map(map_dict)
        test[i] = test[i].map(map_dict)
    return data,test
def FeatureInfoUnique(data,feature=['A1','A2','A3','B1','B2','B3','C1','C2','C3','D1','D2']):
    for i in feature:
        INFO = data[i].unique()
        print("Feature: {} len: {} \nall: {}".format(i,len(INFO),INFO))
def dataInfoImage(data,target,feature,num):
    data['target'] = target['label']
    fig = plt.figure(figsize=(15,20))
    x = 1
    for i in feature:
        plt.subplot(num[0],num[1],x)
        sns.countplot(x=i,hue='target',hue_order=[1,0],data=data),
        sns.despine(bottom=True)
        plt.title(i)
        x += 1
    fig.tight_layout()
#删除出现次数少的样本，不建议
def deleteDATA(data,target,feature):
    data['target'] = target['label']
    for i in feature:
        g = data.groupby(data[i])
        data = g.filter(lambda x : len(x) > 100)
    print(train2.shape)
    return data
def MinMax():
    scale = MinMaxScaler()
    train = scale.fit_transform(train)
    test = scale.fit_transform(test)
def OneHotprocessing(data,test,dataInfo):
    for i in dataInfo:
        concatDFtrain = pd.get_dummies(data[i],prefix=i)
        concatDFtest = pd.get_dummies(test[i],prefix=i)
        data = data.drop(i,axis=1)
        test = test.drop(i,axis=1)
        data = pd.concat([data,concatDFtrain],axis=1)
        test = pd.concat([test,concatDFtest],axis=1)
    return data,test
def dataInfo():
    dataInfo = []
    result = {}
    ffearture = ['A1','A2','A3','B1','B2','B3','C1','C2','C3']
    for i in ffearture:  
        #if len(train1[i].unique()) <= 10:
        print("{}  : UniqueNum: {} ".format(i,len(train1[i].unique())))
        dataInfo.append(i)
    for i in dataInfo:
        result[i] = train1[i].value_counts()

def LGBScore(flag=1):
    lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'num_leaves': 50,
    'num_round': 360,
    'max_depth':12,
    'learning_rate': 0.01,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.8,
    'bagging_freq': 12,
    'scale_pos_weight':5
}
    if flag==1:
        lgb_train = lgb.Dataset(x_train, y_train)
    else:
        lgb_train = lgb.Dataset(train1, target['label'])
    model = lgb.train(lgb_params, lgb_train)
    return model

#根据特征组合取值创建新特征
def combined_feature(data):
    result=[]
    for i in range(len(data["E23"])):
        if data["E23"][i]==5 and data["E24"][i]==3 and data["E25"][i]==1 and data["E26"][i]==3 and data["E28"][i]==3 and data["E29"][i]==10:
            if data["E27"][i]==0 or data["E27"][i]==8:
                result.append(1)
            else:
                result.append(0)
        result.append(0)
    return pd.Series(result)
        
def XGBScore(flag=1):
    xgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary:logitraw',
    'subsample': 0.8,
    'num_leaves': 50,
    'num_round': 360,
    'max_depth':12,
    'learning_rate': 0.01,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.8,
    'lambda_l1': 0.6,
    'lambda_l2':  0,
    'bagging_freq': 12,
    'scale_pos_weight':5
    }
    if flag==1:
        xgb_train = xgb.train(xgb_params,xgb.DMatrix(x_train, y_train))
    else:
        xgb_train = xgb.train(xgb_params,xgb.DMatrix(train1, target['label']))
    return xgb_train


# In[ ]:


'''
'n_estimators':[400, 500, 600, 700, 800]
'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]
'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]
'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]
'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]
'''
'''
cv_params = {'n_estimators':[400, 500, 600, 700, 800]}
other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model,param_grid=cv_params,scoring='roc_auc',cv=5,verbose=1, n_jobs=4)
optimized_GBM.fit(x_train,y_train)
evalute_result = optimized_GBM.grid_scores_
print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
'''

#数据导入
path = sys.path[0]
train0 = pd.read_csv(path + r"\train.csv")
target = pd.read_csv(path + r"\train_label.csv")
test0 = pd.read_csv(path + r"\test.csv")

#数据预处理
train0['date_day'] = pd.Series([i.day for i in pd.to_datetime(train0['date'])])
train0['date_hour'] = pd.Series([i.hour for i in pd.to_datetime(train0['date'])])
test0['date_day'] = pd.Series([i.day for i in pd.to_datetime(test0['date'])])
test0['date_hour'] = pd.Series([i.hour for i in pd.to_datetime(test0['date'])])
train = train0.drop('ID',axis=1)
train = train.drop('date',axis=1)
test = test0.drop('ID',axis=1)
test = test.drop('date',axis=1)
target = target.drop('ID',axis=1)
train1 = copy.deepcopy(train)
######################################################################
#train1["E_combined"]=combined_feature(train1)
#test["E_combined"]=combined_feature(test)
#PCA
tem=pca(train1[["A1","A2","A3"]],2)
tem=pd.DataFrame(tem,columns=["A1_independent","A2_independent"])
train1=train1.join(tem)

tem=pca(test[["A1","A2","A3"]],2)
tem=pd.DataFrame(tem,columns=["A1_independent","A2_independent"])
test=test.join(tem)

tem=pca(train1[["B1","B2","B3"]],3)
tem=pd.DataFrame(tem,columns=["B1_independent","B2_independent","B3_independent"])
train1=train1.join(tem)

tem=pca(test[["B1","B2","B3"]],3)
tem=pd.DataFrame(tem,columns=["B1_independent","B2_independent","B3_independent"])
test=test.join(tem)

###################################################################################
train1 = train1.drop(['E26','E13','E18'],axis=1)
test1 = test.drop(['E26','E13','E18'],axis=1)

#提取正负号特征
ffeature = ['A1','A2','A3','B1','B2','B3','C1','C2','C3'，'E2','E5','E7','E9','E10','E16','E17','E19','E22']
for i in ffeature:
    train1[i+"_symbol"] = train1[i] / abs(train1[i])
    test1[i+"_symbol"] = test1[i] / abs(test1[i])

_1list = [12,11,13,14,15,10,6,18,0,23]
_0list = [i for i in range(0,24)]
_01map = {}
for i in _0list:
    if i in _1list:
        _01map[i] = 1
    else:
        _01map[i] = 0 
_02map = {}
for i in range(1,12):
    if i not in [3,9]:
        _02map[i] = 0
    else:
        _02map[i] = 1

train1['date_day_01'] = train1['date_day'].map(_02map)
train1['date_hour_01'] = train1['date_hour'].map(_01map)
test1['date_day_01'] = test1['date_day'].map(_02map)
test1['date_hour_01'] = test1['date_hour'].map(_01map)
train1,test1 = transStrFeature(train1,test1,['A1','B1','D1','D2'])

#随意尝试特征
train1['A4'] = train1['A2']*train1['A3']/(train1['A2']+train1['A3'])
train1['B4'] = train1['B2']*train1['B3']/(train1['B2']+train1['B3'])

test1['A4'] = test1['A2']*test1['A3']/(test1['A2']+test1['A3'])
test1['B4'] = test1['B2']*test1['B3']/(test1['B2']+test1['B3'])

train1['E1_2'] = train1['E1']*train1['E2']
train1['E1_3'] = train1['E1']*train1['E3']
train1['E3_2'] = train1['E3']*train1['E2']

test1['E1_2'] = test1['E1']*test1['E2']
test1['E1_3'] = test1['E1']*test1['E3']
test1['E3_2'] = test1['E3']*test1['E2']


##########################################################################
#将类别取值特征转化为one-hot编码新特征，命名为：D1_0,D1_1等
train1=pd.get_dummies(train1,columns=["D1","D2","E11"],prefix_sep="_")
test1=pd.get_dummies(test1,columns=["D1","D2","E11"],prefix_sep="_")
#############################################################################################

poly = preprocessing.PolynomialFeatures(degree=2,include_bias=False,interaction_only=True)
MINMAX = MinMaxScaler()

X1 = train1[["A1_independent","A2_independent"]]
X1 = MINMAX.fit_transform(X1)
X1 = poly.fit_transform(X1)
Y1 = test1[["A1_independent","A2_independent"]]
Y1 = MINMAX.fit_transform(Y1)
Y1 = poly.fit_transform(Y1)

train1_1 = pd.DataFrame(X1,columns=['A6_1','A6_2','A6_3'])
test1_1 = pd.DataFrame(Y1,columns=['A6_1','A6_2','A6_3'])

train1 = pd.concat([train1,train1_1],axis=1)
test1 = pd.concat([test1,test1_1],axis=1)

train1 = train1.drop(["A1","A2","A3"],axis=1)
test1 = test1.drop(["A1","A2","A3"],axis=1)

print(train1.info())

x_train,x_test,y_train,y_test = train_test_split(train1, target['label'], test_size = 0.3, random_state = 10)


#模型评价
model_xgb = XGBScore()
model_lgb = LGBScore()
y_prediction_xgb = model_xgb.predict(xgb.DMatrix(x_test))
y_prediction_lgb = model_lgb.predict(x_test)
for i in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
    y_prediction = y_prediction_xgb*(1-i)+y_prediction_lgb*i
    score = roc_auc_score(y_test,y_prediction)
    print('Model:lgb_weight: {} AUC : {}'.format(i,score))



y_submit_xgb = model_xgb.predict(xgb.DMatrix(test1))
y_submit_lgb = model_lgb.predict(test1)
y_submit = y_submit_xgb*0.3+y_submit_lgb*0.7


submit = pd.DataFrame()
submit['ID'] = test0['ID']
submit['label'] = y_submit
submit.to_csv(path + '\\tem\\submit.csv',encoding='utf8',index=False)



#gridsearchCV网格调参
def tainanle():
    gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                             objective = 'binary',
                             metric = 'auc',
                             verbose = 0,
                             learning_rate = 0.01,
                             num_leaves = 35,
                             feature_fraction=0.8,
                             bagging_fraction= 0.9,
                             bagging_freq= 8,
                             lambda_l1= 0.6,
                             lambda_l2= 0,scale_pos_weight=5)
    parameters = {
        'num_leaves': [30,40,50],
        'max_depth':[8,10,12],
        'bagging_freq': [12,8,6]}
    gsc = GridSearchCV(gbm,param_grid=parameters,cv=10,scoring='roc_auc')
    gsc.fit(x_train,y_train)
    return gsc
