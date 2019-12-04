import pandas as pd
import copy
import sys
from data_helper import *
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.models import DeepFM
from sklearn import preprocessing
from deepctr.inputs import  SparseFeat, DenseFeat, get_feature_names
def labelencode(train,list_of_features):
    le = preprocessing.LabelEncoder()
    for each in list_of_features:
        le.fit(train[each].values.tolist()) 
        # transform 以后，这一列数就变成了 [0,  n-1] 这个区间的数，即是  le.classes_ 中的索引
        train[each]=le.transform(train[each].values.tolist())
    return train

if __name__ == "__main__":
    #加载原始数据
    train = pd.read_csv(sys.path[0]+'\\data\\train.csv')
    test = pd.read_csv(sys.path[0]+'\\data\\test.csv')
    labels = pd.read_csv(sys.path[0]+'\\data\\train_label.csv')['label']
    train['label']=labels
    #处理时间信息
    train['date_day'] = pd.Series([i.day for i in pd.to_datetime(train['date'])])
    train['date_hour'] = pd.Series([i.hour for i in pd.to_datetime(train['date'])])
    test['date_day'] = pd.Series([i.day for i in pd.to_datetime(test['date'])])
    test['date_hour'] = pd.Series([i.hour for i in pd.to_datetime(test['date'])])

    #以下将特征分类送入不同的模型
    A_features = ['A1','A2','A3'] 
    B_features = ['B1','B2','B3']
    C_features = ['C' + str(i) for i in range(1, 4)]
    #一定要经过one_hot编码
    D_features = ['D1','D2']
    E_features = ['E' + str(i) for i in range(1, 30)]

    dense_features=C_features
    sparse_features =A_features+B_features+D_features+E_features+['date_day']+['date_hour']
    
    target = ['label']
##################################################################################################
    # 1.Label Encoding for sparse features,and do simple Transformation for dense (对类别特征进行编码)
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
        test[feat] = lbe.fit_transform(test[feat])
    mms = MinMaxScaler(feature_range=(0, 1))#归一化
    data[dense_features] = mms.fit_transform(data[dense_features])
    test[dense_features] = mms.fit_transform(test[dense_features])
    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                           for feat in sparse_features] + [DenseFeat(feat, 1,)
                          for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, valid = train_test_split(data, test_size=0.2,random_state=10)

    train_model_input = {name:train[name] for name in feature_names}
    valid_model_input = {name:valid[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}
    
    # 4.Define Model,train,predict and evaluate
    #dnn_hidden_units用来定义隐藏层数量以及每层神经元个数
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary',dnn_hidden_units=[100,100],dnn_dropout=0.2)
    model.compile("adam", "binary_crossentropy",
                  metrics=['binary_crossentropy'], )

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=2, verbose=2, validation_data=(valid_model_input,valid['target']) )
    pred_ans = model.predict(valid_model_input, batch_size=256)
    print("valid LogLoss", round(log_loss(valid[target].values, pred_ans), 4))
    print("valid AUC", round(roc_auc_score(valid[target].values, pred_ans), 4))
    #进行预测，并写入csv文件
    result=model.predict(test_model_input, batch_size=256)
    
    result=pd.DataFrame(result,columns=['label'])
    submit=pd.DataFrame(test['ID'],columns=['ID'])
    submit=submit.join(result)
    submit.to_csv(sys.path[0]+"\\tem\\"+"result"+'.csv',index=False)
