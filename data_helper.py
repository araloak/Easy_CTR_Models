import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.neural_network import BernoulliRBM 
from collections import Counter
from sklearn.preprocessing import*
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

random.seed(10)

def takeFirst(elem): 
    return elem[1]
#返回一个特征的统计信息
def get_info(feature):
    d=Counter(feature)
    info=d.most_common()
    
    for i in info[0:20]:
        print(i)
    
    info=sorted(info,key=takeFirst,reverse=True)
    for each in info:
        #print(each)
        pass
    return info
def read_from(train_file,label_file):
    data=np.array(pd.read_csv(train_file,'r',encoding='utf-8'))
    train_labels=np.array(pd.read_csv(label_file,'r',encoding='utf-8'))
    result=[]
    labels=[]
    for i,each in enumerate(data):
        single_info=each[0].split(",")
        tem=[]
        for j in single_info[2:]:
            tem.append(float(j))
        result.append(tem)
        
        labels.append(int(train_labels[i][0].split(",")[1]))

    return result,labels

#dim代表降维到的维数
def pca(data,dim,list_of_features):
    train_data=data[list_of_features]
    model_pca = PCA(n_components=dim)
    tem = model_pca.fit(train_data).transform(train_data)
    tem=pd.DataFrame(tem,columns=[list_of_features[0][0]+"pca"+str(i) for i in range(dim)])
    data=data.join(tem)
    return data
    
#Degree是data当前的维度
def add_degree(data,list_of_features):
    MINMAX = MinMaxScaler()
    train_data=MINMAX.fit_transform(data[list_of_features])
    poly = PolynomialFeatures(degree=2,include_bias = False,interaction_only=True)
    X_poly=poly.fit_transform(train_data)
    num_of_new_names=X_poly.shape[1]
    new_data=pd.DataFrame(X_poly,columns=[list_of_features[0][0]+'poly'+str(i) for i in range(num_of_new_names)])
    data=data.join(new_data)
    return data
    
#使用lda经过对训练集在指定属性集合的学习后同时对训练集和测试集进行降维处理
#train，test为Dataframe对象，labels为Series对象；
#list_of_features列表中保存需要进行降维处理的列名，例：list_of_features=["name","age"]
#new_feature是降维后产生的新特征的跟名字，会基于此对产生的新特征进行编号命名，例："mix1"，"mix2"
def lda(train,labels,test,list_of_features,new_feature):
    train_data=train[list_of_features]
    y=labels
    num_of_types=len(labels.unique())
    test_data=test[list_of_features]
    #n_components代表进行LDA降维时降到的维数，取值范围为[1，分类类别数-1]
    model_lda = LinearDiscriminantAnalysis(n_components=num_of_types)
    model_lda.fit(np.array(train_data), y)
    
    tem_train = model_lda.transform(np.array(train_data))
    tem_test=model_lda.transform(np.array(test_data))
    
    tem_train=pd.DataFrame(tem_train,columns=[new_feature+str(i) for i in range(num_of_types)])
    train=train.join(tem_train)
    tem_test=pd.DataFrame(tem_test,columns=[new_feature+str(i) for i in range(num_of_types)])
    test=test.join(tem_test)
    return train, test
    
#kernel表示是用什么核函数
def Svm(data,labels,test_data,kernel):
    result=[]
    labels=np.array(labels)
    data=np.array(data)
    clf = svm.SVC(kernel=kernel)#调参
    clf.fit(data, labels)    
    for i in range(len(test_data)):#循环检测测试数据分类成功的个数
        prediction=clf.predict(np.array([test_data[i]]))
        result.append(prediction[0])
    return result
    
#z-score归一化（针对列数据，一般计算样本之间距离时使用其做归一化处理，比如聚类）  
def standard_scale(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)
    
#正则化（针对行数据）
def normalization(data):
    data_normalized =normalize(data, norm='l2')
    return data_normalized
    
#二值化
def binarization(data,Threshold):
    binarizer = Binarizer().fit(data) 
    Binarizer(copy=True,threshold=Threshold)
    return binarizer.transform(data)
    
def main():
    labels,As,Bs,Cs,Ds,E_ints,E_floats=get_data()

#用于调试data_helper文件
'''
if __name__ =="__main__":
    main()
'''
