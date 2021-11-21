import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import pandas_profiling

#Training Dataset
file='Data_Train.xlsx'
x1=pd.ExcelFile(file)
dataset=x1.parse('Sheet1')
dataset=dataset[(dataset.isnull().sum(axis=1)==0)]
#X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y=y.reshape((len(y),1))
dataset1=dataset
dataset1=dataset1.drop("Price",1)
#dataset1=dataset1.drop("Source",1)
#dataset1=dataset1.drop("Destination",1)
#dataset1=dataset1.drop("Route",1)
#dataset1=dataset1.drop("Dep_Time",1)
#dataset1=dataset1.drop("Total_Stops",1)
#dataset1=dataset1.drop("Arrival_Time",1)
#dataset1=dataset1.drop("Duration",1)

#EDA of training data
profile=dataset.profile_report(title='EDA of Training Data')
profile.to_file(output_file="eda_data_train.html")

#Test Dataset
file2='Test_set.xlsx'
x2=pd.ExcelFile(file2)
test=x2.parse('Sheet1')
s=set(dataset1['Route'])
t=set(test['Route'])
z=s-t


X = dataset1.iloc[:, :].values
X=np.delete(X,2878,0)
y=np.delete(y,2878,0)
#test=test.drop("Source",1)
#test=test.drop("Destination",1)
#test=test.drop("Dep_Time",1)
#test=test.drop("Total_Stops",1)
#test=test.drop("Arrival_Time",1)
#test=test.drop("Duration",1)
#test=test.drop("Route",1)
X_test = test.iloc[:, :].values
X_test1=X_test

#Categorical DataColumns
#X=X.astype(None)
X1=pd.concat([dataset1,test], ignore_index=True)

"""result = []

for i in range(13352):
    ind = 0
    result.append(ind)

r = pd.DataFrame(result, columns=['Month'])
X1 = pd.concat([X1, r], axis=1)"""

#X1=X1.drop("Arrival_Time",1)
co=0
X2=X1.iloc[:,:].values
X2[785,9]="Bad Information"
X2[10510,9]="Bad Information"
X2[8122,9]="Bad Information"
X2=np.delete(X2,2878,0)
for q in range(13352):
    #t = '10:15:30'
    """if(X2[q,7].find('h')>0):
        h,m = X2[q,7].split('h')
        if(m.find('m')>0):
            m,s= m.split('m')
            X2[q,7]=int(h) * 60 +int(m)
        else:
            X2[q,7]=int(h) * 60
    else:
        if(X2[q,7].find('m')>0):
            m,s= X2[q,7].split('m')
            X2[q,7]=int(m)
    if(X2[q,8]=='non-stop'):
        X2[q,8]=int(0)
    else:
        w,v=X2[q,8].split(' ')
        X2[q,8]=int(w)"""
    if(X2[q,9]=='No info'):
        X2[q,9]='No Info'
    """d,mo,yr=X2[q,1].split('/')
    X2[q,1]=d
    X2[q,10]=mo"""
    #dd,tm=X2[q,5].split(':')
    #X2[q,5]=dd
    DAT=set(X1['Dep_Time'])
    TIM=set(test['Dep_Time'])
    NU=DAT-TIM
    DURS=set(X1['Duration'])
    DURT=set(test['Duration'])
    NV=DURS-DURT
for E in range(13285): 
    if((X2[E,4] in z)==True):
        co=co+1
        X2=np.delete(X2,E,0)
        if(E<10613):
            y=np.delete(y,E,0)
            E=E-1
    """if((X2[E,5] in NU)==True):
        X2[E,5]="Untest"
    if((X2[E,7] in NV)==True):
        X2[E,7]="Untested Time" """
    #else:
                #X2[q,3]=int(h) * 60
"""X2[:,1]=X2[:,1].astype(int)    
X2[:,5]=X2[:,5].astype(int)
X2[:,7]=X2[:,7].astype(int)
X2[:,8]=X2[:,8].astype(int)
X2[:,10]=X2[:,10].astype(int)"""

X2=np.delete(X2,8,1)
X2=np.delete(X2,6,1)
X2=np.delete(X2,3,1)
X2=np.delete(X2,2,1)
categorical=[0,1,2,3,4,5,]  
#X2=X1.iloc[:-2671,:].values
#X3=X1.iloc[10683:,:].values
#Encoding Categorical Data
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
for i in categorical:
    X2[:,i]=labelencoder_X.fit_transform(X2[:,i])
    #X_test[:,i]=labelencoder_X.fit_transform(X_test[:,i].astype(str))
#X_train=X.astype(float)
#X_test=X_test.astype(float)
#X2[:,1]=X2[:,1].astype(int)    
#2[:,7]=X2[:,7].astype(int)
#X3=X2[:,5]
#X4=X2[:,4]
#X2=np.delete(X2,4,1)
#X2=np.delete(X2,4,1)

onehotencoder=OneHotEncoder(categorical_features="all")
X2=onehotencoder.fit_transform(X2).toarray()

#X2=onehotencoder.fit_transform(X2).toarray()
#X3=X3.astype(int)
#X4=X4.astype(int)
#X2=np.append(X2,X3[:,None],axis=1)
#X2=np.append(X2,X4[:,None],axis=1)
#X2=np.concatenate((X2,X3),axis=1)
#X_test=onehotencoder.fit_transform(X_test).toarray()
#labelencoder_y=LabelEncoder()
#y=labelencoder_y.fit_transform(y)
X_train=X2[:-2671,:]
X_test=X2[10614:,:]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  
X_train = sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
y=sc.fit_transform(y)
y=y.ravel()
#Importing the Keras Libraries and packages
"""from sklearn.svm import SVR
classifier = SVR(kernel = "linear")
classifier.fit(X_train, y)"""

from sklearn.ensemble import RandomForestRegressor
classifier = RandomForestRegressor(n_estimators = 300, random_state = 0)
classifier.fit(X_train, y)

#Predicting the Test Result
y_pred = classifier.predict(X_test)
y_pred=sc.inverse_transform(y_pred)
#y_pred1=abs(y_pred.astype(int))
y_pred1=y_pred
writer6 = pd.ExcelWriter('submission_final(sreesh).xlsx', engine='xlsxwriter')
resu1=pd.DataFrame(y_pred1, columns=['Price'])
resu1.to_excel(writer6, 'Sheet1',index=False)
writer6.save()
