#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('C:\\Users\\v.omsai\\Downloads\\train.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df.drop(columns=['PassengerId','Ticket','Name','Cabin'],inplace=True)


# In[5]:


df['Survived'].value_counts()


# In[6]:


df['Survived'].unique()


# In[7]:


df['Pclass'].value_counts()


# In[8]:


df['Pclass'].unique()


# In[9]:


df['Sex'].value_counts()


# In[10]:


df['Sex'].unique()


# In[11]:


df['SibSp'].value_counts()


# In[12]:


df['SibSp'].unique()


# In[13]:


df['Parch'].value_counts()


# In[14]:


df['Parch'].unique()


# In[15]:


df['Embarked'].value_counts()


# In[16]:


df.isnull().sum()


# In[17]:


df.dropna(inplace=True)


# In[18]:


continueos = ['Age','Fare']


# In[19]:


df[continueos].describe()


# In[20]:


df['Age'] = df['Age'].astype('int')


# In[21]:


sns.pairplot(df[continueos])


# In[22]:


plt.rcParams['figure.figsize'] = (18,8)

plt.subplot(2,3,1)
sns.histplot(df['Age'],kde=True)

plt.subplot(2,3,2)
sns.histplot(df['Fare'],kde=True)


# In[23]:


sns.heatmap(df[continueos].corr(),annot=True)


# In[24]:


plt.rcParams['figure.figsize'] = (18,8)

plt.subplot(2,4,1)
sns.countplot(x = df['Sex'])

plt.subplot(2,4,2)
sns.countplot(x = df['Embarked'])

plt.subplot(2,4,3)
sns.countplot(x = df['Survived'])

plt.subplot(2,4,4)
sns.countplot(x = df['Pclass'])

plt.subplot(2,4,5)
sns.countplot(x = df['SibSp'])

plt.subplot(2,4,6)
sns.countplot(x = df['Parch'])


# In[25]:


df.isnull().sum()


# In[26]:


df[continueos].skew()


# In[27]:


from scipy.stats import boxcox
df['Fare'],a =boxcox(abs(df['Fare']+0.00001)) 


# In[28]:


df['Sex'] = df['Sex'].map({'male':1,'female':0}).astype('int')


# In[29]:


df.drop(columns=['Embarked'],inplace=True)


# In[30]:


df


# In[31]:


X = df.drop('Survived',axis=1)
y = df['Survived']


# In[32]:


Train=[]
Test=[]
cv=[]
for i in range(1,101):
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=i)
    
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train,y_train)
    
    ypred_train = model.predict(X_train)
    ypred_test = model.predict(X_test)
    
    from sklearn.metrics import accuracy_score
    Train.append(accuracy_score(ypred_train,y_train))
    Test.append(accuracy_score(ypred_test,y_test))
    
    from sklearn.model_selection import cross_val_score
    cv.append(cross_val_score(model,X,y,cv=5,scoring='accuracy').mean())
    
em = pd.DataFrame({'Train':Train,'Test':Test,'cv':cv})
gm = em[(abs(em['Train']-em['Test'])<=0.05) & (abs(em['Test']-em['cv'])<=0.05)]
rs = gm[gm['Test']==gm['Test'].max()].index.to_list()
print('Random_state',rs)


# In[33]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)


# In[34]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[35]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)

ypred_train = model.predict(X_train)
ypred_test = model.predict(X_test)

print(accuracy_score(ypred_train,y_train))
print(accuracy_score(ypred_test,y_test))
print(cross_val_score(model,X,y,cv=5,scoring='accuracy').mean())


# In[36]:


from sklearn.neighbors import KNeighborsClassifier
estimator = KNeighborsClassifier()
param_grid = {'n_neighbors':list(range(1,10))}

from sklearn.model_selection import GridSearchCV
knn_grid = GridSearchCV(estimator,param_grid,cv=5,scoring='accuracy')
knn_grid.fit(X_train,y_train)
knn_grid.best_params_


# In[45]:


from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train,y_train)

ypred_train = knn_model.predict(X_train)
ypred_test = knn_model.predict(X_test)

print(accuracy_score(ypred_train,y_train))
print(accuracy_score(ypred_test,y_test))
print(cross_val_score(knn_model,X,y,cv=5,scoring='accuracy').mean())


# In[38]:


from sklearn.svm import SVC
estimator = SVC()
param_grid = {'C':[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
              'kernel':['linear','rbf','sigmoid','poly']}
svm_grid = GridSearchCV(estimator,param_grid,cv=5,scoring='accuracy')
svm_grid.fit(X_train,y_train)
svm_grid.best_params_


# In[46]:


from sklearn.svm import SVC
svm_model = SVC(C=0.1,kernel='linear')
svm_model.fit(X_train,y_train)

ypred_train = svm_model.predict(X_train)
ypred_test = svm_model.predict(X_test)

print(accuracy_score(ypred_train,y_train))
print(accuracy_score(ypred_test,y_test))
print(cross_val_score(svm_model,X,y,cv=5,scoring='accuracy').mean())


# In[40]:


from sklearn.tree import DecisionTreeClassifier
estimator = DecisionTreeClassifier()
param_grid = {'criterion':['gini','entropy'],
              'max_depth':list(range(1,10))}
tree_grid = GridSearchCV(estimator,param_grid,cv=5,scoring='accuracy')

tree_grid.fit(X_train,y_train)

model_tree = tree_grid.best_estimator_

tree_feat = model_tree.feature_importances_

index = [i for i,x in enumerate(tree_feat) if x>0 ]

X_train_dt = X_train.iloc[:,index]
X_test_dt = X_test.iloc[:,index]

model_tree.fit(X_train_dt,y_train)

ypred_train = model_tree.predict(X_train_dt)
ypred_test = model_tree.predict(X_test_dt)

print(accuracy_score(ypred_train,y_train))
print(accuracy_score(ypred_test,y_test))
print(cross_val_score(model_tree,X_train_dt,y_train,cv=5,scoring='accuracy').mean())


# In[41]:


from sklearn.ensemble import RandomForestClassifier
estimator = RandomForestClassifier()
param_grid = {'n_estimators':list(range(1,10))}
rf_grid = GridSearchCV(estimator,param_grid,cv=5,scoring='accuracy')

rf_grid.fit(X_train,y_train)

model_rf = rf_grid.best_estimator_

rf_feat = model_rf.feature_importances_

index = [i for i,x in enumerate(rf_feat) if x>0]

X_train_rf = X_train.iloc[:,index]
X_test_rf = X_test.iloc[:,index]

model_rf.fit(X_train_rf,y_train)

ypred_train = model_rf.predict(X_train_rf)
ypred_test = model_rf.predict(X_test_rf)

print(accuracy_score(ypred_train,y_train))
print(accuracy_score(ypred_test,y_test))
print(cross_val_score(model_rf,X_train_rf,y_train,cv=5,scoring='accuracy').mean())


# In[42]:


from sklearn.ensemble import AdaBoostClassifier
estimator = AdaBoostClassifier()
param_grid = {'n_estimators':list(range(1,10))}
abc_grid = GridSearchCV(estimator,param_grid,cv=5,scoring='accuracy')
abc_grid.fit(X_train,y_train)
model_abc = abc_grid.best_estimator_
abc_feat = model_abc.feature_importances_

index = [i for i,x in enumerate(abc_feat) if x>0]

X_train_abc = X_train.iloc[:,index]
X_test_abc = X_test.iloc[:,index]

model_abc.fit(X_train_abc,y_train)

ypred_train = model_abc.predict(X_train_abc)
ypred_test = model_abc.predict(X_test_abc)

print(accuracy_score(ypred_train,y_train))
print(accuracy_score(ypred_test,y_test))
print(cross_val_score(model_abc,X_train_abc,y_train,cv=5,scoring='accuracy').mean())


# In[43]:


from sklearn.ensemble import GradientBoostingClassifier
estimator = GradientBoostingClassifier()
param_grid = {'n_estimators':list(range(1,10)),
              'learning_rate':[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}
gbc_grid = GridSearchCV(estimator,param_grid,cv=5,scoring='accuracy')
gbc_grid.fit(X_train,y_train)
model_gbc = gbc_grid.best_estimator_
gbc_feat = model_gbc.feature_importances_

index = [i for i,x in enumerate(gbc_feat) if x>0]

X_train_gbc = X_train.iloc[:,index]
X_test_gbc = X_test.iloc[:,index]

model_gbc.fit(X_train_gbc,y_train)

ypred_train = model_gbc.predict(X_train_gbc)
ypred_test = model_gbc.predict(X_test_gbc)

print(accuracy_score(ypred_train,y_train))
print(accuracy_score(ypred_test,y_test))
print(cross_val_score(model_gbc,X_train_gbc,y_train,cv=5,scoring='accuracy').mean())


# In[44]:


from xgboost import XGBClassifier
estimator = XGBClassifier()
param_grid = {'n_estimators':[10,20,40,100],
              'max_deapth':[3,4,5],
              'gamma':[0,0.15,0.35,0.5,1]}
xgb_grid = GridSearchCV(estimator,param_grid,cv=5,scoring='accuracy')
xgb_grid.fit(X_train,y_train)
model_xgb = xgb_grid.best_estimator_
xgb_feat = model_xgb.feature_importances_

index = [i for i,x in enumerate(xgb_feat) if x>0]

X_train_xgb = X_train.iloc[:,index]
X_test_xgb = X_test.iloc[:,index]

model_xgb.fit(X_train_xgb,y_train)

ypred_train = model_xgb.predict(X_train_xgb)
ypred_test = model_xgb.predict(X_test_xgb)

print(accuracy_score(ypred_train,y_train))
print(accuracy_score(ypred_test,y_test))
print(cross_val_score(model_xgb,X_train_xgb,y_train,cv=5,scoring='accuracy').mean())


# In[49]:


from keras.models import Sequential
ann = Sequential()


# In[167]:


from keras.layers import Dense,Dropout
ann.add(Dense(input_dim =6, units=8, kernel_initializer = 'uniform', activation = 'relu'))
ann.add(Dense(units=8, kernel_initializer = 'uniform', activation = 'relu'))


# In[168]:


ann.add(Dense(units=1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# In[169]:


ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])


# In[170]:


ann.fit(X_train,y_train, epochs=50, batch_size=32)


# In[145]:


ypred = ann.predict(X_test)
ypred = (ypred >0.5)


# In[146]:


from sklearn.metrics import confusion_matrix, accuracy_score
print('Test_Accuracy:',accuracy_score(y_test,ypred))
confusion_matrix(y_test,ypred)


# In[128]:


def Classifier():
    model = Sequential()
    model.add(Dense(input_dim =6,units=6,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model


# In[154]:


from scikeras.wrappers import KerasClassifier
classifiers = KerasClassifier(Classifier,batch_size=32,epochs=100)


# In[156]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(classifiers,X,y,cv=5)


# In[158]:


accuracies.mean()


# In[132]:


estimator = KerasClassifier(Classifier())
param_grid = {'batch_size':[10,32],'epochs':[50,100],'optimizer':['adam','rmsprop']}


# In[133]:


grid = GridSearchCV(estimator,param_grid,cv=5,scoring='accuracy')
grid_results = grid.fit(X_train,y_train)
grid_results.best_params_


# In[134]:


grid_results.best_score_


# In[ ]:




