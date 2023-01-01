import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import Logistics_helpler
#%%
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
print('Shape of Training Data:',train.shape,'Shape of Testing Data',test.shape)
#%%
train.head()
#%%
test.head()
#%%
train1=train.copy()
test1=test.copy()

train1.drop(columns=['PassengerId','Ticket','Cabin'],inplace=True)
test1.drop(columns=['PassengerId','Ticket','Cabin'],inplace=True)
#%%
train1.head()
#%%
def gettitle(name):
    str1=name.split(',')[1]
    str2=str1.split('.')[0]
    str3=str2.strip()
    return str3

for data in [train1,test1]:
    for i in range(data.shape[0]):
        data.Name[i]=gettitle(data.Name[i])

train1.head()

#%%
test1.head()
#%%
train1.isna().sum()
#%%
test1.isna().sum()
#%%
train1.Embarked.value_counts(dropna=False)
#%%
import seaborn as sns
import matplotlib.pyplot as plt

sns.distplot(train1.Age)
plt.show()
#%%
sns.distplot(train1.Fare)
plt.show()
#%%
train2=train1.copy()
test2=test1.copy()

train2.Embarked.fillna('S',inplace=True)
train2.Age.fillna(train2.Age.median(),inplace=True)

test2.Age.fillna(train2.Age.median(),inplace=True)
test2.Fare.fillna(train2.Fare.mean(),inplace=True)


#%%
train2.duplicated().sum()
#%%
train2.drop_duplicates(inplace=True)
train2.duplicated().sum()
#%%
train2.head()
#%%
plt.figure(figsize=(10,5))
n=0
for col in ["Survived",'Pclass','Sex','SibSp','Parch','Embarked']:
    n+=1
    plt.subplot(3,2,n)
    sns.countplot(x = train2[col])
    plt.title(f'Distribution of {col}')
plt.tight_layout()
#%%
plt.figure(figsize=(10,5))
sns.countplot(y=train2.Name)
plt.title('Distribution of Name')
plt.tight_layout()
#%%
plt.figure(figsize=(10,5))
n=0
for col in ['Age','Fare']:
    n+=1
    plt.subplot(2,2,n)
    sns.distplot(x = train2[col])
    plt.title(f'Distribution of {col}')
    n+=1
    plt.subplot(2,2,n)
    sns.boxplot(x = train2[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
#%%
plt.figure(figsize=(10,5))
cor=train2.corr()
sns.heatmap(cor,annot=True,cmap='Blues')
plt.title('Correlation Plot')
plt.tight_layout()
#%%
data=pd.concat([train2,test2],axis=0,ignore_index=True)
data[['Pclass','SibSp','Parch']]=data[['Pclass','SibSp','Parch']].astype(object)
data=pd.get_dummies(data)
data.head()
#%%
train=data.iloc[:train2.shape[0],]
test=data.iloc[train2.shape[0]:,1:]

train.isna().sum()
train.shape, test.shape
#%%
train.Age=(train.Age-train.Age.mean())/train.Age.std()
train.Fare=(train.Fare-train.Fare.mean())/train.Fare.std()

test.Age=(test.Age-train.Age.mean())/train.Age.std()
test.Fare=(test.Fare-train.Fare.mean())/train.Fare.std()

round(train.Age.var()),round(train.Fare.var())
#%%
from sklearn.model_selection import train_test_split

x=train.drop(columns='Survived')
y=train.Survived

xtrain,xval,ytrain,yval=train_test_split(x,y,test_size=0.2,random_state=50,shuffle=True)

xtrain.shape,ytrain.shape,xval.shape,yval.shape

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
lr=model.fit(xtrain,ytrain)
lr.coef_=np.ones((1,43))
from sklearn.model_selection import cross_val_score
train_score=cross_val_score(lr,xtrain,ytrain,scoring='accuracy',cv=10).mean()
val_score=cross_val_score(lr,xval,yval,scoring='accuracy',cv=10).mean()

print('train score:',train_score,'validation score:',val_score)

# train_numpy=np.array(train)
groundTruth=np.array(ytrain)
modelInput=np.array(xtrain)
ltestModel=Logistics_helpler.LogisticsModel(groundTruth,modelInput)
ltestModel.Train()
modelInput_test=np.array(test)
modelOutput_test=ltestModel.RunModel(modelInput_test)
modelOutput_test=modelOutput_test.reshape(modelOutput_test.shape[0])

for i in range(modelOutput_test.shape[0]):
    if modelOutput_test[i]>=0.5:
        modelOutput_test[i]=1
    else:
        modelOutput_test[i]=0

idList=[]
for i in range(modelOutput_test.shape[0]):
    idList.append(892+i)

submission=pd.DataFrame({"PassengerId":idList,"Survived":modelOutput_test})
submission.to_csv("test_submit.csv",index=False)
