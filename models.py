import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier


df = pd.read_csv("E:\\MACHINE LEARNING\\datasets\\Titanic-Dataset.csv")
df.head()



df.isnull().sum()

# KNNImputer apply for age columns 
impute = KNNImputer()
imputed = impute.fit_transform(df[['Age']])
age_df = pd.DataFrame(imputed,columns=['age'])
age_df.head()


 # SimpleImputer apply for embark columns 
sim_impute = SimpleImputer(strategy='most_frequent')
sim_imputed = sim_impute.fit_transform(df[['Embarked']])
embark_df = pd.DataFrame(sim_imputed,columns=['embarked'])
embark_df.head()

df2 = df[['Sex','Fare','Pclass','Survived']]

x = pd.concat([embark_df,age_df,df2],ignore_index=True,axis=1)
x.columns=['embark','age','sex','fare','class','survived']

x.isnull().sum()

plt.scatter(x['age'],x['survived'],s=4)
plt.show()

x = x[x['age'] < 72] # removing the outliers rows 

plt.scatter(x['fare'],x['survived'],s=4)
plt.show()

x  = x[x['fare']<300] # remove the outliers 

x['sex'] = x['sex'].map({
    'male' : 1,
    'female' : 0
})
x.head()

y = x[['survived']]

x.drop('survived',axis=1,inplace=True)

# hot_encoding the class columns 
encode = OneHotEncoder(drop='first',sparse_output=False)
encoded = encode.fit_transform(x[['class','embark']])
class_df = pd.DataFrame(encoded,columns=encode.get_feature_names_out(['class','embark']))
class_df.reset_index(drop=True, inplace=True)
x = x.reset_index(drop=True)
features = pd.concat([x,class_df],axis= 1 )
features.drop('embark',axis=1,inplace=True) # after encding delteing the categorical columns 
features.head(10)

# sgd classifier 
apply_1 = SGDClassifier()
model_1 = apply_1.fit(x_train,y_train)
y_pred_1 = model_1.predict(x_test)
print(accuracy_score(y_test,y_pred_1)*100)

 
# logistic regression without regularization 
apply_2 = LogisticRegression()
model_2 = apply_2.fit(x_train,y_train)
y_pred_2 = model_2.predict(x_test)
accuracy_score(y_pred_2,y_test)*100




# logistic regression without regularization 
param_grid = {
    'penalty' : ['l1','l2','elasticnet',None],
    'C': [0.01,0.1,1,10],
    'solver': ['liblinear','saga'],
    'max_iter' : [100,200,300],
    'l1_ratio': [0.5,0.7]
}

model_3 =GridSearchCV(LogisticRegression(),param_grid,cv=5)
model_3.fit(x_train,y_train)

y_pred_3 =  model_3.predict(x_test)
print(accuracy_score(y_test,y_pred_3)*100)




# random forest classifier
from sklearn.ensemble import RandomForestClassifier
model_4 = RandomForestClassifier(n_estimators=100,
                                  bootstrap=True,
                                  max_depth=10,
                                  min_samples_leaf=5,
                                  max_features='sqrt')
model_4.fit(x_train, y_train)
y_pred = model_4.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
