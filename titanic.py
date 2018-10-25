
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[6]:


titanic = pd.read_csv("train.csv")


# In[7]:


titanic.head()


# In[8]:


titanic_test = pd.read_csv("test.csv")


# In[9]:


titanic_test.head()
#there is no survived column which is our target variable


# In[10]:


titanic.shape


# In[11]:


titanic.describe()
#age has got missing values 


# In[12]:


titanic.info()


# In[13]:


null_columns=titanic.columns[titanic.isnull().any()]
titanic.isnull().sum()
#embarked and cabin also has missing values 


# In[14]:


titanic_test.isnull().sum()


# In[18]:


#visualizations 
titanic.hist(bins=10,figsize=(10,8),grid=True);


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1)


# In[20]:


a = sns.FacetGrid(titanic, col="Sex", row="Survived", margin_titles=True)
a.map(plt.hist, "Age",color="red");


# In[22]:


a = sns.FacetGrid(titanic, hue="Survived", col="Pclass", margin_titles=True,)
a=a.map(plt.scatter, "Fare", "Age")


# In[27]:


corr=titanic.corr()#["Survived"]
plt.figure(figsize=(10, 10))

sns.heatmap(corr,annot=True)
plt.title('Correlation between features');


# In[28]:


#correlation of features with target variable
titanic.corr()["Survived"]
#Pclass Parch age and Fare has highest correlation with survived 


# In[29]:


#filling the missing values 


# In[30]:


# rows which have null Embarked column
titanic[titanic['Embarked'].isnull()]


# In[31]:


sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=titanic);


# In[32]:


titanic["Embarked"] = titanic["Embarked"].fillna('C')


# In[33]:


#for test data
titanic_test.describe()
#age column has a missing values 


# In[34]:


titanic_test[titanic_test['Fare'].isnull()]


# In[35]:


def fill_missing_fare(df):
    median_fare=df[(df['Pclass'] == 3) & (df['Embarked'] == 'S')]['Fare'].median()

       
    df["Fare"] = df["Fare"].fillna(median_fare)
    return df

titanic_test=fill_missing_fare(titanic_test)


# In[36]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelEnc=LabelEncoder()

cat_vars=['Embarked','Sex']
for col in cat_vars:
    titanic[col]=labelEnc.fit_transform(titanic[col])
    titanic_test[col]=labelEnc.fit_transform(titanic_test[col])

titanic.head()


# In[41]:


titanic["TicketNumber"] = titanic["Ticket"].str.extract('(\d{2,})', expand=True)
titanic["TicketNumber"] = titanic["TicketNumber"].apply(pd.to_numeric)


titanic_test["TicketNumber"] = titanic_test["Ticket"].str.extract('(\d{2,})', expand=True)
titanic_test["TicketNumber"] = titanic_test["TicketNumber"].apply(pd.to_numeric)


# In[42]:


titanic[titanic["TicketNumber"].isnull()]


# In[46]:


titanic.TicketNumber.fillna(titanic["TicketNumber"].median(), inplace=True)
titanic_test.TicketNumber.fillna(titanic_test["TicketNumber"].median(), inplace=True)


# In[50]:


from sklearn.ensemble import RandomForestRegressor
#predicting missing values in age using Random Forest
def fill_missing_age(df):
    
    #Feature set
    age_df = df[['Age','Embarked','Fare', 'Parch', 'SibSp',
                 'TicketNumber', 'Pclass']]
    # Split sets into train and test
    train  = age_df.loc[ (df.Age.notnull()) ]# known Age values
    test = age_df.loc[ (df.Age.isnull()) ]# null Ages
    
    # All age values are stored in a target array
    y = train.values[:, 0]
    
    # All the other values are stored in the feature array
    X = train.values[:, 1::]
    
    # Create and fit a model
    regressor = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    regressor.fit(X, y)
    
    # Use the fitted model to predict the missing values
    predictedAges = regressor.predict(test.values[:, 1::])
    
    # Assign those predictions to the full data set
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    
    return df


# In[56]:



titanic_test=fill_missing_age(titanic_test)


# In[58]:


titanic.head()


# In[61]:


titanic.describe()


# In[60]:


titanic_test.describe()


# In[62]:


#feature Scalling
from sklearn import preprocessing

Scaler = preprocessing.StandardScaler().fit(titanic[['Age', 'Fare']])
titanic[['Age', 'Fare']] = Scaler.transform(titanic[['Age', 'Fare']])


Scaler = preprocessing.StandardScaler().fit(titanic_test[['Age', 'Fare']])
titanic_test[['Age', 'Fare']] = Scaler.transform(titanic_test[['Age', 'Fare']])


# In[63]:


# Import the linear regression class
from sklearn.linear_model import LinearRegression

from sklearn.cross_validation import KFold

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age","SibSp", "Parch", "Fare",
              "Embarked"]
target="Survived"
# Initialize our algorithm class
alg = LinearRegression()

# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []


# In[64]:


predictions


# In[65]:


for train, test in kf:

    train_predictors = (titanic[predictors].iloc[train,:])
    
    train_target = titanic[target].iloc[train]
    
    alg.fit(train_predictors, train_target)
    
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)


# In[66]:


predictions


# In[67]:


predictions = np.concatenate(predictions, axis=0)
# Map predictions to outcomes (only possible outcomes are 1 and 0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0


accuracy=sum(titanic["Survived"]==predictions)/len(titanic["Survived"])
accuracy


# In[68]:


from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

predictors = ["Pclass", "Sex", "Fare", "Embarked","Age","Parch"]

# Initialize our algorithm
lr = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)

scores = cross_val_score(lr, titanic[predictors], 
                                          titanic["Survived"],scoring='f1', cv=cv)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())


# In[69]:


from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.model_selection import cross_val_predict

import numpy as np
predictors = ["Pclass", "Sex", "Age","Fare","Embarked","Parch"]

# Initialize our algorithm with the default paramters
# n_estimators is the number of trees we want to make
# min_samples_split is the minimum number of rows we need to make a split
# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)
rf = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, 
                            min_samples_leaf=1)
kf = KFold(titanic.shape[0], n_folds=5, random_state=1)
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)

predictions = cross_validation.cross_val_predict(rf, titanic[predictors],titanic["Survived"],cv=kf)
predictions = pd.Series(predictions)
scores = cross_val_score(rf, titanic[predictors], titanic["Survived"],
                                          scoring='f1', cv=kf)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())


# In[73]:


from sklearn.ensemble import AdaBoostClassifier
predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked","TicketNumber"]
adb=AdaBoostClassifier()
adb.fit(titanic[predictors],titanic["Survived"])
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)
scores = cross_val_score(adb, titanic[predictors], titanic["Survived"], scoring='f1',cv=cv)
print(scores.mean())


# In[71]:


predictors = ["Pclass", "Sex", "Age",
              "Fare","TicketNumber"]
rf = RandomForestClassifier(random_state=1, n_estimators=50, max_depth=9,min_samples_split=6, min_samples_leaf=4)
rf.fit(titanic[predictors],titanic["Survived"])
kf = KFold(titanic.shape[0], n_folds=5, random_state=1)
predictions = cross_validation.cross_val_predict(rf, titanic[predictors],titanic["Survived"],cv=kf)
predictions = pd.Series(predictions)
scores = cross_val_score(rf, titanic[predictors], titanic["Survived"],scoring='f1', cv=kf)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())


# In[74]:


predictions=["Pclass", "Sex", "Age", "Fare", "Embarked","TicketNumber"]
from sklearn.ensemble import VotingClassifier
eclf1 = VotingClassifier(estimators=[
        ('lr', lr), ('rf', rf), ('adb', adb)], voting='soft')
eclf1 = eclf1.fit(titanic[predictors], titanic["Survived"])
predictions=eclf1.predict(titanic[predictors])
predictions

test_predictions=eclf1.predict(titanic_test[predictors])

test_predictions=test_predictions.astype(int)
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": test_predictions
    })

submission.to_csv("titanic_submission.csv", index=False)

