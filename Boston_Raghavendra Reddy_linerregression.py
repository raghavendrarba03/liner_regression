
# coding: utf-8

# In[1]:


#Import packages
import pandas as pd
import numpy as np
import sklearn


# In[2]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity='all' # this is magic command which will execute all the lines in code instead of last one.


# In[3]:


#Import dataset and create a dataframe
from sklearn import datasets
boston=datasets.load_boston()


# In[4]:


#Find out more about this dataset
print(boston.DESCR)


# In[5]:


#create dataframe
df=pd.DataFrame(boston.data,columns=boston.feature_names)
df['House_Price']=boston.target


# In[6]:


df.head()


# In[7]:


df.describe()


# In[8]:


#scale all values between 0 & 1
from sklearn.preprocessing import MinMaxScaler
scld=MinMaxScaler(feature_range=(0,1))
arr_scld=scld.fit_transform(df)
df_scld=pd.DataFrame(arr_scld,columns=df.columns)
df.head()
df.describe()
df_scld.head()
df_scld.describe()


# In[9]:


#Inverse scaling to revert back to same scale
df_scld.head()
df1=pd.DataFrame(scld.inverse_transform(df_scld),columns=df.columns)
df1.head()


# In[10]:


df.count()


# In[11]:


#Add dependent variabels
df['House_Price']=boston.target
df.head()
df.describe()


# In[12]:


#correlation matrix
x=df.corr()
x


# In[15]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().magic('matplotlib inline')
import seaborn as sns
plt.subplots(figsize=(20,20))
sns.heatmap(x,cmap='RdYlGn',annot=True)
plt.show()


# In[16]:


#Create features and labels on the data
x=df.drop('House_Price', axis=1)
y=df['House_Price']
x.head()
y.head()


# In[17]:


import sklearn
from sklearn.cross_validation import train_test_split


# In[18]:


#Create train and test data with 75% and 25% split
train_x, test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=1)
train_x.shape
test_x.shape
train_y.shape
test_y.shape


# In[19]:


#lets import the regression object and define model
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm


# In[20]:


#Fit a model on the train data
lm.fit(train_x,train_y)


# In[21]:


#Evaluate the model
predict_test=lm.predict(test_x)


# In[22]:


#R2 Value
print("Rsquare value for TEST data is-")
np.round(lm.score(test_x, test_y)*100,0)
print("Rsquare value for TRAIN data is-")
np.round(lm.score(train_x,train_y)*100,0)


# In[23]:


#Predict on test and training data
predict_test=lm.predict(test_x)


# In[24]:


#Print the Loss Function - MSE
import numpy as np
from sklearn import metrics
print ("Mean Square Error (MSE) for TEST data is-")
np.round(metrics.mean_squared_error(test_y,predict_test),0)


# In[25]:


from sklearn.metrics import mean_absolute_error
print ("Mean Absolute Error (MAE) for TEST data is-")
np.round(mean_absolute_error(test_y,predict_test),0)


# In[26]:


#Liner regression model fitting and model Evaluation
#Append data
fdf=pd.concat([test_x,test_y],1)
fdf['Predicted']=np.round(predict_test,1)
fdf['Prediction_error']=fdf['House_Price']-fdf['Predicted']
fdf


# In[27]:


plt.subplots(figsize=(20,20))
plt.scatter(fdf.House_Price,fdf.Prediction_error,color='red')
plt.xlabel ('House Price')
plt.ylabel ('Error')
plt.show();


# In[28]:


import seaborn as sns


# In[29]:


#magic command which will show graphs in the same notebook rather than saving some where.
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.show()


# In[30]:


iris=sns.load_dataset('iris')


# In[31]:


iris.head()


# In[32]:


np.round(iris.mean(),2)


# In[33]:


np.round(iris.median(),2)


# In[34]:


iris.count()


# In[35]:


#Counting frequency for categorical values
pd.crosstab(index=iris["species"],columns="Frequency")


# In[36]:


correlation=iris.corr()
correlation


# In[37]:


sns.heatmap(correlation,cmap="RdYlGn",annot=True)
plt.show();
sns.heatmap(correlation,cmap="Blues",annot=True)
plt.show();

