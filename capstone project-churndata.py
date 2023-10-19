#!/usr/bin/env python
# coding: utf-8

# In[92]:



#Problem Statement –You are the Data Scientist at a telecom company “Neo” whose customers are churning out to its competitors. You have to analyse the data of your company and find insights and stop your customers from churning out to other telecom companies.


# # (A) data manipulation

# In[93]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[94]:


a=pd.read_csv("customer_churn _2_.csv")


# In[95]:


a


# In[96]:


#a.	Extract the 5th column & store it in ‘customer_5’
a.iloc[:,4]


# In[97]:


#b.	Extract the 15th column & store it in ‘customer_15’ 
a.iloc[:,15]


# In[98]:


d=a.iloc[:,15]
d.head()


# In[180]:


#c.	Extract all the male senior citizens whose Payment Method is Electronic check & store the result in ‘senior_male_electronic’
e=a[(a['gender']=='Male') & (a['SeniorCitizen']==1) & (a['PaymentMethod']=='Electronic Check')]
e.head()


# In[170]:


#d.	Extract all those customers whose tenure is greater than 70 months or their Monthly charges is more than 100$ & store the result in ‘customer_total_tenure’

f=a[(a['tenure']>70) | (a['MonthlyCharges']>100)]
f.tail()


# In[171]:


#e.	Extract all the customers whose Contract is of two years, payment method is Mailed check & the value of Churn is ‘Yes’ & store the result in ‘two_mail_yes’
g=a[(a['Contract']=='Two year') & (a['PaymentMethod']=='Mailed check') & (a['Churn']=='Yes')]
g


# In[172]:


#f.	Extract 333 random records from the customer_churndataframe& store the result in ‘customer_333’
h=a.sample(n=333)
h


# In[173]:


#g.	Get the count of different levels from the ‘Churn’ column
a["Churn"].value_counts()


# # (B) data Visualization

# In[175]:


#a.	Build a bar-plot for the ’InternetService’ column:
    #i.	Set x-axis label to ‘Categories of Internet Service’
    #ii.	Set y-axis label to ‘Count of Categories’
    #iii.	Set the title of plot to be ‘Distribution of Internet Service’
    #iv.	Set the color of the bars to be ‘orange’


plt.bar(a["InternetService"].value_counts().keys().tolist(),a["InternetService"].value_counts().tolist(),color='orange')
plt.xlabel("Categories of net usage")
plt.ylabel("Count")
plt.title("Distribution of net Service")
plt.show()


# In[176]:


a["InternetService"].value_counts().keys().tolist()


# In[177]:


a["InternetService"].value_counts().tolist()


# In[178]:


#b.	Build a histogram for the ‘tenure’ column:
   #i.	Set the number of bins to be 30
   #ii.	Set the color of the bins  to be ‘green’
   #iii.	Assign the title ‘Distribution of tenure’

plt.hist(a['tenure'],bins=30,color="green")
plt.title("tenure")
plt.show()


# In[109]:


#c.	Build a scatter-plot between ‘MonthlyCharges’ & ‘tenure’. Map ‘MonthlyCharges’ to the y-axis & ‘tenure’ to the ‘x-axis’:
    #i.	Assign the points a color of ‘brown’
    #ii.	Set the x-axis label to ‘Tenure of customer’
    #iii.	Set the y-axis label to ‘Monthly Charges of customer’
    #iv.	Set the title to ‘Tenure vs Monthly Charges’

plt.scatter(x=a['tenure'].head(20),y=a['MonthlyCharges'].head(20),color='brown')
plt.xlabel("tenure")
plt.ylabel("charges")
plt.title('tenure vs Monthly Charges')
plt.grid(True)
plt.show()


# In[110]:


#d.	Build a box-plot between ‘tenure’ & ‘Contract’. Map ‘tenure’ on the y-axis & ‘Contract’ on the x-axis. 

a.boxplot(column=['tenure'],by=['Contract'],showmeans=True)
plt.show()


# In[111]:


import seaborn as sns
sns.boxplot('Contract','tenure',data=a,width=0.3)


# # (C)Linear Regression

# In[113]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[114]:


#a.	Build a simple linear model where dependent variable is ‘MonthlyCharges’ and independent variable is ‘tenure’
   #i.	Divide the dataset into train and test sets in 70:30 ratio. 
   #ii.	Build the model on train set and predict the values on test set
   #iii.	After predicting the values, find the root mean square error
   #iv.	Find out the error in prediction & store the result in ‘error’
   #v.	Find the root mean square error

x=pd.DataFrame(a['tenure'])
y=pd.DataFrame(a['MonthlyCharges'])


# In[115]:


y


# In[116]:


type(x)


# In[117]:


y.shape


# In[118]:


X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.30)


# In[119]:


model=LinearRegression()              ###creating a mode


# In[120]:


model


# In[121]:


model.fit(X_train,Y_train)            ###Training


# In[122]:


X_train


# In[123]:


model.predict(X_test)     ###tested


# In[124]:


from sklearn.metrics import mean_squared_error


# In[125]:


y_pred=model.predict(X_test)


# In[126]:


y_pred


# In[127]:


mse=mean_squared_error(X_test,y_pred)              ##mse=mean square error


# In[128]:


mse


# In[129]:


rms=np.sqrt(mse)    ####root means error


# In[130]:


rms


# # (D)	Logistic Regression:

# In[131]:


#a.	Build a simple logistic regression modelwhere dependent variable is ‘Churn’ & independent variable is ‘MonthlyCharges’
   #i.	Divide the dataset in 65:35 ratio
   #ii.	Build the model on train set and predict the values on test set
   #iii.	Build the confusion matrix and get the accuracy score


# In[132]:


from sklearn.linear_model import LogisticRegression


# In[133]:


x=pd.DataFrame(a['MonthlyCharges'])
y=a['Churn']


# In[134]:


X_train,X_test,Y_train,Y_test=train_test_split(x,y,train_size=0.65,random_state=0)


# In[135]:


model=LogisticRegression()


# In[136]:


model.fit(X_train,Y_train)


# In[137]:


model.predict(X_test)


# In[138]:


y_pred = model.predict(X_test)


# In[139]:


y_pred


# In[140]:


from sklearn.metrics import accuracy_score,confusion_matrix


# In[141]:


accuracy_score(Y_test,y_pred)


# In[142]:


cm=confusion_matrix(Y_test,y_pred)
cm


# In[143]:


#(TP+TN)+(TP+TN+FP+FN)
A=(1837+0)+(1837+0+629)
A


# In[144]:


#Multiple Loistic regression


# In[145]:


x=pd.DataFrame(a.loc[:,['MonthlyCharges','tenure']])
y=a['Churn']


# In[146]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.80,random_state=10)


# In[147]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)


# In[148]:


y_pred = logmodel.predict(x_test)


# In[149]:


y_pred


# In[150]:


cm = confusion_matrix(y_pred,y_test)
cm


# In[151]:


add=accuracy_score(y_test,y_pred)
add


# In[152]:


#b.	Build a multiple logistic regression model where dependent variable is ‘Churn’ & independent variables are ‘tenure’ & ‘MonthlyCharges’
   #i.	Divide the dataset in 80:20 ratio
   #ii.	Build the model on train set and predict the values on test set
   #iii.	Build the confusion matrix and get the accuracy score

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# # (E)Decision Tree

# In[154]:


#a.	Build a decision tree model where dependent variable is ‘Churn’ & independent variable is ‘tenure’
   #i.	Divide the dataset in 80:20 ratio
   #ii.	Build the model on train set and predict the values on test set
   #iii.	Build the confusion matrix and calculate the accuracy

x=pd.DataFrame(a['tenure'])
y=a['Churn']


# In[155]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)


# In[156]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x_train,y_train)


# In[157]:


y_pred = classifier.predict(x_test)


# In[158]:


from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
print(confusion_matrix(y_test,y_pred))
#print(accuracy_score(y_test,y_pred))


# In[159]:


print(accuracy_score(y_test,y_pred))


# # (F)Random Forest

# In[161]:


#a.	Build a Random Forest model where dependent variable is ‘Churn’ & independent variables are ‘tenure’ and ‘MonthlyCharges’
   #i.	Divide the dataset in 70:30 ratio
   #ii.	Build the model on train set and predict the values on test set
   #iii.	Build the confusion matrix and calculate the accuracy


x = a[['MonthlyCharges','tenure'] ]
y = a['Churn']


# In[162]:


x = a[['tenure','MonthlyCharges']]
y = a['Churn']


# In[163]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20)


# In[164]:


from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(x_train,y_train)


# In[165]:


y_pred=clf.predict(x_test)


# In[166]:


cm=confusion_matrix(y_pred,y_test)
cm


# In[167]:


ac=accuracy_score(y_pred,y_test)
ac


# In[168]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))


# In[ ]:




