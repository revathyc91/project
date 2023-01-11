#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv("Weather_Data.csv")


# In[3]:


data.head(5)


# In[4]:


data.shape


# In[5]:


data.info()


# # Features description

# 
# Location---The common name of the location of the weather station
# 
# MinTemp---The minimum temperature in degrees celsius
# 
# MaxTemp---The maximum temperature in degrees celsius
# 
# Rainfall---The amount of rainfall recorded for the day in mm
# 
# Evaporation---The so-called Class A pan evaporation (mm) in the 24 hours to 9am
# 
# Sunshine---The number of hours of bright sunshine in the day.
# 
# WindGustDir---The direction of the strongest wind gust in the 24 hours to midnight
# 
# WindGustSpeed---The speed (km/h) of the strongest wind gust in the 24 hours to midnight
# 
# WindDir9am---Direction of the wind at 9am
# 
# WindDir3pm---Direction of the wind at 3pm
# 
# WindSpeed9am---Wind speed (km/hr) averaged over 10 minutes prior to 9am
# 
# WindSpeed3pm---Wind speed (km/hr) averaged oover 10 minutes prior to 3pm
# 
# Humidity9am---Humidity (percent) at 9am
# 
# Humidity3pm---Humidity (percent) at 3pm
# 
# Pressure9am---Atmospheric pressure (hpa) reduced to mean sea level at 9am
# 
# Pressure3pm---Atmospheric pressure (hpa) reduced to mean sea level at 3pm
# 
# Cloud9am---Fraction of sky obscured by cloud at 9am. This is measured in "oktas", which are a unit of eigths. It records how many eigths of the sky are obscured by cloud. A 0 measure indicates completely clear sky whilst an 8 indicates that it is completely overcast.
# 
# Cloud3pm---Fraction of sky obscured by cloud (in "oktas": eighths) at 3pm. See Cload9am for a description of the values
# 
# Temp9am---Temperature (degrees C) at 9am
# 
# Temp3pm---Temperature (degrees C) at 3pm
# 
# RainToday---Boolean: 1 if precipitation (mm) in the 24 hours to 9am exceeds 1mm, otherwise 0
# 
# RainTomorrow---The target variable. Did it rain tomorrow?

# In[6]:


data.nunique()


# In[7]:


data['Cloud3pm'].unique()


# In[8]:


data.drop('row ID',axis=1,inplace=True)


# In[9]:


data.shape


# In[10]:


data.describe()


# # Exploratory data analysis

# In[11]:


freqgraph=data.select_dtypes(include=['float'])
freqgraph.hist(figsize=(20,15))
plt.show()


# In[12]:


sns.displot(data, x="MinTemp", hue='RainTomorrow', kde=True)
plt.title("Minimum Temperature Distribution", fontsize = 14)
plt.show()


# In[13]:


sns.displot(data, x="MaxTemp", hue='RainTomorrow', kde=True)
plt.title("Maximum Temperature Distribution", fontsize = 14)
plt.show()


# In[14]:


sns.displot(data,x='WindGustSpeed',hue='RainTomorrow',kde=True)
plt.title('Distribution of WindGustSpeed',fontsize=16)
plt.show()


# In[15]:


sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[['MinTemp','MaxTemp','Temp9am','Temp3pm']])


# In[16]:


sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[['WindGustSpeed','WindSpeed9am','WindSpeed3pm']])


# In[17]:


sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[['Humidity9am','Humidity3pm']])


# In[18]:


sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[['Pressure9am','Pressure3pm']])


     
    


# In[19]:


sns.set(style="whitegrid")
plt.figure(figsize=(5, 3))
sns.boxplot(data=data[['Cloud9am','Cloud3pm']])


# In[20]:


sns.set(style="whitegrid")
plt.figure(figsize=(5, 6))
sns.boxplot(data=data[['Rainfall']])


# In[21]:


sns.set(style="whitegrid")
plt.figure(figsize=(5, 6))
sns.boxplot(data=data[['Evaporation','Sunshine']])


# In[22]:


# Correlation Matrix
corr_matrix=data.corr()
plt.subplots(figsize=(18,8))
sns.heatmap(corr_matrix,annot=True,cmap='winter')


# Highly positive correlated : MinTemp and MaxTemp , MinTemp and Temp9am , MinTemp and Temp3pm MaxTemp and Temp9am , MaxTemp and Temp3pm , Pressure9am and Pressure 3pm, Temp9am and Temp 3pm
# 
# Negatively correlated : Sunshine and Cloud9am , Sunshine and Cloud3pm
# 
# The columns MaxTemp,Evaporation,Sunshine,Pressure9am,Pressure3pm,Temp9am and Temp3pm have less negative correlation with the Target variable,'RainTomorrow'

# In[23]:


data.isna().sum()


# In[24]:


# temp9am and temp 3pm highly correlated
#cloud 3pm and 9am
#pressure 3pm and 9am
#humidity 9am and 3 pm
#wing gust speed 9am and 3pm and wind gust speed
#evaporation and max temp
#max temp temp3pm


# In[25]:


col=['Sunshine','Evaporation','Temp9am','Pressure9am','Cloud9am','Humidity9am','WindSpeed9am','WindSpeed3pm']


# In[26]:


data.drop(col,axis=1,inplace=True)


# In[27]:


data.shape


# In[28]:


# Correlation Matrix
corr_matrix=data.corr()
plt.subplots(figsize=(18,8))
sns.heatmap(corr_matrix,annot=True,cmap='winter')


# In[29]:


data.drop('Temp3pm',axis=1,inplace=True)


# In[30]:


data.drop('Cloud3pm',axis=1,inplace=True)


# # PREPROCESSING

# In[31]:


# Filling missing values in Humidity3pm' with mean
data['Humidity3pm']=data['Humidity3pm'].fillna(data['Humidity3pm'].mean())


# In[32]:


obj = data[['WindGustDir','WindDir9am','WindDir3pm']]
num= data[[ 'MinTemp','MaxTemp','Rainfall','WindGustSpeed','Pressure3pm']]   


# In[33]:


for i in num:
   data[i].fillna(data[i].median(),inplace=True)


# In[34]:


for i in obj:
   data[i].fillna(data[i].value_counts().index[0],inplace=True)


# In[35]:


data.isna().sum()


# In[36]:


# Column 'Raintoday' has 979 null values if we fill this with any other values, it may mislead our prediction so drop the null values 


# In[37]:


data= data.dropna(subset=['RainToday'])


# In[38]:


data.shape


# In[39]:


data.isna().sum()


# In[40]:


#Outliers we are checking only for numerical features
sns.set(style="whitegrid")
plt.figure(figsize=(20, 10))
sns.boxplot(data=data[[ 'MinTemp','MaxTemp','Rainfall','WindGustSpeed','Humidity3pm','Pressure3pm']])


# In[41]:


for i in num:
 Q1=np.percentile(data[i],25)
 Q3=np.percentile(data[i],75)
 IQR=Q3-Q1
 low_lim=Q1-1.5*IQR
 up_lim=Q3+1.5*IQR
 outliers=[]
 for x in data[i]:
     if(x<low_lim)or(x>up_lim):
         outliers.append(x)
 data[i]=np.where(data[i]<low_lim,low_lim,np.where(data[i]>up_lim,up_lim,data[i]))


# In[42]:


sns.set(style="whitegrid")
plt.figure(figsize=(20, 10))
sns.boxplot(data=data[[ 'MinTemp','MaxTemp','Rainfall','WindGustSpeed','Humidity3pm','Pressure3pm']])


# In[43]:


# Label Encoding Location,WindDir3pm,WindGustDir and RainToday
from sklearn.preprocessing import LabelEncoder
lab_enc=LabelEncoder()
data['Location']=lab_enc.fit_transform(data['Location'])
data['WindDir3pm']=lab_enc.fit_transform(data['WindDir3pm'])
data['WindDir9am']=lab_enc.fit_transform(data['WindDir9am'])
data['WindGustDir']=lab_enc.fit_transform(data['WindGustDir'])
data['RainToday']=lab_enc.fit_transform(data['RainToday'])


# In[44]:


data.head()


# In[45]:


corr_matrix=data.corr()
plt.subplots(figsize=(18,8))
sns.heatmap(corr_matrix,annot=True,cmap='winter')


# In[46]:


data.drop('Rainfall',axis=1,inplace=True)


# In[47]:


corr_matrix=data.corr()
plt.subplots(figsize=(18,8))
sns.heatmap(corr_matrix,annot=True,cmap='winter')


# In[48]:


data.drop('MinTemp',axis=1,inplace=True)


# In[49]:


data.shape


# In[50]:


data


# In[51]:


corr_matrix=data.corr()
plt.subplots(figsize=(18,8))
sns.heatmap(corr_matrix,annot=True,cmap='winter')


# # standardisation

# In[52]:


X=data.drop(['RainTomorrow'],axis=1)
y=data['RainTomorrow']


# In[53]:


X


# In[54]:


#Standard Scaling 
from sklearn.preprocessing import StandardScaler
std_scl=StandardScaler()


# In[55]:


X1=X.drop(['Location','WindDir9am','WindDir3pm','WindGustDir','RainToday'],axis=1)


# In[56]:


X1


# In[57]:


X1=std_scl.fit_transform(X1)


# In[58]:


X1


# In[59]:


X1=pd.DataFrame(X1,columns=['MaxTemp','WindGustSpeed','Humidity3pm','Pressure3pm'])


# In[60]:


X1


# In[61]:


X=X.drop(['MaxTemp','WindGustSpeed','Humidity3pm','Pressure3pm'],axis=1)


# In[62]:


X


# In[63]:


X.index=X1.index


# In[64]:


X=pd.concat([X,X1],axis=1)


# In[65]:


X


# In[66]:


data.isna().sum()


# In[67]:


X.describe()


# # modeling

# In[68]:


from sklearn.model_selection import train_test_split


# In[69]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)


# # Creating Logistic Regression Model

# In[70]:


from sklearn.linear_model import LogisticRegression
logit_model=LogisticRegression()
logit_model=logit_model.fit(X_train,y_train)
y_pred_logit=logit_model.predict(X_test)


# In[71]:



from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import classification_report


# In[72]:


from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score


# In[73]:


confusion_matrix(y_test,y_pred_logit)


# In[74]:


print('Accuracy score is :',accuracy_score(y_test,y_pred_logit))


# In[75]:


precision_score(y_test,y_pred_logit)


# In[76]:


recall_score(y_test,y_pred_logit)


# In[77]:


f1_score(y_test,y_pred_logit)


# In[78]:


print(classification_report(y_test,y_pred_logit))


# # Creating KNN Model

# In[79]:


from sklearn.neighbors import KNeighborsClassifier


# In[80]:


metric_k=[]
neighbors=range(3,15)

for k in neighbors:
     classifier=KNeighborsClassifier(n_neighbors=k)
     kNN_model=classifier.fit(X_train,y_train)
     y_pred_kNN=kNN_model.predict(X_test)
     acc=accuracy_score(y_test,y_pred_kNN)
     metric_k.append(acc)


# In[81]:


plt.plot(neighbors,metric_k,'o-')
plt.xlabel('k values')
plt.ylabel('Accuracies')
plt.grid()


# In[82]:


accuracy_score(y_test,y_pred_kNN)


# In[83]:


classifier=KNeighborsClassifier(n_neighbors=11)
kNN_model=classifier.fit(X_train,y_train)
y_pred_kNN=kNN_model.predict(X_test)


# In[84]:


confusion_matrix(y_test,y_pred_kNN)


# In[85]:




accuracy_score(y_test,y_pred_kNN)


# In[86]:


precision_score(y_test,y_pred_kNN)


# In[87]:


recall_score(y_test,y_pred_kNN)


# In[88]:


f1_score(y_test,y_pred_kNN)


# In[89]:


print('classification_report :\n',classification_report(y_test,y_pred_kNN))


# # Creating SVM(Kernel='Linear')Model

# In[90]:


from sklearn.svm import SVC


# In[91]:


svm_clf=SVC(kernel='linear')


# In[92]:


#svm_clf=svm_clf.fit(X_train,y_train)
#y_pred_svm=svm_clf.predict(X_test)


# In[93]:


#accuracy_score(y_test,y_pred_svm)


# # Creating SVM(Kernel='rbf')Model

# In[94]:


#svm_clf2=SVC(kernel='rbf')
#svm_clf2=svm_clf2.fit(X_train,y_train)
#y_pred_svm2=svm_clf2.predict(X_test)


# In[95]:


#accuracy_score(y_test,y_pred_svm2)


# In[ ]:





# # Creating DecisionTreeClassifier Model

# In[96]:


X=data.drop(['RainTomorrow'],axis=1)
y=data['RainTomorrow']


# In[97]:


# Splitting into train and test data
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=.2)


# In[98]:


from sklearn.tree import DecisionTreeClassifier


# In[99]:


dt_clf=DecisionTreeClassifier()
dt_clf=dt_clf.fit(X_train,y_train)
y_pred_dt=dt_clf.predict(X_test)


# In[100]:


confusion_matrix(y_test,y_pred_dt)


# In[101]:


accuracy_score(y_test,y_pred_dt)


# In[102]:


precision_score(y_test,y_pred_dt)


# In[103]:


recall_score(y_test,y_pred_dt)


# In[104]:


f1_score(y_test,y_pred_dt)


# In[105]:


print('classification_report :\n',classification_report(y_test,y_pred_dt))


# # Creating RandomForestClassifier Model 

# In[106]:


from sklearn.ensemble import RandomForestClassifier


# In[107]:


rf_clf=RandomForestClassifier()
rf_clf=rf_clf.fit(X_train,y_train)
y_pred_rf=rf_clf.predict(X_test)


# In[108]:


accuracy_score(y_test,y_pred_rf)


# In[109]:


confusion_matrix(y_test,y_pred_rf)


# In[110]:


precision_score(y_test,y_pred_rf)


# In[111]:


f1_score(y_test,y_pred_rf)


# In[112]:


print('classification_report :\n',classification_report(y_test,y_pred_rf))


# In[113]:


import pickle


# In[114]:


pickle.dump(rf_clf,open('model1.pkl','wb') )


# In[115]:


model1=pickle.load(open('model1.pkl','rb'))


# In[116]:


X_test


# In[117]:


y_test


# In[118]:


print(model1.predict([[15,11.0,6,68.5,6,6,63.0,998.25,0]]))


# In[119]:


print(model1.predict([[14,26.7,10,61.0,10,10,65.0,1014.30,1]]))


# In[120]:


print(model1.predict([[38,21.2,11,68.5,11,12,79.0,1019.20,0]]))


# In[121]:


print(model1.predict([[2,30.4,3,30.0,10,2,22.0,1008.7,0]]))


# # Fine Tuning RandomForestClassifier Model

# In[122]:


from sklearn.model_selection import GridSearchCV,RandomizedSearchCV


# In[123]:


parameters={'n_estimators':[50,100,150,250],'max_depth':[3,6,9,None]}


#  # Hyperparameter Tuning-GridSearchCV

# In[124]:



grid_model=GridSearchCV(RandomForestClassifier(),parameters,scoring='f1')
grid_model.fit(X_train,y_train)
print(grid_model.best_params_)


# In[125]:


model_grid=RandomForestClassifier(n_estimators=250,max_depth=None)
rf_grid=model_grid.fit(X_train,y_train)
y_pred_grid=rf_grid.predict(X_test)


# In[126]:


accuracy_score(y_test,y_pred_grid)


# In[127]:


f1_score(y_test,y_pred_grid)


# In[128]:


print('classification_report :\n',classification_report(y_test,y_pred_grid))


# In[129]:


pickle.dump(rf_grid,open('model2.pkl','wb') )


# In[130]:


model2=pickle.load(open('model2.pkl','rb'))


# In[131]:


print(model2.predict([[15,11.0,6,68.5,6,6,63.0,998.25,0]]))


# In[132]:


print(model2.predict([[14,26.7,10,61.0,10,10,65.0,1014.30,1]]))


# In[133]:


print(model2.predict([[38,21.2,11,68.5,11,12,79.0,1019.20,0]]))


# In[134]:


print(model2.predict([[2,30.4,3,30.0,10,2,22.0,1008.7,0]]))


# # Hyperparameter Tuning-RandomizedSearchCV
# 

# In[135]:


random_model=RandomizedSearchCV(RandomForestClassifier(),parameters,scoring='f1')
random_model.fit(X_train,y_train)
print(random_model.best_params_)


# In[136]:


model_random=RandomForestClassifier(n_estimators=250,max_depth=None)
rf_random=model_random.fit(X_train,y_train)
y_pred_random=rf_random.predict(X_test)


# In[137]:


accuracy_score(y_test,y_pred_random)


# In[138]:


f1_score(y_test,y_pred_random)


# In[139]:


print('classification_report :\n',classification_report(y_test,y_pred_random))


# In[140]:


pickle.dump(rf_random,open('model3.pkl','wb') )


# In[141]:


model3=pickle.load(open('model3.pkl','rb'))


# In[142]:


print(model3.predict([[15,11.0,6,68.5,6,6,63.0,998.25,0]]))


# In[143]:


print(model3.predict([[14,26.7,10,61.0,10,10,65.0,1014.30,1]]))


# In[144]:


print(model3.predict([[38,21.2,11,68.5,11,12,79.0,1019.20,0]]))


# In[145]:


print(model3.predict([[2,30.4,3,30.0,10,2,22.0,1008.7,0]]))


# In[ ]:




