#!/usr/bin/env python
# coding: utf-8

# In[85]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error


# In[86]:


car_data=pd.read_csv("car data (1).csv")
car_data


# In[87]:


car_data.isnull().sum()


# In[88]:


print(car_data.Fuel_Type.value_counts())
print(car_data.Seller_Type.value_counts())
print(car_data.Transmission.value_counts())


# In[89]:


le=LabelEncoder()
car_data["Fuel_Type"]=le.fit_transform(car_data["Fuel_Type"])
car_data["Seller_Type"]=le.fit_transform(car_data["Seller_Type"])
car_data["Transmission"]=le.fit_transform(car_data["Transmission"])
car_data


# In[90]:


x=car_data.drop(["Car_Name","Selling_Price"],axis=1)
y=car_data["Selling_Price"]


# In[91]:


x


# In[92]:


y


# In[93]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=0)


# In[104]:




x_train


# In[105]:




y_train


# In[106]:


lr=LinearRegression()
lr.fit(x_train,y_train)


# In[114]:


train_data_prediction=lr.predict(x_train)
r2_scorelr=metrics.r2_score(y_train,train_data_prediction)
print("r2 score :",r2_score)


# In[115]:


plt.scatter(y_train,train_data_prediction)
plt.xlabel("actual price")
plt.ylabel("predicted price")
plt.title("actual vs predicted price ")
plt.show()


# In[121]:


test_data_prediction=lr.predict(x_test)
r2_scorelr=metrics.r2_score(y_test,test_data_prediction)
print("r2 score :",r2_score)


# In[117]:


plt.scatter(y_test,test_data_prediction)
plt.xlabel("actual price")
plt.ylabel("predicted price")
plt.title("actual vs predicted price ")
plt.show()


# In[129]:


ls=Lasso()
ls.fit(x_train,y_train)
train_data_predict=ls.predict(x_train)
r2_scorels1=metrics.r2_score(y_train,train_data_predict)
mse_lasso1= mean_squared_error(y_train,train_data_predict)
mae_lasso1= mean_absolute_error(y_train,train_data_predict)

test_data_predict=ls.predict(x_test)
r2_scorels2=metrics.r2_score(y_test,test_data_predict)
mse_lasso2= mean_squared_error(y_test,test_data_predict)
mae_lasso2= mean_absolute_error(y_test,test_data_predict)

print("r2 score with trained data:",r2_scorels)
print("r2 score with tested data:",r2_scorels)
print("mse with trained data :",mse_lasso1)
print("mse with tested data :",mse_lasso2)
print("mae with trained data :",mae_lasso1)
print("mae with tested data :",mae_lasso2)


# In[131]:


plt.scatter(y_train,train_data_predict)
plt.xlabel("actual price")
plt.ylabel("predicted price")
plt.title("actual vs predicted price ")
plt.show()

plt.scatter(y_test,test_data_predict)
plt.xlabel("actual price")
plt.ylabel("predicted price")
plt.title("actual vs predicted price ")
plt.show()


# In[144]:


rfr=RandomForestRegressor()
rfr.fit(x_train,y_train)
train_data_predict_rfr1=rfr.predict(x_train)
r2_score_rfr1=metrics.r2_score(y_train,train_data_predict_rfr1)
mse_rfr1= mean_squared_error(y_train,train_data_predict)
mae_rfr1= mean_absolute_error(y_train,train_data_predict)

test_data_predict_rfr2=rfr.predict(x_test)
r2_score_rfr2=metrics.r2_score(y_test,test_data_predict_rfr2)
mse_rfr2= mean_squared_error(y_test,test_data_predict)
mae_rfr2= mean_absolute_error(y_test,test_data_predict)

print("r2 score with trained data:",r2_score_rfr1)
print("r2 score with tested data:",r2_score_rfr2)
print("mse with trained data :",mse_rfr1)
print("mse with tested data :",mse_rfr2)
print("mae with trained data :",mae_rfr1)
print("mae with tested data :",mae_rfr2)



# In[143]:


plt.scatter(y_train,train_data_predict_rfr1)
plt.xlabel("actual price")
plt.ylabel("predicted price")
plt.title("actual vs predicted price ")
plt.show()

plt.scatter(y_test,test_data_predict_rfr2)
plt.xlabel("actual price")
plt.ylabel("predicted price")
plt.title("actual vs predicted price ")
plt.show()


# In[147]:


dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
train_data_predict_dt1=rfr.predict(x_train)
r2_score_dt1=metrics.r2_score(y_train,train_data_predict_rfr1)
mse_dt1= mean_squared_error(y_train,train_data_predict)
mae_dt1= mean_absolute_error(y_train,train_data_predict)

test_data_predict_dt2=rfr.predict(x_test)
r2_score_dt2=metrics.r2_score(y_test,test_data_predict_rfr2)
mse_dt2= mean_squared_error(y_test,test_data_predict)
mae_dt2= mean_absolute_error(y_test,test_data_predict)

print("r2 score with trained data:",r2_score_dt1)
print("r2 score with tested data:",r2_score_dt2)
print("mse with trained data :",mse_dt1)
print("mse with tested data :",mse_dt2)
print("mae with trained data :",mae_dt1)
print("mae with tested data :",mae_dt2)


# In[148]:


plt.scatter(y_train,train_data_predict_dt1)
plt.xlabel("actual price")
plt.ylabel("predicted price")
plt.title("actual vs predicted price ")
plt.show()

plt.scatter(y_test,test_data_predict_dt2)
plt.xlabel("actual price")
plt.ylabel("predicted price")
plt.title("actual vs predicted price ")
plt.show()


# In[153]:


knn=KNeighborsRegressor()
knn.fit(x_train,y_train)
train_data_predict_knn1=rfr.predict(x_train)
r2_score_knn1=metrics.r2_score(y_train,train_data_predict_rfr1)
mse_knn1= mean_squared_error(y_train,train_data_predict)
mae_knn1= mean_absolute_error(y_train,train_data_predict)

test_data_predict_knn2=rfr.predict(x_test)
r2_score_knn2=metrics.r2_score(y_test,test_data_predict_rfr2)
mse_knn2= mean_squared_error(y_test,test_data_predict)
mae_knn2= mean_absolute_error(y_test,test_data_predict)

print("r2 score with trained data:",r2_score_knn1)
print("r2 score with tested data:",r2_score_knn2)
print("mse with trained data :",mse_knn1)
print("mse with tested data :",mse_knn2)
print("mae with trained data :",mae_knn1)
print("mae with tested data :",mae_knn2)


# In[154]:


plt.scatter(y_train,train_data_predict_knn1)
plt.xlabel("actual price")
plt.ylabel("predicted price")
plt.title("actual vs predicted price ")
plt.show()

plt.scatter(y_test,test_data_predict_knn2)
plt.xlabel("actual price")
plt.ylabel("predicted price")
plt.title("actual vs predicted price ")
plt.show()


# In[ ]:





# In[ ]:




