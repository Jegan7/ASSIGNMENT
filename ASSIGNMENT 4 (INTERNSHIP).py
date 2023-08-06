#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#import libraries for graphs and vizualitations
import seaborn as sns
import plotly as py
import plotly.express as px


# In[3]:


# library have imported 


# In[4]:


#reading data sets of both airbnb and hrdataset
airbnb=pd.read_csv('Airbnb Dataset 19.csv')
hr_data=pd.read_csv('HRDataset_v14.csv')


# In[5]:


#getting head and tail for airbnb
airbnb


# In[6]:


# getting head and tail of hrdata set
hr_data


# In[7]:


#finding if any missing value is present or not in airbnb
airbnb.isnull().sum()


# In[8]:


#finding if any missing value is present or not in hr_data
hr_data.isnull().sum()


# In[9]:


airbnb.dtypes


# In[10]:


hr_data.dtypes


# TREATMENT OF MISSING VALUE

# In[11]:


#replacing value in missing value in arbnb data using interpolation
airbnb['last_review']=airbnb['last_review'].fillna(method='bfill')
airbnb['reviews_per_month']=airbnb['reviews_per_month'].interpolate(method='linear')
hr_data['DateofTermination']=hr_data['DateofTermination'].fillna(0)
hr_data['ManagerID']=hr_data['ManagerID'].fillna(0)


# In[12]:


#checking if the null value is filled 
airbnb.isnull().sum().sum()
hr_data.isnull().sum().sum()


# TREAMENT OF OUTLIERS

# In[13]:


# using univariate analysis 
plt.figure.figsize=(102,24)

plt.subplot(1,3,1)
sns.scatterplot(y=airbnb['price'],x=airbnb['availability_365'])

plt.subplot(1,3,2)
sns.scatterplot(y=airbnb['price'],x=airbnb['room_type'])

plt.subplot(1,3,3)
sns.scatterplot(y=airbnb['price'],x=airbnb['calculated_host_listings_count'])


plt.suptitle("scatter plot of airbnb using price")
plt.show()


# In[14]:


plt.rcParams['figure.figsize']=(15,28)

plt.subplot(2,2,1)
sns.scatterplot(y=hr_data['EmpID'],x=hr_data['SpecialProjectsCount'])

plt.subplot(2,2,2)
sns.scatterplot(y=hr_data['EmpID'],x=hr_data['SpecialProjectsCount'])

plt.subplot(2,2,3)
sns.scatterplot(y=hr_data['EmpID'],x=hr_data['Absences'])

plt.suptitle("scatter plot of hrdataset using price")
plt.show()


# DATA VIZUALITATION

# In[15]:


# visualizing tha airbnb data


# In[16]:


#visualize the dataset using plotly in airbnb dataset
px.scatter(airbnb , x="price",y="minimum_nights",color="room_type")



#-- in this the minimum night styaed in specific room is ascertained with price


# In[17]:


#convert langitude and latitude into int format
airbnb['longitude']=airbnb['longitude'].astype(int)
airbnb['latitude']=airbnb['latitude'].astype(int)


# In[18]:


#price is seen with the respect of latitude with the minimumnight spent 



px.line_polar(airbnb,r='latitude',theta='price',color='minimum_nights')


# In[19]:


#visulaize dataset with 3d plot
px.scatter_3d(airbnb,x='id',y='host_id',z='number_of_reviews',color='availability_365')


#reviews got according to the id for 365 days


# In[20]:


#visulizing data with scatter plt to get access of heatmaps 
plt.figure(figsize=(8,6))
sns.heatmap(airbnb.corr() , annot=True, cmap='viridis')
plt.title("heat map ")    
plt.show()

# finding the correlation between the data in data set


# In[21]:


#seeing the value in dataset using barplot
sns.boxplot(x=airbnb['price'],y=airbnb['room_type'])

# finding the price according to room 


# In[22]:


#visualizing the hr_data 
hr_data.dtypes


# In[23]:


# creating bar plot with  hr data
px.bar(hr_data,x='EmpID',y='Absences',color='MarriedID')

# bar shows the employee who are absent on basis of marriage id 


# In[24]:


#creating waffle using plotly violin plot
px.violin(hr_data, x='Salary',y='Position',color='Department')

# this shows the range of salary the deiiferent position 


# In[25]:


#histogram to see the range of performance by employees
px.histogram(hr_data,x='EmpID',y='PerformanceScore',color='GenderID')

#in this the performance of employee is based on gender


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




