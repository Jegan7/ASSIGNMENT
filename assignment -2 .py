#!/usr/bin/env python
# coding: utf-8

# # # WEB SCRAPPING 
# 

# scrapping data from toscrape(educational website)

# In[18]:


import requests
from bs4 import BeautifulSoup
import pandas as pd


# In[19]:


url="https://books.toscrape.com/catalogue/page-1.html"


# In[20]:


response = requests.get(url)


# In[26]:


#imported pandas ,requests, beautifulsoup


# In[25]:


books = []

for i in range(1,4):
    url = "https://books.toscrape.com/catalogue/page-1.html"
    response = requests.get(url)
    response = response.content
    soup = BeautifulSoup(response,'html.parser')
    ol = soup.find('ol')
    articles = ol.find_all('article', class_ = 'product_pod')

    for article in articles:
        image = article.find('img')
        title = image.attrs['alt']
        star = article.find('p')
        star = star['class'][1]
        price = article.find('p',class_ = 'price_color').text
        price = float(price[1:])
        books.append([title,price,star])


# In[23]:


df = pd.DataFrame(books,columns = ['Title','Price','Stars'])


# In[24]:


df.to_csv('book.csv')


# In[ ]:





# In[ ]:




