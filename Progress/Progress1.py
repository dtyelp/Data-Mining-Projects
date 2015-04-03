
# coding: utf-8

# In[1]:

import json
import pandas as pd


# In[2]:

data = json.loads(yelp_academic_dataset_review.json)


# In[3]:

data = pandas.io.json.read_json(file://localhost//C://Users//Jvalin//Documents//IPython Notebooks//yelp_academic_dataset_review.json)


# In[7]:

import json
data = []
x = []
fdict  = {}
with open ('yelp_academic_dataset_review.json') as f:
    for line in f:
        data.append(json.loads(line))
        fdict = json.loads(line)
        x.append(fdict.items())


# In[8]:

print (x[3])


# In[9]:

print (x[4])


# In[10]:

import nltk


# In[11]:

df = pd.DataFrame(data.items(), columns=['text', 'textValue'])


# In[13]:

df = pd.DataFrame(fdict.items(), columns=['text', 'review text'])


# In[14]:

pd.DataFrame(fdict)


# In[1]:

pd.DataFrame(x)


# In[2]:

import pandas as pd
pd.Dataframe(x)


# In[3]:

import json
import pandas as pd


# In[2]:

import json
data1 = []
x = []
data = []
y  = []
fdict = {}
bdict = {}

with open ('yelp_academic_dataset_review.json') as f:
    for line in f:
        data.append(json.loads(line))
        fdict = json.loads(line)
        x.append(fdict.items())
  
with open ('yelp_academic_dataset_business.json') as g:
    for line in g:
        data1.append(json.loads(line))
        bdict = json.loads(line)
        y.append(bdict.items())
        



# In[5]:

pd.Dataframe(x)


# In[6]:

print (len(x))


# In[3]:

print (len(y))


# In[23]:

print (y[1])


# In[5]:

print (len(x)/ len(y))


# In[6]:

print (y[1])


# In[7]:

import pandas as pf
pf.DataFrame(y)


# In[13]:




# In[14]:

print (len(zdict))


# In[16]:

print (fdict)


# In[17]:

print (bdict)


# In[18]:

print(zdict)


# In[20]:

print (x[1])
            


# In[ ]:



