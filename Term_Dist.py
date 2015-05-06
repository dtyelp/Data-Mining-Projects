
# coding: utf-8

# In[1]:

#Getting the data
import json
import pandas as pd
data1 = []
x = []
data = []
y  = []
m = []
n = []
odict = {}
bdict = {}
data1 = []
ya = []
fa = []
yb = []
yc = []
w = 0
y_loc = []
yrc = []
with open ('yelp_academic_dataset_business.json') as f:
    for line in f:
        data1.append(json.loads(line))
        bdict = json.loads(line)
        yb.append(bdict["state"])
        ya.append(bdict["business_id"])
        yc.append(bdict["categories"])
        fa.append(bdict["full_address"])
        y_loc.append(bdict["city"])
        
        
import json
import nltk
import pandas as pd
fdict = {}
data = []
xa = []
xb = []
xd = []
xs = []
xv = []
with open ('yelp_academic_dataset_review.json') as f:
    for line in f:
        data.append(json.loads(line))
        fdict = json.loads(line)
        ui  = fdict["business_id"]
        ug  = fdict ["text"]
        ud  = fdict ["date"]
        us  = fdict ["stars"]
        uv  = fdict ["votes"]
        xa.append(ui)
        xb.append(ug)
        xd.append(ud)
        xs.append(us)
        xv.append(uv)


# In[ ]:




# In[2]:

#getting business id of restaurants in charlotte

bus_ind = []
ra  = []
for k in range (len(yb)):
    if 'Charlotte' == y_loc[k]:
        ra.append(k)
for i in range (len(ra)):
    r= ra[i]
    flag  = 0
    x = yc[r]
    k = r
    for j in range (len(x)):
        u = x.pop()
        u = u.lower()
        if (u == 'restaurant' or u == 'restaurants') and (flag != 1):
            bus_ind.append(k)
            flag = 1
            
            
            
res_bid = []
for i in range (len(bus_ind)):
    g = bus_ind[i]
    res_bid.append(ya[g])
    
    
t  = []
for i in range (len(xa)):
    if xa[i] in res_bid:
        t.append(i)

        
        
rev_vote = []
rev_st = []    
rev_txt = []
for i in range (len(t)):
    p = t[i]
    rev_txt.append(xb[p])
    rev_vote.append(xv[p])
    rev_st.append(xs[p])


# In[3]:

#creating tokens of reviews
fr = []
tg  = []
for i in range (len(rev_txt)):
    fr.append(rev_txt[i])
reviw = []
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
b = sorted((stopwords.words('english')))
for i in range (len(rev_txt)):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = tokenizer.tokenize(rev_txt[i])
    tokens = [w for w in tokens if not w.lower() in b]
    rev_txt[i] = (tokens)


# In[4]:

#getting set of common universal nouns
wsj = nltk.corpus.treebank.tagged_words(tagset='universal')
word_tag_fd = nltk.FreqDist(wsj)
fd = [wt[0] for (wt, _) in word_tag_fd.most_common() if wt[1] == 'NOUN']


# In[6]:

#getting count of all nouns from all reviews
nn_cnt = []
for i in range (len(fd)):
    fcv = fd[i]
    q = 0
    for j in range (len(rev_txt)):
        if fcv in rev_txt[j]:
            q = q + 1
    nn_cnt.append(q)


# In[7]:

#Seprating more frequent nouns and their counts (count more than 4%)
g = 0 
k = int((4 * (len(rev_txt)))/100)
imp_wrd = []
imp_wrdcnt = []
for i in range (len(fd)):
    if nn_cnt[i] < k:
        g = g + 1
    else:
        imp_wrd.append(fd[i])
        imp_wrdcnt.append(nn_cnt[i])

print(k)


# In[8]:

#removing proper nouns and getting the new count and set of words
bol = []
ix = []
tg1 = []
fnl_wrd= []
fnl_wrdcnt = []
from nltk import pos_tag, ne_chunk
from nltk.tokenize import SpaceTokenizer
for i in range (len(imp_wrd)):
    tokenizer = SpaceTokenizer()
    toks = tokenizer.tokenize(imp_wrd[i])
    pos = pos_tag(toks) 
    #tg1.append(pos)

    if ([x[1] for x in pos]) == ['NNP'] :
        bol.append([x[0] for x in pos])
        ix .append(i)
    else:
        tg1.append([x[0] for x in pos])

t = 0
for j in range(len(imp_wrd)):
    if j in ix :
        t= t+1
    else:
        fnl_wrd.append(imp_wrd[j])
        fnl_wrdcnt.append(imp_wrdcnt[j])


# In[9]:

#classifyinging reviews with different ratings
o = 0
one  = []
two = []
three = []
four = []
five = []
for i in range (len(rev_st)):
    if rev_st[i] == 1:
        one.append(rev_txt[i])
    elif rev_st[i] == 2 :
        two.append(rev_txt[i])
    elif rev_st[i] == 3 :
        three.append(rev_txt[i])
    elif rev_st[i] == 5:
        five.append(rev_txt[i])
    elif rev_st[i] == 4 :
        four.append(rev_txt[i])
    else:
        o = o +1


# In[10]:

print (len(one))
print(len(two))
print(len(three))
print(len(four))
print(len(five))
print(len(rev_txt))
print (k)


# In[11]:

#getting count of nouns in reviews with rating one leaving the proper nouns
one_nn_cnt = []
for i in range (len(tg1)):
    h = tg1[i]
    k =0
    for j in range (len(one)):
        if h in one[j]:
            k= k+1
    one_nn_cnt.append(k)


# In[12]:

print(len(one_nn_cnt))


# In[12]:

#getting prior probabilities of final words

nn_cnt_sum = float(sum(nn_cnt))
imp_p =[]
for i in range (len(fnl_wrdcnt)):
    x = float(fnl_wrdcnt[i])
    nn_p = x/nn_cnt_sum
    imp_p.append(nn_p)


# In[13]:

#taking reviews in sample data
sd = []
for i in range (len(one)):
    sd.append(one[i])


# In[28]:

#Naive Bayes Probability :count of each noun in each sample and total count in samples 
import statistics
from operator import itemgetter
sp_pp = []
spc_wrd = []
vp = int((10*len(rev_txt))/100)
for i in range(len(fnl_wrd)):
    if fnl_wrdcnt[i] > vp:
        spc_wrd.append(fnl_wrd[i])  
        sp_pp.append(imp_p[i])
        
        
v = 0        
pb = []
trash =[]
for i in range(len(sd)):
    op = sd[i]
    cb = []
    cp = []
    #lim = int((len(spc_wrd))/2)
    for j in range(len(spc_wrd)):
        gf = spc_wrd[j]
        c = 0
        for k in range(len(op)):
            if op[k] == gf:
                c = c +1
        if c != 0:
            cb.append(c)
            cp.append(sp_pp[j])
        else:
            cb.append(0)
            cp.append(0)
    
    bn=[]
    hsm = sum(cb)
    if (hsm != 0):
        for l in range(len(cb)):
            bn.append(cb[l]/hsm)
    
    r = 1
    flag = 0
    for n in range (len(cb)):
        if cb[n] != 0 and cp[n] != 0:
            r =r *  bn[n] *cp[n]
            flag = 1
        else:
            flag = 2

    if flag == 1:
        pb.append(r)
    elif flag == 2:
        pb.append(0)


frc =  max(enumerate(pb), key=itemgetter(1))[0]
print(frc)

fdst = []
for i in range (len(spc_wrd)):
    wd = spc_wrd[i]
    m = 0
    for j in range(len(sd)):
        oy = sd[j]
        if wd in oy:
            m = m +1
    fdst.append(m)





# In[36]:

import matplotlib.pyplot as pyplot
import matplotlib.cm as cm

  
x_list = fdst
label_list = spc_wrd


pyplot.axis("equal")
cs=cm.Set1(np.arange(22)/22.)
pyplot.pie(
        x_list,
        labels=label_list,
        colors = cs, 
        autopct="%1.1f%%"
        )
pyplot.title("Distribution of Terms for Rating of Restaurants (1 star)")
pyplot.show()
    


# In[75]:

fsd = []
for i in range (len(five)):
    fsd.append(five[i])
    


# In[38]:

v = 0        
pb = []
trash =[]
for i in range(len(fsd)):
    op = fsd[i]
    cb = []
    cp = []
    #lim = int((len(spc_wrd))/2)
    for j in range(len(spc_wrd)):
        gf = spc_wrd[j]
        c = 0
        for k in range(len(op)):
            if op[k] == gf:
                c = c +1
        if c != 0:
            cb.append(c)
            cp.append(sp_pp[j])
        else:
            cb.append(0)
            cp.append(0)
    
    bn=[]
    hsm = sum(cb)
    if (hsm != 0):
        for l in range(len(cb)):
            bn.append(cb[l]/hsm)
    
    r = 1
    flag = 0
    for n in range (len(cb)):
        if cb[n] != 0 and cp[n] != 0:
            r =r *  bn[n] *cp[n]
            flag = 1
        else:
            flag = 2

    if flag == 1:
        pb.append(r)
    elif flag == 2:
        pb.append(0)


frc =  max(enumerate(pb), key=itemgetter(1))[0]
print(frc)

fdst1 = []
for i in range (len(spc_wrd)):
    wd = spc_wrd[i]
    m = 0
    for j in range(len(fsd)):
        oy = fsd[j]
        if wd in oy:
            m = m +1
    fdst1.append(m)


# In[ ]:

import matplotlib.pyplot as pyplot
import matplotlib.cm as cm

  
x_list = fdst1
label_list = spc_wrd


pyplot.axis("equal")
cs=cm.Set1(np.arange(22)/22.)
pyplot.pie(
        x_list,
        labels=label_list,
        colors = cs, 
        autopct="%1.1f%%"
        )
pyplot.title("Distribution of Terms for Rating of Restaurants (1 star)")
pyplot.show()
    


# In[37]:

import matplotlib.pyplot as pyplot


import matplotlib.pyplot as mp


#the_grid = GridSpec(2, 2)

#plt.subplot(the_grid[0, 0], aspect=1)


mp.figure(0)
x_list = fdst
label_list = spc_wrd

pyplot.axis("Equal")
cs=cm.Set1(np.arange(22)/22.)


pyplot.pie(
        x_list,
        labels=label_list,
        colors= cs,
        autopct="%1.1f%%"
        )
pyplot.title("Distribution of Terms for Rating of Restaurants (1 star)")



mp.figure(1) 

x_list = fdst1
label_list = spc_wrd

pyplot.axis("Equal")
cs=cm.Set1(np.arange(22)/22.)
pyplot.pie(
        x_list,
        labels=label_list,
        colors= cs,
        autopct="%1.1f%%"
        )
pyplot.title("Distribution of Terms for Rating of Restaurants (5 star)")


mp.show()


# In[ ]:



