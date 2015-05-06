from gensim import corpora, models, similarities
from gensim.models import hdpmodel, ldamodel
from itertools import izip
import nltk
import json
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import string
from pprint import pprint
import sys
import os
import pickle
import re, math
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib


root = 'C:\\ALL\\trend analysis\\yelp_dataset_challenge_academic_dataset\\'
file_reviews = 'yelp_academic_dataset_review.json'
file_business = 'yelp_academic_dataset_business.json'
seed_mexican = 'seed_mexican.txt'
seed_asian = 'seed_chinese.txt'
review_mexican_time = 'review_mexican.txt'
review_asian_time = 'review_asian.txt'
review_american_time = 'review_american.txt'
seed_american = 'seed_american.txt'
common_topics = 'Common_topics.txt'

data = []
business_data = {}
dict_list_restaurant = []
dict_0 = {}
location = "Charlotte"
catagories_val = []
catagories_val_check = "Restaurants"
b_id_restaurant_list = []
review_city = 'review_city.txt'
train_reviews = 'yelp_academic_dataset_review.json'
train_seed_file1 = 'review_city_train.json'
train_file = 'review_city_train.json'
time = ""

review_data = {}
review_list = []
business_id_list = []
review_text_list = []
business_val = ""
seed_threshold = 0.01
seed_im_threshold = 0.1
seed = []
seed1 = []
seed_improved = []


def get_b_id(filename, location):                          #fetch business ids of restaurants based on location 
    catagories_val = []
    with open(os.path.join(root, filename), 'r') as filedata:
        for line in filedata:
            business_data = json.loads(line)
            for ke,val in business_data.items():
                if ke == "business_id":
                    business_id = val
                if ke == "categories":
                    catagories_val = val
                if ke == "city":
                    if "Restaurants" in catagories_val and val == location:
                        dict_list_restaurant.append(business_data)
                        b_id_restaurant_list.append(business_id)
    filedata.close()
    return b_id_restaurant_list
	
	
def cosine_sim(data_query, data):                           #function to get the cosine similarity between two dictionaries of word
    query_dat = {}
    sim = 0.0
    for item, val in data.items():
        if data_query.has_key(item):
            query_dat.update({item: val})
			
    prod = sum([data_query[x] * data[x] for x in query_dat])                            #coding technique referenced from www.stackoverflow.com
    sum_query = sum([data_query[i]**2 for i in data_query.keys()])
    sum_data = sum([data[i]**2 for i in data.keys()])
    norm = math.sqrt(sum_query) * math.sqrt(sum_data)
	
    if norm == 0:
        return sim
    else:
        sim = float(prod) / norm
    return sim
	
	
def tf_vector(text):
    return Counter(text)
	
	
def plot_trend(dates):                                             # matplotlib referred from http://ubuntuforums.org/showthread.php?t=1906642
    list1 = []
    for ke, val in dates.items():
        list1.append(val)
    dates_in_string = ['2010-01-01', '2010-02-01', '2010-03-01', '2010-04-01', '2010-05-01', '2010-06-01', '2010-07-01', '2010-08-01', '2010-09-01', '2010-10-01', '2010-11-01', '2010-12-01']
    dates_datetime = []
    for d in dates_in_string:
       dates_datetime.append(datetime.datetime.strptime(d, '%Y-%m-%d'))
    dates_float = matplotlib.dates.date2num(dates_datetime)
    return(list1, dates_float)
	
	 
def read_seed(seedfile):
    with open(os.path.join(root, seedfile), 'r') as seeddata:
        for line in seeddata:
	        tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
	        seed1 = tokenizer.tokenize(line)
		return seed1
		
def train_seed(seed_data, datafile, bdatafile, location, time):                         #function to train the initial seed list of products based on the a set of number of reviews
    review_data = {}
    review_list = []
    business_id_list = []
    business_val = ""
    seed2 = []
    seed_ext = []
    seed_ext = seed_data
    seed_data2=[]
    date = []
    stop_words = stopwords.words('english')
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    b_id_restaurant_list = get_b_id(bdatafile, location)
    
    with open(os.path.join(root, datafile), 'r') as myfile:
        for line in myfile:
            review_data = json.loads(line)
            
            for ke, val in review_data.items():
                if ke == "business_id":
                    business_val = val
                if ke == "date":
				    date = val.split('-')
				    
                if ke == "text" and business_val in b_id_restaurant_list and date[0] == time:
                    
                    val = val.lower()
                    
                    tokens = tokenizer.tokenize(val)
                    tokens_filtered = [words for words in tokens if words not in stop_words]
                    filter_dict = tf_vector(tokens_filtered)
                    seed_dict = tf_vector(seed_data)
                    sim = cosine_sim(seed_dict, filter_dict)
                    
                    if sim >= 0.27:
                        tagged = nltk.pos_tag(tokens_filtered)
                        filtered = [wt for (wt, wt2) in tagged if wt2 == 'JJ' or wt2=='VBG' or wt2 == 'RB' or wt2 == 'VB' or wt2 == 'VBD' or wt2 == 'VBN' or wt2 == 'NNS' or wt2 == 'IN' or wt2 == 'CC' or wt2 == 'ADV' or wt2 == 'ADJ' or wt2 == 'VBG' or wt2 == 'VBD' or wt2 == 'MD' or wt2 == 'VB' or wt2 == 'VBP' or wt2 == 'VBZ' or wt2 == 'CD']
                        seed2 = [words for words in tokens_filtered if words not in filtered]
                        seed_ext.extend(seed2)
                        seed_data2 = [words for words in seed_ext if words not in stop_words]
                        seed_ext = list(set(seed_data2))
                        
                    
	myfile.close()
    return seed_ext
	
seed_mex = read_seed(seed_mexican)
seed_trainedmex = train_seed(seed_mex, file_reviews, file_business, location, "2014")
seed_asi = read_seed(seed_asian)
seed_trainedasi = train_seed(seed_asi, file_reviews, file_business, location, "2014")
seed_ame = read_seed(seed_american)
seed_trainedame = train_seed(seed_ame, file_reviews, file_business, location, "2014")

def filter_reviews(dump_file, seed_data, review_file, bdatafile, location, time):       #function to filter the reviews based on cosine similarity between the seed itemset in one location, so that these could be further used to fetch more items
    review_data = {}
    review_list = []
    business_id_list = []
    review_text_list = []
    business_val = ""
    c = 0
    seed_impr = seed_data
    tbest = 0
    m_count = {}
    for x in range(1,13):
        m_count[x] = 0
              
    b_id_restaurant_list = get_b_id(bdatafile, location)
    stop_words = stopwords.words('english')
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    print(b_id_restaurant_list)
    with open(os.path.join(root, review_file), 'r') as filedata:
        
        for line in filedata:
            review_data = json.loads(line)
        
            for ke, val in review_data.items():
                if ke == "business_id":
                    business_val = val
                if ke == "text" and business_val in b_id_restaurant_list:
                    val = val.lower()
                if ke == "date":
				    date = val.split('-')
				    
                if ke == "text" and business_val in b_id_restaurant_list and date[0] == time:
                    tokens = tokenizer.tokenize(val)
                    tokens_filtered = [words for words in tokens if words not in stop_words]
				
                    filter_dict2 = tf_vector(tokens_filtered)
                    seed_dict = tf_vector(seed_impr)
                    sim = cosine_sim(seed_dict, filter_dict2)
                    
                    
                    if sim > tbest:
                        tbest = sim
                    if sim > 0.05 and sim < 0.4:
                        for x in range(1,13):
                            if x == int(date[2]):
                            
                                m_count[x] = m_count[x] +1
                        c = c+1
                        review_text_list.append(tokens_filtered)
                    if sim > 0.4:
                        c = c+1
                        tagged = nltk.pos_tag(tokens_filtered)
                        verb_filtered = [wt for (wt, wt2) in tagged if wt2 == 'JJ' or wt2=='VBG' or wt2 == 'RB' or wt2 == 'VB' or wt2 == 'VBD' or wt2 == 'VBN' or wt2 == 'NNS' or wt2 == 'IN' or wt2 == 'CC' or wt2 == 'ADV' or wt2 == 'ADJ' or wt2 == 'VBG' or wt2 == 'VBD' or wt2 == 'MD' or wt2 == 'VB' or wt2 == 'VBP' or wt2 == 'VBZ' or wt2 == 'CD']
                        tokens_filtered2 = [words for words in tokens_filtered if words not in verb_filtered]
                        review_text_list.append(tokens_filtered)
                        seed_impr.extend(tokens_filtered2)
                        seed_data = list(set(seed_impr))
                        

    output = open(dump_file, 'wb')
    pickle.dump(review_text_list, output, -1)
    output.close()
    filedata.close()
    return(m_count, c)
	
new_mex, count_mex = filter_reviews(review_mexican_time,seed_trainedmex, file_reviews, file_business, location, "2014")
print(new_mex, count_mex)
mex_trends, months_mex = plot_trend(new_mex)
print(mex_trends, months_mex)

new_asi, count_asi = filter_reviews(review_asian_time, seed_trainedasi, file_reviews, file_business, location, "2014")
print(new_asi, count_asi)
asi_trends, months_asi = plot_trend(new_asi)
print(asi_trends, months_asi)

new_ame, count_ame = filter_reviews(review_american_time, seed_trainedame, file_reviews, file_business, location, "2014")
print(new_ame, count_ame)
ame_trends, months_ame = plot_trend(new_ame)
print(ame_trends, months_ame)

plt.plot_date(months_mex, mex_trends, linestyle='-', xdate=True, ydate=False)
plt.plot_date(months_asi, asi_trends, linestyle='-', xdate=True, ydate=False)
plt.plot_date(months_ame, ame_trends, linestyle='-', xdate=True, ydate=False)
plt.show()

def lda_subtopics(filename,file_com, gen_topics):              # LDA generation is based on Gensim package, it generates subtopics on the filtered data set. Reviews which talk only about food items, products are only used to generate and fetch more subtopics.
    i =0
    n=10
    lda_res = []
    filtered = []
    lda_prod = []
    topics_un = []
    topics_un = read_seed(file_com)
    data = pickle.load(open(filename, "rb"))
    dict_corpora = corpora.Dictionary(data)
   
	
    filter_words1 = [p for p, freqn in dict_corpora.dfs.iteritems() if freqn == 5]
    dict_corpora.filter_tokens(filter_words1)
    dict_corpora.compactify()
	
    corpus_file = [dict_corpora.doc2bow(x) for x in data]
  
    lda = models.ldamodel.LdaModel(corpus=corpus_file, num_topics=gen_topics)
    
    for topic in lda.show_topics(num_topics=gen_topics, formatted=False):
        i = i + 1
        
        for p, id in topic:
            
            lda_res.append(dict_corpora[int(id)])
    tagged = nltk.pos_tag(lda_res)
    filtered = [wt for (wt, wt2) in tagged if wt2 == 'JJ' or wt2=='VBG' or wt2 == 'RB' or wt2 == 'VB' or wt2 == 'VBD' or wt2 == 'VBN' or wt2 == 'NNS' or wt2 == 'IN' or wt2 == 'CC' or wt2 == 'ADV' or wt2 == 'ADJ' or wt2 == 'VBG' or wt2 == 'VBD' or wt2 == 'MD' or wt2 == 'VB' or wt2 == 'VBP' or wt2 == 'VBZ' or wt2 == 'CD']
    lda_res = [words for words in lda_res if words not in filtered]
    lda_res = [words for words in lda_res if words not in topics_un]
    lda_prod = list(set(lda_res))
	
    return(lda_prod)
    
			
lda_mexican = lda_subtopics(review_mexican_time,common_topics, 10)
print("********************************************************************************************")
print("popular mexican food items/flavours and products in year 2014 at " + location + "-" )
print(lda_mexican)

lda_asian = lda_subtopics(review_asian_time,common_topics, 10)
print("********************************************************************************************")
print("popular asian food items/flavours and products in year 2014 at  " + location + "-" )
print(lda_asian)

lda_american = lda_subtopics(review_american_time,common_topics, 10)
print("********************************************************************************************")
print("popular american food items/flavours and products in year 2014 at " + location + "-" )
print(lda_american)





