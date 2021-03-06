# -*- coding: utf-8 -*-


DOC_ROOT = '/content/docs/'
from zipfile import ZipFile
with ZipFile('/content/docs.zip', 'r') as zip:
  zip.extractall()
  print('Done')

import nltk
import re
import pandas as pd 
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.spatial import distance
from nltk.tokenize import sent_tokenize
from __future__ import division
from itertools import chain
from datetime import datetime
import Datewise as nts
import clustering as clust
import rouge as RF
ar = ['apple.txt','charles_taylor.txt','haiti.txt','steve jobs.txt','Iraq.txt','Libya.txt','Michelle obama.txt','Syria.txt','Yemen.txt','Bill Clinton.txt','fb.txt','ms.txt','trump.txt','MJ.txt']

all_topics=['Apple.Inc','Charles taylor','Haiti','Steve jobs','Iraq','Libya','Michelle obama','Syria','Yemen','Bill Clinton','FB','MS','Trump','MJ']

#generating all articles'timelines-
print("\ngenerating all articles'timelines-\n")
k=[]
for ob in ar:
  with open( DOC_ROOT +ob )as f:
    k.append(nts.datewise(f))
k

#generating all articles'summaries using datewise approach--
print("\ngenerating all articles'summaries using datewise approach--\n")
n=[]
for ob in ar:
    with open( DOC_ROOT +ob )as f:
      n.append(nts.generate_summary(f, 5))
n

#summary list
summary=[(k[i],all_topics[i], n[i]) for i in range(len(k))]
summary.sort()
print("\ntimeline summary list---\n")
summary

Datewise_summary=pd.DataFrame(summary)
Datewise_summary.columns=['Date','Topics','summary']
Datewise_summary

print("-----------------------------------------------------CLUSTERING METHOD------------------------------------------------------------------")

clust_articles=[]
for ob in ar:
  with open( DOC_ROOT +ob )as f:
    line = f.read().replace("\n", " ")
    clust_articles.append(line)
clust_articles

for i in range(len(clust_articles)):
  data=clust_articles[i]
  clust.cluster_all(data)
  print("\n")

print("--------------------------------------------------------------EVALUATION METRICS-----------------------------------------------------------------------------------")

#original articles----
org_art=[]
for ob in ar:
  with open( DOC_ROOT +ob )as f:
    line = f.read().replace("\n", " ")
    org_art.append(line)
org_art

#system_generated_summary for evaluation metrics
gnrtd_summary=n
for i in range(len(gnrtd_summary)):
  gnrtd_summary[i]=' '.join(gnrtd_summary[i])

gnrtd_summary

r = RF.Rouge()
scores_list=[]
#print(r.rouge_l([original article], [system_generated_summary]))#print("\t\t\tPrecision\t recall\t\t\tf-score")
for i in range(len(org_art)):
  scores_list.append(r.rouge_l([org_art[i]],[gnrtd_summary[i]]))
scores = pd.DataFrame(scores_list)

scores.columns = ['Precision','Recall','ROUGE F1-score']
dates=pd.DataFrame(k) 
dates.columns = ['Published Date']
all_art=pd.DataFrame(all_topics) 
all_art.columns = ['Topics']
eval_score= pd.concat([all_art,dates,scores], axis=1, join='inner')
display(eval_score)

max_fscore= eval_score[['Precision','Recall','ROUGE F1-score']].max()
min_fscore=eval_score[['Precision','Recall','ROUGE F1-score']].min()
res= pd.concat([pd.DataFrame(max_fscore),pd.DataFrame(min_fscore)], axis=1, join='inner')
res.columns=['Max','Min']
t=res.transpose()
display(res)
t
