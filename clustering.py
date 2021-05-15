import nltk
import numpy as np
import re
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from scipy.spatial import distance
from nltk.tokenize import sent_tokenize
nltk.download('stopwords') 
nltk.download('punkt')

def cluster_all(s):
  text=str (s)
  sentence = sent_tokenize(text)
  corpus = []
  for i in range(len(sentence)):
    sen = re.sub('[^a-zA-Z]',' ', sentence[i])  
    sen = sen.lower()                            
    sen=sen.split()                         
    sen = ' '.join([i for i in sen if i not in stopwords.words('english')])   
    corpus.append(sen)

  n=500
  all_words = [i.split() for i in corpus]
  model = Word2Vec(all_words, min_count=1,size= n)

  sen_vector=[]
  for i in corpus:
    plus=0
    for j in i.split():
      plus+=model.wv[j]
    plus = plus/len(plus)
    sen_vector.append(plus)
     
    
  n_clusters = 5
  kmeans = KMeans(n_clusters, init = 'k-means++', random_state = 42)
  y_kmeans = kmeans.fit_predict(sen_vector)

  my_list=[]
  for i in range(n_clusters):
     my_dict={}
    
     for j in range(len(y_kmeans)):
        
          if y_kmeans[j]==i:
              my_dict[j] =  distance.euclidean(kmeans.cluster_centers_[i],sen_vector[j])
     min_distance = min(my_dict.values())
     my_list.append(min(my_dict, key=my_dict.get))

                            
  #print(my_list)
  #print(y_kmeans)
  for i in sorted(my_list):
    print(sentence[i])