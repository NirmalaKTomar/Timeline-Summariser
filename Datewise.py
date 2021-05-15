import nltk
import re
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
nltk.download('stopwords')
from datetime import datetime
DOC_ROOT = '/content/docs/'
DEBUG = False
SUMMARY_LENGTH = 3  # number of sentences in final summary
stop_words = set(stopwords.words('english')) 
ideal_sent_length = 20.0

def read_article(f_name):
    filedata = f_name.readlines()
    article = filedata[0].split(". ")
    sentences = []

    for sentence in article:
        print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
    return sentences
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwordords = []
 
    sent1 = [word.lower() for word in sent1]
    sent2 = [word.lower() for word in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
     
    for word in sent1:
        if word in stopwords:
            continue
        vector1[all_words.index(word)] += 1
 
   
    for word in sent2:
        if word in stopwords:
            continue
        vector2[all_words.index(word)] += 1
 
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
     
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for index1 in range(len(sentences)):
        for index2 in range(len(sentences)):
            if index1 == index2: 
                continue 
            similarity_matrix[index1][index2] = sentence_similarity(sentences[index1], sentences[index2], stop_words)

    return similarity_matrix

def generate_summary(f_name, top_n=5):
    summarize_text = []
    sentences =  read_article(f_name)
    #print(sentences)
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)  
    #print("Indexes of top ranked_sentence order are: ",ranked_sentence)   

    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))
    print("\nSummarize Text: \t\n", ". ".join(summarize_text))
    print("\n\n")

    return summarize_text

def datewise(f_name):
             listofdate=[]
             f =  f_name.readlines()  
             content = "" 
             for ele in f: 
                 content += ele  
            # content = fileinlist.read()
             pattern = "\d{4}[/-]\d{2}[/-]\d{2}" 
             dates = re.findall(pattern, content)
             for date in dates:
               if "-" in date:
                 year, month, day  = map(int, date.split("-"))
               else:
                 year, month, day  = map(int, date.split("/"))
               if 1 <= day <= 31 and 1 <= month <= 12:
                 listofdate.append(date)
                 
               return (listofdate)