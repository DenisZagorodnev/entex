#ТУТ ВСЕ ФУНКЦИИ ПРЕДОБРАБОТКИ, ЧИСТКИ, ДОБАВЛЕНИЯ СУЩНОСТЕЙ, МЕТРИК

import re
import math
from collections import Counter

import nltk
import numpy as np

from pymorphy2 import MorphAnalyzer
from emosent_py.emosent.emosent import get_emoji_sentiment_rank
from nltk import ngrams
from natasha import NamesExtractor, MorphVocab, DatesExtractor, MoneyExtractor, AddrExtractor
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from emosent_py.emosent.emosent import get_emoji_sentiment_rank
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from textblob import TextBlob
from sklearn.metrics import silhouette_score, pairwise_distances, mutual_info_score

#косинусное расстояние между предложениями

def cosine_similarity(line_1, line_2):
    
    WORD = re.compile(r"\w+")
    
    words_1 = WORD.findall(line_1)
    
    line_1 = Counter(words_1)
    
    words_2 = WORD.findall(line_2)
    
    line_2 = Counter(words_2)
    
    inters = set(line_1.keys()) & set(line_2.keys())
    
    num = sum([line_1[x] * line_2[x] for x in inters])

    s_1 = sum([line_1[x] ** 2 for x in list(line_1.keys())])
    
    s_2 = sum([line_2[x] ** 2 for x in list(line_2.keys())])
    
    denom = math.sqrt(s_1) * math.sqrt(s_2)

    if not denom:
        
        return 0.0
    
    else:
        
        return float(num) / denom
    
#среднее значение показателя силуэт по всем кластерам
    
def silhouette_avg(X_train_vect, prediction): 
    
    return silhouette_score(X_train_vect,  np.asarray(prediction))

#самые редкие Н слов корпуса

def get_rare_n_words(corpus, n = None):

    vec = CountVectorizer().fit(corpus)
    
    bag_of_words = vec.transform(corpus)
    
    sum_words = bag_of_words.sum(axis=0) 
    
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items() if len(word) > 3]
    
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    return [list(elem)[0] for elem in words_freq[-n:]]


#самые частые Н слов корпуса

def get_freq_n_words(corpus, n = None):

    vec = CountVectorizer().fit(corpus)
        
    bag_of_words = vec.transform(corpus)
        
    sum_words = bag_of_words.sum(axis=0) 
        
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items() if len(word) > 1]
        
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    return [list(elem)[0] for elem in words_freq[:n]]

#метрика Кульбака-Лейблера

def kl_divergence(line_1, line_2):
    
    WORD = re.compile(r"\w+")
    
    words_1 = WORD.findall(line_1)
    
    line_1 = Counter(words_1)
    
    words_2 = WORD.findall(line_2)
    
    line_2 = Counter(words_2)
    
    div = mutual_info_score(np.asarray(line_1),  np.asarray(line_2))
    

    return div




