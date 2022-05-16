#ТУТ ВСЕ ФУНКЦИИ ПРЕДОБРАБОТКИ, ЧИСТКИ, ДОБАВЛЕНИЯ СУЩНОСТЕЙ, МЕТРИК

import re
import math
from collections import Counter

import nltk
from pymorphy2 import MorphAnalyzer
from nltk import ngrams
from natasha import NamesExtractor, MorphVocab, DatesExtractor, MoneyExtractor, AddrExtractor
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from emosent_py.emosent.emosent import get_emoji_sentiment_rank
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from textblob import TextBlob


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




#подсчет частоты частей речи

def count_part_speech_tag(line):
    
    result = TextBlob(line)
    
    pos = dict.fromkeys(['CC', 'DT', 'IN', 'NN', 'NNS', 
                         'RB', 'TO', 'VB', 'VBG', 'JJ', 
                         'NNP', 'VBD', 'CD', 'VBP', 'PRP', 'VBZ', 'JJR', 'VBN'], 0)

    for elem in result.tags:

        if list(elem)[1] in list(pos.keys()):
            
            pos[list(elem)[1]]  += 1
            
        else:
            
            pos[list(elem)[1]] = 0
        
    return pos

#подсчет количества языков

def count_languages(line):
    
    cnt = []
    
    for elem in line.split(' '):
        
        #print(elem)

        #try: 
    
            b = TextBlob(elem)
    
            cnt.append(str(b.detect_language()))
         
       # except:
            
          #  None
        
    return len(list(set(cnt)))


import chardet


def count_languages_chardet(line):
    
    cnt = []
    
    
    for elem in line.split(' '):
        
        #print(elem)

        #try: 
    
            b = chardet.detect(elem.encode('cp1251'))['language']
    
            cnt.append(b)
         
       # except:
            
          #  None

    for elem in list(set(cnt)):
        
        if elem != None:
            
            if len(elem) > 0:
         
                line += ' FOUNDED_' 
        
                line += elem
    
    return line




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

#вынуть из текста Н-граммы

def sub_ngrams(line, n = 2):
    
    grams = ngrams(line.split(), n)

    return [' '.join(gram) for gram in grams]




#пак стопслов

def get_stopwords_bag(fname = "stopwords_aug"):
    
    with open(fname, "r", encoding="utf-8") as stopwords:
    
        words_pack = stopwords.read().splitlines()
    
    return words_pack




#удалить повторяющиеся технические символы

def rm_extra_symbols(text):
    
    for ch in ['\\','`','*','_','{','}','[',']','(',')','>','#','+','-','!','$','\'', ' ']:
        
        try:
            
            text = re.sub(ch + '+', ch, text)
        
        except:
            
            None

    return text

def rm_special(x):
    
    x = re.sub(r'\\\\/', r' ',  x)
    
    x = re.sub(r'\n', r' ',  x)
    
    x = re.sub(r'«', r' ',  x)
    
    x = re.sub(r'»', r' ',  x)
    
    x = re.sub(r'\'', r' ',  x)
    
    return x




morph = MorphAnalyzer()

morph_vocab = MorphVocab()

#стемминг (eng)

def stemming_porter(line):
    
    porter = PorterStemmer()
    
    return porter.stem(line)

def stemming_lancaster(line):
    
    lancaster = LancasterStemmer()
    
    return lancaster.stem(line)

#лемматизация (eng)

def lemmatize(line):
    
    wordnet_lemmatizer = WordNetLemmatizer()
    
    tokenization = nltk.word_tokenize(line)
    
    return [wordnet_lemmatizer.lemmatize(w) for w in tokenization]
        

#приведение текста к нормальной форме

def pymorphy_preproc(line):

    return [morph.parse(word)[0].normal_form for word in line]

#удаление из текста стоп-слов

def rm_stopwords(line, words_pack):
    
    filtered_line = []
    
    for word in line.split(' '):
        
        if word not in  words_pack:
            
            filtered_line.append(word)
            
    return filtered_line

#найти имя в тексте

def sub_names(line):
    
    line = str(line)
    
    extractor = NamesExtractor(morph_vocab)
    
    matches = extractor(line)
    
    if len([_.fact.as_json for _ in matches]) == 3:
        
        return  line + ' NAME_FOUNDED'
            
    return line

#найти дату в тексте

def sub_dates(line):
    
    line = str(line)
    
    extractor = DatesExtractor(morph_vocab)
    
    matches = extractor(line)
    
    if len([_.fact.as_json for _ in matches]):
        
        return  line + ' DATE_FOUNDED'
    
    return line

#найти деньги в тексте

def sub_money(line):
    
    line = str(line)
    
    extractor = MoneyExtractor(morph_vocab)
    
    matches = extractor(line)
    
    if len([_.fact.as_json for _ in matches]):
        
        return  line + ' MONEY_FOUNDED'
    
    return line
    

#найти адрес в тексте
    
def sub_addr(line):
    
    line = str(line)
    
    extractor = AddrExtractor(morph_vocab)
    
    matches = extractor(line)
    
    if len([_.fact.as_json for _ in matches]):
        
        return  line + ' ADDR_FOUNDED'
            
    return line


def rm_numbers(line):
    
    line = ''.join(i for i in line if not i.isdigit())
    
    return line


def rm_punctuation(line):
    
    #line = line.translate(str.maketrans('', '', string.punctuation))
    
    line = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', line)
    
    return line

def make_lowercase(line):
    
    line = line.lower()
    
    return line

def rm_emojies(line):
    
    pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    
    return pattern.sub(r'', line)

 






#вынуть все смайлики 

def extract_emojies(line):
    
    pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    
    result = pattern.findall(line)
    
    return result

#эмоциональный окрас смайлика 

def get_emojies_emotion(emoj):
    
    try:

        return  dict([(key, get_emoji_sentiment_rank(emoj)[key]) for key in ['negative', 'neutral', 'positive', 'sentiment_score']])
        
    except:
        
        return {'negative' : 0, 'neutral' : 0, 'positive' : 0, 'sentiment_score' : 0}
    
#НАЛИЧИЕ негативных или позитивных смайликов

def find_out_emojies(line):
    
        emojies = extract_emojies(line)
        
        neg = 0
        
        pos = 0
        
        neut = 0
    
        for emoj in emojies:
        
            line_emojies = get_emojies_emotion(emoj)
            
            neg += line_emojies['negative']
            
            pos += line_emojies['positive']
            
            neut += line_emojies['neutral']
            
            
        if neg > 0:
        
            line += ' NEG_EMOJ_FOUNDED'
            
        if pos > 0:
        
            line += ' POS_EMOJ_FOUNDED'
            
        if neut > 0:
        
            line += ' NEUT_EMOJ_FOUNDED'
            
        return line

#НАЛИЧИЕ негативных или позитивных скобочки

def extract_hren(line):
    
    negatives = 0
    
    positives = 0
    
    res = re.findall(r":\)|:\(|:с", line)
    
    for elem in res:
        
        if elem in [':(', ':с']:
            
            negatives += 1
            
        else:
            
            positives += 1
            
    if negatives > 0:
        
        line += ' NEG_BRACK_FOUNDED'
        
    if positives > 0:
        
        line += ' POS_BRACK_FOUNDED'
        
    return line
            

