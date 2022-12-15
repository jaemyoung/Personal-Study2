# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 13:29:15 2022

@author: user
"""
#%% package
import pandas as pd
import requests
from requests.exceptions import HTTPError
from bs4 import BeautifulSoup
import re
import numpy as np
import nltk
from keybert import KeyBERT
from nltk.tokenize import regexp_tokenize
from nltk.stem import WordNetLemmatizer
from keyphrase_vectorizers import KeyphraseCountVectorizer
from sentence_transformers import SentenceTransformer

#%% Data
import pickle
with open('data/co_lib_network.pkl', 'wb') as f:
	pickle.dump(co_lib_network, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/sorted_lib_1731.pkl', 'rb') as f:
	co_lib_network = pickle.load(f)
    
#%% 실행
# lib당 description 정보 추출 후 전처리
with open('data/sorted_library_1731.pkl', 'rb') as f:
	sorted_lib = pickle.load(f)
sorted_lib1 = apply_pypi_crawling(sorted_lib)
sorted_lib2 = preprocessing(sorted_lib1)
sorted_lib3 = add_keywords(sorted_lib2,10) # keybert keyphrase 조
sorted_lib4 = add_embeddingvector(sorted_lib3)
node_features = sorted_lib4.set_index(["LIBRARY"])["embedding_vector"]

#pickle로 저장
with open('data/sorted_library_1731.pkl', 'wb') as f:
	pickle.dump(sorted_lib4, f, protocol=pickle.HIGHEST_PROTOCOL)


#%% function
def pypi_crawling(package):
    """pypi에서 package를 검색했을때 나오는 description 정보 가져오기"""
    url = f"https://pypi.org/project/{package}"
        
    # url 불러올때 에러 있는지 확인
    try:
        page = requests.get(url)
        page.raise_for_status()
        
    except HTTPError as Err:
    	print('HTTP 에러가 발생했습니다.')
        
    except Exception as Err:
    	print('다른 에러가 발생했습니다.')
    
    else:
    	print('성공')
        
    result = []
    soup = BeautifulSoup(page.content, 'html.parser')
    try:
        description = soup.find(class_="project-description")
        description_list= description.find_all("p")
    
        for tag in description_list:
            result.append(tag.text)
    except:
        print("description is NoneType")
        result = []
        
    return result

def apply_pypi_crawling(df):
    """pypi에서 검색 후 해당 library의 description이 없거나 UNKNOWN이면 null처리"""
    df["description"] = ""
    for i, lib in enumerate(df["LIBRARY"]):
        df["description"].loc[i] = pypi_crawling(lib)
    
    df["description"] = df["description"].apply(lambda x : [] if "UNKNOWN" in x else x) # desciption이 UNKNOWN일때 None 처리
    
    return df 


def apply_lemma(tok_list):
    """명사의 원형으로 추출"""
    lemma = WordNetLemmatizer()
    result = []
    for tok in tok_list:
        result.append(lemma.lemmatize(tok, pos='n'))
    return result 


def add_keywords(sorted_lib,n):
    """keyBERT로 description에서 keyword 할당"""
    kw_model = KeyBERT()
    vectorizer = KeyphraseCountVectorizer()
    sorted_lib["keywords"] = sorted_lib["description"].apply(lambda x : kw_model.extract_keywords(x, vectorizer=vectorizer, top_n = n))

    return sorted_lib


def attract_key(list_):
    result = []
    for tuple_ in list_:
        result.append(tuple_[0])
        
    return ", ".join(result)

def add_embeddingvector(sorted_lib):
    """할당된 keyword값에 따른 embedding vector 계산"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sorted_lib["embedding_vector"] = sorted_lib["keywords"].apply(lambda x : model.encode(attract_key(x)))

    return sorted_lib

def preprocessing(sorted_lib):
    sorted_lib["description"] = sorted_lib["description"].apply(lambda x : "".join(x)) # list를 문자열로 변환
    sorted_lib["description"] = sorted_lib["description"].apply(lambda x : nltk.regexp_tokenize(x.lower(),'[A-Za-z]+')) # 영어 소문자 제외 제거
    sorted_lib["description"] = sorted_lib["description"].apply(lambda x : " ".join(apply_lemma(x))) # 단어 원형 복원
    sorted_lib["description"] = sorted_lib["description"].apply(lambda x : None if len(x) < 2 else x) # 빈 문자열 찾아서 None으로 변환
    #sorted_lib = sorted_lib.dropna(axis=0) # null 값제거 
    sorted_lib = sorted_lib.fillna("none value")
    #sorted_lib["description"] = np.where(pd.notnull(sorted_lib["description"]) == True, sorted_lib["description"], sorted_lib["LIBRARY"]*10) # null 값을 library 이름으로 대체

    return sorted_lib



