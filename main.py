# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 10:42:59 2022

@author: user
"""

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import Counter
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
from github import Github

from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from sentence_transformers import SentenceTransformer
g = Github("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") #재명   

import re
import nltk
from keybert import KeyBERT
from nltk.tokenize import regexp_tokenize
from nltk.stem import WordNetLemmatizer

'''
# pickle로 저장
with open('', 'wb') as f: 
	pickle.dump(, f, protocol=pickle.HIGHEST_PROTOCOL)

#pickle로 불러오기
with open('', 'rb') as f:
	 = pickle.load(f)
'''
#%% modularity 별 node feature 정보 저장
##co_library
import pickle
with open('data/sorted_library_1448.pkl', 'rb') as f: 
	node_feature = pickle.load(f)  
co_lib = pd.read_csv("data/co_library_216.csv",index_col=False) # gephi에서 가져온 modularity
node_feature["description"][59]

# modulairty_class 정보 추가 
node_feature = node_feature[node_feature["LIBRARY"].isin(list(co_lib["Id"]))].reset_index(drop= True)
node_feature["modularity_class"] = co_lib["modularity_class"]

lib_class_0 = node_feature[node_feature["modularity_class"] == 0]


##lib_coupling
with open('data/have_requirelist_data_1971.pkl', 'rb') as f: 
	repo_list = pickle.load(f)  
lib_coup = pd.read_csv("data/library_coupling_205.csv", index_col=False)

# modularity_class 정보 추가
repo_list = repo_list[repo_list["full_name"].isin(list(lib_coup["Id"]))].reset_index(drop= True)
repo_list["modularity_class"] = lib_coup["modularity_class"]

result1 = []
for idx, name in enumerate(repo_list["full_name"]):
    repo = g.get_repo(name)
    contents = repo.get_contents("")
    for content in contents:
        if content.name.lower() == "readme.md":
            readme_content_1 = content.decoded_content.decode('utf-8')
            #readme_content_2 = nltk.regexp_tokenize(readme_content_1.lower(),'[A-Za-z]+')
            #readme_content_3 = " ".join(apply_lemma(readme_content_2))
            #repo_list["readme"].loc[idx] =readme_content_3
            print('.md 수집 완료')

        elif content.name.lower() == "readme.rst":
            readme_content_1 = content.decoded_content.decode('utf-8')
            #readme_content_2 = nltk.regexp_tokenize(readme_content_1.lower(),'[A-Za-z]+')
            #readme_content_3 = " ".join(apply_lemma(readme_content_2))
            #repo_list["readme"].loc[idx] = readme_content_3
            print('.rst 수집 완료')
            
        else:
            pass
    
    result1.append(readme_content_1)
#%% readme 파일 정규표현식으로 전처리하고 키워드 추출해야됨     
repo_list["readme"] = result1
test = repo_list["description"]

# 정규표현식으로 전처리 추가(보완 필요)
test.replace('{}(.*?){}'.format(re.escape('['), re.escape(']')), '', regex=True, inplace=True)
test.replace('{}(.*?){}'.format(re.escape('('), re.escape(')')), '', regex=True, inplace=True)
test.replace('{}(.*?){}'.format(re.escape('<'), re.escape('>')), '', regex=True, inplace=True)
test.replace('{}(.*?){}'.format(re.escape('```'), re.escape('```')), '', regex=True, inplace=True)
test = test.apply(lambda x: nltk.regexp_tokenize(x.lower(),'[A-Za-z]+'))
test = test.apply(lambda x: " ".join(apply_lemma(x)))
test[0]
# 키워드 추출

kw_model = KeyBERT()
vectorizer = KeyphraseCountVectorizer() 
b = kw_model.extract_keywords(test[86], vectorizer=vectorizer, top_n = 10)

#%% 사용 X
def apply_lemma(tok_list):
#%% function
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





repo_class_0 = repo_list[repo_list["modularity_class"] == 0]
repo_class



# 각 키워드별 내용 확인하기
label_5 = pd.read_csv("data/label_5.csv")
label_5["keywords"]= label_5["keywords"].apply(lambda x : eval(x))
total_list =[key[0] for keywords_list in label_3["keywords"] for key in keywords_list]
counter = Counter(total_list)
sorted_by_value = sorted(counter.items(), key=lambda x: x[1], reverse=True) # value값으로 내림차순
output = pd.DataFrame(sorted_by_value,columns = ["LIBRARY","FREQUENCY"])
output.to_csv("data/label_3_keywords.csv")
    
    

    
    
    

def add_keywords(sorted_lib,n):
    """keyBERT로 description에서 keyword 할당"""
    kw_model = KeyBERT()
    vectorizer = KeyphraseCountVectorizer()
    b= kw_model.extract_keywords(a, vectorizer=vectorizer, top_n = 10)

    return sorted_lib

def attract_key(list_):
    result = []
    for tuple_ in list_:
        result.append(tuple_[0])
        
    return ", ".join(result)

def add_embeddingvector(sorted_lib):
    """할당된 keyword값에 따른 embedding vector 계산"""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sorted_lib["embedding_vectors"] = sorted_lib["keywords"].apply(lambda x : model.encode(attract_key(x)))

    return sorted_lib

with open('data/sorted_library_1448.pkl', 'rb') as f:
	node= pickle.load(f)
    
    
node_feature = add_keywords(node_feature, 5)
node_feature = add_embeddingvector(node_feature)
node_feature_1["count"]= node_feature_1["keywords"].apply(lambda x : len(x))
new_node_feature= node_feature_1[node_feature_1["count"] ==10]


embedding_vectors = list(node["embedding_vectors"])
kmeans = KMeans(n_clusters=7).fit(embedding_vectors)
clusters_7 = kmeans.labels_
node_feature_key10_emb5["labels_7"] = clusters_7


two_dim_embedded_vectors = TSNE(n_components=2,random_state = 1).fit_transform(embedding_vectors)

fig, ax = plt.subplots(figsize=(16,10))
sns.scatterplot(two_dim_embedded_vectors[:,0], two_dim_embedded_vectors[:,1], hue=clusters_7, palette='deep', ax=ax)
plt.show()





node_feature[node_feature["count"] ==2]
plt.hist(node_feature["count"])
plt.hist(data["year"])
plt.title('Distribution of updated repository by year')
plt.xlabel('year')
plt.ylabel('count')
plt.show()
'''
