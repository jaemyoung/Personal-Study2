# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 16:16:42 2022

@author: user
"""

#%%load data
import pandas as pd
data = pd.read_csv("C:/Users/user/Documents/GitHub/개인연구_오픈소스 기반 기업 역량평가/Personal-Study/data/filttered_data_0424.csv")
# PL = python 만 선별
data = data[data["language"] == "Python"].reset_index()

#%% Github token
from github import Github  
# Github Enterprise with custom hostname
g = Github(login_or_token="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") #토큰 입력 
#%% setup.py만 있는 repo 뽑아내기 + setup.py contents + setup.py에 있는 install_requires 가져오기
repo_list = []
require_list = []
decode_list = []
for repo in data["full_name"]:
    try:
        repos = g.get_repo(repo)
        contents = repos.get_contents("")
        for content in contents:
            if content.name == "setup.py":
                setupfile = content.decoded_content.decode('utf-8')
                repo_list.append(repo) # setup.py 가진 repo name
                require_list.append(setupfile) # setup.py의 contents
                decode_list.append(p.findall(setupfile.replace('\n',""))) # install_require 가져오기
    except:
        print("repo not found")
have_setup_data = data[data["full_name"].isin(repo_list)] # isin 함수를 사용하여 setup.py가 있는 repo만을 선별
have_setup_data.to_csv("C:/Users/user/Documents/GitHub/개인연구_오픈소스 기반 기업 역량평가/Personal-Study/data/have_setup_data_865.csv")
#%% setup.py에 있는 install_requires 가져오기
import re
import pandas as pd
have_setup_data = pd.read_csv("C:/Users/user/Documents/GitHub/개인연구_오픈소스 기반 기업 역량평가/Personal-Study/data/have_setup_data_865.csv")
p = re.compile('{}(.*?){}'.format(re.escape('install_requires=['), re.escape('],')),re.MULTILINE) # 특정문자 사이에 존재하는 문자 추출(package 추출)

setupfile = []
require_list = []
for repo in have_setup_data["full_name"]:
    try:
        repos = g.get_repo(repo)
        contents = repos.get_contents("")
        for content in contents:
            if content.name == "setup.py":
                setupfile_content = content.decoded_content.decode('utf-8')
                setupfile.append(setupfile_content) # setup.py의 contents
                require_list.append(p.findall(setupfile.replace('\n',""))) # install_require 가져오기
    except:
        print("error")
    
have_setup_data["setupfile"] = setupfile
have_setup_data["require_list"] = require_list

have_setup_data.to_csv("C:/Users/user/Documents/GitHub/개인연구_오픈소스 기반 기업 역량평가/Personal-Study/data/have_setup_data_865.csv")
#%% 추가 전처리
#데이터 불러오기
have_setup_data = pd.read_csv("C:/Users/user/Documents/GitHub/개인연구_오픈소스 기반 기업 역량평가/Personal-Study/data/have_setup_data_865.csv")
test_pd = have_setup_data["require_list"].replace('[]',None).replace("['']",None).replace('[""]',None).dropna(axis=0).reset_index(drop=False)
test_se = test_pd["require_list"].apply(lambda x : x.replace("['","")).apply(lambda x : x.replace('["',"")).apply(lambda x : x.replace('"]',"")).apply(lambda x : x.replace("']","")).apply(lambda x : x.replace("'",'"'))# []를 없애서 정규표현식 가능하게 하기

#정규표현식으로 '', "" 의 단어들만 추출 -> 문자만 추출 -> 리스트로 저장
p = re.compile('{}(.*?){}'.format(re.escape('"'), re.escape('"')),re.MULTILINE) # 특정문자 사이에 존재하는 문자 추출(package 추출)

#"", '' 안에 있는 단어들만 추출
test_list1 = []
for r_list in test_se:
    test_list1.append(p.findall(r_list))
        
# 영문자 및 특정 특수문자 단어들만 추출
test_list2 = []
for tok in test_list1:
    to = []
    for t in tok:
        to.append(re.sub('[^A-Za-z-_]', '', t.lower())) # 영문자와 특수문자(-,_)만 남기기 ex) opencv_python_headless>=4.1.1.26 -> opencv_python_headless
    test_list2.append(to)
    
# DataFrame에 합치기
test_pd["after_preprocessing"]= test_list2

#전체 리스트로 합쳐서 count 하기  
total_list =[]
for tok in test_list2:
    total_list.extend(tok)
total_list = list(filter(None,total_list))
counter = Counter(total_list)
sorted_by_value = sorted(counter.items(), key=lambda x: x[1], reverse=True) # value값으로 내림차순
output = pd.DataFrame(sorted_by_value,columns = ["LIBRARY","FREQUENCY"])
output.to_csv("C:/Users/user/Documents/GitHub/개인연구_오픈소스 기반 기업 역량평가/Personal-Study/data/sorted_package_821.csv")

#%% co-lib 분석
import numpy
lib_pd = output[7:].reset_index(drop=False) #  상위 7개 제외
lib_list = lib_pd["LIBRARY"]
doc_word_list = test_pd["after_preprocessing"]

#Document term Frequency 작성
def apply_dtm(doc_word_list,package_list):
    total_result = []
    for doc_word in doc_word_list:
        result = []
        for package in package_list:
            if package in doc_word: #doc_word의 keyword가 있으면 1 아니면 0
                result.append(1)
            else:
                result.append(0)
        total_result.append(result)
    return pd.DataFrame(total_result,columns=package_list).to_numpy() # numpy로 변환까지

dtm = apply_dtm(doc_word_list,lib_list)
# library X library network -> undirected, weight O, cut-off : 동시등장횟수
co_lib_network = pd.DataFrame(np.dot(dtm.T,dtm),columns = lib_list,index = lib_list)

#%% lib-coupling 분석
repo_index = test_pd["index"][:340] # 추후 수정(index 864가 없음)
repo_name = have_setup_data.loc[repo_index]["full_name"].append(pd.Series("test"))
lib_coupling_network = pd.DataFrame(np.dot(dtm,dtm.T),columns = repo_name,index = repo_name)

#%% networkx graph 작성
import networkx as nx
G = nx.Graph(co_lib_network)
G = nx.Graph(lib_coupling_network)

# Gephi file 작성 
nx.write_gexf(G, '0912_lib_coupling_network_341.gexf')

# Association strength를 활용한 정규화 -> 나중에
package_network[7][2]/(sum(dtm[7])*sum(dtm[2]))

