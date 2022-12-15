# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 16:16:42 2022

@author: user
"""
#%% package
import pickle
import re
import numpy as np
from tqdm import tqdm 
import pandas as pd
import requests
from requests.exceptions import HTTPError
from collections import Counter
from bs4 import BeautifulSoup
import networkx as nx
    
#%% Github token
from github import Github  
g = Github("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") #재명
#g = Github("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") #근수

# 연도별 데이터 분포 plot 그리기
import matplotlib.pyplot as plt
data["year"] = data["update_date"].apply(lambda x : int(x[:4]))

plt.hist(data["year"])
plt.title('Distribution of updated repository by year')
plt.xlabel('year')
plt.ylabel('count')
plt.show()
#%%load data
import pickle
with open('crawled_data/repo_list_56111.pkl', 'rb') as f:
	data = pickle.load(f)
# preprocessing 순서 -> 1. language = python만 2. setup.py없는 repo 삭제, 3. require_list 없는 repo 삭제
data = data[data["language"] == "Python"].reset_index()
data = add_setupfile(data) #  2210 -> 980개, 56111 -> 5196 개
have_setup_data = data.dropna(subset=["setupfile"]) # setup.py가 있는 repo만을 선별
have_requirelist_data = add_requirelist(have_setup_data)# require_list 추가

#picke 저장
with open('data/have_requirelist_data_2054.pkl', 'wb') as f:
	pickle.dump(have_requirelist_data, f, protocol=pickle.HIGHEST_PROTOCOL)
'''    
with open('data/have_requirelist_data_2054.pkl', 'rb') as f:
    have_requirelist_data=pickle.load(f)'''

# 전체 리스트로 합쳐서 lib count 하기  
sorted_lib = make_sorted_lib(have_requirelist_data["require_list"])
sorted_lib.to_csv("data/sorted_1731.csv")

#picke 저장
with open('data/sorted_lib_1731.pkl', 'wb') as f:
	pickle.dump(sorted_lib, f, protocol=pickle.HIGHEST_PROTOCOL)



#%% Function

def add_setupfile(repo_list):
    '''repo_list에서 setup.py가 있는 repo만 찾아 dataframe에 추가''' 
    repo_list["setupfile"] = None # 열 추가
    tiredness = 0

    for idx, repo in enumerate(repo_list["full_name"]):
        try:
            repos = g.get_repo(repo)
            contents = repos.get_contents("")
            for content in contents:
                if content.name == "setup.py":
                    setupfile_content = content.decoded_content.decode('utf-8')
                    repo_list["setupfile"].loc[idx] =setupfile_content
                    print('{0}번 repository setup.py 수집 완료 \t tiredness : {1}'.format(idx, tiredness))
                else:
                    pass

        except:
            print("error")
            
        time.sleep(np.random.random())
        tiredness += 1
        
        if tiredness == 300:
            print('crawling process get in rest')
            tiredness = 0
            time.sleep(100)

    return repo_list


def double_check_lib(lib):
    """pypi에서 검색가능하며 description이 있는지 체크"""
    if lib =="":
        return None
    
    else:
        # pypi에 검색
        url = f"https://pypi.org/project/{lib}"
    
        # url 불러올때 에러 있는지 확인
        try:
            page = requests.get(url)
            page.raise_for_status()
            
        except HTTPError as Err:
            print('HTTP 에러가 발생했습니다.')
            return None
            
        except Exception as Err:
            print('다른 에러가 발생했습니다.')
            return None
        
        else:
            soup = BeautifulSoup(page.content, 'html.parser')
            description = soup.find(class_="project-description")
                    
            if description:
                result = []
                description_list= description.find_all("p")
                for tag in description_list:
                    result.append(tag.text)
                if "UNKNOWN" not in result:
                    print("성공")
                    return lib
                else:
                    print("description have UNKNOWN")
                    return None
            else:
                print("description is None")
                return None
            
def check_lib(lib):
    """pypi에서 검색가능하며 description이 있는지 체크"""
    if lib =="":
        return None
    
    else:
        # pypi에 검색
        url = f"https://pypi.org/project/{lib}"
    
        # url 불러올때 에러 있는지 확인
        try:
            page = requests.get(url)
            page.raise_for_status()
            
        except HTTPError as Err:
            print('HTTP 에러가 발생했습니다.')
            return None
            
        except Exception as Err:
            print('다른 에러가 발생했습니다.')
            return None
        
        else:
            print('성공')
            return lib
            

def add_requirelist(data):
    '''4단계로 나눠 setup.py에 있는 requirement 정보 가져오기'''
    
    p1 = re.compile('{}(.*?){}'.format(re.escape('install_requires=['), re.escape('],')),re.MULTILINE) # 특정문자 사이에 존재하는 문자 추출(package 추출)
    p2 = re.compile('{}(.*?){}'.format(re.escape('"'), re.escape('"')),re.MULTILINE) # 특정문자 사이에 존재하는 문자 추출(package 추출)

    data["require_list"] = data["setupfile"].apply(lambda x : " ".join(p1.findall(x.replace('\n',"").replace("'",'"'))))
    print("1단계 종료")
    data["require_list"] = data["require_list"].apply(lambda x : p2.findall(x))
    print("2단계 종료")
    data["require_list"] = data["require_list"].apply(lambda x : [(re.sub('[^A-Za-z-_]', '', lib.lower())) for lib in x]) # 영문자와 특수문자(-,_)만 남기기 ex) opencv_python_headless>=4.1.1.26 -> opencv_python_headless
    print("3단계 종료")
    data["require_list"] = data["require_list"].apply(lambda x : list(filter(None,[double_check_lib(lib) for lib in x]))) # list 안 library 확인 후 None 제거
    print("4단계 종료")
    
    data["require_list"] = data["require_list"].apply(lambda x :None if not x else x) # dataframe 안 [] -> None으로 처리
    data = data.dropna(axis=0)  # null제거
    
    return data

def make_sorted_lib(have_requirelist_data):
    total_list =[lib for require_list in have_requirelist_data for lib in require_list]
    counter = Counter(total_list)
    sorted_by_value = sorted(counter.items(), key=lambda x: x[1], reverse=True) # value값으로 내림차순
    output = pd.DataFrame(sorted_by_value,columns = ["LIBRARY","FREQUENCY"])
    
    return output


#%% setup.py만 있는 repo 뽑아내기 + setup.py contents + setup.py에 있는 install_requires 가져오기
'''
def preprocessing(data):
    repo_list = []
    require_list = []
    decode_list = []
    
    tiredness = 0
    for idx,repo in enumerate(data["full_name"]):
        try:
            repos = g.get_repo(repo)
            contents = repos.get_contents("")
            for content in contents:
                if content.name == "setup.py":
                    setupfile = content.decoded_content.decode('utf-8')
                    repo_list.append(repo) # setup.py 가진 repo name
                    require_list.append(setupfile) # setup.py의 contents
                    decode_list.append(p.findall(setupfile.replace('\n',""))) # install_require 가져오기
                    print('{0}번 repository setup.py 수집 완료 \t tiredness : {1}'.format(idx, tiredness))  
        except:
            print("repo not found")

        time.sleep(np.random.random())
        tiredness += 1
        
        if tiredness == 300:
            print('crawling process get in rest')
            tiredness = 0
            time.sleep(100)
            
    have_setup_data = data[data["full_name"].isin(repo_list)] # isin 함수를 사용하여 setup.py가 있는 repo만을 선별
    #have_setup_data.to_csv(".csv")

    return have_setup_data'''
