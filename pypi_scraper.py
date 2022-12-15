import pandas as pd
import requests
from requests.exceptions import HTTPError
from bs4 import BeautifulSoup
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


#%% 실행
lib = pd.read_csv("data/sorted_lib_639.csv")
lib = apply_pypi_crawling(lib)
lib.to_csv("data/sorted_lib_639(+description).csv",index=False)

'''
topic 정보 가져오기
url = f"https://pypi.org/project/pandas"
soup = BeautifulSoup(page.content, 'html.parser')
description = soup.find_all("ul",class_= "sidebar-section__classifiers")
description[1].find_all("li")
description_list= description.find_all("p")
'''