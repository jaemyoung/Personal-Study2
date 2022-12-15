
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 13:46:37 2022

@author: user
"""
from github import Github
import numpy as np
import pandas as pd
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import datetime
from dateutil.relativedelta import relativedelta
import pickle

g = Github("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") #재명
g = Github("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") #근수

# pickle 데이터 저장 및 불러오
import pickle
with open('data/co_lib_network.pkl', 'wb') as f:
	pickle.dump(co_lib_network, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('crawled_data/repo_list(machine-learning~)_28875.pkl', 'rb') as f:
	a = pickle.load(f)
#%% 작성중
query = '"machine-learning" OR "artificial-intelligence" OR "natural-language-processing"OR "nlp" OR"deep-learning"'
star = ">5"
period_list = make_period_list("2012-10-01","2022-10-01", months =1)

repo_list_ = crawling_repo(query,star,period_list)
data = crawling_data(repo_list_)
data_ =data_.drop(2210)


with open('crawled_data/repo_list_56111.pkl', 'wb') as f:
	pickle.dump(data_, f, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('crawled_data/repo_list(machine-learning~)_53901.pkl', 'wb') as f:
	pickle.dump(repo_list_, f, protocol=pickle.HIGHEST_PROTOCOL)

#%% function

def make_period_list(start, end, months):
    '''입력된 기간을 원하는 개월로 나눠 start, end 개월 리스트 저장'''
    start = datetime.date(int(start.split('-')[0]), int(start.split('-')[1]), int(start.split('-')[2]))
    end = datetime.date(int(end.split('-')[0]), int(end.split('-')[1]), int(end.split('-')[2]))
    diff_month = (end.year - start.year) * 12 + end.month - start.month
    result= []
    for i in range(int(diff_month/months)):
        before = start
        after = start + relativedelta(months = months)
        start = after
        result.append((before.isoformat()+".."+after.isoformat()))
        
    return result


def crawling_repo(query,star,period_list):
    """query에 맞는 repo 가져오기"""
    repo_list = []; tiredness = 0 ; idx = 0

    for period in tqdm(period_list):
        count_per_iteration = 0
        search_query = query +' created:' + period + ' stars:' + star
        repositories = g.search_repositories(query=search_query,sort="stars",order="desc")
    
        for repo in repositories:
            repo_list.append(repo)
            print('{0} \t period : {1} \t {2}th data crawling out of {3} total data \t tiredness : {4}'.format(idx, period,  repositories.totalCount, count_per_iteration, tiredness))
            count_per_iteration += 1
            idx += 1
            time.sleep(np.random.random())
            tiredness += 1
            
            if tiredness == 300 or tiredness == 600 :
                #save_data(crawled_data,period)
                print('crawling process get in rest')
                tiredness = 0
                time.sleep(100)
                    
            if len(repo_list) in [i for i in range(5000,60000,5000)]: # 5000 마다 저장
                with open('C:/Users/user/Documents/GitHub/[개인연구]/Personal-Study2/crawled_data/repo_list(machine-learning~)_임시저장.pkl', 'wb') as f:
                    pickle.dump(repo_list, f, protocol=pickle.HIGHEST_PROTOCOL)
                print("임시저장 완료")
                
                
    return repo_list

     
# repo에서 meta 정보 가져오기

def crawling_data(repo_list):
    '''repo에서 meta 정보 가져오기'''    
        
   # main
   
    crawled_data = []
    tiredness = 0
    for idx, repo in enumerate(repo_list):
        try :
            row = [idx, repo.full_name, repo.created_at, repo.updated_at, repo.language, repo.stargazers_count, repo.forks_count, repo.description]

        except :
            print("no description or unknown error")
            row = [idx, repo.full_name, repo.created_at, repo.updated_at, repo.language, repo.stargazers_count, repo.forks_count, None] # description 없을때

    
        crawled_data.append(row)
        print('{0}번 repository 수집 완료 \t tiredness : {1}'.format(idx, tiredness))        
        
        time.sleep(np.random.random())
        tiredness += 1
        
        if tiredness == 300:
            print('crawling process get in rest')
            tiredness = 0
            time.sleep(100)
    
    
    repository_column = ["total_index","full_name", "create_date", "update_date", "language", "star", "fork", "description"]
    data = pd.DataFrame(crawled_data, columns=repository_column)
    #data.to_csv('crawled_data/repo_2210.csv', mode='a', index=False)
    return data




#%% package
from github import Github
from material_ import crawling_material
import numpy as np
import pandas as pd
import time
from dateutil.relativedelta import relativedelta


def rest(tiredness) :
    print('crawling process get in rest')
    
    tiredness = 0
    time.sleep(50)
    
    return tiredness

def save_data(repo_list, period):
    pickle
    
    
    
def save_data(crawled_data, period) :
    
    def data_processing(data, column_list) :
    
    # change list data to string
        for col in column_list:
            data[col] = ['#'.join(map(str, corpus)) for corpus in data[col]] 
        return data

    repository_column = ['total_index', 'repo_id', 'repo_name', 'owner_id', 'owner_type', 'full_name', 'create_date', 'update_date', 'topics', 'language', 'contributors', 'contributor_counts',  
          'stargazer_counts', 'forker_counts', 'keyword', 'readme_url', 'read_length', 'is_it_forked_repo', 'open_issues', 'original_repo', 'contents',"description"]
    
    data = data_processing(pd.DataFrame(crawled_data, columns=repository_column), ['topics', 'contributors'])
    data.to_csv('crawled_data/repo' + '_' + str(period)  + '.csv', mode='a', index=False)
    print('csv saved \n')

def make_period_list(start, end, months):
    '''입력된 기간을 원하는 개월로 나눠 start, end 개월 리스트 저장'''
    start = datetime.date(int(start.split('-')[0]), int(start.split('-')[1]), int(start.split('-')[2]))
    end = datetime.date(int(end.split('-')[0]), int(end.split('-')[1]), int(end.split('-')[2]))
    diff_month = (end.year - start.year) * 12 + end.month - start.month
    result= []
    for i in range(int(diff_month/months)):
        before = start
        after = start + relativedelta(months = months)
        start = after
        result.append((before.isoformat()+".."+after.isoformat()))
        
    return result


def find_owner_type(string) :
    if string == None :
        owner_type = 'user'
    else :
        owner_type = 'organization'
    
    return owner_type

def crawling_data(repo, crawled_data, idx, keyword) :
    try :
    	contributors = [contributor.id for contributor in repo.get_contributors()]
    	url = repo.url
    	owner_type = crawling_material.find_owner_type(repo.organization)
    
    	try :
            row = [idx, repo.id, repo.name, repo.owner.id, owner_type, repo.full_name, repo.created_at, repo.updated_at, repo.get_topics(), repo.language, 
               contributors, len(contributors), repo.stargazers_count, repo.forks_count, keyword, crawling_material.url_organizer(url), repo.get_readme().size, repo.fork, 
               repo.open_issues, repo.parent, repo.contents_url,repo.description] # description 추가

    	except :
        	row = [idx, repo.id, repo.name, repo.owner.id, owner_type, repo.full_name, repo.created_at, repo.updated_at, repo.get_topics(), repo.language, 
               contributors, len(contributors), repo.stargazers_count, repo.forks_count, keyword, crawling_material.url_organizer(url), None, repo.fork, 
               repo.open_issues, repo.parent, repo.contents_url,repo.description] # readme 없을 떄
    
    	crawled_data.append(row)
        
    except :
        print('error occur')
    
    return crawled_data
    
#%%
def crawling_data(repo, crawled_data, idx, keyword) :
    try :
    	contributors = [contributor.id for contributor in repo.get_contributors()]
    	url = repo.url
    	owner_type = crawling_material.find_owner_type(repo.organization)
    
    	try :
            row = [idx, repo.id, repo.name, repo.owner.id, owner_type, repo.full_name, repo.created_at, repo.updated_at, repo.get_topics(), repo.language, 
               contributors, len(contributors), repo.stargazers_count, repo.forks_count, keyword, crawling_material.url_organizer(url), repo.get_readme().size, repo.fork, 
               repo.open_issues, repo.parent, repo.contents_url,repo.description] # description 추가

    	except :
        	row = [idx, repo.id, repo.name, repo.owner.id, owner_type, repo.full_name, repo.created_at, repo.updated_at, repo.get_topics(), repo.language, 
               contributors, len(contributors), repo.stargazers_count, repo.forks_count, keyword, crawling_material.url_organizer(url), None, repo.fork, 
               repo.open_issues, repo.parent, repo.contents_url,repo.description] # readme 없을 떄
    
    	crawled_data.append(row)
        
    except :
        print('error occur')
    
    return crawled_data


def crawling_user(user_list) :
    # modify 'repos'
    # can not crawling where user contribute and fork
    
    doc_idx = 0; idx = 0; tiredness = 0; crawled_data = []
    
    for user_id in user_list : 
        user = git.get_user_by_id(user_id)
        repos = [repo for repo in user.get_repos()]
        followers = [follower for follower in user.get_followers()]
        followings = [following for following in user.get_followings()]
        organizations = [organization for organization in user.get_orgs()]
        
        row = [idx, user_id, user.name, repos, len(repos), user.company, user.email, user.location, followers, len(followers), followings, len(followings), organizations, user.contributions,
               user.url]
        
        crawled_data.append(row)
        idx += 1; tiredness += 1
        
        if tiredness == 300 :
            save_data(crawled_data, doc_idx, mode='user')
            tiredness = crawling_material.rest(tiredness)
            doc_idx += 1
        
    
def save_data(crawled_data, year, mode) :
    
    if mode == 'repo' :
        data = crawling_material.data_processing(pd.DataFrame(crawled_data, columns=crawling_material.repository_column), ['topics', 'contributors'])
        data.to_csv('C:/Users/user/Documents/GitHub/GitHub-crawler/crawled_data/repo' + '_' + str(year)  + '.csv', mode='a', index=False)
        print('csv saved \n')
        
    elif mode == 'user' :
        data = crawling_material.data_processing(pd.DataFrame(crawled_data, columns=crawling_material.user_column), ['repo_id', 'followers', 'following', 'organization_list'])
        data.to_csv('C:/Users/user/Documents/GitHub/GitHub-crawler/crawled_data/user' + str(year) + '.csv', index=False)


def search_by_keyword(start_date, end_date, save_point) :
    
    # watchers have error -> print stargazer data 
    # variable declare
    crawled_data = []; tiredness = 0 ; doc_idx = 0; idx = 0
    
    for period in crawling_material.make_periods_list(start_date, end_date) :  
        for keyword in crawling_material.keywords :
            if idx < save_point : 
                #idx += crawling_material.number_of_repos[keyword][period]
                idx += save_point
                break
            
            else :
                try :
                    count_per_iteration = 0
                    query = '+'.join([keyword]) +' created:' + str(period)
                    result = git.search_repositories(query, sort='stars', order='desc')
                    
                    for repo in result :
                        crawled_data = crawling_data(repo, crawled_data, idx, keyword)
                            
                        print('{0} \t keyword : {1}, period : {2} \t {3}th data crawling out of {4} total data \t tiredness : {5}'.format(idx, keyword, period, 
                                                                                                                                              result.totalCount, count_per_iteration, tiredness))
                        count_per_iteration += 1
                        
                        time.sleep(np.random.random())
                        tiredness += 1
                        idx += 1
                        if tiredness == 300 or tiredness == 600 :
                            save_data(crawled_data, start_date[:4], mode='repo')
                            tiredness = crawling_material.rest(tiredness)
                            doc_idx+=1
                            crawled_data.clear()
                except :
                    print('repository does not exist')
                    
    save_data(crawled_data, start_date, mode='repo')


#%%
if __name__ == '__main__' :
   
    # set constant 
    ACCESS_TOKEN = ["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","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","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",
                    "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","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"]
    #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 새롬누나
    #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 민찬
    #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 재명
    #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 현호

    SAVE_POINT = 112
    
    git = Github("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") #7/08 재명 토큰
    git = Github("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") #7/11 새롬 토큰

    # topics 
    # machine-leaning
    # processed : image-processing, deep-learning
    # complete : aritificial-intelligence, autonomous-vehicle, automl, nlp, speech-recognition
    search_by_keyword('2015-06-01','2016-01-01', SAVE_POINT)
    
    
    del git
