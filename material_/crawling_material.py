# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 22:19:14 2021

@author: user
"""
import datetime
import time
    
# variable declare   
        
repository_column = ['total_index', 'repo_id', 'repo_name', 'owner_id', 'owner_type', 'full_name', 'create_date', 'update_date', 'topics', 'language', 'contributors', 'contributor_counts',  
          'stargazer_counts', 'forker_counts', 'keyword', 'readme_url', 'read_length', 'is_it_forked_repo', 'open_issues', 'original_repo', 'contents',"description"]

user_column = ['total_index', 'user_id', 'user_name', 'repo_list', 'repo_count', 'company', 'email', 'location', 'followers', 'follower_count', 'following', 'following_count', 
               'organization_list', 'contributed_repo_count', 'forked_repo', 'forked_repo_count', 'readme_size', 'url']

keywords = ['user:microsoft user:IBM user:aws user:facebook user:google']



number_of_repos = {'user:microsoft user:IBM user:aws user:facebook user:google' : {'2015-03-01' :50}}


# define method

def data_processing(data, column_list) :
    
    # change list data to string
    for col in column_list :
        data[col] = ['#'.join(map(str, corpus)) for corpus in data[col]] 
    
    return data


def url_organizer(url) :
    return url[0:8] + url[12:23] + url[29:]


def find_owner_type(string) :
    if string == None :
        owner_type = 'user'
    else :
        owner_type = 'organization'
    
    return owner_type

def make_user_id_set(data) :
    owner = list(set(data['owner_id']))
    contributors = list(set(data['contributors']))
    
    users = sorted(list(set(owner+contributors)))
    return users    


def rest(tiredness) :
    print('crawling process get in rest')
    
    tiredness = 0
    #time.sleep(500)
    
    return tiredness


def make_periods_list(start, end) :
    start = datetime.date(int(start.split('-')[0]), int(start.split('-')[1]), int(start.split('-')[2]))
    end = datetime.date(int(end.split('-')[0]), int(end.split('-')[1]), int(end.split('-')[2]))
    time_delta = end - start
        
    date_result = [(start + datetime.timedelta(days=day)).isoformat() for day in range(time_delta.days)]
        
    return date_result
            
    
            
    
    