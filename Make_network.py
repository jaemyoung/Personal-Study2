# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 13:40:45 2022

@author: user
"""
#%% package
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

#%% data
with open('data/have_requirelist_data_1971.pkl', 'rb') as f:
    have_requirelist_data = pickle.load(f)
    
with open('data/sorted_library_1448.pkl', 'rb') as f:
	sorted_lib = pickle.load(f)

#%% network 구축
lib_list = sorted_lib["LIBRARY"]
repo_lib_list = have_requirelist_data["require_list"]
dtm = apply_dtm(repo_lib_list,lib_list)

# library X library network -> co-lib 분석
co_lib_matrix = np.dot(dtm.T,dtm)
np.fill_diagonal(co_lib_matrix,0,wrap =True) # 대각 성분 0으로 변환
co_lib_network = pd.DataFrame(co_lib_matrix, columns = lib_list,index = lib_list)
co_lib_network_normalize = normalize(co_lib_network,sorted_lib) # 정규화 실행

G1 = nx.Graph(co_lib_network_normalize)

# pickle 저장
with open('data/co_lib_network_normalize(1448X1448).pkl', 'wb') as f:
	pickle.dump(co_lib_network_normalize, f, protocol=pickle.HIGHEST_PROTOCOL)
    
# gephi file 저장
nx.write_gexf(G1, 'Gephi_file/co_lib_network_normalize(1448X1448).gexf') # network 저장

# repo X repo network -> lib-coupling 분석
lib_coupling_matrix = np.dot(dtm,dtm.T)
np.fill_diagonal(lib_coupling_matrix,0,wrap =True) # 대각 성분 0으로 변환
lib_coupling_network = pd.DataFrame(lib_coupling_matrix, columns =  have_requirelist_data["full_name"],index = have_requirelist_data["full_name"])
#lib_coupling_network_normalize = normalize(lib_coupling_network,sorted_lib) # 정규화 실행
#lib_coupling_network1 = lib_coupling_network.applymap(lambda x : 1 if x>=1 else 0)
G2 = nx.Graph(lib_coupling_network) # 중복이 있다고..?

# pickle 저장
with open('data/lib_coupling_network(1971X1971).pkl', 'wb') as f:
	pickle.dump(lib_coupling_network, f, protocol=pickle.HIGHEST_PROTOCOL)
# gephi file 저장
nx.write_gexf(G2, 'Gephi_file/lib_coupling_network(1971X1971).gexf') # network 저장

#%% function
def normalize(co_lib_network,sorted_lib):
    """ Association strength를 활용한 정규화 -> 나중에 논문 참조"""
    result = co_lib_network.copy()
    for i, idx_i in enumerate(co_lib_network.index):
        for j, idx_j in enumerate(co_lib_network.index):
            c_i = sorted_lib["FREQUENCY"][i]
            c_j = sorted_lib["FREQUENCY"][j]
            c_ij = co_lib_network[idx_i][idx_j]
            P_ij = c_ij/(c_i*c_j)
            result[idx_i][idx_j] = P_ij
            
    return result

def apply_dtm(doc_word_list,package_list):
    """Document term Frequency 작성"""
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


#%% Network structure 
"""
#Degree distribution

degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
dmax = max(degree_sequence)

#connectec componnent of G
fig = plt.figure("Degree of a random graph", figsize=(8, 8))
# Create a gridspec for adding subplots of different sizes
axgrid = fig.add_gridspec(5, 4)

ax0 = fig.add_subplot(axgrid[0:3, :])
Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
pos = nx.spring_layout(Gcc, seed=10396953)
nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20)
nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
ax0.set_title("Connected components of G")
ax0.set_axis_off()

# Degree Rank plot

ax1 = fig.add_subplot(axgrid[3:, :2])
ax1.plot(degree_sequence, "b-", marker="o")
ax1.set_title("Degree Rank Plot")
ax1.set_ylabel("Degree")
ax1.set_xlabel("Rank")

# Degree histogram plot 
ax2 = fig.add_subplot(axgrid[3:, 2:])
ax2.bar(*np.unique(degree_sequence, return_counts=True))
ax2.set_title("Degree histogram")
ax2.set_xlabel("Degree")
ax2.set_ylabel("# of Nodes")

fig.tight_layout()
plt.show()

#%% Coefficient
from networkx.algorithms import approximation

approximation.average_clustering(G, trials=10000, seed=100)


#%% connected component

[len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]

largest_cc = max(nx.connected_components(G), key=len) # largest graph
S = [G.subgraph(c).copy() for c in nx.connected_components(G)] # list of sub graph 
nx.draw(S[0]) 

#%% path length

nx.draw(G)
nx.average_shortest_path_length(S[0])
"""