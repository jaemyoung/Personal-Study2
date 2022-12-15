 # -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 17:06:06 2022

@author: user
"""
    
#%% package

import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import FullBatchLinkGenerator
from stellargraph.layer import GCN, LinkEmbedding

from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, HinSAGE, link_classification, AttentionalAggregator

from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import Attri2VecLinkGenerator, Attri2VecNodeGenerator
from stellargraph.layer import Attri2Vec, link_classification

from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec


from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.model_selection import train_test_split

from stellargraph import globalvar
from stellargraph import datasets
from IPython.display import display, HTML
get_ipython().run_line_magic('matplotlib', 'inline')

import networkx as nx

import multiprocessing
import pickle

import itertools
import numpy as np
#%% 불러오기

with open('data/sorted_library_1731.pkl', 'rb') as f:
	node_features = pickle.load(f)

with open('data/co_lib_network(1731X1731).pkl', 'rb') as f:
	co_lib_network = pickle.load(f)
    
node_features = node_features.set_index(["LIBRARY"])["embedding_vector"]
adj_mat = co_lib_network.applymap(lambda x : 1 if x >= 1 else 0) # 가중치 없는 네트워크로 진행
adj_mat = co_lib_network
# 그래프 정의
G = sg.StellarGraph.from_networkx(nx.from_pandas_adjacency(adj_mat), node_features = node_features)
print(G.info())

#%% 훈련 파라미터

RANDOM_SEED = 1004
layer_sizes = [16,16]
epochs = 500

#%% train, test, final로 edge 분할

# Define an edge splitter on the original graph G:
edge_splitter_test = EdgeSplitter(G)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
# reduced graph G_test with the sampled links removed:
G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
    p=0.1, method="global", keep_connected=True, seed = RANDOM_SEED)

# Define an edge splitter on the reduced graph G_test:
edge_splitter_train = EdgeSplitter(G_test)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test, and obtain the
# reduced graph G_train with the sampled links removed:
G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
    p=0.1, method="global", keep_connected=True)


#%% train, test, final generator와 flow 작성 -> GCN 적용

train_gen = FullBatchLinkGenerator(G_train, method="gcn")
train_flow = train_gen.flow(edge_ids_train, edge_labels_train)

test_gen = FullBatchLinkGenerator(G_test, method="gcn")
test_flow = train_gen.flow(edge_ids_test, edge_labels_test)


#%% 모델 정의
gcn = GCN(layer_sizes=[16, 16], activations=["relu", "relu"], generator=train_gen) # dropout제외
x_inp, x_out = gcn.in_out_tensors()

prediction = LinkEmbedding(activation="softmax", method="ip")(x_out)
prediction = keras.layers.Reshape((-1,))(prediction)

model = keras.Model(inputs=x_inp, outputs=prediction)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss=keras.losses.binary_crossentropy,
    metrics=["binary_accuracy"])


# In[25]: 모델 실행

def activation(train_flow,test_flow):

    init_train_metrics = model.evaluate(train_flow)
    init_test_metrics = model.evaluate(test_flow)
    
    print("\nTrain Set Metrics of the initial (untrained) model:")
    for name, val in zip(model.metrics_names, init_train_metrics):
        print("\t{}: {:0.4f}".format(name, val))
    
    print("\nTest Set Metrics of the initial (untrained) model:")
    for name, val in zip(model.metrics_names, init_test_metrics):
        print("\t{}: {:0.4f}".format(name, val))
    
    
    history = model.fit(train_flow, epochs=epochs, validation_data=test_flow, verbose=2, shuffle=True)
    sg.utils.plot_history(history)
    print("\nMake Plot Completed!")
    train_metrics = model.evaluate(train_flow)
    test_metrics = model.evaluate(test_flow)
    
    print("\nTrain Set Metrics of the trained model:")
    for name, val in zip(model.metrics_names, train_metrics):
        print("\t{}: {:0.4f}".format(name, val))
    
    print("\nTest Set Metrics of the trained model:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

activation(train_flow,test_flow)

#%% final 적용

def make_res_list_word(pred, true, dichotomization, th=0.5): 
    # cut_off 를 기준으로 1 or 0으로 나누었으면 dichotomization = True
    # th(threshold)는 edge가 생겼다고 보는 수치
    
    if dichotomization == True:
        result = np.where(pred > th, 1, 0) - true
    elif dichotomization == False:
        result = np.where(pred > th, 1, 0) - np.where(true >= 1, 1, 0)

    app_edges = dict()
    #van_edges = dict()

    for r, l, score  in zip(result, edge_ids_final, pred):
        if l[0] == l[1]:
            continue
    
        if r == 1:
            #if (l[1],l[0]) not in app_edges:
            app_edges[l] = score
        #elif r == -1:
            #if (l[1],l[0]) not in van_edges:
                #van_edges[l] = score
    
    print('생성된 Edges의 수: ' + str(len(app_edges)/2))
    #print('소멸된 Edges의 수: ' + str(len(van_edges)))
    
    sorted_app_edges = sorted(app_edges.items(), key = lambda item: item[1], reverse = True)
    #sorted_van_edges = sorted(van_edges.items(), key = lambda item: item[1], reverse = False)
    
    app = [sorted_app_edges[i][0] for i in range(0,len(sorted_app_edges),2)]
    #van_50 = [sorted_van_edges[i][0] for i in range(0,50)]
    
    return pd.DataFrame({'app':app})

edge_ids_final = list(itertools.product(adj_mat.columns, adj_mat.columns)) # node 순서쌍 정의
edge_labels_final = adj_mat.to_numpy().ravel() #true -> adj의 value

final_gen = FullBatchLinkGenerator(G, method = 'gcn')
final_flow = final_gen.flow(edge_ids_final, edge_labels_final)

pred = model.predict(final_flow).ravel()
app = make_res_list_word(pred, edge_labels_final, dichotomization = False, th = 0.9)

#app.to_excel('result.xlsx')