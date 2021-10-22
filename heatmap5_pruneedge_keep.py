# Author:Chengxi Li
# AgglomerativeClustering
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
import json
from itertools import combinations
from functools import reduce
import itertools
import functools
from scipy import stats
import os
import glob
import re
import pickle
import hashlib
from numpy import arange
import copy
import random
vectorfile = "seg2vec_30.npy"
vecnpy = np.load(vectorfile, allow_pickle=True)
vecdic = vecnpy.item()
segkeys = list(vecnpy.item().keys())
class Buildcluster(object):
    def __init__(self, clusterdic15,testuser=-1,cluster_distance_threshold=15,cls_num=295):
        # self.tp = "0.2"
        # self.session = 0
        self.cluster_distance_threshold = cluster_distance_threshold
        self.cls_num = cls_num
        self.plot_out = 1
        self.clusterdic = clusterdic15
        self.digest = pd.read_csv("digest_output/connect_cls"+str(cluster_distance_threshold)+"/cluster_digests/cluster_digest_data.csv")
        self.cluster_hashes = {}
        self.axis_hash = {}
        self.axis_hash_pre = {}
        connect_cls = {}
        for cluster_index, segment_labels in self.clusterdic.items():
            cluster_hash = hashlib.sha256(" ".join(segment_labels).encode('utf-8')).hexdigest()
            self.cluster_hashes[cluster_index] = cluster_hash
            connect_cls['cluster_label'+str(cluster_index)] = segment_labels
        connect_cls = dict(sorted(connect_cls.items(),key=lambda x: int(x[0].split('cluster_label')[1])))
        with open('pop5_graph_pruneedge_keep/'+'connect_cls'+str(cls_num)+'.json', 'w') as fp:
            json.dump(connect_cls,fp,indent=4)


    def make_pairs(self,cls):
        for k, v in cls.items():
            for i in range(len(v)-1):
                yield (v[i], v[i + 1])
    def get_matrix(self,pairs,dim):
        m = np.zeros((dim,dim))
        for p in pairs:
            x = p[0]
            y = p[1]
            if x =='start':
                leftindex = 0
            elif x == 'end':
                leftindex = dim-1
            else:
                leftindex = x+1
            if y == 'start':
                rightindex = 0
            elif y =='end':
                rightindex = dim -1
            else:
                rightindex = y+1
            self.axis_hash[p] = (leftindex,rightindex)
            m[leftindex][rightindex] += 1
        for i in range(len(m)):
            row = m[i]
            n = sum(row)
            if n>0:
                m[i] = [f/n for f in row]
        
        return m
    def get_matrix_pre(self,pairs,dim):
        m = np.zeros((dim,dim))
        for p in pairs:
            x = p[0]
            y = p[1]
            leftindex = x
            rightindex = y
            m[leftindex][rightindex] += 1
        for i in range(len(m)):
            row = m[i]
            n = sum(row)
            if n>0:
                m[i] = [f/n for f in row]
        
        return m
   

    def markov(self):
        seg2cls = {}
        for k, v in self.clusterdic.items():
            for s in v:
                if s not in seg2cls:
                    seg2cls[s] = [k]
                else:
                    if k not in  seg2cls[s]:
                        seg2cls[s].append(k)
        self.user2cls = {}
        segcsv = "newSearchseg/alluserseg.csv"
        segdf = pd.read_csv(segcsv)
        stack = []
        for i, item in segdf.iterrows():
            if item['User ID'] not in self.user2cls:
                self.user2cls[item['User ID']] = []
            seg = item['seg']
            if seg in stack: 
                continue
            else:
                curcls = seg2cls[seg]
                assert(len(curcls)==1)
                self.user2cls[item['User ID']].append(int(curcls[0]))
                stack.append(seg)
      
        for k in self.user2cls:
            self.user2cls[k] = ['start']+ self.user2cls[k]+['end']
        self.e2user = {}
        for k,v in self.user2cls.items():
            for i in range(len(v)-1):
                e = (v[i],v[i+1])
                if e not in self.e2user:
                    self.e2user[e] = []
                if k not in self.e2user[e]:
                    self.e2user[e].append(k)
        self.node2user = {}
        for k, v in self.user2cls.items():
            for n in v:
                if n not in self.node2user:
                    self.node2user[n] = []
                if k not in self.node2user[n]:
                    self.node2user[n].append(k)

        pairs = self.make_pairs(self.user2cls)
        pairs = list(pairs)
        gram_dict = {}
        for w1, w2 in pairs:
            if w1 in gram_dict.keys():
                gram_dict[w1].append(w2)
            else:
                gram_dict[w1] = [w2]
        m = self.get_matrix(pairs,len(self.clusterdic)+2)
        edges = list(set(pairs))
        nodes = list(set(self.clusterdic.keys()))
        pop_list = []
        for topthrethold in [5]:
            pop_list.append(topthrethold)
            th_nodes = list(filter(lambda x: len(self.node2user[x])>=topthrethold,nodes))

            fpairs = [p for p in pairs if p[0] in th_nodes and p[1] in th_nodes]

            user2cls_left = {}
            for k, v in self.user2cls.items():
                if k not in user2cls_left:
                    user2cls_left[k] = []
                for eachcls in v:
                    if eachcls in th_nodes:
                        user2cls_left[k].append(eachcls)
            # reget using left nodes
            fpairs = self.make_pairs(user2cls_left)
            pair2user = {}
            for k, v in user2cls_left.items():
                for i in range(len(v)-1):
                    onepair = (v[i], v[i + 1])
                    if onepair not in pair2user:
                        pair2user[onepair]=[]
                    pair2user[onepair].append(k)
            
            edgepop = 2
            fpairs = map(lambda x: x[0],filter(lambda x:len(set(x[1]))>=edgepop,pair2user.items()))
            fpairs = list(fpairs)
            fm = self.get_matrix(fpairs,len(self.clusterdic)+2)
            fnodes = th_nodes
            sendm10 = self.get_matrix_pre(fpairs,len(self.clusterdic))
            np.save("adjmatrix_5pop_2edgepop"+str(self.cluster_distance_threshold), sendm10)
     
def cal_centro(segkeys):
    res = np.zeros((len(segkeys), len(segkeys)))
    for i1 in range(len(segkeys)-1):
        for i2 in range(i1+1, len(segkeys)):
            g1 = segkeys[i1]
            g2 = segkeys[i2]
            e1 = vecdic[g1]
            e2 = vecdic[g2]
            res[i1][i2] = np.sqrt(
                np.sum((np.mean(e1, axis=0)-np.mean(e2, axis=0))**2))
    res += res.T
    return res

def hmodel(X, cls_threshold):
    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(n_clusters=cls_threshold,affinity='precomputed', linkage='average')
    model = model.fit(X)

    preds = model.labels_
    clusterdic = {}
    plot = 0
    for x, y in zip(segkeys, preds):
        if y not in clusterdic:
            clusterdic[y] = [x]
        else:
            clusterdic[y].append(x)
    # print(clusterdic)
    return clusterdic, model

X = cal_centro(segkeys)
xcls_list = []
intradis_list = []
intrastd_list = []
finalsegnum_list = []
inthrethold_dict = {}
for cluster_distance_threshold in [290]:
    xcls_list.append(cluster_distance_threshold)
    clusterdic15, model15 = hmodel(X, cluster_distance_threshold)
    total_people = 20
    bdcluster = Buildcluster(clusterdic15,cluster_distance_threshold=cluster_distance_threshold,cls_num=len(clusterdic15))
    bdcluster.markov()
