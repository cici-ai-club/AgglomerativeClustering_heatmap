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
# a vector file and we load vector which contains segement and event to vector information
vectorfile = "seg2vec_30.npy"
vecnpy = np.load(vectorfile, allow_pickle=True)
vecdic = vecnpy.item() # key would be segment and value would be the events vectors
segkeys = list(vecnpy.item().keys()) 
class Buildcluster(object):
    def __init__(self, clusterdic15,cluster_distance_threshold=15):
        self.cluster_distance_threshold = cluster_distance_threshold
        self.clusterdic = clusterdic15
        self.axis_hash = {}
        self.axis_hash_pre = {}

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
        # we get a mapping from segments to clusters
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
        # we get the mapping from user to clusters
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
        # we reformated so that each user starts with "start" and ends with end (optional) 
        for k in self.user2cls:
            self.user2cls[k] = ['start']+ self.user2cls[k]+['end']
        self.node2user = {}
        for k, v in self.user2cls.items():
            for n in v:
                if n not in self.node2user:
                    self.node2user[n] = []
                if k not in self.node2user[n]:
                    self.node2user[n].append(k)
        # make before-next pairs from clusters for each user
        pairs = self.make_pairs(self.user2cls)
        pairs = list(pairs)
        # we get adjaency matrix for these clusters
        m = self.get_matrix(pairs,len(self.clusterdic)+2)
        # we get all the edges which connect one cluster to another
        edges = list(set(pairs))
        # we get all the nodes which are the cluster nodes
        nodes = list(set(self.clusterdic.keys()))
        # we prune the graph
        for topthrethold in [5]:
            # we prune the nodes by selecting those nodes contains at least 5 users
            th_nodes = list(filter(lambda x: len(self.node2user[x])>=topthrethold,nodes))
            # we prune the edges for the left nodes
            fpairs = [p for p in pairs if p[0] in th_nodes and p[1] in th_nodes]

            # regenerate adjacency matrix using left nodes
            user2cls_left = {}
            for k, v in self.user2cls.items():
                if k not in user2cls_left:
                    user2cls_left[k] = []
                for eachcls in v:
                    if eachcls in th_nodes:
                        user2cls_left[k].append(eachcls)
            fpairs = self.make_pairs(user2cls_left)
            pair2user = {}
            for k, v in user2cls_left.items():
                for i in range(len(v)-1):
                    onepair = (v[i], v[i + 1])
                    if onepair not in pair2user:
                        pair2user[onepair]=[]
                    pair2user[onepair].append(k)
            
            edgepop = 2
            # we prune the edges even down so that each edge should at least have 2 people
            fpairs = map(lambda x: x[0],filter(lambda x:len(set(x[1]))>=edgepop,pair2user.items()))
            fpairs = list(fpairs)
            # we get final adjacency matrix from the left edges and nodes
            fm = self.get_matrix(fpairs,len(self.clusterdic)+2)
            fnodes = th_nodes
            # get adjacency matrix without start and end
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
            # calculate distance betwwen two segments
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
    return clusterdic, model # we return the the clusters which maps segement to clusters
# we calculate the distance between different segements
X = cal_centro(segkeys)
for cluster_distance_threshold in [290]:
    # we define we would like to get 290 clusters from the AgglomerativeClustering 
    clusterdic15, model15 = hmodel(X, cluster_distance_threshold)
    total_people = 20
    # build clusters and get some basic information from it
    bdcluster = Buildcluster(clusterdic15,cluster_distance_threshold=cluster_distance_threshold)
    # we build markov model from with the clusters
    bdcluster.markov()
