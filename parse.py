import random
import numpy as np
import numpy.linalg as LA
import networkx as nx
import os
import scipy.sparse as sps
from sklearn.cluster import KMeans


#解析存储在特定文件结构中的图形数据集
def parse_dataset(dir, DS):
    prefix = dir + '/' + DS + '/' + DS
    A = prefix + '_A.txt'
    offsets = np.loadtxt(prefix +'_graph_indicator.txt', dtype=np.int_, delimiter=',') - 1
    offs = np.append([0], np.append(np.where((offsets[1:] - offsets[:-1])>0)[0]+1, len(offsets)))
    labels = np.loadtxt(prefix+'_graph_labels.txt', dtype=np.float64).reshape(-1)
    A_data = np.loadtxt(prefix+'_A.txt', dtype=np.int_, delimiter=',') - 1
    A_mat = sps.csr_matrix((np.ones(A_data.shape[0]), (A_data[:, 0], A_data[:, 1])), dtype=np.int_)

    As = []
    for i in range(1, len(offs)):
        As.append(A_mat[offs[i-1]:offs[i],offs[i-1]:offs[i]])

    am = [np.array(sps.csr_matrix.todense(x.astype(np.float64))) for x in As]
    am_corrected = []
    label_corrected = []
    N = len(am)
    for i in range(N):
        d = sum(am[i], 0)
        if not np.any(d == 0):
            am_corrected.append(am[i])
            label_corrected.append(labels[i])
    return am_corrected, label_corrected


def parse_dataset_node(dir, DS):
    prefix = dir + '/' + DS + '/' + DS
    A = prefix + '_A.txt'
    offsets = np.loadtxt(prefix +'_graph_indicator.txt', dtype=np.int_, delimiter=',') - 1
    offs = np.append([0], np.append(np.where((offsets[1:] - offsets[:-1])>0)[0]+1, len(offsets)))
    labels = np.loadtxt(prefix+'_graph_labels.txt', dtype=np.float64).reshape(-1)
    A_data = np.loadtxt(prefix+'_A.txt', dtype=np.int_, delimiter=',') - 1
    A_mat = sps.csr_matrix((np.ones(A_data.shape[0]), (A_data[:, 0], A_data[:, 1])), dtype=np.int_)
    allnode_labels = np.loadtxt(prefix+'_node_labels.txt', dtype=np.int32).reshape(-1)

    As = []
    for i in range(1, len(offs)):
        As.append(A_mat[offs[i-1]:offs[i],offs[i-1]:offs[i]])

    am = [np.array(sps.csr_matrix.todense(x.astype(np.float64))) for x in As]
    am_corrected = []
    label_corrected = []
    allnode_labels_corrected = []
    N = len(am)
    left = 0
    for i in range(N):
        d = sum(am[i], 0)
        right = left+am[i].shape[0]
        if not np.any(d == 0):
            am_corrected.append(am[i])
            label_corrected.append(labels[i])
            allnode_labels_corrected.append(allnode_labels[left:right])
        left = right
    return am_corrected, label_corrected, allnode_labels_corrected