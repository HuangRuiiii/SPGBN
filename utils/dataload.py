import sys
import os
import pickle
import numpy as np
import scanpy as scanpy
import anndata as ad
from scipy.sparse import coo_matrix

def load_data(path):
    """
    loads data from a pickle file.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
        X_all = data['X_all']
        Y_all = data['Y_all']
        train_indices = data['train_indices']
        test_indices = data['test_indices']
        WO = data['WO']
    return X_all, Y_all, train_indices, test_indices, WO

def preprocess(path = './data/20news-bydate/', save = None, Truncate = None):
    '''
    Output:
        X_all: 2-dim array (shape = word_num * doc_num) records the number of each word appears in each doc;
        Y_all: 1-dim array (length = doc_num) records the label of each doc;
        train_label: 1-dim array (length = train_doc_num);
        test_label: 1-dim array (length = test_doc_num);
        WO: list (length = word_num) records the filtered words.
    Note:
        The function don't filter doc classes.
    '''
    #Load data
    train_data = np.loadtxt(path + 'train.data').astype(int)
    test_data = np.loadtxt(path + 'test.data').astype(int)

    test_data[:,0] = test_data[:,0] + np.max(train_data[:,0])
    train_test = np.vstack((train_data, test_data))

    row = train_test[:,1] - 1  # word index
    col = train_test[:,0] - 1  # doc index
    data = train_test[:,2]  # word count
    X_all = coo_matrix((data, (row, col))).toarray()

    #Load Label
    train_label = np.loadtxt(path + 'train.label').astype(int)
    test_label = np.loadtxt(path + 'test.label').astype(int)
    Y_all = np.hstack((train_label, test_label))

    #Filter according to words
    with open('./data/stop-word-list.txt', 'r') as file:
        stopwords = [line.strip() for line in file.readlines()]
    with open('./data/20news-bydate/vocabulary.txt', 'r') as file:
        WO = [line.strip() for line in file.readlines()]

    # Filter stop words
    dex = [1 if word not in stopwords else 0 for word in WO]
    WO = [word for i, word in enumerate(WO) if dex[i]]
    X_all = X_all[np.array(dex).astype(bool), :]

    # Filter infrequency words
    tmp = np.sum(X_all, axis = 1)
    tmp = (tmp >= 5)
    WO = [word for i, word in enumerate(WO) if tmp[i]]
    X_all = X_all[tmp, :]

    # Truncate the number of words
    if Truncate != None:
        dex = np.argsort(np.sum(X_all, axis = 1))[::-1]
        WO = [WO[i] for i in dex[:Truncate]]
        X_all = X_all[dex[:Truncate], :]
    
    return X_all, Y_all, train_label, test_label, WO