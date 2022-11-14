"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import collections
import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def s_items(self):
        raise NotImplementedError

    @property
    def t_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError


class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self,config = world.config,path="../data/"):
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        self.s_item = 0
        self.t_item = 0
        self.s_dataset = config['s_dataset']
        self.t_dataset = config['t_dataset']
        self.s_path = self.s_dataset + '_' + self.t_dataset
        self.t_path = self.t_dataset + '_' + self.s_dataset
        s_train_file = path + self.s_path + '/train.txt'
        t_train_file = path + self.t_path + '/train.txt'
        s_test_file = path + self.s_path + '/test.txt'
        t_test_file = path + self.t_path + '/test.txt'
        
        self.path = path + self.s_path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        self.n_train, self.n_test = 0, 0
        self.exist_users = []
        self.ratingList = []
        self.train_items, self.test_items = {}, {}

        with open(s_train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split('\t')
                    uid, iid = int(l[0]), int(l[1])
                    if uid in self.train_items.keys():
                        self.train_items[uid].append(iid)
                    else:
                        self.train_items[uid] = [iid]
                    self.n_user = max(self.n_user, uid)
                    self.m_item = max(self.m_item, iid)
        with open(s_test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split('\t')
                    uid, iid = int(l[0]), int(l[1])
                    if uid in self.test_items.keys():
                        self.test_items[uid].append(iid)
                    else:
                        self.test_items[uid] = [iid]
                    self.m_item = max(self.m_item, iid)
        
        self.n_user += 1
        self.s_item = self.m_item + 1

        with open(t_train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split('\t')
                    uid, iid = int(l[0]), int(l[1])+self.s_item
                    self.train_items[uid].append(iid)
                    self.m_item = max(self.m_item, iid)
        with open(t_test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split('\t')
                    uid, iid = int(l[0]), int(l[1])+self.s_item
                    # self.test_items[uid].append(iid)
                    self.m_item = max(self.m_item, iid)

        self.m_item += 1
        self.t_item = self.m_item - self.s_item

        # 下面处理trainUser、trainItem、testUser、testItem

        for i in range(len(self.train_items.keys())):
            trainUser.extend([list(self.train_items.keys())[i]] * len(self.train_items[list(self.train_items.keys())[i]]))
            trainItem.extend(self.train_items[list(self.train_items.keys())[i]])
            self.traindataSize += len(self.train_items[list(self.train_items.keys())[i]])
            # print(len(self.train_items[list(self.train_items.keys())[i]]))
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        # print(len(trainItem))

        for i in range(len(self.test_items.keys())):
            testUser.extend([list(self.test_items.keys())[i]] * len(self.test_items[list(self.test_items.keys())[i]]))
            testItem.extend(self.test_items[list(self.test_items.keys())[i]])
            self.testDataSize += len(self.test_items[list(self.test_items.keys())[i]])
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)



        # with open(train_file) as f:
        #     for l in f.readlines():
        #         if len(l) > 0:
        #             l = l.strip('\n').split(' ')
        #             items = [int(i) for i in l[1:]]
        #             uid = int(l[0])
        #             # trainUniqueUsers.append(uid)
        #             trainUser.extend([uid] * len(items))
        #             trainItem.extend(items)
        #             self.m_item = max(self.m_item, max(items))
        #             self.n_user = max(self.n_user, uid)
        #             self.traindataSize += len(items)
        # # self.trainUniqueUsers = np.array(trainUniqueUsers)
        # self.trainUser = np.array(trainUser)
        # self.trainItem = np.array(trainItem)

        # with open(test_file) as f:
        #     for l in f.readlines():
        #         if len(l) > 0:
        #             l = l.strip('\n').split(' ')
        #             items = [int(i) for i in l[1:]]
        #             uid = int(l[0])
        #             # testUniqueUsers.append(uid)
        #             testUser.extend([uid] * len(items))
        #             testItem.extend(items)
        #             self.m_item = max(self.m_item, max(items))
        #             self.n_user = max(self.n_user, uid)
        #             self.testDataSize += len(items)
        # self.m_item += 1
        # self.n_user += 1
        # # self.testUniqueUsers = np.array(testUniqueUsers)
        # self.testUser = np.array(testUser)
        # self.testItem = np.array(testItem)
        
        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        # print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        # print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item

    @property
    def s_items(self):
        return self.s_item

    @property
    def t_items(self):
        return self.t_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                # print(adj_mat)
                # print(R)
                # print(adj_mat)
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
            
            # print(self.Graph)
            # print(norm_adj)
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems

    def read_neg_file(self):
        try:
            test_neg = self.path + '/test_neg.txt'
            test_neg_f = open(test_neg ,'r')
        except:
            negativeList = None
            return negativeList
        negativeList = []
        line = test_neg_f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            negatives = []
            for x in arr[1:]:  # arr[0] = (user, pos_item)
                item = int(x)
                negatives.append(item)
            negativeList.append(negatives)
            line = test_neg_f.readline()
        return negativeList