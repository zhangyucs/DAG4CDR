'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import world
import torch
from torch import nn, optim
import numpy as np
from torch import log
from dataloader import BasicDataset
from time import time
from model import LightGCN
from model import PairWiseModel
from sklearn.metrics import roc_auc_score
import random
import os
        
class BPRLoss:
    def __init__(self, 
                 recmodel : PairWiseModel, 
                 config : dict):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)
        
    def stageOne(self, users, pos, neg, domain_type):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg, domain_type)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        return loss.cpu().item()

def UniformSample_original(users, dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]


        # 在这里需要分别为两域pos_item匹配对应域的neg_item
        # dataset.s_items
        # dataset.t_items
        # dataset.m_items

        # while True:
        #     negitem = np.random.randint(0, dataset.m_items)
        #     if negitem in posForUser:
        #         continue
        #     else:
        #         break
        
        while True:
            if positem < dataset.s_items:    
                negitem = np.random.randint(0, dataset.s_items)
                if negitem in posForUser:
                    continue
                else:
                    break
            else:
                negitem = np.random.randint(dataset.s_items, dataset.m_items)
                if negitem in posForUser:
                    continue
                else:
                    break
        
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S), [total, sample_time1, sample_time2]

# ===================end samplers==========================
# =====================utils====================================

def set_seed(seed):
    np.random.seed(seed)   
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def getFileName():
    if world.model_name == 'mf':
        file = f"mf-{world.config['latent_dim_rec']}.pth.tar"
    elif world.model_name == 'lgn':
        file = f"lgn-{world.config['s_dataset']}-{world.config['t_dataset']}-{world.config['lightGCN_n_layers']}-{world.config['latent_dim_rec']}.pth.tar"
    return os.path.join(world.FILE_PATH,file)

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result

def find_best_epoch(ndcg_loger, hit_loger, save_log_file = './log.txt'):
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)
    hit_10 = hit[:,-1]
    best_rec_0 = max(hit_10)
    idx = list(hit_10).index(best_rec_0)
    ff = open(save_log_file,'w+')
    print('{:*^40}'.format(world.config['s_dataset']+' part'))
    ff.write('{:*^40}'.format(world.config['s_dataset']+' part')+'\n')
    final_perf = "Best Iter=[%d]@[%s]\t hit=%s, ndcg=%s" % \
                 (idx-1, world.config['topk'], str(['%.4f'%i for i in list(hit[idx])]), str(['%.4f'%i for i in list(ndcgs[idx])]))
    print(final_perf)
    ff.write(final_perf+'\n')
    return final_perf

def early_stopping(log_hr, best_hr, stopping_step , flag_step=100):
    # early stopping strategy:
    # assert expected_order in ['acc', 'dec']

    if log_hr >= best_hr :
        stopping_step = 0
        best_hr = log_hr
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(flag_step, log_hr))
        should_stop = True
    else:
        should_stop = False
    return best_hr, stopping_step, should_stop

# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]
    
    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset : BasicDataset
    r_all = np.zeros((dataset.m_items, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)

def hit_at_k(r, k):
    sum_r = 0.
    for _r in r:
        if np.sum(np.array(_r)[:k]):
            sum_r += 1.
    return sum_r

def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    sum_ndcg = 0.
    for _r in r:
        dcg_max = dcg_at_k(sorted(_r, reverse=True), k, method)
        if not dcg_max:
            sum_ndcg += 0.
        else:
            sum_ndcg += dcg_at_k(_r, k, method) / dcg_max
    return sum_ndcg

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# ====================end Metrics=============================
# =========================================================