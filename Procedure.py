'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score


CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    allusers = list(range(dataset.n_users))
    S, sam_time = utils.UniformSample_original(allusers, dataset)
    print(f"BPR[sample time][{sam_time[0]:.1f}={sam_time[1]:.2f}+{sam_time[2]:.2f}]")
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri_s = bpr.stageOne(batch_users, batch_pos, batch_neg, domain_type='source')
        aver_loss += cri_s
        cri_t = bpr.stageOne(batch_users, batch_pos, batch_neg, domain_type='target')
        aver_loss += cri_t
        
    aver_loss = aver_loss / total_batch
    return f"[BPR[aver loss{aver_loss:.3e}]"
    
    
def test_one_batch(X, savepath):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    # print(r)
    f = open(savepath, 'a+')
    for r_ in r:
        f.write(str(r_)+'\n')
    f.close()
    pre, recall, ndcg, hr, ndcg_K = [], [], [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
        hr.append(utils.hit_at_k(r, k))

    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg),
            'hr':np.array(hr)}
        
            
def Test(dataset, Recmodel, epoch, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    savepath = '../result/'+world.config['s_dataset']+'_'+world.config['t_dataset']+'/'
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks)),
               'hr': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []

        neg_test_list = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            neg_test_list= dataset.read_neg_file()
            neg_test_ndarray = np.array(neg_test_list)
            item_list = neg_test_ndarray[batch_users][:]
            item_list = item_list.tolist()
            for i in range(len(batch_users)):
                item_list[i].extend(groundTrue[i])
            item_list = torch.Tensor(item_list).long()
            item_list = item_list.to(world.device)

            # 传入组建好的list，使得最后一位为真实值，在计算topk时固定999为真实值
            rating = Recmodel.getUsersRating(batch_users_gpu, item_list, domain_type='source')
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(torch.full((len(batch_users),1),world.config['neg_item_num'], dtype=torch.int))
        assert total_batch == len(users_list)

        X = zip(rating_list, groundTrue_list)
        # if multicore == 1:
        #     pre_results = pool.map(test_one_batch, X)
        # else:
        #     pre_results = []
        #     for x in X:
        #         pre_results.append(test_one_batch(x, savepath+str(epoch)))
        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x, savepath+str(epoch)))

        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
            results['hr'] += result['hr']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        results['hr'] /= float(len(users))
        
        if multicore == 1:
            pool.close()
    print(results)
    return results
