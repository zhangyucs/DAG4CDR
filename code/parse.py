'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse

# COMMON_USERS

def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int,default=1024,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-6,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=1,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=1,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=400,
                        help="the batch size of users for testing")
    # parser.add_argument('--dataset', type=str,default='gowalla',
    #                     help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')

    parser.add_argument('--s_dataset', type=str, default='cloth',
                        help="available source datasets: []")
    parser.add_argument('--t_dataset', type=str, default='sport',
                        help="available target datasets: []")

    parser.add_argument('--neg_item_num', type=int, default=999,
                        help="negative item num")
    parser.add_argument('--save_flag', type=int, default=1,
                        help="save or not")

    parser.add_argument('--l2', type=int, default=1, help='use l2 normlized')

    parser.add_argument('--A_type', type=int, default=3, help='0 for without, 1 for node, 2 for domain, 3 for all')

    return parser.parse_args()
