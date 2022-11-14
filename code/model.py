"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
# from signal import SIG_SETMASK
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import utils


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users, domain_type):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg, domain_type):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.num_nodes = self.num_items + self.num_users

        self.num_items_s = self.dataset.s_items
        self.num_items_t = self.dataset.t_items
        self.num_nodes_s = self.num_users + self.num_items_s
        self.num_nodes_t = self.num_users + self.num_items_t

        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']

        # self.embedding_user = torch.nn.Embedding(
        #     num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        # self.embedding_item = torch.nn.Embedding(
        #     num_embeddings=self.num_items, embedding_dim=self.latent_dim)


        ################################## renew ##################################
        self.embedding_user_s = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_user_t = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        ################################## renew ##################################


        if self.config['pretrain'] == 0:

            # nn.init.normal_(self.embedding_user.weight, std=0.1)
            # nn.init.normal_(self.embedding_item.weight, std=0.1)

            nn.init.normal_(self.embedding_user_s.weight, std=0.1)
            nn.init.normal_(self.embedding_user_t.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            # self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            # self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))

            self.embedding_user_s.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_user_t.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()

        # self.Graph = self.dataset.getSparseGraph()


        ################################## renew ##################################
        self.Graph_s = self.dataset.getSparseGraph()
        self.Graph_t = self.dataset.getSparseGraph()
        ################################## renew ##################################


        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob, domain_type):
        if domain_type == 'source':
            if self.A_split:
                graph = []
                for g in self.Graph_s:
                    graph.append(self.__dropout_x(g, keep_prob))
            else:
                graph = self.__dropout_x(self.Graph_s, keep_prob)
        else:
            if self.A_split:
                graph = []
                for g in self.Graph_t:
                    graph.append(self.__dropout_x(g, keep_prob))
            else:
                graph = self.__dropout_x(self.Graph_t, keep_prob)
        return graph
    
    def computer(self, domain_type):
        """
        propagate methods for lightGCN
        """

        # users_emb = self.embedding_user.weight
        # items_emb = self.embedding_item.weight

        if domain_type == 'source':
            users_emb = self.embedding_user_s.weight
        else:
            users_emb = self.embedding_user_t.weight

        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        # print(world.config['A_type'])
        if self.config['dropout']:
            if self.training:
                # print("droping")
                g_droped = self.__dropout(self.keep_prob, domain_type)
            else:
                if domain_type == 'source':
                    g_droped = self.Graph_s
                else:
                    g_droped = self.Graph_t
        else:
            if domain_type == 'source':
                g_droped = self.Graph_s
            else:
                g_droped = self.Graph_t
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            if self.config['l2']:
                all_emb = torch.nn.functional.normalize(all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users, item_list, domain_type):
        item_list = item_list.long()
        all_users, all_items = self.computer(domain_type)
        users_emb = all_users[users.long()]
        # items_emb = all_items[item_list.long()]
        items_emb = all_items
        r = torch.ones(1)
        r = []
        # 这里用r做一个张量拼接，以组合全部的rating
        for i in range(np.shape(users_emb)[0]):
            _items_emb = items_emb[item_list[i].long()]
            _rating = self.f(torch.matmul(users_emb[i], _items_emb.t()))
            _rating = _rating.unsqueeze_(0)
            r.append(_rating)
        r = torch.cat(r, dim = 0)

        # rating = self.f(torch.matmul(users_emb, items_emb.t()))

        return r
    
    def getEmbedding(self, users, pos_items, neg_items, domain_type):
        all_users, all_items = self.computer(domain_type)
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        
        if domain_type == 'souece':
            users_emb_ego = self.embedding_user_s(users)
        else:
            users_emb_ego = self.embedding_user_t(users)

        # users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg, domain_type):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long(), domain_type)
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss

    def domain_level_attention(self, domain_type):
        with torch.no_grad():
            if domain_type == 'source':
                nodes_emb = torch.cat([self.embedding_user_s.weight, self.embedding_item.weight], axis=0)
                g_indices = self.Graph_s.indices()
                values = self.Graph_s.values()
            else:
                nodes_emb = torch.cat([self.embedding_user_t.weight, self.embedding_item.weight], axis=0)
                g_indices = self.Graph_t.indices()
                values = self.Graph_t.values()

            all_sim_score = torch.zeros(1, device=world.device)
            all_nodes = torch.arange((self.num_users+self.num_items)/2).long()

            # 在这里建立两组mask
            # user_mask - u_mask
            # item_mask - ~u_mask
            
            u_mask = torch.ones(len(values))

            for i in range(len(g_indices[0])):
                if g_indices[0][i] > self.num_users:
                    u_mask[i] = 0
            u_mask = u_mask.bool()
            indices = g_indices.t()

            u_indices = indices[u_mask].t()
            u_values = values[u_mask]
            i_indices = indices[~u_mask].t()
            i_values = values[~u_mask]
            
            # g_indices = self.Graph.indices()
            Graph_sum = []
            # Different datasets may have different sensitivities to functions that calculate similarity
            funs = [nn.LeakyReLU(), torch.sigmoid]
            for k in range(len(funs)):
                all_sim_score = torch.zeros(1, device=world.device)
                for batch_nodes in utils.minibatch(all_nodes, batch_size=self.config['bpr_batch_size']):
                    if funs[k] == None:
                        batch_sim_score = torch.matmul(nodes_emb[batch_nodes], nodes_emb.t())
                    else:
                        batch_sim_score = funs[k](torch.matmul(nodes_emb[batch_nodes], nodes_emb.t()))
                    batch_indices = (u_indices[0] >= batch_nodes[0]) & (u_indices[0] <= batch_nodes[-1])
                    batch_indices = u_indices[:, batch_indices]
                    batch_sim_score = batch_sim_score[batch_indices[0] - batch_nodes[0], batch_indices[1]]
                    all_sim_score = torch.cat([all_sim_score, batch_sim_score])

                all_sim_score = all_sim_score[1:]
                graph = torch.sparse.softmax(torch.sparse_coo_tensor(u_indices, all_sim_score, self.Graph_s.shape), dim=1).to(world.device).coalesce()
                if k == 0:
                    Graph_sum = graph
                else:
                    Graph_sum = graph + Graph_sum
            u_Graph = (1 / len(funs)) * Graph_sum
            u_Graph = u_Graph.coalesce()

            # 最后这里抽出两个图的indices和values，将两者拼接，重新组合还原回最初形状的图
            all_Graph_indices = torch.cat([u_Graph.indices(), i_indices], dim=1)
            all_Graph_values = torch.cat([u_Graph.values(), i_values])

            if domain_type == 'source':
                self.Graph_s = torch.sparse_coo_tensor(all_Graph_indices, all_Graph_values, self.Graph_s.shape).to(world.device).coalesce()
            else:
                self.Graph_t = torch.sparse_coo_tensor(all_Graph_indices, all_Graph_values, self.Graph_t.shape).to(world.device).coalesce()
    

    # def source_node_level_attention(self):
    #     with torch.no_grad():
    #         # nodes_emb = torch.cat([self.embedding_user.weight, self.embedding_item.weight], axis=0)
    #         nodes_emb = torch.cat([self.embedding_user.weight, self.embedding_item.weight[:self.num_items_s]], axis=0)
    #         all_sim_score = torch.zeros(1, device=world.device)
    #         all_nodes = torch.arange(self.num_users+self.num_items_s).long()

    #         # 在这里建立两组mask
    #         # source_mask - s_mask
    #         # target_mask - ~s_mask
    #         g_indices = self.Graph.indices()
    #         values = self.Graph.values()
    #         s_mask = torch.ones(len(values))

    #         for i in range(len(g_indices[0])):
    #             if g_indices[0][i] >= self.num_users+self.num_items_s or g_indices[1][i] >= self.num_users+self.num_items_s:
    #                 s_mask[i] = 0
    #         s_mask = s_mask.bool()
    #         indices = g_indices.t()

    #         s_indices = indices[s_mask].t()
    #         s_values = values[s_mask]
    #         t_indices = indices[~s_mask].t()
    #         t_values = values[~s_mask]
    #         # g_indices = self.Graph.indices()
    #         Graph_sum = []
    #         # Different datasets may have different sensitivities to functions that calculate similarity
    #         funs = [nn.LeakyReLU(), torch.sigmoid]
    #         for k in range(len(funs)):
    #             all_sim_score = torch.zeros(1, device=world.device)
    #             for batch_nodes in utils.minibatch(all_nodes, batch_size=self.config['bpr_batch_size']):
    #                 if funs[k] == None:
    #                     batch_sim_score = torch.matmul(nodes_emb[batch_nodes], nodes_emb.t())
    #                 else:
    #                     batch_sim_score = funs[k](torch.matmul(nodes_emb[batch_nodes], nodes_emb.t()))
    #                 batch_indices = (s_indices[0] >= batch_nodes[0]) & (s_indices[0] <= batch_nodes[-1])
    #                 batch_indices = s_indices[:, batch_indices]
    #                 batch_sim_score = batch_sim_score[batch_indices[0] - batch_nodes[0], batch_indices[1]]
    #                 all_sim_score = torch.cat([all_sim_score, batch_sim_score])

    #             all_sim_score = all_sim_score[1:]
    #             graph = torch.sparse.softmax(torch.sparse_coo_tensor(s_indices, all_sim_score, self.Graph.shape), dim=1).to(world.device).coalesce()
    #             if k == 0:
    #                 Graph_sum = graph
    #             else:
    #                 Graph_sum = graph + Graph_sum
    #         s_Graph = (1 / len(funs)) * Graph_sum
    #         s_Graph = s_Graph.coalesce()

    #         # 最后这里抽出两个图的indices和values，将两者拼接，重新组合还原回最初形状的图
    #         all_Graph_indices = torch.cat([s_Graph.indices(), t_indices], dim=1)
    #         all_Graph_values = torch.cat([s_Graph.values(), t_values])

    #         self.Graph = torch.sparse_coo_tensor(all_Graph_indices, all_Graph_values, self.Graph.shape).to(world.device).coalesce()


    # def target_node_level_attention(self):
    #     with torch.no_grad():
    #         # nodes_emb = torch.cat([self.embedding_user.weight, self.embedding_item.weight], axis=0)
    #         nodes_emb = torch.cat([self.embedding_user.weight, self.embedding_item.weight[self.num_items_s:]], axis=0)
    #         all_sim_score = torch.zeros(1, device=world.device)
    #         all_nodes = torch.arange(self.num_users+self.num_items_t+self.num_items_s).long()

    #         # 在这里建立两组mask
    #         # source_mask - s_mask
    #         # target_mask - ~s_mask
    #         g_indices = self.Graph.indices()
    #         values = self.Graph.values()
    #         t_mask = torch.ones(len(values))

    #         for i in range(len(g_indices[0])):
    #             # if g_indices[0][i] < self.num_users+self.num_items_s and g_indices[1][i] < self.num_users+self.num_items_s:
    #             if g_indices[0][i] < self.num_users+self.num_items_s:
    #                 t_mask[i] = 0
    #         t_mask = t_mask.bool()
    #         indices = g_indices.t()

    #         t_indices = indices[t_mask].t()
    #         t_values = values[t_mask]
    #         s_indices = indices[~t_mask].t()
    #         s_values = values[~t_mask]

    #         # g_indices = self.Graph.indices()
    #         Graph_sum = []
    #         # Different datasets may have different sensitivities to functions that calculate similarity
    #         funs = [nn.LeakyReLU(), torch.sigmoid]
    #         for k in range(len(funs)):
    #             all_sim_score = torch.zeros(1, device=world.device)
    #             for batch_nodes in utils.minibatch(all_nodes, batch_size=self.config['bpr_batch_size']):
    #                 if funs[k] == None:
    #                     batch_sim_score = torch.matmul(nodes_emb[batch_nodes], nodes_emb.t())
    #                 else:
    #                     batch_sim_score = funs[k](torch.matmul(nodes_emb[batch_nodes], nodes_emb.t()))
    #                 batch_indices = (t_indices[0] >= batch_nodes[0]) & (t_indices[0] <= batch_nodes[-1])
    #                 batch_indices = t_indices[:, batch_indices]
    #                 for i in range(len(batch_indices[0])):
    #                     if batch_indices[0][i] < self.num_users:
    #                         batch_indices[1][i] = batch_indices[1][i] - self.num_items_s
    #                 batch_sim_score = batch_sim_score[batch_indices[0] - batch_nodes[0], batch_indices[1]]
    #                 all_sim_score = torch.cat([all_sim_score, batch_sim_score])

    #             all_sim_score = all_sim_score[1:]
    #             graph = torch.sparse.softmax(torch.sparse_coo_tensor(t_indices, all_sim_score, self.Graph.shape), dim=1).to(world.device).coalesce()
    #             if k == 0:
    #                 Graph_sum = graph
    #             else:
    #                 Graph_sum = graph + Graph_sum
    #         t_Graph = (1 / len(funs)) * Graph_sum
    #         t_Graph = t_Graph.coalesce()

    #         # 最后这里抽出两个图的indices和values，将两者拼接，重新组合还原回最初形状的图
    #         all_Graph_indices = torch.cat([t_Graph.indices(), s_indices], dim=1)
    #         all_Graph_values = torch.cat([t_Graph.values(), s_values])

    #         self.Graph = torch.sparse_coo_tensor(all_Graph_indices, all_Graph_values, self.Graph.shape).to(world.device).coalesce()


    def node_level_attention(self, domain_type):
        with torch.no_grad():
            if domain_type == 'source':
                nodes_emb = torch.cat([self.embedding_user_s.weight, self.embedding_item.weight], axis=0)
                g_indices = self.Graph_s.indices()
            else:
                nodes_emb = torch.cat([self.embedding_user_t.weight, self.embedding_item.weight], axis=0)
                g_indices = self.Graph_t.indices()

            all_sim_score = torch.zeros(1, device=world.device)
            all_nodes = torch.arange(self.num_nodes).long()
            # g_indices = self.Graph.indices()
            Graph_sum = []
            # Different datasets may have different sensitivities to functions that calculate similarity
            funs = [nn.LeakyReLU(), torch.sigmoid]
            for k in range(len(funs)):
                all_sim_score = torch.zeros(1, device=world.device)
                for batch_nodes in utils.minibatch(all_nodes, batch_size=self.config['bpr_batch_size']):
                    if funs[k] == None:
                        batch_sim_score = torch.matmul(nodes_emb[batch_nodes], nodes_emb.t())
                    else:
                        batch_sim_score = funs[k](torch.matmul(nodes_emb[batch_nodes], nodes_emb.t()))
                    batch_indices = (g_indices[0] >= batch_nodes[0]) & (g_indices[0] <= batch_nodes[-1])
                    batch_indices = g_indices[:, batch_indices]
                    batch_sim_score = batch_sim_score[batch_indices[0] - batch_nodes[0], batch_indices[1]]
                    all_sim_score = torch.cat([all_sim_score, batch_sim_score])
                all_sim_score = all_sim_score[1:]
                graph = torch.sparse.softmax(torch.sparse_coo_tensor(g_indices, all_sim_score, self.Graph_s.shape), dim=1).to(world.device).coalesce()
                if k == 0:
                    Graph_sum = graph
                else:
                    Graph_sum = graph + Graph_sum
            if domain_type == 'source':
                self.Graph_s = (1 / len(funs)) * Graph_sum
                self.Graph_s = self.Graph_s.coalesce()
            else:
                self.Graph_t = (1 / len(funs)) * Graph_sum
                self.Graph_t = self.Graph_t.coalesce()


    def forward(self, users, items):
        
        # compute embedding
        all_users, all_items = self.computer()
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma