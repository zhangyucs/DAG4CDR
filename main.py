from distutils.command.config import config
import world
import utils
from world import cprint
import torch
import numpy as np
import time
import Procedure
from os.path import join
# from utils import early_stopping, find_best_epoch
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}") 
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")

best_hr, best_ndcg = 0., 0.
stopping_step = 0
ndcg_loger, hit_loger = [], []

for epoch in range(world.TRAIN_epochs):
    print('======================')
    print(f'EPOCH[{epoch}/{world.TRAIN_epochs}]')
    start = time.time()
    if epoch %5 == 0:
        cprint("[TEST]")
        result = Procedure.Test(dataset, Recmodel, epoch, world.config['multicore'])
        if result['hr'][-1] >= best_hr and world.config['save_flag'] == 1:
            cprint(f'[saved]')
            torch.save(Recmodel.state_dict(), weight_file)
        best_hr, stopping_step, should_stop = utils.early_stopping(result['hr'][-1], best_hr,stopping_step, flag_step=15)
        if should_stop == True:
            break
    hit_loger.append(result['hr'])
    ndcg_loger.append(result['ndcg'])

    if world.config['A_type'] == 1:
        # node_level_atttention
        Recmodel.node_level_attention('source')
        Recmodel.node_level_attention('target')
        # Recmodel.source_node_level_attention()
        # Recmodel.target_node_level_attention()
    if world.config['A_type'] == 2:
        # domain_level_attention
        Recmodel.domain_level_attention('source')
        Recmodel.domain_level_attention('target')
    if world.config['A_type'] == 3:
        # node_level_atttention
        Recmodel.node_level_attention('source')
        Recmodel.node_level_attention('target')
        # domain_level_attention
        Recmodel.domain_level_attention('source')
        Recmodel.domain_level_attention('target')

    output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch)

    print(f"[TOTAL TIME] {time.time() - start}")

final_perf_s = utils.find_best_epoch(ndcg_loger, hit_loger, save_log_file = './'+world.config['s_dataset']+'_'+world.config['t_dataset']+'.log')