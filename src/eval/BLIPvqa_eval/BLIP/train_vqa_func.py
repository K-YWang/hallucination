'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
# import ruamel_yaml as yaml
# try:
#     import ruamel_yaml as yaml
# except ModuleNotFoundError:
#     import ruamel.yaml as yaml

from ruamel.yaml import YAML

import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import sys
sys.path.append("..") 
from models.blip_vqa import blip_vqa
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.vqa_dataset import vqa_collate_fn
from data.utils import save_result


def train(model, data_loader, optimizer, epoch, device):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i,(image, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, weights = image.to(device,non_blocking=True), weights.to(device,non_blocking=True)

        loss = model(image, question, answer, train=True, n=n, weights=weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, device, config) :
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50

    result = []

    if config['inference']=='rank':
        answer_list = data_loader.dataset.answer_list
        answer_candidates = model.tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)
        answer_candidates.input_ids[:,0] = model.tokenizer.bos_token_id

    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device,non_blocking=True)

        if config['inference']=='generate':
            answers = model(image, question, train=False, inference='generate')

            for answer, ques_id in zip(answers, question_id):
                ques_id = int(ques_id.item())
                result.append({"question_id":ques_id, "answer":answer})

        elif config['inference']=='rank':
            answer_ids = model(image, question, answer_candidates, train=False, inference='rank', k_test=config['k_test'])

            for ques_id, answer_id in zip(question_id, answer_ids):
                result.append({"question_id":int(ques_id.item()), "answer":answer_list[answer_id]})

        elif config['inference'] == 'vqa_prob': #pred yes probability
            probs = model(image, question, train=False, inference="vqa_prob")
            for prob, ques_id in zip(probs, question_id):
                result.append({"question_id":int(ques_id.item()), "answer":prob})


    return result


def VQA(evaluate, device, seed, distributed, config, result_dir, output_dir):
    # utils.init_distributed_mode(args)

    device = torch.device(device)

    # fix the seed for reproducibility
    seed = seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating vqa datasets")
    datasets = create_dataset('vqa', config)

    if distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)
    else:
        samplers = [None, None]

    train_loader, test_loader = create_loader(datasets,samplers,
                                              batch_size=[config['batch_size_train'],config['batch_size_test']],
                                              # num_workers=[4,4],is_trains=[True, False],
                                              num_workers=[1,1],is_trains=[True, False],
                                              collate_fns=[vqa_collate_fn,None])
    #### Model ####
    print("Creating model")
    model = blip_vqa(pretrained=config['pretrained'], image_size=config['image_size'],
                       vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])

    model = model.to(device)

    model_without_ddp = model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()
    for epoch in range(0, config['max_epoch']):
        if not evaluate:
            if distributed:
                train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

            train_stats = train(model, train_loader, optimizer, epoch, device)

        else:
            break

        if utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }
            with open(os.path.join(output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(output_dir, 'checkpoint_%02d.pth'%epoch))

        dist.barrier()

    vqa_result = evaluation(model_without_ddp, test_loader, device, config)
    result_file = save_result(vqa_result, result_dir, 'vqa_result')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    return vqa_result


def VQA_main(ann_root,output_dir,inference='vqa_prob'): #annotation path, output path
    config = 'configs/vqa.yaml' #todo config file
    evaluate = True
    device = 'cuda'
    seed = 42
    distributed = False

    # config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
    yaml = YAML(typ='rt')  # 'rt' = round-trip, 保留注释和格式
    with open(config, 'r') as f:
        config = yaml.load(f)


    config['ann_root']=ann_root
    config['inference'] = inference

    result_dir = os.path.join(output_dir, 'result')

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(output_dir, 'config.yaml'), 'w'))

    result = VQA(evaluate, device, seed, distributed, config, result_dir, output_dir)
    return result #list("question_id":0,"answer":yes)
