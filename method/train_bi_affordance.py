"""
    Train the full model
"""

import os
import time
import sys
import shutil
import random
from time import strftime
from argparse import ArgumentParser
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from PIL import Image
from subprocess import call
from data import SAPIENVisionDataset
import method_utils as utils
# from pointnet2_ops.pointnet2_utils import furthest_point_sample
from tensorboardX import SummaryWriter
from method_utils import sample_points_fps


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'blender_utils'))


def train(conf, train_data_list, val_data_list):
    # create training and validation datasets and data loaders
    data_features = ['init_pc', 'target_pc', 'transformed_disassembly_dir', 'ctpt1', 'ctpt2', 'pickup_dir1', 'pickup_dir2', 'success']
    init_pts_channel = 3
    target_pts_channel = 4

    model_def = utils.get_model_module(conf.model_version)
    network = model_def.Network(conf.feat_dim, conf.cp_feat_dim, conf.dir_feat_dim, z_dim=conf.z_dim, 
                                lbd_kl=conf.lbd_kl, lbd_dir=conf.lbd_dir, pts_channel=init_pts_channel, target_pts_channel=target_pts_channel)
    utils.printout(conf.flog, '\n' + str(network) + '\n')

    network_opt = torch.optim.Adam(network.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    network_lr_scheduler = torch.optim.lr_scheduler.StepLR(network_opt, step_size=conf.lr_decay_every, gamma=conf.lr_decay_by)
    network.to(conf.device)
    utils.optimizer_to_device(network_opt, conf.device)

    # create logs
    if not conf.no_console_log:
        header = '     Time    Epoch     Dataset    Iteration     LR      TotalLoss   C2loss   P2loss    A2loss    C1Loss    P1loss    A1Loss    KL2      Dir2      KL1     Dir1'
    if not conf.no_tb_log:
        # https://github.com/lanpa/tensorboard-pytorch
        train_writer = SummaryWriter(os.path.join(conf.exp_dir, 'tb', 'train'))
        val_writer = SummaryWriter(os.path.join(conf.exp_dir, 'tb', 'val'))


    # load dataset
    train_dataset = SAPIENVisionDataset(data_features, train_test_type='Train', buffer_max_num=conf.train_buffer_max_num,
                                        succ_proportion=conf.train_succ_proportion, fail1_proportion=conf.train_fail1_proportion, fail2_proportion=conf.train_fail2_proportion, fail3_proportion=conf.train_fail3_proportion, fail4_proportion=conf.train_fail4_proportion,
                                        assigned_category=conf.assigned_category, obj_asset_dir=conf.object_asset_dir)
    val_dataset = SAPIENVisionDataset(data_features, train_test_type='Test1', buffer_max_num=conf.val_buffer_max_num,
                                      succ_proportion=conf.val_succ_proportion, fail1_proportion=conf.val_fail1_proportion, fail2_proportion=conf.val_fail2_proportion, fail3_proportion=conf.val_fail3_proportion, fail4_proportion=conf.val_fail4_proportion,
                                      assigned_category=conf.assigned_category, obj_asset_dir=conf.object_asset_dir)
    ### load data for the current epoch
    train_dataset.load_data(train_data_list)
    utils.printout(conf.flog, str(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True,
                                                   pin_memory=True, num_workers=0, drop_last=True,
                                                   collate_fn=utils.collate_feats, worker_init_fn=utils.worker_init_fn)
    val_dataset.load_data(val_data_list)
    utils.printout(conf.flog, str(val_dataset))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=False,
                                                 pin_memory=True, num_workers=0, drop_last=True,
                                                 collate_fn=utils.collate_feats, worker_init_fn=utils.worker_init_fn)
    train_num_batch = len(train_dataloader)
    val_num_batch = len(val_dataloader)
    print('train_num_batch: %d, val_num_batch: %d' % (train_num_batch, val_num_batch))

    last_train_console_log_step, last_val_console_log_step = None, None

    # start training
    start_time = time.time()
    start_epoch = 0

    # train for every epoch
    for epoch in range(start_epoch, conf.epochs):
        ### print log
        if not conf.no_console_log:
            utils.printout(conf.flog, f'training run {conf.exp_name}')
            utils.printout(conf.flog, header)

        train_batches = enumerate(train_dataloader, 0)
        val_batches = enumerate(val_dataloader, 0)

        train_ep_loss, train_cnt = 0, 0
        val_ep_loss, val_cnt = 0, 0
        val_fraction_done = 0.0
        val_batch_ind = -1

        for train_batch_ind, batch in train_batches:
            train_fraction_done = (train_batch_ind + 1) / train_num_batch
            train_step = epoch * train_num_batch + train_batch_ind

            log_console = not conf.no_console_log and (last_train_console_log_step is None or \
                                                       train_step - last_train_console_log_step >= conf.console_log_interval)
            if log_console:
                last_train_console_log_step = train_step

            # save checkpoint
            if epoch % 1 == 0 and train_batch_ind == 0:
                with torch.no_grad():
                    utils.printout(conf.flog, 'Saving checkpoint ...... ')
                    torch.save(network.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-network.pth' % epoch))
                    torch.save(network_opt.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-optimizer.pth' % epoch))
                    torch.save(network_lr_scheduler.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-lr_scheduler.pth' % epoch))
                    utils.printout(conf.flog, 'DONE')

            # set models to training mode
            network.train()
            # forward pass (including logging)
            total_loss = forward(batch=batch, data_features=data_features, network=network, conf=conf, is_val=False, \
                                 step=train_step, epoch=epoch, batch_ind=train_batch_ind, num_batch=train_num_batch, start_time=start_time, \
                                 log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=train_writer, lr=network_opt.param_groups[0]['lr'])

            # optimize one step
            network_opt.zero_grad()
            total_loss.backward()
            network_opt.step()
            network_lr_scheduler.step()

            train_ep_loss += total_loss
            train_cnt += 1

            # validate one batch
            while val_fraction_done <= train_fraction_done and val_batch_ind + 1 < val_num_batch:
                val_batch_ind, val_batch = next(val_batches)

                val_fraction_done = (val_batch_ind + 1) / val_num_batch
                val_step = (epoch + val_fraction_done) * train_num_batch - 1

                log_console = not conf.no_console_log and (last_val_console_log_step is None or \
                                                           val_step - last_val_console_log_step >= conf.console_log_interval)
                if log_console:
                    last_val_console_log_step = val_step

                # set models to evaluation mode
                network.eval()
                with torch.no_grad():
                    # forward pass (including logging)
                    val_loss = forward(batch=val_batch, data_features=data_features, network=network, conf=conf, is_val=True, \
                                 step=val_step, epoch=epoch, batch_ind=val_batch_ind, num_batch=val_num_batch, start_time=start_time, \
                                 log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=val_writer,
                                 lr=network_opt.param_groups[0]['lr'])
                val_ep_loss += val_loss
                val_cnt += 1

        utils.printout(flog, "epoch: %d, total_train_loss: %f, total_val_loss: %f" % (epoch, train_ep_loss / train_cnt, val_ep_loss / val_cnt))



def forward(batch, data_features, network, conf, \
            is_val=False, step=None, epoch=None, batch_ind=0, num_batch=1, start_time=0, \
            log_console=False, log_tb=False, tb_writer=None, lr=None):
    torch.cuda.empty_cache()

    batch_size = conf.batch_size
    
    init_pcs = torch.cat(batch[data_features.index('init_pc')], dim=0).float().to(device)
    init_pcs, _, _ = sample_points_fps(init_pcs, conf.num_point_per_shape)

    target_pcs = torch.cat(batch[data_features.index('target_pc')], dim=0).float().contiguous().to(device)
    target_pcs, _, _ = sample_points_fps(target_pcs, conf.num_point_per_shape)

    transformed_disassembly_dir = torch.tensor(np.array(batch[data_features.index('transformed_disassembly_dir')])).float().view(batch_size, -1).to(device)
    
    ctpt1 = torch.tensor(np.array(batch[data_features.index('ctpt1')])).float().view(batch_size, -1).to(device)
    ctpt2 = torch.tensor(np.array(batch[data_features.index('ctpt2')])).float().view(batch_size, -1).to(device)
    dir1 = torch.tensor(np.array(batch[data_features.index('pickup_dir1')])).float().view(batch_size, -1).to(device)
    dir2 = torch.tensor(np.array(batch[data_features.index('pickup_dir2')])).float().view(batch_size, -1).to(device)
    
    succ_label = torch.tensor(np.array(batch[data_features.index('success')])).float().view(batch_size, -1).to(device)
    succ_mask = (succ_label == 1).long()
    
    target_feats = network.get_target_feats(target_pcs)
    disassembly_dir_feats = network.get_disassembly_dir_feats(transformed_disassembly_dir)
    cp1_feats = network.get_cp_feats(ctpt1)
    cp2_feats = network.get_cp_feats(ctpt2)
    dir1_feats = network.get_dir_feats(dir1)
    dir2_feats = network.get_dir_feats(dir2)
    net1, net2 = network.get_pointnet_feats(init_pcs.clone(), cp1=ctpt1, cp2=ctpt2)
    
    
    # actor2
    actor2_losses = network.get_loss_actor2(target_feats, disassembly_dir_feats, net1, net2, cp1_feats, cp2_feats, dir2, dir1_feats, dir2_feats)
    kl2_loss = torch.mean(actor2_losses['kl'][succ_mask])
    dir2_loss = torch.mean(actor2_losses['dir'][succ_mask])
    actor2_loss = torch.mean(actor2_losses['tot'][succ_mask])
    
    # actor1
    actor1_losses = network.get_loss_actor1(target_feats, disassembly_dir_feats, net1, cp1_feats, dir1, dir1_feats)
    kl1_loss = torch.mean(actor1_losses['kl'][succ_mask])
    dir1_loss = torch.mean(actor1_losses['dir'][succ_mask])
    actor1_loss = torch.mean(actor1_losses['tot'][succ_mask])
    
    # critic2
    critic2_score = network.forward_critic2(target_feats, disassembly_dir_feats, net1, net2, cp1_feats, cp2_feats, dir1_feats, dir2_feats)   # after sigmoid
    critic2_loss = torch.mean(network.get_CE_loss(critic2_score, succ_label))
    
    # affordance2
    aff2_score = network.forward_aff2(target_feats, disassembly_dir_feats, net1, net2, cp1_feats, cp2_feats, dir1_feats)
    with torch.no_grad():
        aff2_gt = network.get_aff2_gt(target_feats, disassembly_dir_feats, net1, net2, cp1_feats, cp2_feats, dir1_feats, rvs=100, topk=10)
    aff2_loss = torch.mean(network.get_L1_loss(aff2_score.view(batch_size, -1), aff2_gt.view(batch_size, -1)))
    
    # critic1
    critic1_score = network.forward_critic1(target_feats, disassembly_dir_feats, net1, cp1_feats, dir1_feats)
    with torch.no_grad():
        critic1_gt = network.get_critic1_gt(target_feats, init_pcs.clone(), disassembly_dir_feats, ctpt1, cp1_feats, dir1_feats, rvs_ctpt=10, rvs_dir=10, topk=10)
    critic1_loss = torch.mean(network.get_L1_loss(critic1_score.view(batch_size, -1), critic1_gt.view(batch_size, -1)))
    
    # affordance1
    aff1_score = network.forward_aff1(target_feats, disassembly_dir_feats, net1, cp1_feats)
    with torch.no_grad():
        aff1_gt = network.get_aff1_gt(target_feats, disassembly_dir_feats, net1, cp1_feats, rvs=100, topk=10)
    aff1_loss = torch.mean(network.get_L1_loss(aff1_score.view(batch_size, -1), aff1_gt.view(batch_size, -1)))


    total_loss = actor2_loss + actor1_loss + conf.lbd_critic*critic2_loss + conf.lbd_critic*aff2_loss + conf.lbd_critic*critic1_loss + conf.lbd_critic*aff1_loss

    # display information
    data_split = 'train'
    if is_val:
        data_split = 'val'

    # log to console
    if log_console:
        utils.printout(conf.flog, \
                        f'''{strftime("%H:%M:%S", time.gmtime(time.time() - start_time)):>9s} '''
                        f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                        f'''{data_split:^10s} '''
                        f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                        f'''{lr:>5.2E} '''
                        f'''{total_loss.item():>10.5f}'''
                        f'''{critic2_loss.item():>10.5f}'''
                        f'''{actor2_loss.item():>10.5f}'''
                        f'''{aff2_loss.item():>10.5f}'''
                        f'''{critic1_loss.item():>10.5f}'''
                        f'''{actor1_loss.item():>10.5f}'''
                        f'''{aff1_loss.item():>10.5f}'''                        
                        f'''{kl2_loss.item():>10.5f}'''
                        f'''{dir2_loss.item():>10.5f}'''
                        f'''{kl1_loss.item():>10.5f}'''
                        f'''{dir1_loss.item():>10.5f}'''                    
                        )
        conf.flog.flush()

    # log to tensorboard
    if log_tb and tb_writer is not None:
        loss_dict = {'total_loss': total_loss, 
                     'critic2_loss': critic2_loss, 'actor2_loss': actor2_loss, 'aff2_loss': aff2_loss, 
                     'critic1_loss': critic1_loss, 'actor1_loss': actor1_loss, 'aff1_loss': aff1_loss, 
                     'kl2_loss': kl2_loss, 'dir2_loss': dir2_loss, 'kl1_loss': kl1_loss, 'dir1_loss': dir1_loss}
        for key, value in loss_dict.items():
            tb_writer.add_scalar(key, value.item(), step)
        tb_writer.add_scalar('lr', lr, step)

    return total_loss


if __name__ == '__main__':
    ### get parameters
    parser = ArgumentParser()

    # main parameters (required)
    parser.add_argument('--exp_suffix', type=str, help='exp suffix')
    parser.add_argument('--model_version', type=str, help='model def file')
    parser.add_argument('--train_data_dir', type=str, nargs='+', help='data directory')
    parser.add_argument('--val_data_dir', type=str, nargs='+', help='data directory')
    parser.add_argument('--assigned_category', type=str, default=None)
    parser.add_argument('--object_asset_dir', type=str, default='../assets/object/everyday2pieces_selected')
    
    parser.add_argument('--train_succ_proportion', type=float, nargs='+', default=[0.5], help='data directory')
    parser.add_argument('--train_fail1_proportion', type=float, nargs='+', default=[0.2], help='data directory')
    parser.add_argument('--train_fail2_proportion', type=float, nargs='+', default=[0.2], help='data directory')
    parser.add_argument('--train_fail3_proportion', type=float, nargs='+', default=[0.1], help='data directory')
    parser.add_argument('--train_fail4_proportion', type=float, nargs='+', default=[0.1], help='data directory')

    parser.add_argument('--val_succ_proportion', type=float, nargs='+', default=[0.5], help='data directory')
    parser.add_argument('--val_fail1_proportion', type=float, nargs='+', default=[0.2], help='data directory')
    parser.add_argument('--val_fail2_proportion', type=float, nargs='+', default=[0.2], help='data directory')
    parser.add_argument('--val_fail3_proportion', type=float, nargs='+', default=[0.1], help='data directory')
    parser.add_argument('--val_fail4_proportion', type=float, nargs='+', default=[0.1], help='data directory')

    # main parameters (optional)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
    parser.add_argument('--seed', type=int, default=-1, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--log_dir', type=str, default='../logs/bi_affordance', help='exp logs directory')
    parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if exp_dir exists [default: False]')
    parser.add_argument('--resume', action='store_true', default=False, help='resume if exp_dir exists [default: False]')

    # network settings
    parser.add_argument('--num_point_per_shape', type=int, default=8192)
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--cp_feat_dim', type=int, default=32)
    parser.add_argument('--dir_feat_dim', type=int, default=32)
    parser.add_argument('--z_dim', type=int, default=10)

    # training parameters
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--train_buffer_max_num', type=int, default=20000)
    parser.add_argument('--val_buffer_max_num', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_decay_by', type=float, default=0.9)
    parser.add_argument('--lr_decay_every', type=float, default=5000)

    # CAVE
    parser.add_argument('--lbd_kl', type=float, default=1.0)
    parser.add_argument('--lbd_dir', type=float, default=1.0)
    parser.add_argument('--lbd_critic', type=float, default=0.1)

    # logging
    parser.add_argument('--no_tb_log', action='store_true', default=False)
    parser.add_argument('--no_console_log', action='store_true', default=False)
    parser.add_argument('--console_log_interval', type=int, default=10, help='number of optimization steps beween console log prints')


    # parse args
    conf = parser.parse_args()

    ### prepare before training
    # make exp_name
    conf.exp_name = f'exp-{conf.model_version}-{conf.exp_suffix}'

    if conf.overwrite and conf.resume:
        raise ValueError('ERROR: cannot specify both --overwrite and --resume!')

    # mkdir exp_dir; ask for overwrite if necessary; or resume
    conf.exp_dir = os.path.join(conf.log_dir, conf.exp_name)
    print('exp_dir: ', conf.exp_dir)
    if os.path.exists(conf.exp_dir):
        if not conf.resume:
            if not conf.overwrite:
                response = input('A training run named "%s" already exists, overwrite? (y/n) ' % conf.exp_name)
                if response != 'y':
                    exit(1)
            shutil.rmtree(conf.exp_dir)
    else:
        if conf.resume:
            raise ValueError('ERROR: no training run named %s to resume!' % conf.exp_name)
    if not conf.resume:
        os.makedirs(conf.exp_dir)
        os.mkdir(os.path.join(conf.exp_dir, 'ckpts'))


    # control randomness
    if conf.seed < 0:
        conf.seed = random.randint(1, 10000)
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)

    # save config
    if not conf.resume:
        torch.save(conf, os.path.join(conf.exp_dir, 'conf.pth'))

    # file log
    if conf.resume:
        flog = open(os.path.join(conf.exp_dir, 'train_log.txt'), 'a+')
    else:
        flog = open(os.path.join(conf.exp_dir, 'train_log.txt'), 'w')
    conf.flog = flog

    # backup command running
    utils.printout(flog, ' '.join(sys.argv) + '\n')
    utils.printout(flog, f'Random Seed: {conf.seed}')

    # set training device
    device = torch.device(conf.device)
    utils.printout(flog, f'Using device: {conf.device}\n')
    conf.device = device
    

    train_data_list = utils.get_data_list(conf.train_data_dir, flog, 'train')
    val_data_list = utils.get_data_list(conf.val_data_dir, flog, 'val')

    ### start training
    train(conf, train_data_list, val_data_list)

    ### before quit
    flog.close()
