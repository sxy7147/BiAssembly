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
from data import SAPIENVisionDataset
import method_utils as utils
# from pointnet2_ops.pointnet2_utils import furthest_point_sample
from tensorboardX import SummaryWriter
from method_utils import sample_points_fps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'blender_utils'))
 

def train(conf, train_data_list, val_data_list):
    # create training and validation datasets and data loaders
    data_features = ['imaginary_pc', 'imaginary_disassembly_dir', 'init_pc', 'target_transformation']

    model_def = utils.get_model_module(conf.model_version)
    network = model_def.Network(conf.dir_feat_dim, z_dim=conf.z_dim,
                                lbd_kl=conf.lbd_kl, lbd_dir=conf.lbd_dir, lbd_rot=conf.lbd_rot, lbd_tran=conf.lbd_tran)
    utils.printout(conf.flog, '\n' + str(network) + '\n')

    network_opt = torch.optim.Adam(network.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    network_lr_scheduler = torch.optim.lr_scheduler.StepLR(network_opt, step_size=conf.lr_decay_every, gamma=conf.lr_decay_by)
    network.to(conf.device)
    utils.optimizer_to_device(network_opt, conf.device)

    # create logs
    if not conf.no_console_log:
        header = '     Time    Epoch     Dataset    Iteration     LR      TotalLoss DisasmbLoss KLDisasmbLoss TransRotLoss TransTranLoss KLTransLoss'
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

    ### load data
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
    
    imaginary_pcs = torch.cat(batch[data_features.index('imaginary_pc')], dim=0).float().to(device)    # B * P * N * 3
    imaginary_pcs = imaginary_pcs.reshape(batch_size * 2, -1, 3)
    imaginary_pcs, _, _ = sample_points_fps(imaginary_pcs, conf.num_point_per_shape)
    imaginary_pcs = imaginary_pcs.reshape(batch_size, 2, conf.num_point_per_shape, 3)    # B * P * N * 3

    imaginary_disassembly_dir = torch.tensor(np.array(batch[data_features.index('imaginary_disassembly_dir')])).float().view(batch_size, -1).to(device)
    target_transformation = torch.tensor(np.array(batch[data_features.index('target_transformation')])).float().view(batch_size, -1).to(device)

    # losses = network.forward(imaginary_pcs, imaginary_disassembly_dir, init_pcs, target_transformation)
    losses = network.forward_new(imaginary_pcs, imaginary_disassembly_dir, init_pcs, target_transformation)
    disassembly_loss = losses['disassembly']
    kl_disassembly_loss = losses['kl_disassembly']
    transformation_rot_loss = losses['transformation_rot']
    transformation_tran_loss = losses['transformation_tran']
    kl_transformation_loss = losses['kl_transformation']
    total_loss = losses['tot']

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
                        f'''{disassembly_loss.item():>10.5f}'''
                        f'''{kl_disassembly_loss.item():>10.5f}'''
                        f'''{transformation_rot_loss.item():>10.5f}'''
                        f'''{transformation_tran_loss.item():>10.5f}'''
                        f'''{kl_transformation_loss.item():>10.5f}'''
                        )
        conf.flog.flush()

    # log to tensorboard
    if log_tb and tb_writer is not None:
        tb_writer.add_scalar('total_loss', total_loss.item(), step)
        tb_writer.add_scalar('disassembly_loss', disassembly_loss.item(), step)
        tb_writer.add_scalar('kl_disassembly_loss', kl_disassembly_loss.item(), step)
        tb_writer.add_scalar('transformation_rot_loss', transformation_rot_loss.item(), step)
        tb_writer.add_scalar('transformation_tran_loss', transformation_tran_loss.item(), step)
        tb_writer.add_scalar('kl_transformation_loss', kl_transformation_loss.item(), step)
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

    parser.add_argument('--train_succ_proportion', type=float, nargs='+', default=[1], help='data directory')
    parser.add_argument('--train_fail1_proportion', type=float, nargs='+', default=[0], help='data directory')
    parser.add_argument('--train_fail2_proportion', type=float, nargs='+', default=[0], help='data directory')
    parser.add_argument('--train_fail3_proportion', type=float, nargs='+', default=[0], help='data directory')
    parser.add_argument('--train_fail4_proportion', type=float, nargs='+', default=[0], help='data directory')

    parser.add_argument('--val_succ_proportion', type=float, nargs='+', default=[1], help='data directory')
    parser.add_argument('--val_fail1_proportion', type=float, nargs='+', default=[0], help='data directory')
    parser.add_argument('--val_fail2_proportion', type=float, nargs='+', default=[0], help='data directory')
    parser.add_argument('--val_fail3_proportion', type=float, nargs='+', default=[0], help='data directory')
    parser.add_argument('--val_fail4_proportion', type=float, nargs='+', default=[0], help='data directory')
    
    # main parameters (optional)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
    parser.add_argument('--seed', type=int, default=-1, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--log_dir', type=str, default='../logs/diassembly_predictor', help='exp logs directory')
    parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if exp_dir exists [default: False]')
    parser.add_argument('--resume', action='store_true', default=False, help='resume if exp_dir exists [default: False]')

    # network settings
    parser.add_argument('--num_point_per_shape', type=int, default=2048)
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
    parser.add_argument('--lbd_rot', type=float, default=1.0)
    parser.add_argument('--lbd_tran', type=float, default=1.0)

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
