import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
from pdb import set_trace

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print('BASE_DIR:', BASE_DIR)
sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(BASE_DIR, 'vnn'))

from vnn.modules import *
from vnn.dgcnn import DGCNN_New
from vnn.utils import *

# https://github.com/erikwijmans/Pointnet2_PyTorch
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG



class PointNet2SemSegSSG(PointNet2ClassificationSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[3, 32, 32, 64],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=True,
            )
        )
        
        # self.global_module = \
        #     PointnetSAModule(
        #         mlp=[512, 512, 512, 1024], use_xyz=True
        #     )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 3, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, self.hparams['feat_dim'], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.hparams['feat_dim']),
            nn.ReLU(True),
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 使用自适应平均池化汇聚全局特征
        self.global_fc = nn.Sequential(
            nn.Linear(512, 256),  # 输入维度根据具体情况调整
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
        )

    def forward(self, pointcloud):
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.fc_layer(l_features[0])
    
    
    def forward_global(self, pointcloud):
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        
        # xyz, features = self.global_module(xyz, features)
        global_features = self.global_pool(features)
        global_features = global_features.squeeze(dim=-1)
        global_features = self.global_fc(global_features)

        # return features.squeeze(-1)
        return global_features
    


class cVAEEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cVAEEncoder, self).__init__()

        self.hidden_dim = 128
        self.mlp1 = nn.Linear(input_dim, self.hidden_dim)
        self.mlp2 = nn.Linear(self.hidden_dim, output_dim)
        self.mlp3 = nn.Linear(output_dim, output_dim)
        self.get_mu = nn.Linear(output_dim, output_dim)
        self.get_logvar = nn.Linear(output_dim, output_dim)

        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')
        self.L1Loss = nn.L1Loss(reduction='none')

    def forward(self, inputs):
        net = torch.cat(inputs, dim=-1)
        net = F.leaky_relu(self.mlp1(net))
        net = F.leaky_relu(self.mlp2(net))
        net = self.mlp3(net)
        mu = self.get_mu(net)
        logvar = self.get_logvar(net)
        noise = torch.Tensor(torch.randn(*mu.shape)).to(net.device)
        z = mu + torch.exp(logvar / 2) * noise
        return z, mu, logvar
    

class cVAEDecoder(nn.Module):
    def __init__(self, input_dim, output_dim=6):
        super(cVAEDecoder, self).__init__()

        self.hidden_dim = 128
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, output_dim)
        )

    def forward(self, inputs):
        net = torch.cat(inputs, dim=-1)
        net = self.mlp(net)
        return net
    
    
class cVAE(nn.Module):
    def __init__(self, feat_dim, dir_feat_dim=32, rot_feat_dim=32, tran_feat_dim=32, z_dim=32, lbd_kl=1.0, lbd_dir=1.0, lbd_rot=1.0, lbd_tran=1.0):
        super(cVAE, self).__init__()
        self.feat_dim = feat_dim
        self.z_dim = z_dim
        self.lbd_kl = lbd_kl
        self.lbd_dir = lbd_dir
        self.lbd_rot = lbd_rot
        self.lbd_tran = lbd_tran
                
        self.disassembly_encoder = cVAEEncoder(input_dim=feat_dim * 2 * 3 + dir_feat_dim, output_dim=z_dim)
        self.disassembly_decoder = cVAEDecoder(input_dim=feat_dim * 2 * 3 + z_dim, output_dim=3)
        
        self.init_pointnet2 = PointNet2SemSegSSG({'feat_dim': feat_dim})
        
        self.trans_encoder = cVAEEncoder(input_dim=feat_dim * 2 * 3 + dir_feat_dim + feat_dim + rot_feat_dim + tran_feat_dim, output_dim=z_dim)
        self.trans_decoder = cVAEDecoder(input_dim=feat_dim * 2 * 3 + dir_feat_dim + feat_dim + z_dim,                        output_dim=6+3)
        
        self.mlp_dir = nn.Linear(3, dir_feat_dim)
        self.mlp_transformation = nn.Linear(6 + 3, rot_feat_dim + tran_feat_dim)
        
        self.CosineEmbeddingLoss = nn.CosineEmbeddingLoss(reduction='none')
        self.L1Loss = nn.L1Loss(reduction='none')
        
        
    def KL(self, mu, logvar):
        mu = mu.view(mu.shape[0], -1)
        logvar = logvar.view(logvar.shape[0], -1)
        loss = 0.5 * torch.sum(mu * mu + torch.exp(logvar) - 1 - logvar, 1)
        # high star implementation
        # torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var, 1))
        loss = torch.mean(loss)
        return loss
    
    # input sz bszx3x2
    def bgs(self, d6s):
        bsz = d6s.shape[0]
        b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
        a2 = d6s[:, :, 1]
        b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)

    # batch geodesic loss for rotation matrices
    def bgdR(self, Rgts, Rps):
        Rds = torch.bmm(Rgts.permute(0, 2, 1), Rps)
        Rt = torch.sum(Rds[:, torch.eye(3).bool()], 1)  # batch trace
        # necessary or it might lead to nans and the likes
        theta = torch.clamp(0.5 * (Rt - 1), -1 + 1e-6, 1 - 1e-6)
        return torch.acos(theta)

    # 6D-Rot loss
    # input sz bszx6
    def get_6d_rot_loss(self, pred_6d, gt_6d):
        pred_Rs = self.bgs(pred_6d.reshape(-1, 2, 3).permute(0, 2, 1))
        gt_Rs = self.bgs(gt_6d.reshape(-1, 2, 3).permute(0, 2, 1))
        theta = self.bgdR(gt_Rs, pred_Rs)
        return theta
    
    
    def forward_disassembly_dir(self, global_equ_feats, disassembly_dir_feats):
        z_all, mu, logvar = self.disassembly_encoder([global_equ_feats, disassembly_dir_feats])
        recon_dir = self.disassembly_decoder([global_equ_feats, z_all])
        return recon_dir, mu, logvar
    
    def forward_target_transformation(self, global_equ_feats, disassembly_dir_feats, init_feats, target_trans_feats):
        z_all, mu, logvar = self.trans_encoder([global_equ_feats, disassembly_dir_feats, init_feats, target_trans_feats])
        recon_trans = self.trans_decoder([global_equ_feats, disassembly_dir_feats, init_feats, z_all])
        return recon_trans, mu, logvar


    # def get_loss(self, global_equ_feats, global_inv_feats, disassembly_dir):
    def get_loss(self, global_equ_feats, disassembly_dir, init_pcs, target_trans):
        batch_size = global_equ_feats.shape[0]
        disassembly_dir_feats = self.mlp_dir(disassembly_dir)
        global_equ_feats = global_equ_feats.reshape(batch_size, 2 * self.feat_dim * 3)
        recon_dir, mu_dir, logvar_dir = self.forward_disassembly_dir(global_equ_feats, disassembly_dir_feats)
        
        # recon_dir = recon_dir / torch.norm(recon_dir, dim=1, keepdim=True)
        dir_loss = self.CosineEmbeddingLoss(recon_dir, disassembly_dir, torch.ones(disassembly_dir.shape[0]).to(device=disassembly_dir.device))
        dir_loss = dir_loss.mean()
        kl_dir_loss = self.KL(mu_dir, logvar_dir)
        
        init_pcs = init_pcs.repeat(1, 1, 2)
        init_feats = self.init_pointnet2.forward_global(init_pcs)
        target_trans_feats = self.mlp_transformation(target_trans)
        recon_transformation, mu_trans, logvar_trans = self.forward_target_transformation(global_equ_feats, disassembly_dir_feats, init_feats, target_trans_feats)
        
        rot_loss = self.get_6d_rot_loss(recon_transformation[:, :6], target_trans[:, :6]).mean()
        tran_loss = self.L1Loss(recon_transformation[:, 6:], target_trans[:, 6:]).mean()
        kl_transformation_loss = self.KL(mu_trans, logvar_trans)
        
        losses = {}
        losses['disassembly'] = dir_loss
        losses['kl_disassembly'] = kl_dir_loss
        losses['transformation_rot'] = rot_loss
        losses['transformation_tran'] = tran_loss
        losses['kl_transformation'] = kl_transformation_loss
        losses['tot'] = self.lbd_kl * (kl_dir_loss + kl_transformation_loss) \
                        + self.lbd_dir * dir_loss \
                        + self.lbd_rot * rot_loss \
                        + self.lbd_tran * tran_loss
        return losses

    
    def predict_disassembly_dir(self, global_equ_feats):
        batch_size = global_equ_feats.shape[0]
        global_equ_feats = global_equ_feats.reshape(batch_size, 2 * self.feat_dim * 3)
        # z_all = torch.Tensor(torch.randn(batch_size, self.z_dim)).to(global_equ_feats.device)
        z_all = torch.zeros((batch_size, self.z_dim)).to(global_equ_feats.device)

        recon_dir = self.disassembly_decoder([global_equ_feats, z_all])
        recon_dir = recon_dir / torch.norm(recon_dir, dim=1, keepdim=True)
        return recon_dir
        
        
    def predict_target_transformation(self, global_equ_feats, init_pcs, recon_disassembly_dir):
        batch_size = global_equ_feats.shape[0]
        global_equ_feats = global_equ_feats.reshape(batch_size, 2 * self.feat_dim * 3)
        
        init_pcs = init_pcs.repeat(1, 1, 2)
        init_feats = self.init_pointnet2.forward_global(init_pcs)
        
        recon_disassembly_dir_feats = self.mlp_dir(recon_disassembly_dir)
        # z_all = torch.Tensor(torch.randn(batch_size, self.z_dim)).to(global_equ_feats.device)
        z_all = torch.zeros((batch_size, self.z_dim)).to(global_equ_feats.device)
        
        recon_trans = self.trans_decoder([global_equ_feats, recon_disassembly_dir_feats, init_feats, z_all])
        recon_rot = recon_trans[:, :6]
        recon_tran = recon_trans[:, 6:]
        
        recon_rot = recon_rot.reshape(-1, 2, 3).permute(0, 2, 1)
        recon_rot = self.bgs(recon_rot)
        recon_rot = recon_rot.permute(0, 2, 1)
        recon_rot = recon_rot[:, :2, :]
        recon_rot = recon_rot.reshape(batch_size, 6)
        
        return torch.cat([recon_rot, recon_tran], dim=-1)
    
     


class Network(nn.Module):
    def __init__(self, dir_feat_dim, z_dim, rot_feat_dim=32, tran_feat_dim=32, lbd_kl=1.0, lbd_dir=1.0, lbd_rot=1.0, lbd_tran=1.0):
        super(Network, self).__init__()
        # super().__init__(cfg)
        # self.cfg = cfg
        
        self.pc_feat_dim = 64
        self.dir_feat_dim = dir_feat_dim
        self.rot_feat_dim = rot_feat_dim
        self.tran_feat_dim = tran_feat_dim
        self.z_dim = z_dim
        self.lbd_kl = lbd_kl
        self.lbd_dir = lbd_dir
        self.lbd_rot = lbd_rot
        self.lbd_tran = lbd_tran
        
        self.encoder_type = "vn_dgcnn"
        self.regressor_type = "original_cvae"
        
        self.encoder = self.init_encoder()
        self.pose_predictor = self.init_pose_predictor()
        
        self.close_eps = 0.05
        self.iters = 0
        self.flag = True

    def init_encoder(self):
        if self.encoder_type == "dgcnn":
            encoder = DGCNN_New(feat_dim=self.pc_feat_dim)
        elif self.encoder_type == "vn_dgcnn":
            encoder = VN_DGCNN(feat_dim=self.pc_feat_dim)     # the vector neuron version of DGCNN
        return encoder

    def init_pose_predictor(self):
        if self.regressor_type == "original":
            pose_predictor = Ori_Regressor(pc_feat_dim=self.pc_feat_dim)
        if self.regressor_type == "vnn":
            pose_predictor = VN_Regressor(pc_feat_dim=self.pc_feat_dim)
        elif self.regressor_type == "original_cvae":
            pose_predictor = cVAE(feat_dim=self.pc_feat_dim * 2, dir_feat_dim=self.dir_feat_dim, rot_feat_dim=self.rot_feat_dim, tran_feat_dim=self.tran_feat_dim,
                                  z_dim=self.z_dim, lbd_kl=self.lbd_kl, lbd_dir=self.lbd_dir, lbd_rot=self.lbd_rot, lbd_tran=self.lbd_tran)
        return pose_predictor
    

    def _extract_part_feats(self, part_pcs, part_valids):
        """Extract per-part point cloud features."""
        # part valid (B, P)
        B, P, _, N = part_pcs.shape  # [B, P, 3, N]
        valid_mask = part_valids == 1
        # shared-weight encoder
        valid_pcs = part_pcs[valid_mask]  # [n, 3, N]
        valid_feats_equiv, valid_feats_inv = self.encoder(valid_pcs)

        equiv_pc_feats = torch.zeros(B, P, self.pc_feat_dim * 2, 3).type_as(
            valid_feats_equiv
        )
        equiv_pc_feats[valid_mask] = valid_feats_equiv  # [B, P, self.pc_feat_dim*2, 4]

        inv_pc_feats = torch.zeros(
            B, P, self.pc_feat_dim * 2, self.pc_feat_dim * 2
        ).type_as(valid_feats_inv)
        inv_pc_feats[valid_mask] = valid_feats_inv
        return equiv_pc_feats, inv_pc_feats
    

    def _get_global_feats(self, feats):
        global_feats = torch.sum(
            feats, dim=1, keepdims=False
        )  # [B, 2*feat_dim, 2*feat_dim]
        return global_feats  # (batch_size, num_parts, N, 3), (batch_size, N, 3), (batch_size, 2*feat_dim, 2*feat_dim)


    def _recon_pts(self, inv_feats, part_valids):
        # inv_feats (B, P, 2*feat_dim, 2*feat_dim)
        # part_valids (B, P)
        B, P, C, _ = inv_feats.shape
        valid_mask = part_valids == 1
        valid_inv_feats = inv_feats[valid_mask]  # [n, 2*feat_dim, 2*feat_dim]
        global_inv_feats = torch.sum(
            inv_feats, dim=1, keepdims=False
        )  # [B, 2*feat_dim, 2*feat_dim]
        return global_inv_feats  # (batch_size, num_parts, N, 3), (batch_size, N, 3), (batch_size, 2*feat_dim, 2*feat_dim)
    
    
    def get_object_feature(self, gt_pcs):
        B, P, _, _ = gt_pcs.shape  # [B, P, N, 3]
        gt_pcs = gt_pcs.permute(0, 1, 3, 2) # B * P * 3 * N
        part_valids = torch.ones(gt_pcs.shape[0], gt_pcs.shape[1]).to(gt_pcs.device)
        gt_equiv_feats, gt_inv_feats = self._extract_part_feats(gt_pcs, part_valids)  # (batch_size, num_parts, 2*feat_dim, 3), (batch_size, num_parts, 2*feat_dim, 2*feat_dim)
        # print('gt_equiv_feats: ', gt_equiv_feats.shape)     # N * P * self.pc_feat_dim * 3
        # print('gt_inv_feats: ', gt_inv_feats.shape)         # N * P * self.pc_feat_dim * self.pc_feat_dim
        
        global_inv_feats = self._get_global_feats(gt_inv_feats)
        global_inv_feats = global_inv_feats.unsqueeze(1).repeat(1, P, 1, 1)     
        
        global_equ_feats = self._get_global_feats(gt_equiv_feats)
        global_equ_feats = global_equ_feats.unsqueeze(1).repeat(1, P, 1, 1)
        # print('global_equ_feats: ', global_equ_feats.shape) # N * P * self.pc_feat_dim * 3
        # print('global_inv_feats: ', global_inv_feats.shape) # N * P * self.pc_feat_dim * self.pc_feat_dim
        
        return global_equ_feats, global_inv_feats
     

    def get_object_feature_new(self, gt_pcs):
        B, P, _, _ = gt_pcs.shape  # [B, P, N, 3]
        gt_pcs = gt_pcs.permute(0, 1, 3, 2) # B * P * 3 * N
        part_valids = torch.ones(gt_pcs.shape[0], gt_pcs.shape[1]).to(gt_pcs.device)
        equiv_feats, inv_feats = self._extract_part_feats(gt_pcs, part_valids)  # (batch_size, num_parts, 2*feat_dim, 3), (batch_size, num_parts, 2*feat_dim, 2*feat_dim)
        # print('gt_equiv_feats: ', gt_equiv_feats.shape)     # N * P * self.pc_feat_dim * 3
        # print('gt_inv_feats: ', gt_inv_feats.shape)         # N * P * self.pc_feat_dim * self.pc_feat_dim
        
        global_inv_feats = self._recon_pts(inv_feats, part_valids)
        global_inv_feats = global_inv_feats.unsqueeze(1).repeat(1, P, 1, 1)
        
        GF = torch.bmm(
            global_inv_feats.reshape(
                -1, self.pc_feat_dim * 2, self.pc_feat_dim * 2
            ),
            equiv_feats.reshape(-1, self.pc_feat_dim * 2, 3),
        ).reshape(
            B, P, -1, 3
        )
        
        return GF
        

    def forward(self, gt_pcs, disassembly_dir, init_pcs, target_trans):
        global_equ_feats, global_inv_feats = self.get_object_feature(gt_pcs)
        losses = self.pose_predictor.get_loss(global_equ_feats, disassembly_dir, init_pcs, target_trans)
        return losses

    def forward_new(self, gt_pcs, disassembly_dir, init_pcs, target_trans):
        global_equ_feats = self.get_object_feature_new(gt_pcs)
        losses = self.pose_predictor.get_loss(global_equ_feats, disassembly_dir, init_pcs, target_trans)
        return losses

    def predict(self, gt_pcs, init_pcs):
        global_equ_feats, global_inv_feats = self.get_object_feature(gt_pcs)
        disassembly_dir = self.pose_predictor.predict_disassembly_dir(global_equ_feats)
        target_transformation = self.pose_predictor.predict_target_transformation(global_equ_feats, init_pcs, disassembly_dir)
        return disassembly_dir, target_transformation
    
    def predict_new(self, gt_pcs, init_pcs):
        global_equ_feats = self.get_object_feature_new(gt_pcs)
        disassembly_dir = self.pose_predictor.predict_disassembly_dir(global_equ_feats)
        target_transformation = self.pose_predictor.predict_target_transformation(global_equ_feats, init_pcs, disassembly_dir)
        return disassembly_dir, target_transformation
        
