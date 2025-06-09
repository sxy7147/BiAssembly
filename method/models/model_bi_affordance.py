import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# https://github.com/erikwijmans/Pointnet2_PyTorch
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG
import inference_utils

    
    
class PointNet2SemSegSSG(PointNet2ClassificationSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[self.hparams['pts_channel'], 32, 32, 64],
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

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + self.hparams['pts_channel'], 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, self.hparams['feat_dim'], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.hparams['feat_dim']),
            nn.ReLU(True),
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_fc = nn.Sequential(
            nn.Linear(512, 256),  
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




class MLPs(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=128):
        super(MLPs, self).__init__()
        self.hidden_dim = hidden_dim
        self.mlp1 = nn.Linear(input_dim, self.hidden_dim)
        self.mlp2 = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, inputs):
        input_net = torch.cat(inputs, dim=-1)
        hidden_net = F.leaky_relu(self.mlp1(input_net))
        net = self.mlp2(hidden_net)
        return net



class ActorEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorEncoder, self).__init__()
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
    
    def KL(self, mu, logvar):
        mu = mu.view(mu.shape[0], -1)
        logvar = logvar.view(logvar.shape[0], -1)
        loss = 0.5 * torch.sum(mu * mu + torch.exp(logvar) - 1 - logvar, 1)
        # high star implementation
        # torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var, 1))
        # loss = torch.mean(loss)
        return loss


class ActorDecoder(nn.Module):
    def __init__(self, input_dim, output_dim=6):
        super(ActorDecoder, self).__init__()
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

    

class Network(nn.Module):
    def __init__(self, feat_dim, cp_feat_dim=32, dir_feat_dim=32, z_dim=128, lbd_kl=1.0, lbd_dir=1.0, pts_channel=4, dir_dim=6, target_pts_channel=3):
        super(Network, self).__init__()
        
        self.affordance1 = MLPs(input_dim=feat_dim * 2 + dir_feat_dim + cp_feat_dim * 1)
        self.affordance2 = MLPs(input_dim=feat_dim * 3 + dir_feat_dim + cp_feat_dim * 2 + dir_feat_dim * 1)
        
        self.critic1 = MLPs(input_dim=feat_dim * 2 + dir_feat_dim + cp_feat_dim * 1 + dir_feat_dim * 1)
        self.critic2 = MLPs(input_dim=feat_dim * 3 + dir_feat_dim + cp_feat_dim * 2 + dir_feat_dim * 2)
        
        self.actor1_encoder = ActorEncoder(input_dim=feat_dim* 2 + dir_feat_dim + cp_feat_dim + dir_feat_dim, output_dim=z_dim)
        self.actor1_decoder = ActorDecoder(input_dim=feat_dim* 2 + dir_feat_dim + cp_feat_dim + z_dim, output_dim=dir_dim)
        
        self.actor2_encoder = ActorEncoder(input_dim=feat_dim* 3 + dir_feat_dim + cp_feat_dim * 2 + dir_feat_dim * 2, output_dim=z_dim)
        self.actor2_decoder = ActorDecoder(input_dim=feat_dim* 3 + dir_feat_dim + cp_feat_dim * 2 + dir_feat_dim + z_dim, output_dim=dir_dim)
        
        self.init_pointnet2 = PointNet2SemSegSSG({'feat_dim': feat_dim, 'pts_channel': pts_channel})
        self.target_pointnet2 = PointNet2SemSegSSG({'feat_dim': feat_dim, 'pts_channel': target_pts_channel})

        self.mlp_disassembly_dir = nn.Linear(3, dir_feat_dim)
        self.mlp_cp = nn.Linear(pts_channel, cp_feat_dim)
        self.mlp_dir = nn.Linear(dir_dim, dir_feat_dim)

        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')
        self.BCELoss_withoutSigmoid = nn.BCELoss(reduction='none')
        self.L1Loss = nn.L1Loss(reduction='none')
        self.CosineEmbeddingLoss = nn.CosineEmbeddingLoss(reduction='none')

        self.feat_dim = feat_dim
        self.z_dim = z_dim
        self.lbd_kl = lbd_kl
        self.lbd_dir = lbd_dir
        self.pts_channel = pts_channel
        self.dir_dim = dir_dim


    def get_CE_loss(self, pred_logits, gt_labels):
        loss = self.BCELoss_withoutSigmoid(pred_logits, gt_labels.float())
        return loss
        
    def get_L1_loss(self, pred_score, gt_score):
        loss = self.L1Loss(pred_score, gt_score)
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
    
    def get_6d_rot_loss(self, pred_6d, gt_6d):
        pred_Rs = self.bgs(pred_6d.reshape(-1, 2, 3).permute(0, 2, 1))
        gt_Rs = self.bgs(gt_6d.reshape(-1, 2, 3).permute(0, 2, 1))
        theta = self.bgdR(gt_Rs, pred_Rs)
        return theta
    
    
    def repeat(self, pcs):
        xyz = pcs[:, :, :3]
        f = pcs[:, :, 3:]
        pcs_repeated = torch.cat((xyz, xyz, f), dim=2)
        return pcs_repeated

    def expand_features(self, features, rvs):
        expanded_features = []
        for feat in features:
            feat_dim = feat.shape[-1]
            expanded_features.append(feat.unsqueeze(dim=1).repeat(1, rvs, 1).reshape(-1, feat_dim))
        return expanded_features
    
    def get_target_feats(self, target_pcs):
        target_pcs = self.repeat(target_pcs)
        target_feats = self.target_pointnet2.forward_global(target_pcs) # [B*P, F] 
        return target_feats
    
    def get_disassembly_dir_feats(self, disassembly_dir):
        disassembly_dir_feats = self.mlp_disassembly_dir(disassembly_dir)
        return disassembly_dir_feats
    
    def get_cp_feats(self, cp):
        cp_feats = self.mlp_cp(cp)
        return cp_feats
    
    def get_dir_feats(self, dir):
        dir_feats = self.mlp_dir(dir)
        return dir_feats
    
    
    def get_pointnet_feats(self, init_pcs, cp1, cp2=None):
        init_pcs[:, 0] = cp1
        if cp2 is not None:
            init_pcs[:, 1] = cp2
        init_pcs = self.repeat(init_pcs)
        whole_feats = self.init_pointnet2(init_pcs)
        
        net1 = whole_feats[:, :, 0]
        if cp2 is None:
            return net1, None
        else:
            net2 = whole_feats[:, :, 1]
            return net1, net2
    
    
    def get_pointnet_feats_rvs(self, init_pcs, cp1, cp2=None, rvs_cp1=1, rvs_cp2=1):
        bs = init_pcs.shape[0]
        init_pcs[:, 0: rvs_cp1] = cp1.reshape(bs, rvs_cp1, self.pts_channel)
        if cp2 is not None:
            init_pcs[:, rvs_cp1: rvs_cp1+rvs_cp2] = cp2.reshape(bs, rvs_cp2, self.pts_channel)
        init_pcs = self.repeat(init_pcs)
        whole_feats = self.init_pointnet2(init_pcs)
        
        net1 = whole_feats[:, :, 0: rvs_cp1].permute(0, 2, 1).reshape(bs * rvs_cp1, -1)
        if cp2 is None:
            return net1, None 
        else:
            net2 = whole_feats[:, :, rvs_cp1: rvs_cp1 + rvs_cp2].permute(0, 2, 1).reshape(bs * rvs_cp2, -1)
            return net1, net2
    
    
    def get_pointnet_all_feats(self, init_pcs):
        bs, num_pts = init_pcs.shape[0], init_pcs.shape[1]
        init_pcs = self.repeat(init_pcs)
        whole_feats = self.init_pointnet2(init_pcs)
        net = whole_feats.permute(0, 2, 1).reshape(bs * num_pts, -1)
        return net
            
    
    def forward_critic2(self, target_feats, disassembly_dir_feats, net1, net2, cp1_feats, cp2_feats, dir1_feats, dir2_feats, rvs_ctpt=1, rvs_dir=1):        
        # features = [target_feats, disassembly_dir_feats, net1, net2, cp1_feats, cp2_feats, dir1_feats, dir2_feats]
        expanded_features_1 = self.expand_features([target_feats, disassembly_dir_feats, net1, cp1_feats, dir1_feats], rvs_ctpt * rvs_dir)
        expanded_features_2 = self.expand_features([net2, cp2_feats], rvs_dir)
        expanded_features = expanded_features_1[0:3] + expanded_features_2[0:1] + expanded_features_1[3:4] + expanded_features_2[1:2] + expanded_features_1[4:5] + [dir2_feats]
                
        # pred_result_logits = self.critic2(features)
        pred_result_logits = self.critic2(expanded_features)
        pred_scores = torch.sigmoid(pred_result_logits)
        return pred_scores

    
    def forward_critic1(self, target_feats, disassembly_dir_feats, net1, cp1_feats, dir1_feats, rvs_ctpt=1, rvs_dir=1):        
        features = [target_feats, disassembly_dir_feats, net1, cp1_feats, dir1_feats]
        expanded_features_1 = self.expand_features(features[: 2], rvs_ctpt * rvs_dir)
        expanded_features_2 = self.expand_features(features[2: 4], rvs_dir)
        features = expanded_features_1 + expanded_features_2 + [dir1_feats]
        
        pred_result_logits = self.critic1(features)
        pred_scores = torch.sigmoid(pred_result_logits)
        return pred_scores


    def forward_aff2(self, target_feats, disassembly_dir_feats, net1, net2, cp1_feats, cp2_feats, dir1_feats):                
        pred_result_logits = self.affordance2([target_feats, disassembly_dir_feats, net1, net2, cp1_feats, cp2_feats, dir1_feats])
        pred_scores = torch.sigmoid(pred_result_logits)
        return pred_scores
    

    def forward_aff1(self, target_feats, disassembly_dir_feats, net1, cp1_feats):
        pred_result_logits = self.affordance1([target_feats, disassembly_dir_feats, net1, cp1_feats])
        pred_scores = torch.sigmoid(pred_result_logits)
        return pred_scores
    
        
    def forward_actor1(self, target_feats, disassembly_dir_feats, net1, cp1_feats, dir1_feats):        
        z_all, mu, logvar = self.actor1_encoder([target_feats, disassembly_dir_feats, net1, cp1_feats, dir1_feats])
        recon_dir1 = self.actor1_decoder([target_feats, disassembly_dir_feats, net1, cp1_feats, z_all])
        return recon_dir1, mu, logvar
    
        
    def get_loss_actor1(self, target_feats, disassembly_dir_feats, net1, cp1_feats, dir1, dir1_feats):
        recon_dir1, mu, logvar = self.forward_actor1(target_feats, disassembly_dir_feats, net1, cp1_feats, dir1_feats)
        dir_loss = self.get_6d_rot_loss(recon_dir1, dir1)
        dir_loss = dir_loss
        kl_loss = self.actor1_encoder.KL(mu, logvar)
        losses = {}
        losses['kl'] = kl_loss
        losses['dir'] = dir_loss
        losses['tot'] = self.lbd_kl * kl_loss + self.lbd_dir * dir_loss
        return losses 


    def forward_actor2(self, target_feats, disassembly_dir_feats, net1, net2, cp1_feats, cp2_feats, dir1_feats, dir2_feats):        
        z_all, mu, logvar = self.actor2_encoder([target_feats, disassembly_dir_feats, net1, net2, cp1_feats, cp2_feats, dir1_feats, dir2_feats])
        recon_dir2 = self.actor2_decoder([target_feats, disassembly_dir_feats, net1, net2, cp1_feats, cp2_feats, dir1_feats, z_all])
        return recon_dir2, mu, logvar
        
    def get_loss_actor2(self, target_feats, disassembly_dir_feats, net1, net2, cp1_feats, cp2_feats, dir2, dir1_feats, dir2_feats):
        recon_dir2, mu, logvar = self.forward_actor2(target_feats, disassembly_dir_feats, net1, net2, cp1_feats, cp2_feats, dir1_feats, dir2_feats)
        dir_loss = self.get_6d_rot_loss(recon_dir2, dir2)
        dir_loss = dir_loss
        kl_loss = self.actor2_encoder.KL(mu, logvar)
        losses = {}
        losses['kl'] = kl_loss
        losses['dir'] = dir_loss
        losses['tot'] = self.lbd_kl * kl_loss + self.lbd_dir * dir_loss
        return losses
        

    def inference_affordance2(self, target_feats, init_pcs, disassembly_dir_feats, cp1, cp1_feats, dir1_feats, rvs_ctpt=1):
        bs, num_pts = init_pcs.shape[:2]
        
        cp2 = init_pcs.view(bs * num_pts, -1).clone()
        cp2_feats = self.mlp_cp(cp2)
        
        net1, _ = self.get_pointnet_feats_rvs(init_pcs.clone(), cp1, cp2=None, rvs_cp1=rvs_ctpt)
        net2 = self.get_pointnet_all_feats(init_pcs.clone())
        
        # features = [target_feats, disassembly_dir_feats, net1, net2, cp1_feats, cp2_feats, dir1_feats]
        if rvs_ctpt == 1:
            features = [target_feats, disassembly_dir_feats, net1, cp1_feats, dir1_feats]
            expanded_features = self.expand_features(features, num_pts)
            expanded_features = expanded_features[: 3] + [net2] + [expanded_features[3]] + [cp2_feats] + [expanded_features[4]]
        elif rvs_ctpt > 1:
            expanded_features_1 = self.expand_features([target_feats, disassembly_dir_feats], rvs_ctpt * num_pts)
            expanded_features_2 = self.expand_features([net1, cp1_feats, dir1_feats], num_pts)
            expanded_features_3 = self.expand_features([net2, cp2_feats], rvs_ctpt)
            expanded_features = expanded_features_1 + [expanded_features_2[0]] + [expanded_features_3[0]] + [expanded_features_2[1]] + [expanded_features_3[1]] + [expanded_features_2[2]]
        
        pred_result_logits = self.affordance2(expanded_features)
        pred_scores = torch.sigmoid(pred_result_logits)
        return pred_scores
    
    
    def inference_affordance1(self, target_feats, init_pcs, disassembly_dir_feats):
        bs = init_pcs.shape[0]
        num_pts = init_pcs.shape[1]
        
        cp1 = init_pcs.view(bs * num_pts, -1).clone()
        cp1_feats = self.mlp_cp(cp1)
        
        net1 = self.get_pointnet_all_feats(init_pcs.clone())
        
        # features = [target_feats, disassembly_dir_feats, net1, cp1_feats]
        features = [target_feats, disassembly_dir_feats]
        expanded_features = self.expand_features(features, num_pts)
        expanded_features = expanded_features + [net1, cp1_feats]
        
        pred_result_logits = self.affordance1(expanded_features)
        pred_scores = torch.sigmoid(pred_result_logits)
        return pred_scores


    def actor2_sample(self, target_feats, disassembly_dir_feats, net1, net2, cp1_feats, cp2_feats, dir1_feats, rvs_ctpt=1, rvs_dir=1, z_random=True):
        bs = target_feats.shape[0]
        
        if z_random is True:
            z_all = torch.Tensor(torch.randn(bs * rvs_ctpt * rvs_dir, self.z_dim)).to(net1.device)
        else:
            z_all = torch.Tensor(torch.zeros(bs * rvs_ctpt * rvs_dir, self.z_dim)).to(net1.device)

        # features = [target_feats, disassembly_dir_feats, net1, net2, cp1_feats, cp2_feats, dir1_feats, z_all]
        expanded_features_1 = self.expand_features([target_feats, disassembly_dir_feats, net1, cp1_feats, dir1_feats], rvs_ctpt * rvs_dir)
        expanded_features_2 = self.expand_features([net2, cp2_feats], rvs_dir)
        expanded_features = expanded_features_1[0:3] + expanded_features_2[0:1] + expanded_features_1[3:4] + expanded_features_2[1:2] + expanded_features_1[4:5] + [z_all]
              
        recon_dir2 = self.actor2_decoder(expanded_features)
        recon_dir2 = self.bgs(recon_dir2.reshape(-1, 2, 3).permute(0, 2, 1)).permute(0, 2, 1)
        return recon_dir2[:, :2, :]
    

    def actor1_sample(self, target_feats, disassembly_dir_feats, net1, cp1_feats, rvs_ctpt=1, rvs_dir=1, random_z=True):
        bs = target_feats.shape[0]
        
        if random_z is True:
            z_all = torch.Tensor(torch.randn(bs * rvs_ctpt * rvs_dir, self.z_dim)).to(net1.device)
        else:
            z_all = torch.Tensor(torch.zeros(bs * rvs_ctpt * rvs_dir, self.z_dim)).to(net1.device)
        
        # features = [target_feats, disassembly_dir_feats, net1, cp1_feats, z_all]
        expanded_features_1 = self.expand_features([target_feats, disassembly_dir_feats], rvs_ctpt * rvs_dir)
        expanded_features_2 = self.expand_features([net1, cp1_feats], rvs_dir)
        expanded_features = expanded_features_1 + expanded_features_2 + [z_all]
                      
        recon_dir1 = self.actor1_decoder(expanded_features)
        recon_dir1 = self.bgs(recon_dir1.reshape(-1, 2, 3).permute(0, 2, 1)).permute(0, 2, 1)
        return recon_dir1[:, :2, :]
    
    
    def get_aff2_gt(self, target_feats, disassembly_dir_feats, net1, net2, cp1_feats, cp2_feats, dir1_feats, rvs=100, topk=10):
        bs = cp1_feats.shape[0]
        with torch.no_grad():
            recon_dir2 = self.actor2_sample(target_feats, disassembly_dir_feats, net1, net2, cp1_feats, cp2_feats, dir1_feats, rvs_dir=rvs).contiguous().view(bs * rvs, self.dir_dim)
            dir2_feats = self.mlp_dir(recon_dir2)
            gt_scores = self.forward_critic2(target_feats, disassembly_dir_feats, net1, net2, cp1_feats, cp2_feats, dir1_feats, dir2_feats, rvs_dir=rvs)
            gt_score = gt_scores.view(bs, rvs, 1).topk(k=topk, dim=1)[0].mean(dim=1)
        return gt_score


    def get_aff1_gt(self, target_feats, disassembly_dir_feats, net1, cp1_feats, rvs=100, topk=10):
        bs = net1.shape[0]
        with torch.no_grad():
            recon_dir1 = self.actor1_sample(target_feats, disassembly_dir_feats, net1, cp1_feats, rvs_dir=rvs).contiguous().view(bs * rvs, self.dir_dim)
            dir1_feats = self.mlp_dir(recon_dir1)
            gt_scores = self.forward_critic1(target_feats, disassembly_dir_feats, net1, cp1_feats, dir1_feats, rvs_dir=rvs)  # dir1: B*6; dir2: (B*rvs) * 6
            gt_score = gt_scores.view(bs, rvs, 1).topk(k=topk, dim=1)[0].mean(dim=1)
        return gt_score


    def get_critic1_gt(self, target_feats, init_pcs, disassembly_dir_feats, cp1, cp1_feats, dir1_feats, rvs_ctpt=10, rvs_dir=10, topk=10):
        bs = init_pcs.shape[0]
        num_pts = init_pcs.shape[1]
        with torch.no_grad():
            aff2_scores = self.inference_affordance2(target_feats, init_pcs, disassembly_dir_feats, cp1, cp1_feats, dir1_feats, rvs_ctpt=1)
            selected_ckpts_idx = aff2_scores.view(bs, num_pts, 1).topk(k=rvs_ctpt, dim=1)[1]   # B * rvs_ctpt * 1  (idx)
            selected_ckpts_idx = selected_ckpts_idx.view(bs * rvs_ctpt, 1)                                     # (B * rvs_ctpt) * 1
            pcs_idx = torch.tensor(range(bs)).reshape(bs, 1).unsqueeze(dim=1).repeat(1, rvs_ctpt, 1).reshape(bs * rvs_ctpt, 1)
            selected_ctpts = init_pcs[pcs_idx, selected_ckpts_idx].reshape(bs * rvs_ctpt, self.pts_channel)                                         # (B * rvs_ctpt) * 3

            selected_ctpts_feats = self.get_cp_feats(selected_ctpts)
            net1, net2 = self.get_pointnet_feats_rvs(init_pcs.clone(), cp1, selected_ctpts, rvs_cp1=1, rvs_cp2=rvs_ctpt)
            recon_dir2 = self.actor2_sample(target_feats, disassembly_dir_feats, net1, net2, cp1_feats, selected_ctpts_feats, dir1_feats, rvs_ctpt=rvs_ctpt, rvs_dir=rvs_dir).contiguous().reshape(bs * rvs_ctpt * rvs_dir, self.dir_dim)    # (B * rvs_ctpt * rvs) * 6
            dir2_feats = self.get_dir_feats(recon_dir2)
            gt_scores = self.forward_critic2(target_feats, disassembly_dir_feats, net1, net2, cp1_feats, selected_ctpts_feats, dir1_feats, dir2_feats, rvs_ctpt=rvs_ctpt, rvs_dir=rvs_dir)   # dir1: B*6; dir2: (B*rvs) * 6
            gt_score = gt_scores.view(bs, rvs_ctpt * rvs_dir, 1).topk(k=topk, dim=1)[0].mean(dim=1)

        return gt_score
    
    
    def inference(self, init_pcs, target_pcs, transformed_disassembly_dir, object1_mask, object2_mask, aff_topk, critic_topk,
                  num_ctpt1=10, num_dir1=10, num_pair1=1, num_ctpt2=10, num_dir2=10, z_random=True):
        assert num_pair1 == 1
        
        batch_size, num_point_per_shape, pts_channel = init_pcs.shape
        target_feats = self.get_target_feats(target_pcs)
        disassembly_dir_feats = self.get_disassembly_dir_feats(transformed_disassembly_dir.contiguous().reshape(1, -1))
        
        # aff1
        aff1_scores = self.inference_affordance1(target_feats, init_pcs.clone(), disassembly_dir_feats).view(batch_size, num_point_per_shape)  # B * N
        aff1_scores = aff1_scores * object1_mask
        batch_idx, selected_idx = inference_utils.select_topk_indices_randomly(aff1_scores, topk_ratio=aff_topk, num_selected=num_ctpt1)
        position1s = init_pcs.clone()[batch_idx, selected_idx].view(batch_size * num_ctpt1, pts_channel)
        
        # actor1
        position1s_feats = self.get_cp_feats(position1s)
        net1, _ = self.get_pointnet_feats_rvs(init_pcs.clone(), position1s, cp2=None, rvs_cp1=num_ctpt1)
        net1 = net1.reshape(batch_size * num_ctpt1, -1)
        dir1s = self.actor1_sample(target_feats, disassembly_dir_feats, net1, position1s_feats, rvs_ctpt=num_ctpt1, rvs_dir=num_dir1, random_z=z_random).contiguous().view(batch_size * num_ctpt1 * num_dir1, 6)
        
        # critic1 
        dir1s_feats = self.get_dir_feats(dir1s)
        critic_scores = self.forward_critic1(target_feats, disassembly_dir_feats, net1, position1s_feats, dir1s_feats, rvs_ctpt=num_ctpt1, rvs_dir=num_dir1).view(batch_size, num_ctpt1 * num_dir1)
        batch_idx, selected_idx = inference_utils.select_topk_indices_randomly(critic_scores, topk_ratio=critic_topk, num_selected=num_pair1)
        position1 = position1s.view(batch_size, num_ctpt1, pts_channel)[batch_idx, selected_idx // num_dir1].view(batch_size * num_pair1, pts_channel) 
        dir1 = dir1s.view(batch_size, num_ctpt1 * num_dir1, 6)[batch_idx, selected_idx].view(batch_size * num_pair1, 6)
            
        # aff2
        position1_feats = self.get_cp_feats(position1)
        dir1_feats = self.get_dir_feats(dir1)
        aff2_scores = self.inference_affordance2(target_feats, init_pcs.clone(), disassembly_dir_feats, position1, position1_feats, dir1_feats, rvs_ctpt=num_pair1).view(batch_size * num_pair1, num_point_per_shape)
        aff2_scores = aff2_scores * object2_mask
        batch_idx, selected_idx = inference_utils.select_topk_indices_randomly(aff2_scores, topk_ratio=aff_topk, num_selected=num_ctpt2)
        position2s = init_pcs.clone()[batch_idx, selected_idx].view(batch_size * num_pair1 * num_ctpt2, pts_channel)
        
        # actor2
        position2s_feats = self.get_cp_feats(position2s)
        net1, net2 = self.get_pointnet_feats_rvs(init_pcs.clone(), position1, cp2=position2s, rvs_cp1=num_pair1, rvs_cp2=num_pair1*num_ctpt2)
        dir2s = self.actor2_sample(target_feats, disassembly_dir_feats, net1, net2, position1_feats, position2s_feats, dir1_feats, rvs_ctpt=num_pair1 * num_ctpt2, rvs_dir=num_dir2, z_random=z_random).contiguous().view(batch_size * num_pair1 * num_ctpt2 * num_dir2, 6)
        dir2s_feats = self.get_dir_feats(dir2s)
        
        # critic2
        critic_scores = self.forward_critic2(target_feats, disassembly_dir_feats, net1, net2, position1_feats, position2s_feats, dir1_feats, dir2s_feats, rvs_ctpt=num_pair1 * num_ctpt2, rvs_dir=num_dir2).view(batch_size, num_pair1 * num_ctpt2 * num_dir2)
        batch_idx, selected_idx = inference_utils.select_topk_indices_randomly(critic_scores, topk_ratio=critic_topk, num_selected=1)
        position2 = position2s.reshape(batch_size, num_pair1 * num_ctpt2, pts_channel)[batch_idx, selected_idx // num_dir2].view(batch_size, pts_channel)
        dir2 = dir2s.view(batch_size, num_pair1 * num_ctpt2 * num_dir2, 6)[batch_idx, selected_idx].view(batch_size, 6)
    
        return position1, dir1, position2, dir2, aff1_scores, aff2_scores
    
        
