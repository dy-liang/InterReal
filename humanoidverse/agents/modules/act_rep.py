from __future__ import annotations
from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
from .modules import BaseModule
import numpy as np
from humanoidverse.agents.modules import utils
import math

class Encoder(nn.Module):
    def __init__(self, actor, action_dim, batchsize, device):
        super(Encoder, self).__init__()
        self.encoder = actor.enc_module
        self.action_dim = action_dim
        self.len = 4
        self.network = CFAPredictor(latent_state_dim = 128,
                                   act_seq_in_dim = self.len * self.action_dim,
                                   rew_seq_in_dim = 4,
                                   hidden_dim = 256,
                                    act_seq_out_dim = 4,
                                   omg_seq_out_dim = 4,
                                   output_dim = self.len, l=1,
                                   omega_opt_mode="min_mu",
                                   num_sample=100).to(device)
        self.opt_num = 4
        self.opt_mode = "max"
        self.pred_temp = 0.1
        self.as_fc = True

    def calc_psi(self, a_seq, w_seq):
        if self.as_fc:
            a_seq = self.network.forward_as(a_seq) #
        inner_product = w_seq @ a_seq.t() #
        psi_targ_cos = np.pi / 2 * torch.cos(inner_product) # (num_sample, batch_size*num_aug)
        psi_targ_sin = np.pi / 2 * torch.sin(inner_product) # (num_sample, batch_size*num_aug)
        return torch.stack([psi_targ_cos, psi_targ_sin], dim=-1) # (num_sample, batch_size*num_aug, 2)
    
    def update_extr(self, a_seq, enc_s, batch_size, num_aug=1):
        labels = torch.arange(batch_size).long().to(a_seq.device)

        w_seq = self.network.omega
        psi_targ = self.calc_psi(a_seq, w_seq)  # (num_sample, batch_size*num_aug, 2)
        
        psi_cos, psi_sin = self.network(enc_s, w_seq).chunk(2, -1) 
        psi = torch.stack([psi_cos, psi_sin], dim=0).transpose(0, -1) 
        # MSE
        psi_error = (psi - psi_targ.unsqueeze(0)).pow(2)
        if batch_size !=1:
            psi_error = utils.rank(psi_error,
                                psi_error.mean([-1, -2, -3]).detach(),
                                dim=0,
                                num=self.opt_num,
                                mode=self.opt_mode) # (opt_num, num_sample, batch_size*num_aug, 2)
        loss_psi_mse = psi_error.sum(0).sum(-1).mean() * num_aug

        # Cross Entropy
        psi_cl_targ = psi_targ.transpose(0, 1).reshape(psi_targ.size(1), -1).view(
            batch_size, num_aug, -1) # (batch_size, num_aug, num_sample*2)
        psi_cl = psi.transpose(1, 2).reshape(psi.size(0), psi.size(2), -1).view(
            psi.size(0), batch_size, num_aug, -1) # (output_dim, batch_size, num_aug, num_sample*2)

        loss_psi_cl, acc = utils.compute_cl_loss(psi_cl[:,:,0], psi_cl_targ[:,0], labels, None, self.pred_temp, True)

        if batch_size !=1:
            loss_psi_cl = utils.rank(loss_psi_cl,
                                        loss_psi_cl.detach(),
                                        dim=0,
                                        num=self.opt_num,
                                        mode=self.opt_mode) # (opt_num,)
        loss_psi = loss_psi_mse + loss_psi_cl.sum()
        return loss_psi
    
LOG_SIG_MIN = -6
LOG_STD_MAX = 2
class CFAPredictor(nn.Module):
    def __init__(self, latent_state_dim, act_seq_in_dim, rew_seq_in_dim, hidden_dim,
                 act_seq_out_dim=5, omg_seq_out_dim=5, output_dim=1, l=2, output_mod=None,
                 ensemble=False, num_ensemble=1, activation=utils.Swish, rs_fc=False,
                 omega_opt_mode=None, num_sample=256):
        super(CFAPredictor, self).__init__()

        # Initialize hyperparameters
        self.output_dim = 2 * output_dim
        self.num_ensemble = num_ensemble
        self.hidden_depth = l
        self.num_sample = num_sample
        self.omega_opt_mode = omega_opt_mode

        # Initialize modules
        self.aseq_fc = utils.mlp(act_seq_in_dim, 0, act_seq_out_dim, 0, output_mod)
        self.omeg_fc = utils.mlp(rew_seq_in_dim, 0, omg_seq_out_dim, 0, output_mod)
        self.rseq_fc = utils.mlp(rew_seq_in_dim, 0, rew_seq_in_dim, 0, output_mod) if rs_fc \
            else None
        
        self.predictor = utils.mlp(
            latent_state_dim + omg_seq_out_dim,
            hidden_dim, self.output_dim, l, output_mod, ensemble=ensemble,
            channel_dim=num_ensemble, activation=activation
        )

        # Initialize a random variable
        self._init_omega(omega_opt_mode, rew_seq_in_dim)

        self.infos = dict()
        self.apply(utils.weight_init)

    def _init_omega(self, omega_opt_mode, rs_dim):
        assert omega_opt_mode in [None, 'min_mu', 'min_all'], print(omega_opt_mode)
        self.omega_mu = nn.Parameter(torch.zeros(rs_dim, requires_grad=True))
        self.omega_logstd = nn.Parameter(
            torch.ones(rs_dim, requires_grad=True) * math.atanh(-LOG_SIG_MIN/(LOG_STD_MAX - LOG_SIG_MIN))
        )
        if omega_opt_mode is None:
            self.omega_mu.requires_grad = False
            self.omega_logstd.requires_grad = False
        elif omega_opt_mode == 'min_mu':
            self.omega_logstd.requires_grad = False
        elif omega_opt_mode == 'min_all':
            self.omega_logstd.requires_grad = True
            self.omega_mu.requires_grad = True
    @property
    def omega(self):
        # output.shape == torch.Size([num_sample, rew_seq_dim])
        if self.omega_opt_mode != 'sample':
            log_std = torch.tanh(self.omega_logstd)
            log_std = LOG_SIG_MIN + log_std * (LOG_STD_MAX - LOG_SIG_MIN)
            std = torch.exp(log_std)
            noise = torch.randn(self.num_sample, log_std.size(0)).to(log_std.device)

            self.infos['omega_mu'] = self.omega_mu.detach()
            self.infos['omega_std'] = std.detach()
            return self.omega_mu + noise * std
        return self.omega_mu

    def _get_input(self, h, h_ws):
        # if h_as.ndim == 3:
        #     h = h.unsqueeze(0).expand(h_as.size(0), *h.size())
        cat = h #torch.cat([h, h_as], dim=-1)
        if h_ws.ndim == 4:
            cat = cat.unsqueeze(1).expand(*h_ws.size()[:3], cat.size(-1))
        elif h_ws.ndim == 3:
            cat = cat.unsqueeze(0).expand(h_ws.size(0), *cat.size())
        return torch.cat([cat, h_ws], dim=-1)

    def forward_as(self, action_sequence: torch.tensor):
        # action_sequence: torch.Size([batch_size, seq_dim, act_dim])
        # case1: output.shape == torch.Size([batch_size, act_seq_out_dim])
        # case2: output.shape == torch.Size([num_ensemble, batch_size, act_seq_out_dim])
        assert action_sequence.ndim == 3
        h_as = self.aseq_fc(action_sequence.view(action_sequence.size(0), -1))
        self.infos['h_as'] = h_as[0] if h_as.ndim == 3 else h_as
        return h_as

    def forward_ws(self, omega_sequence: torch.tensor, batch_size: int):
        # omega_sequence: torch.Size([num_sample, rew_seq_dim])
        # case1: output.shape == torch.Size([num_sample, batch_size, rew_seq_dim])
        # case2: output.shape == torch.Size([num_ensemble, num_sample, batch_size, rew_seq_dim])
        assert omega_sequence.ndim == 2
        omega_sequence = omega_sequence.unsqueeze(1).expand(
            omega_sequence.size(0), batch_size, omega_sequence.size(-1)
        ).unsqueeze(0)
        h_ws = self.omeg_fc(omega_sequence)
        self.infos['h_ws'] = h_ws[0].mean(0)
        return h_ws.squeeze(0)

    def forward_rs(self, reward_sequence: torch.tensor):
        # reward_sequence: torch.Size([batch_size, rew_seq_dim])
        # output: torch.Size([batch_size, rew_seq_dim])
        assert reward_sequence.ndim == 2
        h_rs = self.rseq_fc(reward_sequence) if self.rseq_fc is not None else reward_sequence
        self.infos['h_rs'] = h_rs
        return h_rs

    def forward(self, latent_state, omega_sequence):
        # case1: output.shape == torch.Size([num_sample, batch_size, output_dim])
        # case2: output.shape == torch.Size([num_ensemble, num_sample, batch_size, output_dim])
        batch_size = latent_state.size(0)
        #latent_a_seq = self.forward_rs(r_sequence)
        latent_w_seq = self.forward_ws(omega_sequence, batch_size)
        input = self._get_input(latent_state, latent_w_seq)
        return self.predictor(input)