from __future__ import annotations
from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
from .modules import BaseModule

def AvgL1Norm(x, eps=1e-8):
	return x/x.abs().mean(-1, keepdim=True).clamp(min=eps)

class PPOActor(nn.Module):
    def __init__(self,
                obs_dim_dict,
                module_config_dict,
                num_actions,
                init_noise_std):
        super(PPOActor, self).__init__()
        self.module_config_dict = module_config_dict
        # if "use_enc" in module_config_dict and module_config_dict.use_enc:
        #     latent_state = module_config_dict.encoder.latent_state
        #     obs_dim_dict["latent_state"]=latent_state

        #     module_config_dict_enc = self._process_module_config(module_config_dict.encoder, latent_state)  
        #     module_config_dict_actor = self._process_module_config(module_config_dict.enc_actor, num_actions)
        #     self.enc_module = BaseModule(obs_dim_dict, module_config_dict_enc)
        # else:
        #     module_config_dict_actor = self._process_module_config(module_config_dict.actor, num_actions)

        # self.act_module = BaseModule(obs_dim_dict, module_config_dict_actor)
        
        hide_dim = 100

        self.actor_model = nn.Sequential(   #s+a -> h
            nn.Linear(hide_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions))

        self.f_zs_enc_shared = nn.Sequential( # [s,h,h] -> a  #shared!
            nn.Linear(obs_dim_dict["actor_obs"], 512),
            nn.ReLU(),
            nn.Linear(512, hide_dim))

        self.f_s_enc = nn.Sequential( # [s,h,h] -> a 
            nn.Linear(obs_dim_dict["actor_obs"], 512),
            nn.ReLU(),
            nn.Linear(512, hide_dim)) 

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def _process_module_config(self, module_config_dict, num_actions):
        for idx, output_dim in enumerate(module_config_dict['output_dim']):
            if output_dim == 'robot_action_dim':
                module_config_dict['output_dim'][idx] = num_actions
        return module_config_dict

    #@property
    def actor(self, actor_obs):
        out_zs = self.f_zs_enc_shared(actor_obs).detach()
        out_zs = AvgL1Norm(out_zs)
        out_s = self.f_s_enc(actor_obs)
        out_s = AvgL1Norm(out_s)
        out_zs_s = torch.cat([out_zs, out_s], 1)

        act_mu = self.actor_model(out_zs_s)

        return act_mu
    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, actor_obs): # o + z
        mean = self.actor(actor_obs)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, actor_obs, **kwargs):
        self.update_distribution(actor_obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, actor_obs):
        actions_mean = self.actor(actor_obs)
        return actions_mean
    
    def to_cpu(self):
        self.actor = deepcopy(self.actor).to('cpu')
        self.std.to('cpu')

class PPOCritic(nn.Module):
    def __init__(self,
                obs_dim_dict,
                module_config_dict,
                num_actions):
        super(PPOCritic, self).__init__()

        hide_dim = 100
        critic_dim = obs_dim_dict["actor_obs"] +  obs_dim_dict["critic_obs"]
        self.f_za_enc = nn.Sequential(       #h+a -> h
            nn.Linear(critic_dim+num_actions, 512),
            nn.ReLU(),
            nn.Linear(512, hide_dim))
        
        self.transition_model_shared = nn.Sequential( #h+a -> h'   #shared!
            nn.Linear(hide_dim+num_actions+obs_dim_dict["critic_obs"], 1024),
            nn.ReLU(),
            nn.Linear(1024, hide_dim))

        self.critic_model = nn.Sequential(       #h+a -> h
            nn.Linear(hide_dim * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 1))
        
        #self.critic_module = BaseModule(obs_dim_dict, module_config_dict)

    #@property
    def critic(self, obs, f_zs_enc_shared):
        critic_obs, critic_obs_priv, actions = obs["actor_obs"], obs["critic_obs"], obs["actions"]
        out_zs = f_zs_enc_shared(critic_obs).detach()
        out_zs = AvgL1Norm(out_zs)
        out_next_zs = self.transition_model_shared(torch.cat((out_zs, critic_obs_priv, actions), dim=1)).detach()

        all_obs = torch.cat((critic_obs, critic_obs_priv), dim=1)

        out_sa = self.f_za_enc(torch.cat((all_obs, actions), dim=1))
        out_sa = AvgL1Norm(out_sa)

        critic_in = torch.cat((out_next_zs,out_zs, out_sa), dim=1)

        value = self.critic_model(critic_in)

        return value
    
    def reset(self, dones=None):
        pass
    
    def evaluate(self, obs, f_zs_enc_shared, **kwargs):
        value = self.critic(obs, f_zs_enc_shared)
        return value

# Deprecated: TODO: Let Wenli Fix this
class PPOActorFixSigma(PPOActor):
    def __init__(self,                 
                 obs_dim_dict,
                network_dict,
                network_load_dict,
                num_actions,):
        super(PPOActorFixSigma, self).__init__(obs_dim_dict, network_dict, network_load_dict, num_actions, 0.0)
        
    def update_distribution(self, obs_dict):
        mean = self.actor(obs_dict)['head']
        self.distribution = mean

    @property
    def action_mean(self):
        return self.distribution
    
    def get_actions_log_prob(self, actions):
        raise NotImplementedError
    
    def act(self, obs_dict, **kwargs):
        self.update_distribution(obs_dict)
        return self.distribution