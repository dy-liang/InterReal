import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from humanoidverse.agents.meta_agent.utils import soft_update, hard_update
from humanoidverse.agents.meta_agent.model import GaussianPolicy, QNetwork
from humanoidverse.agents.meta_agent.replay_memory import ParallelReplayBuffer
import numpy as np

class HighSAC(object):
    def __init__(self, device, num_envs):
        num_inputs = 547
        self.gamma = 0.99
        self.tau = 0.005
        self.action_range = [-1., 1.]

        self.target_update_interval = 2
        self.automatic_entropy_tuning = True
        self.hidden_size = 256
        self.device = device
        self.action_nums = 19
        self.lr = 3e-4
        self.init_temperature = 0.1
        self.critic = QNetwork(num_inputs, self.action_nums, self.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.critic_target = QNetwork(num_inputs, self.action_nums, self.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.buffer = ParallelReplayBuffer(capacity=40960*2, state_dim=num_inputs, action_dim=self.action_nums, num_envs=num_envs)

        if self.automatic_entropy_tuning is True:
            self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
            self.log_alpha.requires_grad = True
            # set target entropy to -|A|
            self.target_entropy = -self.action_nums
            self.alpha_optim = Adam([self.log_alpha], lr=self.lr)

        self.actor = GaussianPolicy(num_inputs, self.action_nums, self.hidden_size).to(self.device)
        self.policy_optim = Adam(self.actor.parameters(), lr=self.lr)

        self.train()
        self.critic_target.train()
        print(self.actor,self.critic)

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(-1., 1.0)
        return action.detach()

    def update_critic(self, obs, action, reward, next_obs, step):
        dist = self.actor(next_obs)

        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + ( self.gamma * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        # Optimize the critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        return critic_loss.item()

    def update_actor_and_alpha(self, obs, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # optimize the actor
        self.policy_optim.zero_grad()
        actor_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            self.alpha_optim.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.alpha_optim.step()

        return actor_loss.item(), self.alpha.item(), -log_prob.mean()
    
    def update_parameters(self, memory, batch_size=256, iter=2):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)

        critic_loss = self.update_critic(state_batch, action_batch, reward_batch, next_state_batch, iter)

        actor_loss, alpha, log_prob = self.update_actor_and_alpha(state_batch, iter)

        if iter % self.target_update_interval == 0:
            self.soft_update_params(self.critic, self.critic_target,
                                    self.tau)

        return critic_loss, actor_loss, alpha, log_prob

    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data +
                                    (1 - tau) * target_param.data)
        
    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()