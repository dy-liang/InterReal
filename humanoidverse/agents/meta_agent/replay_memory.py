import numpy as np

class ParallelReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, num_envs=2):
        self.capacity = int(capacity)  
        self.num_envs = num_envs  
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.position = 0  
        self.epoch = 0
        self.full = False
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)

    def push(self, states, actions, rewards, next_states):
        assert states.shape == (self.num_envs, self.state_dim), f"预期维度({self.num_envs}, {self.state_dim})"
        idx = self.position
        if idx + self.num_envs > self.capacity: 
            delta_idx = idx+self.num_envs - self.capacity
        else:
            delta_idx = self.num_envs
        if idx + self.num_envs >= self.capacity: self.full = True

        self.states[idx:idx+delta_idx] = states.cpu().numpy()[:delta_idx]
        self.actions[idx:idx+delta_idx] = actions.cpu().numpy()[:delta_idx]
        self.rewards[idx:idx+delta_idx] = rewards.cpu().numpy().reshape(-1,1)[:delta_idx]
        self.next_states[idx:idx+delta_idx] = next_states.cpu().numpy()[:delta_idx]
        self.position = (self.position + self.num_envs) % self.capacity

    def sample(self, batch_size):

        idxs = np.random.choice(self.position if not self.full else self.capacity, size=batch_size)
        if not self.full:
            idxs_next = np.where(idxs+1*self.num_envs>=self.position, idxs, idxs+1*self.num_envs)
        else:
            idxs_next = np.where(idxs+1*self.num_envs>=self.capacity, idxs, idxs+1*self.num_envs)
        return self.states[idxs], self.actions[idxs], self.rewards[idxs_next], self.next_states[idxs_next]