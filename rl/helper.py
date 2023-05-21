import torch
import numpy as np
from termcolor import colored

class Episode(object):
    """Storage object for a single episode."""

    def __init__(self, cfg, init_obs, init_state=None):
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.obs = torch.empty(
            (cfg.episode_length + 1, *init_obs.shape),
            dtype=torch.uint8,
            device=self.device,
        )
        self.obs[0] = torch.tensor(init_obs, dtype=torch.uint8, device=self.device)
        self.state = torch.empty(
            (cfg.episode_length + 1, *init_state.shape),
            dtype=torch.float32,
            device=self.device,
        )
        self.state[0] = torch.tensor(
            init_state, dtype=torch.float32, device=self.device
        )
        self.action = torch.empty(
            (cfg.episode_length, cfg.action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.reward = torch.empty(
            (cfg.episode_length,), dtype=torch.float32, device=self.device
        )
        self.cumulative_reward = 0
        self.done = False
        self._idx = 0

    @classmethod
    def from_trajectory(cls, cfg, obs, states, action, reward, done=None):
        """Constructs an episode from a trajectory."""
        episode = cls(cfg, obs[0], states[0])
        episode.obs[1:] = torch.tensor(
            obs[1:], dtype=episode.obs.dtype, device=episode.device
        )
        episode.state[1:] = torch.tensor(
            states[1:], dtype=episode.state.dtype, device=episode.device
        )
        episode.action = torch.tensor(
            action, dtype=episode.action.dtype, device=episode.device
        )
        episode.reward = torch.tensor(
            reward, dtype=episode.reward.dtype, device=episode.device
        )
        episode.cumulative_reward = torch.sum(episode.reward)
        episode.done = True
        episode._idx = cfg.episode_length
        return episode

    def __len__(self):
        return self._idx

    @property
    def first(self):
        return len(self) == 0

    def __add__(self, transition):
        self.add(*transition)
        return self

    def add(self, obs, state, action, reward, done):
        self.obs[self._idx + 1] = torch.tensor(
            obs, dtype=self.obs.dtype, device=self.obs.device
        )
        self.state[self._idx + 1] = torch.tensor(
            state, dtype=self.state.dtype, device=self.state.device
        )
        self.action[self._idx] = action
        self.reward[self._idx] = reward
        self.cumulative_reward += reward
        self.done = done
        self._idx += 1


class ReplayBuffer(object):
    """
    Storage and sampling functionality for training MoDem.
    Uses prioritized experience replay by default.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.capacity = 2 * cfg.train_steps + 1
        obs_shape = (3, *cfg.obs_shape[-2:])
        self._state_dim = cfg.state_dim
        self._obs = torch.empty(
            (self.capacity + 1, *obs_shape), dtype=torch.uint8, device=self.device
        )
        self._last_obs = torch.empty(
            (self.capacity // cfg.episode_length, *cfg.obs_shape),
            dtype=torch.uint8,
            device=self.device,
        )
        self._action = torch.empty(
            (self.capacity, cfg.action_dim), dtype=torch.float32, device=self.device
        )
        self._reward = torch.empty(
            (self.capacity,), dtype=torch.float32, device=self.device
        )
        self._state = torch.empty(
            (self.capacity, self._state_dim), dtype=torch.float32, device=self.device
        )
        self._last_state = torch.empty(
            (self.capacity // cfg.episode_length, self._state_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self._priorities = torch.ones(
            (self.capacity,), dtype=torch.float32, device=self.device
        )
        self._eps = 1e-6
        self.idx = 0

    def __len__(self):
        return self.idx

    def __add__(self, episode: Episode):
        self.add(episode)
        return self

    def add(self, episode: Episode):
        obs = episode.obs[:-1, -3:]
        if episode.obs.shape[1] == 3:
            last_obs = episode.obs[-self.cfg.frame_stack :].view(
                self.cfg.frame_stack * 3, *self.cfg.obs_shape[-2:]
            )
        else:
            last_obs = episode.obs[-1]
        self._obs[self.idx : self.idx + self.cfg.episode_length] = obs
        self._last_obs[self.idx // self.cfg.episode_length] = last_obs
        self._action[self.idx : self.idx + self.cfg.episode_length] = episode.action
        self._reward[self.idx : self.idx + self.cfg.episode_length] = episode.reward
        states = torch.tensor(episode.state, dtype=torch.float32)
        self._state[
            self.idx : self.idx + self.cfg.episode_length, : self._state_dim
        ] = states[:-1]
        self._last_state[
            self.idx // self.cfg.episode_length, : self._state_dim
        ] = states[-1]
        max_priority = (
            1.0
            if self.idx == 0
            else self._priorities[: self.idx].max().to(self.device).item()
        )
        mask = (
            torch.arange(self.cfg.episode_length)
            >= self.cfg.episode_length - self.cfg.horizon
        )
        new_priorities = torch.full(
            (self.cfg.episode_length,), max_priority, device=self.device
        )
        new_priorities[mask] = 0
        self._priorities[self.idx : self.idx + self.cfg.episode_length] = new_priorities
        self.idx = (self.idx + self.cfg.episode_length) % self.capacity

    def update_priorities(self, idxs, priorities):
        self._priorities[idxs] = priorities.squeeze(1).to(self.device) + self._eps

    def _get_obs(self, arr, idxs):
        obs = torch.empty(
            (self.cfg.batch_size, self.cfg.frame_stack * 3, *arr.shape[-2:]),
            dtype=arr.dtype,
            device=torch.device("cuda"),
        )
        obs[:, -3:] = arr[idxs].cuda(non_blocking=True)
        _idxs = idxs.clone()
        mask = torch.ones_like(_idxs, dtype=torch.bool)
        for i in range(1, self.cfg.frame_stack):
            mask[_idxs % self.cfg.episode_length == 0] = False
            _idxs[mask] -= 1
            obs[:, -(i + 1) * 3 : -i * 3] = arr[_idxs].cuda(non_blocking=True)
        return obs.float()

    def sample(self):
        probs = self._priorities[: self.idx] ** self.cfg.per_alpha
        probs /= probs.sum()
        total = len(probs)
        idxs = torch.from_numpy(
            np.random.choice(
                total, self.cfg.batch_size, p=probs.cpu().numpy(), replace=True
            )
        ).to(self.device)
        weights = (total * probs[idxs]) ** (-self.cfg.per_beta)
        weights /= weights.max()
        obs = (
            self._get_obs(self._obs, idxs)
            if self.cfg.frame_stack > 1
            else self._obs[idxs].cuda(non_blocking=True).float()
        )
        next_obs_shape = (3 * self.cfg.frame_stack, *self._last_obs.shape[-2:])
        next_obs = torch.empty(
            (self.cfg.horizon + 1, self.cfg.batch_size, *next_obs_shape),
            dtype=torch.float32,
            device=obs.device,
        )
        action = torch.empty(
            (self.cfg.horizon + 1, self.cfg.batch_size, *self._action.shape[1:]),
            dtype=torch.float32,
            device=self.device,
        )
        reward = torch.empty(
            (self.cfg.horizon + 1, self.cfg.batch_size),
            dtype=torch.float32,
            device=self.device,
        )
        state = self._state[idxs, : self._state_dim]
        next_state = torch.empty(
            (self.cfg.horizon + 1, self.cfg.batch_size, *state.shape[1:]),
            dtype=torch.float32,
            device=state.device,
        )
        for t in range(self.cfg.horizon + 1):
            _idxs = idxs + t
            next_obs[t] = (
                self._get_obs(self._obs, _idxs + 1)
                if self.cfg.frame_stack > 1
                else self._obs[_idxs + 1].cuda(non_blocking=True)
            )
            action[t] = self._action[_idxs]
            reward[t] = self._reward[_idxs]
            next_state[t] = self._state[_idxs + 1, : self._state_dim]

        mask = (_idxs + 1) % self.cfg.episode_length == 0
        next_obs[-1, mask] = (
            self._last_obs[_idxs[mask] // self.cfg.episode_length]
            .to(next_obs.device, non_blocking=True)
            .float()
        )
        state = state.cuda(non_blocking=True)
        next_state[-1, mask] = (
            self._last_state[_idxs[mask] // self.cfg.episode_length, : self._state_dim]
            .to(next_state.device)
            .float()
        )
        next_state = next_state.cuda(non_blocking=True)
        next_obs = next_obs.cuda(non_blocking=True)
        action = action.cuda(non_blocking=True)
        reward = reward.unsqueeze(2).cuda(non_blocking=True)
        idxs = idxs.cuda(non_blocking=True)
        weights = weights.cuda(non_blocking=True)
        return obs, next_obs, action, reward, state, next_state, idxs, weights
    
    
def prefill(arr, chunk_size=20_000):
	capacity = arr.shape[0]
	for i in range(0, capacity+1, chunk_size):
		chunk = min(chunk_size, capacity-i)
		arr[i:i+chunk] = np.random.randn(chunk, *arr.shape[1:])
		
	return arr

def prefill_memory(capacity, obs_shape):
	obses = []
	if len(obs_shape) > 1:
		c, h, w = obs_shape
		for _ in range(capacity):
			frame = np.ones((c, h, w), dtype=np.uint8)
			obses.append(frame)
	else:
		for _ in range(capacity):
			obses.append(np.ones(obs_shape[0], dtype=np.float32))
	print(colored("prefill replay buffer...", color="cyan") )
	return np.stack(obses, axis=0)


class EfficientPrioritizedReplayBuffer():
	def __init__(self,
		obs_shape: tuple,
		state_shape: tuple,
		action_shape: tuple,
		capacity: int,
		batch_size: int,
		prioritized_replay: bool,
		alpha: float,
		beta: float,
		ensemble_size: int,
		device: torch.device='cuda',
		prefilled=True,
		episode_length=50,
		observation_type="image",
		use_single_image=False,
		):
		self.capacity = capacity
		self.batch_size = batch_size
		self.prioritized_replay = prioritized_replay
		self.device = device
		self.state_shape = state_shape
		self.episode_length = episode_length
		self.obs_shape = obs_shape
		self.observation_type = observation_type

		state = len(obs_shape) == 1

		self.use_single_image = use_single_image
		if self.use_single_image:
			obs_shape = list(obs_shape)
			obs_shape[0] = int(obs_shape[0] / 2)
			obs_shape = tuple(obs_shape)

		if prefilled:
			self._obs = prefill_memory(capacity, obs_shape)
			self._last_obs = prefill_memory(capacity//self.episode_length, obs_shape)
			if self.state_shape:
				self._state = prefill_memory(capacity, state_shape)
				self._last_state = prefill_memory(capacity//self.episode_length, state_shape)

		else:
			self._obs = np.empty((capacity, *obs_shape), dtype=np.float32 if state else np.uint8)
			self._last_obs = np.empty((capacity, *obs_shape), dtype=np.float32 if state else np.uint8)

			if self.state_shape:
				self._state = np.empty((capacity, *state_shape), dtype=np.float32)
				self._last_state = np.empty((capacity, *state_shape), dtype=np.float32)
	
		self._action = prefill(np.empty((capacity, *action_shape), dtype=np.float32))
		self._reward = prefill(np.empty((capacity,), dtype=np.float32))
	
		self._alpha = alpha
		self._beta = beta
		self._ensemble_size = ensemble_size
		self._priority_eps = 1e-6
		self._priorities = np.ones((capacity, ensemble_size), dtype=np.float32)

		self.ep_idx = 0
		self.idx = 0
		self.full = False

	def obs_process(self, obs):
		if self.observation_type in ["image", "state+image"]:
			return torch.as_tensor(obs).cuda().float().div(255)
		elif self.observation_type == "state":
			return torch.as_tensor(obs).cuda().float()
		else:
			raise NotImplementedError
			
	def add(self, obs, state, action, reward, next_obs, next_state):
		obs = obs[:3] if self.use_single_image else obs
		next_obs = next_obs[:3] if self.use_single_image else next_obs
		self._obs[self.idx] = obs
		self._action[self.idx] = action
		self._reward[self.idx] = reward

		if (self.idx+1)%self.episode_length==0:
			self._last_obs[self.idx//self.episode_length] = next_obs 
			if self.state_shape:
				self._last_state[self.idx//self.episode_length] = next_state

		if self.state_shape:
			self._state[self.idx] = state

		if self.prioritized_replay:
			if self.full:
				self._max_priority = self._priorities.max()
			elif self.idx == 0:
				self._max_priority = 1.0
			else:
				self._max_priority = self._priorities[:self.idx].max()
			new_priorities = self._max_priority # the latest one has the max priority
			self._priorities[self.idx] = new_priorities

		self.idx = (self.idx + 1) % self.capacity
		self.full = self.full or self.idx == 0

	@property
	def max_priority(self):
		return self._max_priority if self.prioritized_replay else 0

	def update_priorities(self, idxs, priorities:np.ndarray, idx:int=None):
		if not self.prioritized_replay:
			return
		self._priorities[idxs, idx] = priorities + self._priority_eps # add epsilon for > 0

	def get_next_obs(self, idxs):

		new_idxs = (idxs+1) % self.capacity
		next_obs = []
		for idx in new_idxs:
			if ( (idx) % self.episode_length==0):
				next_obs.append( self._last_obs[ (idx)//self.episode_length -1])
			else:
				next_obs.append(self._obs[idx])
		return np.stack(next_obs, axis=0)


	def get_next_state(self, idxs):
		new_idxs = (idxs+1) % self.capacity
		next_state = []
		for idx in new_idxs:
			if ( (idx) % self.episode_length==0):
				next_state.append( self._last_state[(idx)//self.episode_length -1])
			else:
				next_state.append(self._state[idx])
		return np.stack(next_state, axis=0)


    		
	def uniform_sample(self, idx=None):
		if idx is None:
			idx = 0
		# uniform sampling
		probs = self._priorities[:, idx] if self.full else self._priorities[:self.idx, idx]
		probs[:] = 1
		probs[self.idx-1] = 0
		probs /= probs.sum()
		total = len(probs)
		idxs = np.random.choice(total, self.batch_size, p=probs, replace=False)
		obs, next_obs = self._obs[idxs], self.get_next_obs(idxs)

		obs = self.obs_process(obs)
		next_obs = self.obs_process(next_obs)
		actions = torch.as_tensor(self._action[idxs]).cuda()
		rewards = torch.as_tensor(self._reward[idxs]).cuda()

		if self.state_shape:
			state, next_state = self._state[idxs], self.get_next_state(idxs)
			state, next_state = torch.as_tensor(state).cuda(), torch.as_tensor(next_state).cuda()
		else:
			state, next_state = None, None

		return obs, state, actions, rewards.unsqueeze(1), next_obs, next_state

	def prioritized_sample(self, idx=None):
		if idx is None:
			idx = 0
		probs =self._priorities[:, idx]** self._alpha if self.full else self._priorities[:self.idx, idx] ** self._alpha
		probs[self.idx-1] = 0
		probs /= probs.sum()
		total = len(probs)
		idxs = np.random.choice(total, self.batch_size, p=probs, replace=False)
		weights = (total * probs[idxs]) ** (-self._beta)
		weights /= weights.max()

		obs, next_obs = self._obs[idxs], self.get_next_obs(idxs)
		obs = self.obs_process(obs)
		next_obs = self.obs_process(next_obs)


		actions = torch.as_tensor(self._action[idxs]).cuda()
		rewards = torch.as_tensor(self._reward[idxs]).cuda()

		if self.state_shape:
			state, next_state = self._state[idxs], self.get_next_state(idxs)
			state, next_state = torch.as_tensor(state).cuda(), torch.as_tensor(next_state).cuda()
		else:
			state, next_state = None, None

		return obs, state, actions, rewards.unsqueeze(1), next_obs, next_state, idxs, weights

	def sample(self, idx=None, step=None):
		if self.prioritized_replay:
			return self.prioritized_sample(idx)
		else:
			return self.uniform_sample(idx)

	def save(self, fp=None):
		pass