

import numpy as np
import pickle as pickle
from sandbox.rocky.tf.misc import tensor_utils


class VecEnvExecutor(object):
    def __init__(self, envs, max_path_length):
        print("Initializing VecEnvExecutor")
        self.envs = envs
        self._action_space = envs[0].action_space
        self._observation_space = envs[0].observation_space
        self.ts = np.zeros(len(self.envs), dtype='int')
        self.max_path_length = max_path_length

    def step(self, action_n):
        #import pdb; pdb.set_trace()
        # if batch_dim == 0:
        #     action_n = [action_n[i,:] for i in range(action_n.shape[0])]
        # elif batch_dim == -1:
        #     action_n = [action_n[:,i] for i in range(action_n.shape[-1])]
        # else:
        #     print("You must specify whether batch dim is first or last")
        #     raise NotImplementedError

        all_results = [env.step(a) for (a, env) in zip(action_n, self.envs)]
        obs, rewards, dones, env_infos = list(map(list, list(zip(*all_results))))
        dones = np.asarray(dones)
        rewards = np.asarray(rewards)
        self.ts += 1
        if self.max_path_length is not None:
            dones[self.ts >= self.max_path_length] = True
        for (i, done) in enumerate(dones):
            if done:
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        #import pdb; pdb.set_trace()
        return obs, rewards, dones, tensor_utils.stack_tensor_dict_list(env_infos)

    def reset(self):
        #import pdb; pdb.set_trace()
        results = [env.reset() for env in self.envs]
        self.ts[:] = 0
        return results

    @property
    def num_envs(self):
        return len(self.envs)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def terminate(self):
        pass
