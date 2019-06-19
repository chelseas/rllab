import joblib
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout, deterministic_rollout
from rllab.envs.gym_env import GymEnv
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.envs.vary_wrapper import VaryMassRolloutWrapper


def sim(file="",isdeterm=True, speedup=1, max_path_length=200, animated=True, wrap_env=False)
    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]
    if isdeterm:
        rollout_fn = deterministic_rollout
    else:
        rollout_fn = rollout 

    with tf.Session() as sess:
        data = joblib.load(file)
        policy = data['policy']
        if not wrap_env:
            env = data['env']
        else:
            env = VaryMassRolloutWrapper(data['env'])
        while True:
            path = rollout_fn(env, policy, max_path_length=max_path_length,
                           animated=animated, speedup=speedup, always_return_paths=True)
            if not query_yes_no('Continue simulation?'):
                break

sim(file="Users/Chelsea/Dropbox/AAHAA/src/rllab/data/local/experiment/curriculum_training_11142")