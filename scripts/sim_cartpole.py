import argparse

import joblib
import tensorflow as tf

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import deterministic_rollout
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    args = parser.parse_args()

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]
    with tf.Session() as sess:
        data = joblib.load(args.file)
        policy = data['policy']
        env = data['env']
        while True:
            path = deterministic_rollout(env, policy, max_path_length=args.max_path_length,
                           animated=True, speedup=args.speedup, always_return_paths=True)
            #obs = path["observations"]
            #print("observations 0, x,xdot: ", obs[0,0:2])
            #print("observations 0, theta, thetadot",obs[0,2:4]*180/np.pi)
            #print("observations end, x,xdot: ", obs[-1,0:2])
            #print("observations end, theta, thetadot",obs[-1,2:4]*180/np.pi)
            print("sum of rewards: ", np.sum(path["rewards"]))
            if not query_yes_no('Continue simulation?'):
                break
