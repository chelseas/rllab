

from rllab.misc import logger
from rllab.misc import ext
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.algos.mypolopt import MyPolopt
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from sandbox.rocky.tf.misc import tensor_utils
from rllab.core.serializable import Serializable
import tensorflow as tf
from rllab.sampler.utils import rollout, dagger_rollout
import time
from rllab.misc import special

import numpy as np

class Dagger(Serializable):
    """
    DAgger? or BC?.
    """

    def __init__(
            self,
            env,
            policy,
            expert,
            decision_rule,
            optimizer=None,
            optimizer_args=None,
            numtrajs=1, # num trajs to gather between each training pass
            n_itr=500,
            start_itr=0,
            batch_size=512,
            max_path_length=500,
            discount=0.99,
            plot=False,
            fixed_horizon=False,
            pause_for_plot=False,
            **kwargs):
        Serializable.quick_init(self, locals())
        self.policy=policy
        self.env=env
        self.expert=expert
        self.decision_rule = decision_rule
        self.numtrajs = numtrajs
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.max_path_length=max_path_length
        self.discount = discount
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.fixed_horizon = fixed_horizon
        if self.policy.state_info_keys is None:
            self.policy.state_info_keys = ["hidden_in"]
        if optimizer is None:
            default_args = dict(
                batch_size=None,
                max_epochs=1,
            )
            if optimizer_args is None:
                optimizer_args = default_args
            else:
                optimizer_args = dict(default_args, **optimizer_args)
            with self.decision_rule.novice_sess.as_default():
                optimizer = FirstOrderOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.opt_info = None
        with self.decision_rule.novice_sess.as_default():
            self.init_opt()

    @overrides
    def init_opt(self):
        is_recurrent = int(self.policy.recurrent)

        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )

        labels = tf.placeholder(dtype=tf.int32, shape=[None,None], name="expert_actions")
        _ = qj(labels, 'labels', t=True)

        dist = self.policy.distribution

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name='old_%s' % k)
            for k, shape in dist.dist_info_specs
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] * (1 + is_recurrent) + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
            }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]
        print("state_info_vars_list: ", state_info_vars_list)

        if is_recurrent:
            valid_var = tf.placeholder(tf.float32, shape=[None, None], name="valid")
        else:
            valid_var = None

        # get the outputs of the network (truly: a dictionary containing action probabilities)
        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        probs = dist_info_vars['prob']

        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)

        # The gradient of the surrogate objective is the policy gradient
        # shape of valid_var: [numtrajs, maxPathLength]
        if is_recurrent:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=probs,
                labels=labels)
            surr_obj = tf.reduce_sum(losses * valid_var) / tf.reduce_sum(valid_var)
            mean_kl = tf.reduce_sum(kl * valid_var) / tf.reduce_sum(valid_var)
            max_kl = tf.reduce_max(kl * valid_var)

        input_list = [obs_var, labels] + state_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        self.optimizer.update_opt(loss=surr_obj, target=self.policy, inputs=input_list)

        f_kl = tensor_utils.compile_function(
            inputs=input_list + old_dist_info_vars_list,
            outputs=[mean_kl, max_kl],
        )
        self.opt_info = dict(
            f_kl=f_kl,
        )

    @overrides
    def optimize_policy(self, itr, samples_data):
        logger.log("optimizing policy")
        obs = samples_data["observations"]
        labels = samples_data["agent_infos"]["expert_action"] 
        inputs = [obs, labels]
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        print("state_info_list: ", state_info_list)
        inputs += tuple(state_info_list)
        if self.policy.recurrent:
            inputs += (samples_data["valids"],)
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        loss_before = self.optimizer.loss(inputs)
        self.optimizer.optimize(inputs) #self.decision_rule.novice_sess)
        loss_after = self.optimizer.loss(inputs)
        logger.record_tabular("LossBefore", loss_before)
        logger.record_tabular("LossAfter", loss_after)
        tf.summary.scalar("loss", loss_after)

        mean_kl, max_kl = self.opt_info['f_kl'](*(list(inputs) + dist_info_list))
        logger.record_tabular('MeanKL', mean_kl)
        tf.summary.scalar('mean_kl', mean_kl)
        logger.record_tabular('MaxKL', max_kl)

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            env=self.env,
        )

    """
    Do rollouts using DAgger
    """
    def do_rollouts(self):
        paths = []
        for i in range(self.numtrajs):
            path = dagger_rollout(self.env, 
                self.decision_rule,
                max_path_length=self.max_path_length)
            #print("reward: ", sum(path["rewards"]))
            #print("time chose novice action: ", sum(path["agent_infos"]["chose_novice"].astype(int)))
            paths.append(path)
        return paths

    """
    Do some novice rollouts to test the policy
    """
    def novice_rollouts(self):
        paths = []
        with self.decision_rule.novice_sess.as_default():
            for i in range(100):
                path = dagger_rollout(self.env, 
                    self.policy,
                    max_path_length=self.max_path_length, animated=False)
                paths.append(path)

        average_reward = sum([sum(path["rewards"]) for path in paths])/len(paths)
        logger.record_tabular("NoviceAverageReturn", average_reward)
        return True

    """
    Do some expert rollouts for kicks
    """
    def expert_rollouts(self):
        paths = []
        with self.decision_rule.expert_sess.as_default():
            for i in range(100):
                path = dagger_rollout(self.env, 
                    self.expert,
                    max_path_length=self.max_path_length, animated=False)
                paths.append(path)
        average_reward = sum([sum(path["rewards"]) for path in paths])/len(paths)
        logger.record_tabular("ExpertAverageReturn", average_reward)
        return True


    """
    Filter out the best 1000 paths
    """
    def filter_paths(self, paths):
        if len(paths) < 100:
            best_paths = paths
        else:
            sorted_paths = sorted(paths, key=lambda k: sum(k['rewards']), reverse=True)
            best_paths = sorted_paths[:100]
        
        return best_paths

    """
    Concat and pad data so its of the form: [batchsize==numtrajs, maxsteps, var_dim]
    """
    def process_samples(self, itr, paths, log=True):

        # filter out and return the best 1000 paths
        # paths = self.filter_paths(paths)

        returns = []
        for path in paths:
            path["returns"] = special.discount_cumsum(path["rewards"], self.discount)
            returns.append(path["returns"])

        max_path_length = max([len(path["observations"]) for path in paths])
        # make all paths the same length (pad extra advantages with 0)
        obs = [path["observations"] for path in paths]
        obs = tensor_utils.pad_tensor_n(obs, max_path_length)

        actions = [path["actions"] for path in paths]
        actions = tensor_utils.pad_tensor_n(actions, max_path_length)

        rewards = [path["rewards"] for path in paths]
        rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

        returns = [path["returns"] for path in paths]
        returns = tensor_utils.pad_tensor_n(returns, max_path_length)

        agent_infos = [path["agent_infos"] for path in paths]
        agent_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in agent_infos]
            )

        env_infos = [path["env_infos"] for path in paths]
        env_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos]
            )

        valids = [np.ones_like(path["returns"]) for path in paths]
        valids = tensor_utils.pad_tensor_n(valids, max_path_length)

        average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

        undiscounted_returns = [sum(path["rewards"]) for path in paths]

        ent = np.sum(self.policy.distribution.entropy(agent_infos) * valids) / np.sum(valids)

        samples_data = dict(
                observations=obs,
                actions=actions,
                advantages=[None],
                rewards=rewards,
                returns=returns,
                valids=valids,
                agent_infos=agent_infos,
                env_infos=env_infos,
                paths=paths,
            )

        # log data
        if log==True:
            logger.record_tabular('Iteration', itr)
            logger.record_tabular('AverageDiscountedReturn',
                                      average_discounted_return)
            logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
            logger.record_tabular('NumTrajs', len(paths))
            logger.record_tabular('Entropy', ent)
            logger.record_tabular('Perplexity', np.exp(ent))
            logger.record_tabular('StdReturn', np.std(undiscounted_returns))
            logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
            logger.record_tabular('MinReturn', np.min(undiscounted_returns))

        return samples_data

    # todo: need somewhere to store ALL aggregated observation and action data.
    # (maybe store ALL paths, and call process_samples on ALL aggregated paths?)
    # def store_paths(paths):
    #     new_obs = []
    #     new_acts = []
    #     new_hiddens = []
    #     for path in paths:
    #         observations = path['observations']
    #         hiddens = path['agent_infos']['hiddens']
    #         expert_actions = path['agent_infos']['expert_actions']

    def train(self, summary_writer=None, summary=None, store_paths=False): # sess=None
        #created_session = True if (sess is None) else False
        #if sess is None:
        #    sess = tf.Session()
        #    sess.__enter__()   
        #sess.run(tf.global_variables_initializer())
        
        start_time = time.time()
        if not hasattr(self, 'all_paths'):
            self.all_paths = [] # a list of path dicts
        for itr in range(self.start_itr, self.n_itr):
            self.itr = itr
            itr_start_time = time.time()
            with logger.prefix('itr #%d | ' % itr):
                #import pdb; pdb.set_trace()
                logger.log("Obtaining samples...")
                ############################################
                print("Collecting paths with dagger beta of: ", self.decision_rule.beta0 * self.decision_rule.beta_decay ** self.decision_rule.epoch)
                logger.record_tabular('DAgger Beta: ', self.decision_rule.beta0 * self.decision_rule.beta_decay ** self.decision_rule.epoch)
                paths = self.do_rollouts()
                # decay beta by incrementing dagger epoch
                self.decision_rule.epoch += 1
                # log stats (e.g. avg reward n stuff) for newest paths
                _ = self.process_samples(itr, paths, log=True)
                # but get 1 giant batch of data for training (old and new)
                self.all_paths.extend(paths) 
                print("total paths: ", len(self.all_paths))
                samples_data = self.process_samples(itr, self.all_paths, log=False)
                
                ############################################
                logger.log("Logging diagnostics...")
                self.log_diagnostics(paths)
                with self.decision_rule.novice_sess.as_default():
                    logger.log("Optimizing policy...")
                    self.optimize_policy(itr, samples_data)
                    logger.log("Saving snapshot...")
                    params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
                    logger.save_itr_params(itr, params) # this saves everything in params 
                if store_paths:
                    logger.log("Storing paths...")
                    params['paths'] = paths
                logger.log("Saved")
                logger.record_tabular('Time', time.time() - start_time)
                logger.record_tabular('ItrTime', time.time() - itr_start_time)
                
                # test newly optimized policy with some novice rollouts
                logger.log("Doing Novice rollouts...")
                self.novice_rollouts()

                logger.log("Doing Expert rollouts...")
                self.expert_rollouts()

                logger.dump_tabular(with_prefix=False)

                
                if self.plot:
                    with self.decision_rule.novice_sess.as_default():
                        rollout(self.env, self.policy, animated=True, max_path_length=self.max_path_length)
                        if self.pause_for_plot:
                            input("Plotting evaluation run: Press Enter to "
                              "continue...")

        #if created_session:
        #    sess.close()

    def log_diagnostics(self, paths):
        self.env.log_diagnostics(paths)
        self.policy.log_diagnostics(paths)
