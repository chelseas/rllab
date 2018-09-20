import numpy as np
import sandbox.rocky.tf.core.layers as L
import tensorflow as tf
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.core.network import SimpleRNNNetwork, MLP
from sandbox.rocky.tf.distributions.recurrent_categorical import RecurrentCategorical
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.policies.base import StochasticPolicy

from rllab.core.serializable import Serializable
from rllab.misc import special
from rllab.misc.overrides import overrides


class CategoricalSimpleRNNPolicy(StochasticPolicy, LayersPowered, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            hidden_dim=32, # dim of hidden state
            layer_dim=32, # dim of true "hidden layers" inside recurrent cell
            feature_network=None,
            state_include_action=False, 
            hidden_nonlinearity=tf.nn.relu,
            rnn_layer_cls=L.SimpleRNNLayer,
    ):
        """
        :param env_spec: A spec for the env.
        :param hidden_dim: dimension of hidden layer
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :return:
        """
        with tf.variable_scope(name):
            assert isinstance(env_spec.action_space, Discrete)
            Serializable.quick_init(self, locals())
            super(CategoricalSimpleRNNPolicy, self).__init__(env_spec)

            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            if state_include_action:
                input_dim = obs_dim + action_dim
            else:
                input_dim = obs_dim

            l_input = L.InputLayer(
                shape=(None, None, input_dim),
                name="input"
            )

            if feature_network is None:
                feature_dim = input_dim
                l_flat_feature = None
                l_feature = l_input
            else:
                feature_dim = feature_network.output_layer.output_shape[-1]
                l_flat_feature = feature_network.output_layer
                l_feature = L.OpLayer(
                    l_flat_feature,
                    extras=[l_input],
                    name="reshape_feature",
                    op=lambda flat_feature, input: tf.reshape(
                        flat_feature,
                        tf.stack([tf.shape(input)[0], tf.shape(input)[1], feature_dim])
                    ),
                    shape_op=lambda _, input_shape: (input_shape[0], input_shape[1], feature_dim)
                )

            prob_network = SimpleRNNNetwork(
                input_shape=(feature_dim,),
                input_layer=l_feature,
                output_dim=env_spec.action_space.n,
                hidden_dim=hidden_dim,
                layer_dim=layer_dim,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=None, # not used 
                # tho right now maybe its easy to find as the only softmax in the network
                rnn_layer_cls=rnn_layer_cls,
                name="prob_network"
            )

            self.prob_network = prob_network
            self.feature_network = feature_network
            self.l_input = l_input
            self.state_include_action = state_include_action

            flat_input_var = tf.placeholder(dtype=tf.float32, shape=(None, input_dim), name="flat_input")
            if feature_network is None:
                feature_var = flat_input_var
            else:
                feature_var = L.get_output(l_flat_feature, {feature_network.input_layer: flat_input_var})

            # this function is called to produce the action_weights !!
            print("prev hiddens?", prob_network.step_prev_state_layer.input_var)
            self.f_step_prob = tensor_utils.compile_function(
                # inputs
                [
                    flat_input_var, #observations
                    prob_network.step_prev_state_layer.input_var # previous hiddens or init hiddens?
                ],
                # output(s?)
                L.get_output([
                    prob_network.step_output_layer, # layers for which to 'get_output'
                    # step_output_layer applies a FC layer to the output h. This may be the op I want to grab
                    prob_network.step_hidden_layer  # this is the hiddens, being passed forward for the next recurrence
                ], {prob_network.step_input_layer: feature_var} # a dictionary for inputs. mapping
                    # step_input_layer is an InputLayer , and
                    # feature_var is is the input placeholder (for observations) (called "flat_input")
                ) # question -- why pass inputs twice? The first part of the compile functino seems to pass inputs
                # AND we have this dictionary of inputs?? SO CONFUSING
            )

            self.input_dim = input_dim
            self.action_dim = action_dim
            self.hidden_dim = hidden_dim

            self.prev_actions = None
            self.prev_hiddens = None
            self.dist = RecurrentCategorical(env_spec.action_space.n)

            out_layers = [prob_network.output_layer]
            if feature_network is not None:
                out_layers.append(feature_network.output_layer)

            LayersPowered.__init__(self, out_layers)

    @overrides #  dist_info_sym accepts obs_var of shape [batch_size, max_steps, obs_dim], which it breaks down into tensors of shape [batch_size, obs_dim] before passing it through the RNNlayer's step funtion
    def dist_info_sym(self, obs_var, state_info_vars):
        n_batches = tf.shape(obs_var)[0]
        n_steps = tf.shape(obs_var)[1]
        obs_var = tf.reshape(obs_var, tf.stack([n_batches, n_steps, -1]))
        obs_var = tf.cast(obs_var, tf.float32)
        if self.state_include_action:
            prev_action_var = tf.cast(state_info_vars["prev_action"], tf.float32)
            all_input_var = tf.concat(axis=2, values=[obs_var, prev_action_var])
        else:
            all_input_var = obs_var
        if self.feature_network is None:
            d= dict(
                    prob=L.get_output( # calls layer.get_output_for (the function with tf.scan)
                        self.prob_network.output_layer, # turns out to be rnnlayer
                        {self.l_input: all_input_var}
                    )
                )
            #    d= dict(
            #        prob=L.get_output(
            #            self.prob_network.step_output_layer,
            #            {self.prob_network.step_input_layer: obs_var , self.prob_network.step_prev_state_layer: hiddens}
            #        )
            #    )
            return d
        else:
            flat_input_var = tf.reshape(all_input_var, (-1, self.input_dim))
            return dict(
                prob=L.get_output(
                    self.prob_network.output_layer,
                    {self.l_input: all_input_var, self.feature_network.input_layer: flat_input_var}
                )
            )

    @property
    def vectorized(self):
        return True

    """
    Function to reset only some of a batch?
    """
    def reset(self, dones=None):
        if dones is None:
            dones = [True]
        dones = np.asarray(dones)
        if self.prev_actions is None or len(dones) != len(self.prev_actions):
            self.prev_actions = np.zeros((len(dones), self.action_space.flat_dim))
            self.prev_hiddens = np.zeros((len(dones), self.hidden_dim))
        self.prev_actions[dones] = 0.
        self.prev_hiddens[dones] = self.prob_network.hid_init_param.eval()
        #print("prev hiddens after reset: ", self.prev_hiddens)  # hiddens are of shape [batch_dim, hid_dim]

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    @overrides # get_actions takes observations of the form [num_trajs==batch_dim, obs_dim] (NO STEP DIM!)
    def get_actions(self, observations): 
        flat_obs = self.observation_space.flatten_n(observations)
        if self.state_include_action:
            assert self.prev_actions is not None
            all_input = np.concatenate([
                flat_obs,
                self.prev_actions
            ], axis=-1)
        else:
            all_input = flat_obs
        # actions_weights is the hiddens with a dense layer applid on top, and a softmax on to of that to get action weights
        # hidden_vec -- hidden states, saved for next recurrence
        #
        # f_step_prob expects observations in the shape [n_batch, obs_dim] (with NO step dimension, because it expects you to ask for actions one step at a time)
        # consequently, prev_hiddens is also of size [batch_dim, hid_dim]
        action_weights, hidden_vec = self.f_step_prob(all_input, self.prev_hiddens) # all_input == observations
        # hidden_vec is of shape [batch_dim==numtrajs, hid_dim]

        # action weights:  #[batch?, n_actions] shape and an np.ndarray (type)
        #print("action_weights.shape: ", action_weights.shape)
        #print("type(action_weights): ", type(action_weights))
        # eps-greedy policy: 
        # e = np.random.rand()
        # if e < 0.3:
        #     # choose randomly
        #     actions = np.array([int(np.round(np.random.rand()*(action_weights.shape[1]-1))) for i in range(action_weights.shape[0])])
        # else:
        #     actions = np.argmax(action_weights, axis=1)
        actions = np.argmax(action_weights, axis=1)
        prev_actions = self.prev_actions
        self.prev_actions = self.action_space.flatten_n(actions)
        self.prev_hiddens = hidden_vec
        agent_info = dict(prob=action_weights)
        if self.state_include_action:
            agent_info["prev_action"] = np.copy(prev_actions)
        return actions, agent_info

    @property
    @overrides
    def recurrent(self):
        return True

    @property
    def distribution(self):
        return self.dist

    @property
    def state_info_specs(self):
        if self.state_include_action:
            return [
                ("prev_action", (self.action_dim,)),
            ]
        else:
            return [] #[
                #("hidden_in", (self.hidden_dim,))
            #]
