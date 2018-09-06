from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.rocky.tf.algos.npo import NPO
from rllab.core.serializable import Serializable


class PPO(NPO, Serializable):
    """
    Penalized Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict(name="optimizer")
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        super(PPO, self).__init__(optimizer=optimizer, **kwargs)