from sandbox.rocky.tf.envs.base import TfEnv
from rllab.core.serializable import Serializable
import numpy as np

"""
A class to wrap TfEnv to allow for curriculum training. 
"""
class VaryWrapper(TfEnv, Serializable):
	def __init__(self, wrapped_env):
		Serializable.quick_init(self, locals())
		super().__init__(wrapped_env)

	def set_param(self, paramname, paramvalue):
		obj = self._wrapped_env.env.env.env
		if not hasattr(obj, paramname):
			# the object should have this attribute already
			raise AttributeError
		setattr(obj, paramname, paramvalue)

	def get_param(self, paramname):
		obj = self._wrapped_env.env.env.env
		return getattr(obj, paramname)

class VaryMassEnv(VaryWrapper, Serializable):
	def __init__(self, wrapped_env, m0, mf, iters):
		Serializable.quick_init(self, locals())
		super().__init__(wrapped_env)
		self.m0 = m0
		self.mf = mf
		self.iters = iters
		self.calc_gamma(iters - 20)

	def calc_gamma(self, iters):
		if iters > 0:
			self.gamma = (self.mf/self.m0)**(1/iters)
		else:
			self.gamma = 1

	def decay_mass(self, iteration):
		mnew = self.m0*(self.gamma**iteration)
		self.set_param('m', mnew)

	def set_mass_randomly(self):
		mrand = np.random.rand()*(self.mf - self.m0) + self.m0
		self.set_param('m', mrand)


class VaryMassRolloutWrapper():
	def __init__(self, varymassenv):
		self.env = varymassenv
		self.observation_space = self.env._wrapped_env.observation_space
		self.action_space = self.env._wrapped_env.action_space

	def reset(self):
		self.env.set_mass_randomly()
		o = self.env._wrapped_env.reset()
		return o

	def render(self):
		self.env._wrapped_env.render()

	def step(self, action):
		return self.env._wrapped_env.step(action)

	def get_state(self):
		return self.env.get_param('state')

	def set_state(self, s):
		self.env.set_param('state', s)

	def test_set_state(self):
		self.render()
		input('enter to continue')
		for i in range(8):
			angle = i*(np.pi/4)
			self.set_state(np.array([angle, 0.0]))
			self.render()
			input('enter to continue')

	def set_mass_randomly(self):
		self.env.set_mass_randomly()

	

		
