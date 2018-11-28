"""shepherd_gym - Gym environment implementation of dog shepherding task"""

import shepherd_gym.models
from gym.envs.registration import register
from shepherd_gym.shepherd_sim import ShepherdSim

register(
    id='Shepherd-v0',
    entry_point='shepherd_gym.envs:ShepherdEnv',
)

register(
    id='Shepherd-v1',
    entry_point='shepherd_gym.envs:ShepherdEnv',
    kwargs={'fixed_reset' : True, 'info_mode' : 1}
)