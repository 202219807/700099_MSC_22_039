# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs football_env on OpenAI's ddpg."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
from absl import app
from absl import flags

# baselines
from baselines import logger
from baselines.bench import monitor
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.ddpg.ddpg import DDPG
from baselines.ddpg.models import mlp

# gfootball
import gfootball.env as football_env
from gfootball.examples import models  

# tensorflow
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('level', 'academy_empty_goal_close', 'Defines type of problem being solved')
flags.DEFINE_enum('state', 'extracted_stacked', ['extracted', 'extracted_stacked'], 'Observation to be used for training.')
flags.DEFINE_enum('reward_experiment', 'scoring', ['scoring', 'scoring,checkpoints'], 'Reward to be used for training.')
flags.DEFINE_enum('policy', 'cnn', ['cnn', 'lstm', 'mlp', 'impala_cnn', 'gfootball_impala_cnn'], 'Policy architecture')
flags.DEFINE_integer('num_timesteps', int(2e6), 'Number of timesteps to run for.')
flags.DEFINE_integer('num_envs', 8, 'Number of environments to run in parallel.')
flags.DEFINE_integer('nsteps', 128, 'Number of environment steps per epoch; ' 'batch size is nsteps * nenv')
flags.DEFINE_integer('noptepochs', 4, 'Number of updates per epoch.')
flags.DEFINE_integer('nminibatches', 8, 'Number of minibatches to split one epoch to.')
flags.DEFINE_integer('save_interval', 100, 'How frequently checkpoints are saved.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('lr', 0.00008, 'Learning rate')
flags.DEFINE_float('ent_coef', 0.01, 'Entropy coeficient')
flags.DEFINE_float('gamma', 0.993, 'Discount factor')
flags.DEFINE_float('cliprange', 0.27, 'Clip range')
flags.DEFINE_float('max_grad_norm', 0.5, 'Max gradient norm (clipping)')
flags.DEFINE_bool('render', False, 'If True, environment rendering is enabled.')
flags.DEFINE_bool('dump_full_episodes', False, 'If True, trace is dumped after every episode.')
flags.DEFINE_bool('dump_scores', False, 'If True, sampled traces after scoring are dumped.')
flags.DEFINE_string('load_path', None, 'Path to load initial checkpoint from.')

"""
In this implementation, we use a simple multi-
layer perceptron (MLP) with two hidden layers, 
each having 256 units. 

Actor network uses a tanh activation function in 
the output layer to ensure that the actions are 
within the range [-1, 1], which is often suitable 
for continuous action spaces. The critic network 
takes both observations and actions as input and 
outputs a single value representing the Q-value.

Customize the network architecture by adjusting 
the hidden_sizes list or using different activa-
tion functions. 
"""

# Define the actor network architecture
def create_actor_network(observation_space, action_space):
    assert isinstance(observation_space, gym.spaces.Box)
    assert isinstance(action_space, gym.spaces.Box)
    
    # Define the sizes of the hidden layers in the actor network
    hidden_sizes = [256, 256]
    
    # Create the actor network using a multi-layer perceptron (MLP)
    actor_network = mlp(
        input_shape=observation_space.shape,
        output_size=action_space.shape[0],  # Assuming action_space.shape[0] gives the number of actions
        hidden_sizes=hidden_sizes,
        activation=tf.nn.relu,
        output_activation=tf.nn.tanh  # Assuming the actions are normalized to [-1, 1]
    )
    
    return actor_network

# Define the critic network architecture
def create_critic_network(observation_space, action_space):
    assert isinstance(observation_space, gym.spaces.Box)
    assert isinstance(action_space, gym.spaces.Box)
    
    # Define the sizes of the hidden layers in the critic network
    hidden_sizes = [256, 256]
    
    # Create the critic network using a multi-layer perceptron (MLP)
    # Concatenate the observation and action spaces as input to the critic network
    critic_network = mlp(
        input_shape=observation_space.shape[0] + action_space.shape[0],
        output_size=1,  # The critic network outputs a single value (Q-value)
        hidden_sizes=hidden_sizes,
        activation=tf.nn.relu,
        output_activation=None  # No activation function for the critic's output
    )
    
    return critic_network

def create_single_football_env(iprocess):
  """Creates gfootball environment."""
  env = football_env.create_environment(
      env_name=FLAGS.level, stacked=('stacked' in FLAGS.state),
      rewards=FLAGS.reward_experiment,
      logdir=logger.get_dir(),
      write_goal_dumps=FLAGS.dump_scores and (iprocess == 0),
      write_full_episode_dumps=FLAGS.dump_full_episodes and (iprocess == 0),
      render=FLAGS.render and (iprocess == 0),
      dump_frequency=50 if FLAGS.render and iprocess == 0 else 0)
  env = monitor.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(),
                                                               str(iprocess)))
  return env

def train(_):
    """Trains a DDPG policy."""
    vec_env = SubprocVecEnv([
        (lambda _i=i: create_single_football_env(_i))
        for i in range(FLAGS.num_envs)
    ], context=None)

    import tensorflow.compat.v1 as tf
    ncpu = multiprocessing.cpu_count()
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()

    # Define the neural network architecture for the actor and critic networks
    actor_network = create_actor_network(vec_env.observation_space, vec_env.action_space)
    critic_network = create_critic_network(vec_env.observation_space, vec_env.action_space)

    # Create the DDPG agent
    agent = DDPG(actor_network=actor_network,
                 critic_network=critic_network,
                 gamma=FLAGS.gamma,
                 lr_actor=FLAGS.lr,
                 lr_critic=FLAGS.lr,
                 clipnorm=FLAGS.max_grad_norm)

    # Train the agent
    agent.learn(total_timesteps=FLAGS.num_timesteps, env=vec_env)

if __name__ == '__main__':
    app.run(train)