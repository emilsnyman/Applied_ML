#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

# Common imports
import numpy as np
import os
import PIL
import base64
import imageio
import io
import shutil
import tempfile
import zipfile
import IPython

# to make this notebook's output stable across runs
# np.random.seed(42)
# tf.random.set_seed(42)

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# To get smooth animations
import matplotlib.animation as animation
mpl.rc('animation', html='jshtml')

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rl"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    
def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim

def plot_observation(obs):
    # Since there are only 3 color channels, you cannot display 4 frames
    # with one primary color per frame. So this code computes the delta between
    # the current frame and the mean of the other frames, and it adds this delta
    # to the red and blue channels to get a pink color for the current frame.
    obs = obs.astype(np.float32)
    img = obs[..., :3]
    current_frame_delta = np.maximum(obs[..., 3] - obs[..., :3].mean(axis=-1), 0.)
    img[..., 0] += current_frame_delta
    img[..., 2] += current_frame_delta
    img = np.clip(img / 150, 0, 1)
    plt.imshow(img)
    plt.axis("off")
    
class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")
            
def save_frames(trajectory):
    global frames
    frames.append(tf_env.pyenv.envs[0].render(mode="rgb_array"))
    
def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch
    
def create_zip_file(dirname, base_filename):
    return shutil.make_archive(base_filename, 'zip', dirname)

def unzip_file(zip_file,dirname):
    if not os.path.exists(zip_file):
        print('Zip file does not exist')
        return
    return shutil.unpack_archive(zip_file, dirname,'zip')


# In[2]:


from tf_agents.environments import suite_gym
from tf_agents.environments.wrappers import ActionRepeat
import tf_agents.environments.wrappers
from functools import partial
from gym.wrappers import TimeLimit
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.eval.metric_utils import log_metrics
import logging
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.utils.common import function
from tf_agents.utils import common
from tf_agents.policies.policy_saver import PolicySaver


# In[3]:


# tf.random.set_seed(42)
# np.random.seed(42)

max_episode_steps = 27000 # <=> 108k ALE frames since 1 step = 4 frames
environment_name = "KungFuMasterNoFrameskip-v4"

env = suite_atari.load(
    environment_name,
    max_episode_steps=max_episode_steps,
    gym_env_wrappers=[AtariPreprocessing, FrameStack4])
#     ,env_wrappers=[partial(ActionRepeat, times=4)])

tf_env = TFPyEnvironment(env)


# In[4]:


preprocessing_layer = keras.layers.Lambda(
                          lambda obs: tf.cast(obs, np.float32) / 255.)
conv_layer_params=[(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
fc_layer_params=[512]

q_net = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    preprocessing_layers=preprocessing_layer,
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params,
    dropout_layer_params=[0.1]
    )


# In[5]:


train_step = tf.Variable(0)
update_period = 4 # run a training step every 4 collect steps
optimizer = keras.optimizers.RMSprop(learning_rate=2.5e-4, rho=0.95, momentum=0.0,
                                     epsilon=0.00001, centered=True)

epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.0, # initial ε
    decay_steps=250000 // update_period, # <=> 1,000,000 ALE frames
    end_learning_rate=0.01) # final ε

agent = DqnAgent(tf_env.time_step_spec(),
                 tf_env.action_spec(),
                 q_network=q_net,
                 optimizer=optimizer,
                 target_update_period=2000, # <=> 32,000 ALE frames
                 td_errors_loss_fn=keras.losses.Huber(reduction="none"),
                 gamma=0.99, # discount factor
                 train_step_counter=train_step,
                 epsilon_greedy=lambda: epsilon_fn(train_step))
agent.initialize()


# In[6]:


replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=100000) # reduce if OOM error


# In[7]:


tempdir = os.curdir
checkpoint_dir = os.path.join(tempdir, 'lastModelCheckpoint')

zip_file_name = 'lastModelCheckpoint.zip'
unzip_file(zip_file_name,checkpoint_dir)

# Policy Saver
# policy_dir = os.path.join(tempdir, 'policy')
# tf_policy_saver = policy_saver.PolicySaver(agent.policy)

# saved_policy = tf.saved_model.load(policy_dir)
# saved_policy


# In[8]:


checkpoint_dir


# In[9]:


train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=train_step)


# In[14]:


train_checkpointer.initialize_or_restore()
train_step = agent.train_step_counter
train_step


# In[15]:


agent.collect_policy.get_initial_state(tf_env.batch_size)


# In[16]:


policy_dir = os.path.join(tempdir, 'policy')
saved_policy = tf.saved_model.load(policy_dir)


# In[16]:


train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
    ]

logging.getLogger().setLevel(logging.INFO)
log_metrics(train_metrics)

replay_buffer_observer = replay_buffer.add_batch

collect_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=update_period)

collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)


# In[17]:


#initial_collect_policy = saved_policy#(tf_env.time_step_spec(),tf_env.action_spec())

init_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer.add_batch, ShowProgress(20000)],
    num_steps=20000) # <=> 80,000*4 ALE frames

final_time_step, final_policy_state = init_driver.run()


# In[20]:


def train_agent(n_iterations):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f}".format(
            iteration, train_loss.loss.numpy()), end="")
        if iteration % 1000 == 0:
            log_metrics(train_metrics)
    
#     my_policy = agent.policy
#     saver = PolicySaver(my_policy)
#     saver.save('savedPolicy')
            
dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=3).prefetch(3)


# In[26]:


train_agent(n_iterations=10000)


# In[29]:


frames = []

watch_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,#saved_policy,
    observers=[save_frames, ShowProgress(1000)],
    num_steps=1000)

final_time_step, final_policy_state = watch_driver.run()

plot_animation(frames)


# In[42]:


image_path = os.path.join("images", "rl", "KungFu12.gif")
frame_images = [PIL.Image.fromarray(frame) for frame in frames[0:1000]]
frame_images[0].save(image_path, format='GIF',
                     append_images=frame_images[1:],
                     save_all=True,
                     duration=40,
                     loop=0)


# In[28]:


train_checkpointer.save(train_step)
policy_dir = os.path.join(tempdir, 'savedPolicy')
tf_policy_saver = PolicySaver(agent.policy)
tf_policy_saver.save(policy_dir)

checkpoint_zip_filename = create_zip_file(checkpoint_dir, os.path.join(tempdir,'exported_cp'))


# In[ ]:




