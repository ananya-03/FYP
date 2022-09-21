#@title Import Brax and some helper modules

import functools
import time


import gym

import brax

from brax import envs
from brax import jumpy as jp
from brax.envs import to_torch
from brax.io import html
from brax.io import image
import jax
from jax import numpy as jnp
import streamlit.components.v1 as components
import torch
v = torch.ones(1, device='cuda')  # init torch cuda before jax



#@title Visualizing pre-included Brax environments { run: "auto" }
#@markdown Select an environment to preview it below:

environment = "ant"  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'reacher', 'walker2d', 'fetch', 'grasp', 'ur5e']
env = envs.create(env_name=environment)
state = env.reset(rng=jp.random_prngkey(seed=0))

components.html(html.render(env.sys, [state.qp]))


rollout = []
for i in range(100):
  # wiggle sinusoidally with a phase shift per actuator
  action = jp.sin(i * jp.pi / 15 + jp.arange(0, env.action_size) * jp.pi)
  state = env.step(state, action)
  rollout.append(state)

# jit compile env.step:
state = jax.jit(env.step)(state, jnp.ones((env.action_size,)))


for _ in range(100):
  state = jax.jit(env.step)(state, jnp.ones((env.action_size,)))

components.html(html.render(env.sys, [s.qp for s in rollout]))

components.image(image.render(env.sys, [s.qp for s in rollout], width=320, height=240))

