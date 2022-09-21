import numpy as np
from brax import envs
from brax.io import html
import streamlit.components.v1 as components
import brax.jumpy as jp

env = envs.create(env_name='ant')
state = env.reset(rng=jp.random_prngkey(seed=0))
components.html(html.render(env.sys, [state.qp]), height=500)

# %%time
rollout = []
for i in range(100):
  # wiggle sinusoidally with a phase shift per actuator
  action = jp.sin(i * jp.pi / 15 + jp.arange(0, env.action_size) * jp.pi)
  state = env.step(state, action)
  rollout.append(state)
# jit compile env.step:
state = jax.jit(env.step)(state, jnp.ones((env.action_size,)))
# %%time
for _ in range(100):
  state = jax.jit(env.step)(state, jnp.ones((env.action_size,)))
  
components.html(html.render(env.sys, [s.qp for s in rollout]),height=500)
Image(image.render(env.sys, [s.qp for s in rollout], width=320, height=240))
