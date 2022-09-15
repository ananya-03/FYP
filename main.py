import numpy as np
from brax import envs
from brax.io import html
import streamlit.components.v1 as components

env = envs.create(env_name='humanoid')
state = env.reset(rng=jp.random_prngkey(seed=0))
components.html(html.render(env.sys, [state.qp]), height=500)