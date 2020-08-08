#!/usr/bin/env python


import copy
import enum
import time
import numpy
import pandas
import itertools
import matplotlib
import collections


import networkx

from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool
from flatland.utils.rendertools import AgentRenderVariant
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator


from multiroll import *
from multiroll.timeme import *


numpy.random.seed(1)


STEP_ACTIVE = True
DISPLAY_ACTIVE = True
PROFILING = True
DEBUG = True
PLOT_STEPS = 5

H = 50
W = 50
N_STEPS = 100
N_AGENTS = 5
N_CITIES = 8
N_CONNECTIVITY = 8
SEED = 14


gen_schedule = sparse_schedule_generator({1:1})


env = RailEnv(
        width=H,
        height=W,
        rail_generator=sparse_rail_generator(
            max_num_cities=N_CITIES,
            seed=SEED,
            grid_mode=False,
            max_rails_between_cities=2,
            max_rails_in_city=2,
            ),
        schedule_generator=gen_schedule,
        number_of_agents=N_AGENTS,
        obs_builder_object=multiroll.observation.PlaceholderObs(),
        remove_agents_at_target=True,
        record_steps=False
        )


env_renderer = RenderTool(env,
                          gl='PGL',
                          show_debug=True,
                          screen_height=1080,
                          screen_width=1920)


def main():
    timeme_reset()
    env.reset()
    env_renderer.reset()
    timeme('Flatland - Reset: ')

    g = multiroll.graph.MyGraph(env, env_renderer, debug_is_enabled=DEBUG)
    timeme('Graph Setup: ')
    if DISPLAY_ACTIVE or STEP_ACTIVE:
        g.visualise()
        input('####Start testbed?')
        timeme_reset()

    for step in range(N_STEPS):
        print('##IT', step)

        g.update_agent_states()
        timeme('Graph - Update agents: ')

        controls = g.controls()
        timeme('Graph Controls: ')

        env.step(controls)
        timeme('Flatland - Env.step(): ')


        if DISPLAY_ACTIVE and (not step % PLOT_STEPS): 
            g.visualise()
            timeme('Graph - Visualise: ')

        if STEP_ACTIVE:
            input('## --> Continue?'
                  + '\n##############')
            timeme_reset()

    if DISPLAY_ACTIVE:
        g.visualise()
        input('##Testbed: Completed! Press any key to close.')

if __name__ == "__main__":
    args = multiroll.parser.parse_args()
    DISPLAY_ACTIVE = args.display_active
    STEP_ACTIVE = args.step_active

    main()

