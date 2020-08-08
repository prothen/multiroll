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
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters

from flatland.utils.rendertools import AgentRenderVariant


import graph
import display
import observation

from timeme import *
from parser import *


numpy.random.seed(1)


STEP_ACTIVE = True
DISPLAY_ACTIVE = True
PROFILING = True
DEBUG = True
PLOT_STEPS = 5

H = 150
W = 150
N_STEPS = 100
N_AGENTS = 1
# 400 did not work
# 200 worked: 5ms (just fetching controls)
N_CITIES = 8
N_CONNECTIVITY = 8
SEED = 14


params_malfunction = MalfunctionParameters(
        malfunction_rate=30, 
        min_duration=3,  
        max_duration=20  
        )

ratios = {1.: 0.25,  
          1. / 2.: 0.25,
          1. / 3.: 0.25,
          1. / 4.: 0.25} 
gen_schedule = sparse_schedule_generator({1:1})

class SimpleObs(ObservationBuilder):
    """
    Simplest observation builder. The object returns observation vectors with 5 identical components,
    all equal to the ID of the respective agent.
    """
    def __init__(self):
        self.observation_space = [5]

    def reset(self):
        return

    def get(self, handle):
        observation = handle * numpy.ones((self.observation_space[0],))
        return observation

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
        #malfunction_generator_and_process_data=malfunction_from_params(
        #    params_malfunction),
        obs_builder_object=SimpleObs(),  #GlobalObsForRailEnv(),
        remove_agents_at_target=True,
        record_steps=True
        )


# agent_render_variant=AgentRenderVariant.BOX_ONLY,
env_renderer = RenderTool(env,
                          gl='PGL',
                          show_debug=False,
                          screen_height=1080,
                          screen_width=1920)


timeme_reset()
env.reset()
env_renderer.reset()
timeme('Flatland - Reset: ')

graph.set_env(env)
g = graph.MyGraph(env_renderer, debug_is_enabled=DEBUG)

timeme('Graph Setup: ')



def main():
    if DISPLAY_ACTIVE or STEP_ACTIVE:
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
            input('## --> Continue?')
            timeme_reset()

    if DISPLAY_ACTIVE:
        g.visualise(show=True)
        input('##Testbed: Completed! Press any key to close.')

if __name__ == "__main__":
    args = parser.parse_args()
    DISPLAY_ACTIVE = args.display_active
    STEP_ACTIVE = args.step_active

    main()

