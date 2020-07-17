#!/usr/bin/env python

import numpy
import pandas
import networkx
import matplotlib

import itertools
import copy
import enum
import collections
# from multiprocessing import Process, Value

import graph

# from flatland.core.grid.grid4 import Grid4TransitionsEnum
# from flatland.core.grid.grid4_utils import get_new_position

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv
from flatland.utils.rendertools import RenderTool, AgentRenderVariant
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters



class EmptyObservation(ObservationBuilder):
    def __init__(self):
        super().__init__()
        print(self.__dict__)

    def reset(self):
        pass

    def get(self, handle: int = 0):
        #print(self.env.agents[0].status)
        print('custom observation')
        return 'lol'

numpy.random.seed(1)

params_malfunction = MalfunctionParameters(
        malfunction_rate=30,  # Rate of malfunction occurence
        min_duration=3,  # Minimal duration of malfunction
        max_duration=20  # Max duration of malfunction
        )

speed_ration_map = {1.: 0.25,  # Fast passenger train
                    1. / 2.: 0.25,  # Fast freight train
                    1. / 3.: 0.25,  # Slow commuter train
                    1. / 4.: 0.25}  # Slow freight train

env = RailEnv(
        width=40,
        height=40,
        rail_generator=sparse_rail_generator(
            max_num_cities=2,  # Number of cities in map (where train stations are)
            seed=14,  # Random seed
            grid_mode=False,
            max_rails_between_cities=2,
            max_rails_in_city=8,
            ),
        schedule_generator=sparse_schedule_generator(
            speed_ration_map),
        number_of_agents=1,
        malfunction_generator_and_process_data=malfunction_from_params(
            params_malfunction),  # Malfunction data generator
        obs_builder_object=GlobalObsForRailEnv(),  #EmptyObservation(), #GlobalObsForRailEnv(),
        remove_agents_at_target=True,
        record_steps=True
        )

env_renderer = RenderTool(env, gl="PILSVG",
                          agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                          show_debug=True,
                          screen_height=1000,
                          screen_width=1000)


if __name__ == "__main__":
    print('start test program')

    env.reset()
    env_renderer.reset()
    env_renderer.render_env(show=True, show_inactive_agents=True, show_observations=False, show_predictions=False)
    # import sys
    # sys.exit(0)
    g = graph.MyGraph(env)
    for step in range(500):
        print('Step {}'.format(step))
        env.step(dict((a,0) for a in range(env.get_num_agents())))
        print('Step {}'.format(step))
    env_renderer.render_env(show=True, show_predictions=False, show_observations=False)

    input('press to close')
    #g = MyGraph()
    # select random agent initial position (Use GridTransitionMap)
    
    # select available actions (decode the uint16_t code)
    # if multiple transitions create node with coordinate
