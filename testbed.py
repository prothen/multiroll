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


numpy.random.seed(1)


STEP_ACTIVE = False
DISPLAY_ACTIVE = True
PROFILING = True
DEBUG = False
PLOT_STEPS = 5

H = 50
W = 50
N_STEPS = 500
N_AGENTS = 40
# 400 did not work
# 200 worked: 5ms (just fetching controls)
N_CITIES = 8
N_CONNECTIVITY = 32
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
        obs_builder_object=GlobalObsForRailEnv(),
        remove_agents_at_target=True,
        record_steps=True
        )


#agent_render_variant=AgentRenderVariant.BOX_ONLY,
env_renderer = RenderTool(env,
                          gl='PGL',
                          show_debug=False,
                          screen_height=1080,
                          screen_width=1920)

timestamp = time.time()


def start_timeme():
    global timestamp
    timestamp = time.time()


def timeme(message):
    global timestamp
    print(message, '\n\t({:4}s)'.format(time.time() - timestamp))
    timestamp = time.time()

def main():
    global DISPLAY_ACTIVE
    timeme('start')
    env.reset()
    env_renderer.reset()
    timeme('RESET: ')
    print('##Testbed: Instantiate MyGraph.')

    graph.set_env(env)
    g = graph.MyGraph(env_renderer, debug_is_enabled=DEBUG)

    timeme('Graph Setup: ')
    for step in range(N_STEPS):
        print('##IT', step)
        start_timeme()
        controls = g.controls()
        timeme('Graph Controls: ')
        print('##Testbed: Apply controls:\n', controls)
        env.step(controls)
        timeme('Env step: ')
        g.update_agent_states()
        timeme('Graph Update agents: ')
        print(DISPLAY_ACTIVE)
        print(type(DISPLAY_ACTIVE))
        if DISPLAY_ACTIVE:
            if (not step % PLOT_STEPS):
                print('visualise request')
                aoutohaeun
                #g.visualise()
        timeme('Graph visualise: ')
        #if STEP_ACTIVE:
        #    input('## --> Continue?')

    g.visualise(show=True)
    input('##Testbed: Completed! Press any key to close.')

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--display', type=str2bool, dest='display_active', help='Enable visualisation', default=False)

if __name__ == "__main__":
    args = parser.parse_args()
    DISPLAY_ACTIVE = args.display_active
    main()
