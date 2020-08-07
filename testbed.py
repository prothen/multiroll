#!/usr/bin/env python


import copy
import enum
import time
import numpy
import pandas
import networkx
import itertools
import matplotlib
import collections


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

DEBUG = False
H = 50
W = 50
N_AGENTS = 100
N_CITIES = 4
SEED = 14
STEP_ACTIVE = False
PLOT_STEPS = 20


numpy.random.seed(1)

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

import time
timestamp = time.time()

def start_timeme():
    global timestamp
    timestamp = time.time()
    
def timeme(message):
    global timestamp
    print(message, '\n\t({:4}s)'.format(time.time() - timestamp))
    timestamp = time.time()

if __name__ == "__main__":
    timeme('start')
    env.reset()
    env_renderer.reset()
    timeme('RESET: ')
    print('##Testbed: Instantiate MyGraph.')

    graph.set_env(env)
    g = graph.MyGraph(debug_is_enabled=DEBUG)
    timeme('Graph Setup: ')
    #print('##Testbed: Graph initialised! ({:.4}s)'.format(time.time()-t0))
    for step in range(500):
        print('##IT', step)
        start_timeme()
        controls = g.controls()
        timeme('Graph Controls: ')
        #print('##Testbed: Apply controls:\n', controls)
        env.step(controls)
        timeme('Env step: ')
        g.update_agent_states()
        timeme('Graph Update agents: ')
        if not step % PLOT_STEPS:
            g.visualise(env_renderer)
        timeme('Graph visualise: ')
        if STEP_ACTIVE:
            input('## --> Continue?')

    g.visualise(env_renderer)
    env_renderer.gl.show()
    input('##Testbed: Completed! Press any key to close.')


# env.step(dict((a,0) for a in range(env.get_num_agents())))
