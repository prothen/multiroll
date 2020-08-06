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
gen_schedule = sparse_schedule_generator(ratios)

env = RailEnv(
        width=50,
        height=50,
        rail_generator=sparse_rail_generator(
            max_num_cities=4,
            seed=14,
            grid_mode=False,
            max_rails_between_cities=2,
            max_rails_in_city=2,
            ),
        schedule_generator=gen_schedule,
        number_of_agents=1,
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


if __name__ == "__main__":
    print('Run testbed.')
    env.reset()
    env_renderer.reset()
    print('Initialise graph.')
    t0 = time.time()
    graph.set_env(env)
    g = graph.MyGraph()
    print('Graph creation completed!\n\t--> {}s'.format(time.time()-t0))
    for step in range(500):
        print(step)
        controls = g.controls()
        print(controls)
        env.step(controls)
        g.visualise(env_renderer)
        env_renderer.render_env(
                show=True,
                show_agents=True, 
                show_predictions=False, 
                show_observations=False)

    env_renderer.render_env(
            show=True, 
            show_agents=True, 
            show_predictions=False, 
            show_observations=False)
    g.visualise(env_renderer)
    env_renderer.gl.show()
    input('press to close')


# env.step(dict((a,0) for a in range(env.get_num_agents())))
