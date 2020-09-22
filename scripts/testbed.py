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

from multiroll.config import Config

numpy.random.seed(1)

config = Config()

STEP_ACTIVE = True
DISPLAY_ACTIVE = True
DEBUG = True
ROLL_IT = True

H = 40
W = 60
SEED = 14
N_AGENTS = 400
N_CITIES = 9
PLOT_STEPS = 2
N_SIM_STEPS = 100
N_CONNECTIVITY = 4
N_PREDICT_STEPS = 30


gen_schedule = sparse_schedule_generator({1:1})


env = RailEnv(
        width=H,
        height=W,
        rail_generator=sparse_rail_generator(
            max_num_cities=N_CITIES,
            seed=SEED,
            grid_mode=False,
            max_rails_between_cities=10,
            max_rails_in_city=N_CONNECTIVITY,
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

    controller = multiroll.rollout.Rollout(
                    rollit=ROLL_IT,
                    prediction_steps=N_PREDICT_STEPS,
                    env=env,
                    env_renderer=env_renderer,
                    debug_is_enabled=DEBUG)
    timeme('Multiroll Setup: ')

    if config.active('DISPLAY') or config.active('STEP'):
        controller.visualise()
        #input('####Start testbed?')
        timeme_reset()

    for step in range(N_SIM_STEPS):
        print('##IT', step)

        controller.update_agent_states()
        timeme('Multiroll - Update measurements: ')

        controls = controller.controls()
        timeme('Multiroll - Controls: ')

        env.step(controls)
        timeme('Flatland - Env.step(): ')

        if config.active('DISPLAY') and (not step % PLOT_STEPS):
            controller.visualise()
            timeme('Multiroll - Visualise: ')

        if config.active('STEP'):
            input('## --> Continue?'
                  + '\n##############')
            timeme_reset()

    if config.active('DISPLAY'):
        controller.visualise()
        input('##Testbed: Completed! Press any key to close.')

if __name__ == "__main__":
    config.parse_args()
    main()

