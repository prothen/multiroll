#!/bin/env python

import networkx
import matplotlib.pyplot

from constants import *

from graph import Direction

env_renderer = None

def set_env_renderer(env_renderer_arg):
    global env_renderer
    env_renderer = env_renderer_arg

def renderer(func):
    def wrapper(*args, **kwargs):
        return func(env_renderer, *args, **kwargs)
    return wrapper

def report_states(states):
    """ Report states via their states_containers and their metrics. 

        Note:
            Expects a list of graph.StateContainers
    """
    for state_container in state_containers:
        print('\tState: \t{}'.format(state_container.state))
        print('\tControls: \t{}'.format(state_container.controls))

@renderer
def show_path(env_renderer, path):
    """ """
    # todo
    pass

@renderer
def show_agents(env_renderer, agents):
    """ Show states defined in StatesContainer through Flatland env_renderer. 
    
        Todo:
            Update directions
    """
    for agent in agents:
        state = agent.state
        target = agent.target

        env_renderer.gl.scatter(*(state.c, -state.r), color=Color.STATE, layer=1, marker="o", s=Dimension.STATE)
        env_renderer.gl.scatter(*(target.c, -target.r), color=Color.TARGET, layer=1, marker="o", s=Dimension.TARGET)
        #env_renderer.render_env(
        #        show=True, 
        #        show_agents=True, 
         #       show_predictions=False, 
         #       show_observations=False)
@renderer
def show_states(env_renderer, states, color=Color.STATE, dimension=Dimension.STATE):
    """ Show states defined in StatesContainer through Flatland env_renderer. 

        Todo:
            Update directions
    """
    for state in states:
        env_renderer.gl.scatter(*(state.c, -state.r), color=color,
                                layer=1, marker="o", s=dimension)
        env_renderer.render_env(
                show=True, 
                show_agents=True, 
                show_predictions=False, 
                show_observations=False)

@renderer
def show(env_renderer):
    env_renderer.render_env(
            show=True, 
            show_agents=True, 
            show_predictions=False, 
            show_observations=False)
    env_renderer.gl.show()


def show_graph(graph):
    networkx.draw(self._graph)
    matplotlib.pyplot.show()
    input('Showing graph...\n Press any key to continue')

