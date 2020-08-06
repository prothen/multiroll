#!/bin/env python

from graph import Direction

# Plot related
Transition2Color = dict()
Transition2Color[Direction.N] = 'r'
Transition2Color[Direction.E] = 'r'
Transition2Color[Direction.S] = 'r'
Transition2Color[Direction.W] = 'r'


# Plot related
Direction2Target = dict()
Direction2Target[Direction.N] = [-1, 0]
Direction2Target[Direction.E] = [0, 1]
Direction2Target[Direction.S] = [1, 0]
Direction2Target[Direction.W] = [1, -1]
"""
    fl/utils/renderutils
    l.137: grid2pixels
        r -> -y
        c -> x

    fl/utils/graphicslayer
    l. 52: color rgb tuple 255 (int, int, int)

    fl/utils/:
        GraphicsLayer -> PILGL -> PILSVG  -> PGL 
"""

def report_states(states):
    """ Report states via their states_containers and their metrics. 
        
        Note:
            Expects a list of graph.StateContainers
    """
    for state_container in state_containers:
        print('\tState: \t{}'.format(state_container.state))
        print('\tControls: \t{}'.format(state_container.controls))

def show_path(env_renderer, path):
    """ """
    # todo
    pass

def show_states(env_renderer, states):
    """ Show states defined in StatesContainer through Flatland env_renderer. 
    
        Todo:
            Update directions
    """
    for state in states:
        #env_renderer.gl.scatter(*(state.c, -state.r), color='r', layer=1, marker="o", s=25) 
        env_renderer.gl.scatter(*(state.c, -state.r), color=(0,255,0), layer=1, marker="o", s=5) 
        env_renderer.renderer.plot_single_agent((state.r, state.c),
        #                                        state.d, 'r', target=(0,0),
        #                                        selected=True)
