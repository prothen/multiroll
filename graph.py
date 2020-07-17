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

# from flatland.core.grid.grid4 import Grid4TransitionsEnum
# from flatland.core.grid.grid4_utils import get_new_position

#from flatland.envs.rail_env import RailEnv
#from flatland.envs.rail_generators import sparse_rail_generator
#from flatland.envs.schedule_generators import sparse_schedule_generator
#
#from flatland.core.env_observation_builder import ObservationBuilder
#from flatland.utils.rendertools import RenderTool, AgentRenderVariant
#from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters


# Defines a state with its row column and direction
State = collections.namedtuple('State', ['r', 'c', 'd'])
# Stores Initial and Goal Vertex and corresponding distance in cell count 
Edge = collections.namedtuple('Edge', ['vertex_1', 'vertex_2', 'attributes'])
# Stores Goal Vertex and distance in cell count
PartialEdge = collections.namedtuple('PartialEdge', ['vertex', 'distance'])


class Direction(object):
    N = 0
    E = 1
    S = 2
    W = 3
    n_max = 4


dynamics = dict()
dynamics[Direction.N] = (lambda state: State(state.r-1, state.c, state.d))
dynamics[Direction.E] = (lambda state: State(state.r, state.c+1, state.d))
dynamics[Direction.S] = (lambda state: State(state.r+1, state.c, state.d))
dynamics[Direction.W] = (lambda state: State(state.r, state.c-1, state.d))


class Controls(enum.IntEnum):
    NONE = 0
    L = 1
    F = 2
    R = 3
    S = 4


class MyGraph(object):
    """
        Note
            Call only after env.rail has been populated. (after env.reset())
    """
    def __init__(self, env):
        self.graph = networkx.Graph()
        self.n_vertices = 0  # Value('i', 0)
        self.vertices = dict()
        
        # see RailEnvTransitions
        # GridTransitionMap (core.Transition
        self.T = env.rail 

    def _state_unkown(self, state: State):
        return state in self.vertices.keys()

    def _is_vertex(self, state: State):
        return np.nonzero([int(d) for d in 
                           bin(T.get_transitions(*state) & (0xF << d))[2:]
                           ]).shape[0] < 2

    def _get_controls(self, state):
        return np.nonzero([int(d) for d in bin(T.get_transitions(*state))[2:]])[0]

    def _find_vertices(self, state: State, n_path=0):
        vertices = list()
        if self._is_vertex(state):
            print('Found next vertex after {}, return'.format(n_path))
            return vertices.append([PartialEdge(state, n_path)])
        # todo: ensure to remove non-actions (NONE and STOP) -> should be automatically done
        control = self._get_controls(state)
        state = dynamics[control](state) 
        return self._find_vertices(state, n_path=n_path+1)

    def _explore_direction(self, state : State):
        """ """
        # Abort vertex creation if already known or not enough transitions
        if self._state_known(state) or self._is_not_vertex(state):
           return list()

        # semaphore access
        self.n_vertices += 1
        self.vertices[state] = self.n_vertices
        # semaphore stop

        # Find all vertices that are connected to the current vertex
        # Todo: return also attributes found during vertices search
         
        controls = self._get_controls(state) 
        connected_vertex_edges = list()
        for control in controls:
            connected_vertex_edges = (self._find_vertices(state), control)
        for partial_edge, control in connected_vertex_edges:
            if partial_edge.vertex not in self.vertices.keys():
                edge_list += self._explore_coordinate(partial_edge.vertex)
            edge_list += [Edge(vertices[state], vertices[partial_edge.vertex], 
                {'dist': partial_edge.distance, 'control': control})]
        return edge_list

    def _explore_coordinate(self, state: State):
        edge_list = list()
        for d in range(Directions.n_max):
            state.d = d 
            edge_list += self._explore_direction(state)
        return edge_list

    def initialise(self, state: State):
        # get all agents positions, select randomly and start process pool
        initial_state = State(0, 0, 1)
        edge_list = self._explore_coordinate(initial_state)
        for edge in edge_list:
            g.add_edge(edge.vertex_1, edge.vertex_2, edge.attributes)


class Agent(object):
    def __init__(self):
        self.vertex = 0
        self.position = State()
        self.target = State()

    def initialise(self):
        # Find agents next vertex
        # update next vertex whenever transition takes place
        # find all closest target vertices and compute path from target cell to target vertices
        # dict[vertices] = path 
        # each agent has attribute on vertex with the static residual path (distance)
        # define residual path from target vertex to target cell (static)
        pass


if __name__ == "__main__":
    print('start test program')

    env.reset()
    env_renderer.reset()
    env_renderer.render_env(show=True, show_inactive_agents=True, show_observations=False, show_predictions=False)
    # import sys
    # sys.exit(0)
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
