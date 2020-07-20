#!/usr/bin/env python

import numpy
import pandas
import networkx
import matplotlib

import itertools
import copy
import enum
import collections

from typing import Optional
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

        Todo
            - fetch transition map -> take nonzero elements -> find edges
            - for each vertex -> go in all directions up to next state
    """
    def __init__(self, env):
        self.graph = networkx.Graph()
        self.n_vertices = 0  # Value('i', 0)
        self.vertices = dict()
        
        self.env = env
        # see RailEnvTransitions
        # GridTransitionMap (core.Transition
        self.T = env.rail 

    def _vertex_known(self, state: State):
        return state in self.vertices.keys()

    def _is_valid(self, state: State):
        return any(self.T.get_transitions(*state))

    def _is_vertex(self, state: State):
        return len(numpy.nonzero(self.T.get_transitions(*state))[0]) > 1 

    def _get_controls(self, state):
        """ Return control for non-vertex node."""
        return numpy.nonzero(self.T.get_transitions(*state))[0]

    def _find_vertices(self, state: State, control, n_path=1):
        print('Find vertex: \n\t--> {0}'.format(state))
        state = dynamics[int(control)](state) 
        if not self._is_valid(state):
            raise RuntimeError()
        if self._is_vertex(state):
            print('\n\t\t--> FOUND a VERTEX: {0}'.format(state))
            print('\n\t\t--> AFTER {}'.format(n_path))
            return PartialEdge(state, n_path)
        control = self._get_controls(state)
        print('\n\t--> Control: {0}'.format(control))
        return self._find_vertices(state, control, n_path+1)

    def _explore_direction(self, state : State):
        """ """
        print('Explore direction: {0}'.format(state))
        # Abort vertex creation if already known or not enough transitions
        if self._vertex_known(state) or not self._is_vertex(state):
            print('-->(known,not-a-vertex) ({},{})'.format(
                self._vertex_known(state),
                not self._is_vertex(state)))
            return list()

        print('FOUND a new vertex:')
        # semaphore access
        self.n_vertices += 1
        self.vertices[state] = self.n_vertices
        # semaphore stop

        # Find all vertices that are connected to the current vertex
        # Todo: return also attributes found during vertices search
         
        controls = self._get_controls(state) 
        print('Found controls: {}'.format(controls))
        connected_vertex_edges = list()
        for control in controls:
            print('Control: {}'.format(control))
            connected_vertex_edges += [(self._find_vertices(state, control), control)]
            print(connected_vertex_edges)
        edge_list = list()
        for partial_edge, control in connected_vertex_edges:
            print('###########################################')
            print('\t\t TEST IF MORE VERTICES CAN BE FOUND')
            if partial_edge.vertex not in self.vertices.keys():
                edge_list += self._explore_direction(partial_edge.vertex)
            edge_list += [Edge(vertices[state], partial_edge.vertex, 
                {'dist': partial_edge.distance, 'control': control})]
        return edge_list

    def _explore_coordinate(self, state: State):
        print('Explore coordinate: {0}'.format(state))
        edge_list = list()
        for d in range(Direction.n_max):
            si = State(state.r, state.c, d)
            if not self._is_valid(si):
                continue
            elif self._is_vertex(si) and not self._vertex_known(si):
                print('Found vertex that is unknown')
                edge_list += self._explore_direction(state)
                pass
                # run logic
            else:
                print('Not a vertex, carry along')
                # Can be explored in the future but requires to complete in both directions
                # reporting edges
                # state = self._next_vertex(si)
                pass
        return edge_list
    
    def _next_vertex(self, state: State):
        print('Find next vertex {}'.format(state))
        # translate last valid state
        i=0
        while not self._is_vertex(state):
            print('IT{} Search at {}'.format(i,state))
            control = self._get_controls(state)
            print('--> Control: {0}'.format(control))
            state = dynamics[control[0]](state) 
            print('--> State: {0}'.format(state))
            i += 1
        print('IT{} Search at {}'.format(i,state))
        print('--> FOUND vertex: {}'.format(self._is_vertex(state)))
        return state

    def _find_first_vertex(self, state): 
        print('FIND first vertex')
        valid_backup = state
        for d in range(Direction.n_max):
            state = State(state.r, state.c, d)
            valid_backup = [state
                            if self._is_valid(state)
                            else valid_backup][0]
            if not self._is_vertex(state):
                continue
            return state
        print('No vertex in any direction from initial state')
        print('-> exploring now')
        state = self._next_vertex(state)
        return state

    def initialise(self, state: Optional[State]=None):
        # get all agents positions, select randomly and start process pool
        p = self.env.agents[0].initial_position # State(0, 0, 1)
        d = self.env.agents[0].initial_direction # State(0, 0, 1)
        # print('Initial Position of Agent 0: \n\t--> {0}'.format(p))
        # print('Initial Dir of Agent 0: \n\t--> {0}'.format(d))
        s0 = State(*(p+(d,)))
        sv = self._find_first_vertex(s0)
        # check if vertex, if not find next vertex
        print('Initial vertex: {}'.format(sv))
        edge_list = self._explore_coordinate(sv)
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
        # print('Step {}'.format(step))
        env.step(dict((a,0) for a in range(env.get_num_agents())))
        # print('Step {}'.format(step))
    env_renderer.render_env(show=True, show_predictions=False, show_observations=False)

    input('press to close')
    #g = MyGraph()
    # select random agent initial position (Use GridTransitionMap)
    
    # select available actions (decode the uint16_t code)
    # if multiple transitions create node with coordinate
