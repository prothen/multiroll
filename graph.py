#!/usr/bin/env python

import copy
import enum
import numpy
import pandas
import networkx
import itertools
import matplotlib
import collections
from typing import Optional

# Defines a state with its row column and direction
State = collections.namedtuple('State', ['r', 'c', 'd'])
# Defines a state with its row column and direction
ControlDirection = collections.namedtuple('ControlDirection', ['control', 'direction'])
# Stores Initial and Goal Vertex and corresponding distance in cell count 
Edge = collections.namedtuple('Edge', ['vertex_1', 'vertex_2', 'attributes'])
# Stores Goal Vertex and distance in cell count
PartialEdge = collections.namedtuple('PartialEdge', ['vertex', 'distance'])

class Control(enum.IntEnum):
    NONE = 0
    L = 1
    F = 2
    R = 3
    S = 4
    COUNT = 5

class Direction(enum.IntEnum):
    N = 0
    E = 1
    S = 2
    W = 3
    COUNT = 4

Transition2Color = dict()
Transition2Color[Direction.N] = 'r'
Transition2Color[Direction.E] = 'r'
Transition2Color[Direction.S] = 'r'
Transition2Color[Direction.W] = 'r'

Direction2Target = dict()
Direction2Target[Direction.N] = [-1, 0]
Direction2Target[Direction.E] = [0, 1]
Direction2Target[Direction.S] = [1, 0]
Direction2Target[Direction.W] = [1, -1]

Dynamics = dict()
Dynamics[Direction.N] = (lambda state: State(state.r-1, state.c, Direction.N))
Dynamics[Direction.E] = (lambda state: State(state.r, state.c+1, Direction.E))
Dynamics[Direction.S] = (lambda state: State(state.r+1, state.c, Direction.S))
Dynamics[Direction.W] = (lambda state: State(state.r, state.c-1, Direction.W))

Tests = [[Control.L, -1], 
         [Control.F, 0],
         [Control.R, 1]]

FlipControlDirection = dict()
FlipControlDirection[Direction.N] = (lambda control: ControlDirection(control.control, Direction.S))
FlipControlDirection[Direction.E] = (lambda control: ControlDirection(control.control, Direction.W))
FlipControlDirection[Direction.S] = (lambda control: ControlDirection(control.control, Direction.N))
FlipControlDirection[Direction.W] = (lambda control: ControlDirection(control.control, Direction.E))

class MyGraph(object):
    """ """
    def __init__(self, env):
        self.graph = networkx.Graph()

        self.n_vertices = 0  # Value('i', 0)

        self.vertices = dict()
        self.states = dict()
        
        self.env = env
        self.grid = env.rail.grid
        railway = numpy.nonzero(self.grid)

        self._show_transitions = False

        def _bits(i, value):
            """ Return direction dependent control bits. """
            return (value >> (3 - i) * 4) & 0xF
       
        def _all_control_bits(r, c):
            """ Return list of control_bits for all directions. """
            return [_bits(d, self.grid[r][c]) for d in range(Direction.COUNT)]
        
        def _valid_directions(bits):
            """ Return indices of valid directions."""
            return [idx for idx, val in enumerate(bits) if val != 0]

        def _vertex(control_bits):
            """ Return true if control_bits are from a vertex. """
            return (bin(control_bits).count("1") > 1)

        def _vertex_directions(all_control_bits, valid_idxs):
            """ Return indices of vertices at current coordinate. """
            return [idx for idx in valid_idxs if _vertex(all_control_bits[idx])  != 0]
    
        def _is_railway(state: State):
            return (self.grid[state.r][state.c] >> (3 - state.d) * 4) & 0xF

        def _directions2controls(directions, direction_agent):
            """ """
            ds = directions
            ds_idxs = [int(i) for i in format(directions,'04b')] 
            allowed = numpy.nonzero(ds_idxs)[0]
            da = direction_agent
 
            controls = [ControlDirection(control, Direction((da + o)%4)) for  (control, o) 
                         in Tests if ((da + o)%4) in allowed]

            if not any(controls):
                raise RuntimeError()
            return controls

        def _controls(all_control_bits, valid_directions):
            """ """
            controls = list()
            for d in valid_directions:
                directions = all_control_bits[d]
                controls += [_directions2controls(directions, Direction(d))] 
            return controls

        states = self.states
        vertices = self.vertices
        for r, c in zip(*railway):
            all_control_bits = _all_control_bits(r, c)
            valid_directions = _valid_directions(all_control_bits)
            vertex_directions = _vertex_directions(all_control_bits, valid_directions) 
            controls = _controls(all_control_bits, valid_directions)
            for d, controls in zip(valid_directions,controls):
                state = State(r, c, d)
                updated_controls = list()
                updated_controls += controls
                print('\nStart:')
                print(updated_controls)
                for i, control in enumerate(controls):
                    state_test = Dynamics[control.direction](state)
                    if not _is_railway(state_test):
                        print('Also change the control') # --> this needs to change the direction of the vehicle as well!
                        print('Found end of railway at {}'.format(state_test))
                        print('--> old Control {}'.format(control))
                        updated_controls[i] = FlipControlDirection[control.direction](control)
                        print('--> new Control {}'.format(updated_controls[i]))
                 
                print(updated_controls)
                print('END:')
                states[state] = updated_controls
                if d in vertex_directions:
                    vertices[state] = None

        def find_vertices(vertex: State):
            edges = list()
            controls = states[vertex]
            state = vertex
            for control in controls:
                print('Start exploring {}:'.format(state))
                path = 0
                controls_i = [control]
                while len(controls_i) < 2:
                    controls_i = controls_i[0]
                    print('{:05d}'.format(path), Direction(controls_i.direction),
                          'from {}'.format(state))
                    #print('--> Control:', Direction(controls_i.direction))
                    state = Dynamics[controls_i.direction](state)
                    #print('to states \n\t{}'.format(state))
                    controls_i = states[state]
                    path += 1
                edges.append(Edge(vertex, state, path))
                #print('Found Vertex {} for {}'.format(state, vertex))
            return edges
        
        #edge_list = list()
        #for vertex in vertices.keys():
        #    # Return found vertex and its path length
        #    edge_list += find_vertices(vertex)
        ##self.report_vertices()
        #print(edge_list)

    def report_vertices(self):
        for vertex in self.vertices.keys():
            print('\tVertex: \t{}'.format(vertex))

    def show_vertices(self, env_renderer):
        v = list(self.vertices.keys())[0] 
        controls = self.states[v]
        l = list()
        for v in self.vertices.keys():
            for control in controls:
                l += [Direction2Target[control.direction]]
                c = Transition2Color[control.direction]
                env_renderer.renderer.plot_single_agent((v.r, v.c), v.d, 'r', target=(v.r+1, v.c),selected=True)

                if self._show_transitions:
                    env_renderer.renderer.plot_transition(
                            position_row_col=(v.r, v.c),
                            transition_row_col=l,
                            color=c
                            )


if __name__ == "__main__":
    print('Graph - Testbed')
    env.reset()
    env_renderer.reset()

    for step in range(500):
        env.step(dict((a,0) for a in range(env.get_num_agents())))
    env_renderer.render_env(show=True, show_predictions=False, show_observations=False)
    input('press to close')
