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

__author__ = "Philipp Rothenhäusler"
__maintainer__ = "Philipp Rothenhäusler"
__email__ = "philipp.rothenhaeusler@kthformulastudent.com"
__copyright__ = "Copyright 2019, KTH Formula Student"
__status__ = "Development"
__license__ = "BSD"
__version__ = "1.0"

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

dynamics = dict()
dynamics[Direction.N] = (lambda state: State(state.r-1, state.c, state.d))
dynamics[Direction.E] = (lambda state: State(state.r, state.c+1, state.d))
dynamics[Direction.S] = (lambda state: State(state.r+1, state.c, state.d))
dynamics[Direction.W] = (lambda state: State(state.r, state.c-1, state.d))

Tests = [[Control.L, -1], 
         [Control.F, 0],
         [Control.R, 1]]


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
    
        def _directions2controls(directions, direction_agent):
            """ """
            ds = directions
            ds_idxs = [int(i) for i in format(directions,'04b')] 
            allowed = numpy.nonzero(ds_idxs)[0]
            da = direction_agent
 
            controls = [ControlDirection(control, (da + o)%4) for  (control, o) 
                         in Tests if ((da + o)%4) in allowed]

            if not any(controls):
                raise RuntimeError()
            return controls

        def _controls(all_control_bits, valid_directions):
            """ """
            controls = list()
            for d in valid_directions:
                directions = all_control_bits[d]
                controls += [_directions2controls(directions, d)] 
            return controls

        states = self.states
        vertices = self.vertices
        for r, c in zip(*railway):
            all_control_bits = _all_control_bits(r, c)
            valid_directions = _valid_directions(all_control_bits)
            vertex_directions = _vertex_directions(all_control_bits, valid_directions) 
            controls = _controls(all_control_bits, valid_directions)

            for d, controls in zip(valid_directions,controls):
                si = State(r, c, d)
                states[si] = controls
                if d in vertex_directions:
                    vertices[si] = None

        #self.report_vertices()

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
