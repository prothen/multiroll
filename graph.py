#!/usr/bin/env python

import copy
import enum
import numpy
import pandas
import ctypes
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
EdgeControl = collections.namedtuple('Edge', ['vertex_1', 'vertex_2', 'feed_forward', 'length'])
# Stores Goal Vertex and distance in cell count
PartialEdge = collections.namedtuple('PartialEdge', ['vertex', 'distance'])


# System states
is_true = dict()
is_true.update(graph_complete=False)


# All States -> ControlDirection (control and physical direction)
states = dict()
# All States that are vertices
vertices = dict()
# All edges indexed by integer ID
edges = dict()
# An edge indexed queue (updated each iteration) ->  
queue = None

# Amount of edges (iTODO: set n_edges after parsing all)
n_edges = None
# Prediction horizon
prediction_horizon = 20

# TODO:
#       - how to dynamically update directionality of edges (usage constraint)


class Simulation(object):
    """ Only to be run if graph is completed. """
    def __init__(self, edges, N):
        self.N = N
        self.edges = edges
        self.occupancy = dict()

    def initialise(self):
        for id in range(n_edges):
            self.occupancy[k] = Queue(k)

    def update(self):
        # parse all agents


class Agent(object):
    def __init__(self, agent):
        self.a = agent
        self.id = agent.handle
        # Current time evolution
        self.k = 0
        # Evolution update interval
        self.k_progress = self.find_dynamics
    
    # Todo: Numba
    def find_dynamics(self):
        v = 0
        k = 0
        while v < 1:
            v += self.a.speed
            k += 1
        return k


# TODO: wrapper around agents for queue in stack
entities = dict()
class Entity(object):
    def __init__(self, v, agent_id):
        self.v = v
        self.idx = 0
        self.id = agent_id

# Queue for each edge
class Queue(object):
    def __init__(self, edge_id, N):
        self.edge_id = edge_id
        self.N = N

        self.size = edges[edge_id].n_path

        # uint16 supports more than 400 agents
        self.stack = dict()

    def initialise(self);
        for k in range(self.N):
            self.stack[k] = (ctypes.c_uint16*self.size)()

    def enter(self, entity_id):
        # TODO: if agent enters -> update edge weight each iteration 
        #       --> with decaying additional weight
        # add entity 
        # todo: check if safety distance is guarenteed
        # last train in queue
        # SPEED difference:
        # vl ve -> (max vel both) TODO: decide on logic here
        # kr = kl / ke > path length -> train e will reach l after edge
        # --> also step kr (reach) < kl_idx
        # --> progress of kl is greater than kl
        # MALFUNCTION:
        # k_mal (env malfunction upper bound)
        # --> create penalty for violating 
        if self.stack[0]:
            print('collision on entrance!')
        self.stack[0] = entity_id

    def evolve(self):
        # pop last one 
        for entity_id in self.stack_entities:
            if entities[entity_id].progress():
                # ec
                # sc
                if (ec >> 1) & sc):
                    sc |= ec
                    continue


class Direction(enum.IntEnum):
    N = 0
    E = 1
    S = 2
    W = 3
    COUNT = 4


class Control(enum.IntEnum):
    NONE = 0
    L = 1
    F = 2
    R = 3
    S = 4
    COUNT = 5


Tests = [[Control.L, -1],
         [Control.F, 0],
         [Control.R, 1]]

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


FlipDirection = dict()
FlipDirection[Direction.N] = Direction.S
FlipDirection[Direction.E] = Direction.W
FlipDirection[Direction.S] = Direction.N
FlipDirection[Direction.W] = Direction.E


# Update a Physics environment transition that leads to non-railway as dead-end simulation
# https://gitlab.aicrowd.com/flatland/flatland/blob/master/flatland/envs/rail_env.py#L85
# -> agent is moving to cell it came from
FlipControlDirection = (lambda control: ControlDirection(Control.F, FlipDirection[Direction.S]))
#FlipControlDirection = dict()
#FlipControlDirection[Direction.N] = (lambda control: ControlDirection(Control.NONE, Direction.S))
#FlipControlDirection[Direction.E] = (lambda control: ControlDirection(Control.NONE, Direction.W))
#FlipControlDirection[Direction.S] = (lambda control: ControlDirection(Control.NONE, Direction.N))
#FlipControlDirection[Direction.W] = (lambda control: ControlDirection(Control.NONE, Direction.E))


# Dynamics of environment
Dynamics = dict()
Dynamics[Direction.N] = (lambda state: State(state.r-1, state.c, Direction.N))
Dynamics[Direction.E] = (lambda state: State(state.r, state.c+1, Direction.E))
Dynamics[Direction.S] = (lambda state: State(state.r+1, state.c, Direction.S))
Dynamics[Direction.W] = (lambda state: State(state.r, state.c-1, Direction.W))


# Control of current state to transition state
Simulate = dict()
Simulate[Control.NONE] = (lambda state, control: State(state.r, state.c,
                            Dynamics[control.direction](state).direction))
Simulate[Control.L] = (lambda state, control: Dynamics[control.direction](state))
Simulate[Control.F] = (lambda state, control: Dynamics[control.direction](state))
Simulate[Control.R] = (lambda state, control: Dynamics[control.direction](state))
Simulate[Control.S] = (lambda state, control: State(state.r, state.c, state.d))



class MyGraph(object):
    """ """
    def __init__(self, env, debug=False):
        self._verbose = debug
        self.graph = networkx.Graph()

        self.n_vertices = 0  # Value('i', 0)
        global vertices
        global states
        self.vertices = vertices
        self.states = states

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

        #states = self.states
        #vertices = self.vertices
        # TODO: create arrays with railway dimension for N E S W 
        #       --> 0x04 --> 0x04 << 1 ...
        #       --> do & for all
        #       --> 
        for r, c in zip(*railway):
            all_control_bits = _all_control_bits(r, c)
            valid_directions = _valid_directions(all_control_bits)
            vertex_directions = _vertex_directions(all_control_bits, valid_directions)
            controls = _controls(all_control_bits, valid_directions)

            for d, controls in zip(valid_directions,controls):
                state = State(r, c, d)
                valid_controls = controls
                for i, control in enumerate(controls):
                    state_i = Dynamics[control.direction](state)
                    if not _is_railway(state_i):
                        print('\t\t->DEADEND')
                        valid_controls[i] = FlipControlDirection(control)
                states[state] = valid_controls
                if d in vertex_directions:
                    vertices[state] = None

        def find_vertices(vertex: State):
            """ Return the connected vertices, its path and control.
                
                Note:
                    The exploration is based on the deadend compliant state->control
                    dictionary and creates an edge to itself.
            """
            # TODO: update weight for each allocated shortest path vertices
            #       --> same path
            #           Ak
            #           p &= Ak
            #           --> select edge & related weights
            #           --> create dict of own vertices ids (path)
            #           --> update
            #           define vertices
            #               -> store same vertices in same weight (ctypes address)
            #           explore vertices
            #               -> find vertices in opposite direction on exploration
            #               -> store pointer to all collision states
            #           #for k in range(N):
            #               
            edges = list()
            path = dict()
            controls = states[vertex]
            for control in controls:
                state = vertex
                n_path = 0
                controls_i = [control]
                # avoid lookup of vertices dictionary with reusing controls
                # TODO: detect railway collisions (multiple use for different edges)
                while len(controls_i) < 2:
                    controls_i = controls_i[0]
                    path[n_path] = controls_i
                    state = Simulate[controls_i.control](state, controls_i)
                    controls_i = states[state]
                    n_path += 1
                print('Self loop detected') if vertex == state else None
                print(path)
                edges.append(Edge(vertex, state, path, n_path))
            return edges

        def connect_targets(env):
            for agent in env.agents:
                print(agent.target_position)
                # get r, c grid 
                # get valid transitions
                # explore each to next vertex

        edge_list = list()
        for vertex in vertices.keys():
            pass
            # Return found vertex and its path length
            edge_list += find_vertices(vertex)
        self.report_vertices()


    def report_vertices(self):
        for vertex in self.vertices.keys():
            pass
            #print('\tVertex: \t{}'.format(vertex))
            #print('\tControl: \t{}'.format(self.states[vertex]))

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
