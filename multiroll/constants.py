#!/bin/env python

import enum
import collections


# Defines a coordinate with row and column
Coordinate = collections.namedtuple('Coordinate', ['r', 'c'])
# Defines a state with its row column and direction
State = collections.namedtuple('State', ['r', 'c', 'd'])
# Defines the tuple of both the control and its corresponding direction (state-dependent)
ControlDirection = collections.namedtuple('ControlDirection', ['control', 'direction'])
# Defines a state control tuple where control is a ControlDirection type
StateControl = collections.namedtuple('State', ['state', 'control'])
# Store a pair of nodes for condensed edge creation
Pair = collections.namedtuple('Pair', ['vertex_1', 'vertex_2'])
# Define an edge with path priority of goal traversibility and its collective container
Edge = collections.namedtuple('Edge', ['pair', 'priority', 'path', 'length', 'container_id'])


class Direction(enum.IntEnum):
    N = 0
    E = 1
    S = 2
    W = 3


class Control(enum.IntEnum):
    NONE = 0
    L = 1
    F = 2
    R = 3
    S = 4


class Priority(enum.IntEnum):
    """ Priority definition for edges. """
    # Edge goal is intersection
    NONE = 0
    # Low amount of traversability 2 choices
    LOW = 1
    # High amount of traversability 3 choices
    HIGH = 2


class CoordinateType(enum.IntEnum):
    NORMAL_RAILWAY = 0
    INTERSECTION = 1


class StateType(enum.IntEnum):
    NONE = 0
    NODE = 1
    INTERSECTION = 3
    VERTEX = 5

class EdgeDirection(enum.IntEnum):
    FORWARD = 1
    BACKWARD = -1


class AgentStatus(enum.IntEnum):
    NONE = 0
    INITIALISED = 1
    ON_PATH = 2
    ON_NODE = 4


class PathStatus(enum.IntEnum):
    NONE = 0
    INFEASIBLE = 1
    FEASIBLE = 2


class AgentMode(enum.IntEnum):
    # Once infeasibility is encountered on existing graph
    STALE = 0
    # If graph is updated and not yet active
    EXPLORING = 1
    # If currently active and following feasible path
    ACTIVE = 2


class GraphActivity(enum.IntEnum):
    ZERO = 0
    AGENT_ACTIVE = 1


Tests = [[Control.L, -1],
         [Control.F, 0],
         [Control.R, 1]]


class GlobalStatus(enum.IntEnum):
    NONE = 0
    HAS_RAILWAY = 1
    HAS_EDGES = 2
    HAS_GRAPH = 4


FlipDirection = dict()
FlipDirection[Direction.N] = Direction.S
FlipDirection[Direction.E] = Direction.W
FlipDirection[Direction.S] = Direction.N
FlipDirection[Direction.W] = Direction.E


# Update a Physics environment transition that leads to non-railway as dead-end simulation
# https://gitlab.aicrowd.com/flatland/flatland/blob/master/flatland/envs/rail_env.py#L85
# -> agent is moving to cell it came from
ApplyDeadendControl = (lambda control:
        ControlDirection(Control.F, FlipDirection[control.direction]))

# Dynamics of environment
Dynamics = dict()
Dynamics[Direction.N] = (lambda state: State(state.r-1, state.c, Direction.N))
Dynamics[Direction.E] = (lambda state: State(state.r, state.c+1, Direction.E))
Dynamics[Direction.S] = (lambda state: State(state.r+1, state.c, Direction.S))
Dynamics[Direction.W] = (lambda state: State(state.r, state.c-1, Direction.W))


Simulator = (lambda state, control: Dynamics[control.direction](state))


# LEGACY Control of current state to transition state
Simulate = dict()
Simulate[Control.NONE] = (lambda state, control: State(state.r, state.c,
                            Dynamics[control.direction](state).direction))
Simulate[Control.L] = (lambda state, control: Dynamics[control.direction](state))
Simulate[Control.F] = (lambda state, control: Dynamics[control.direction](state))
Simulate[Control.R] = (lambda state, control: Dynamics[control.direction](state))
Simulate[Control.S] = (lambda state, control: State(state.r, state.c, state.d))


class Color:
    STATE = (255,0,0,)
    TARGET = (0,255,0)
    DEBUG = (0,0,255)


class Dimension:
    STATE = 30
    TARGET = 30
    DEBUG = 100


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
