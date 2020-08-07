#!/bin/env python

import enum
import collections


# Defines a coordinate with row and column
Coordinate = collections.namedtuple('Coordinate', ['r', 'c'])
# Defines a state with its row column and direction
State = collections.namedtuple('State', ['r', 'c', 'd'])
# Defines the tuple of both the control and its corresponding direction (state-dependent)
ControlDirection = collections.namedtuple('ControlDirection', ['control', 'direction'])
# Defines a state control tuple where control is a ControlDirection type (track duplicate EdgeCreation)
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

    @staticmethod
    def reverse(edge_direction):
        if EdgeDirection.FORWARD:
            return EdgeDirection.BACKWARD
        return EdgeDirection.FORWARD


class EdgeActionType(enum.IntEnum):
    """ Possible edge related actions on graph. """
    NONE = 0
    ADD = 1
    REMOVE = 2


class VoteStatus(enum.IntEnum):
    """ Semantic structuring of voting related states. 

        Note:
            Whenever VOTED or UNVOTED is set, ELECTED
            is reset implicitly. Setting ELECTED is
            done only through '|' and tested through
            '&'.
                e.g. if var & VoteStatus.ELECTED
    """
    # No votes submitted
    NONE = 0
    # Votes received and either pending or elected (in graph)
    ELECTED = 1
    # Votes received and some prioritisation is expected
    VOTED = 2
    # No votes received and all edges are to be returned
    UNVOTED = 4


class AgentStatus(enum.IntEnum):
    NONE = 0
    INITIALISED = 1
    INFEASIBLE_PATH = 2
    FEASIBLE_PATH = 3

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


