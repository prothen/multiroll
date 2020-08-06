#!/usr/bin/env python
"""
    Multiagent rollout implemented on flatland environment.


    Note:
        This module expects to be initialised with

            import graph
            graph.set_env(env)
            ...
            graph.MyGraph()

        before usage of any subsequently defined classes.
"""


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

# Defines a coordinate with row and column
Coordinate = collections.namedtuple('Coordinate', ['r', 'c'])
# Defines a state with its row column and direction
State = collections.namedtuple('State', ['r', 'c', 'd'])
# Defines a state with its row column and direction
ControlDirection = collections.namedtuple('ControlDirection', ['control', 'direction'])
# Defines a state control tuple to track of exploration
StateControl = collections.namedtuple('State', ['state', 'control_direction'])
# Store a pair of vertices
Pair = collections.namedtuple('Pair', ['vertex_1', 'vertex_2'])
# Define an edge with its corresponding feed_forward control (autopilot until next intersection)
Edge = collections.namedtuple('Edge', ['pair', 'priority', 'feed_forward', 'length'])


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


class GlobalContainer(object):
    # Define environment
    env = None
    # Define grid with uint16 transition bit encoding
    grid = None
    # All railway coordinates
    railway = dict()
    # Railway coordinates to IDs
    railway_ids = dict()
    # All States -> ControlDirection (control and physical direction)
    states = dict()
    # All nodes (vertices and intersections) indexed by state and state_container value
    nodes = dict()
    # All States that are vertices and their StatesContainer
    vertices = dict()
    # All states that are intersections and their StatesContainer
    intersections = dict()
    # All pairs of vertices (combine similar ones) and their ID as value
    pairs = dict()
    # All EdgeContainers indexed by a unique ID, partitioned in edges that share railway cells
    edges = dict()
    # A collection of all unique edges indexed by their StateControl and linked to the EdgeContainer
    edge_collection = dict()
    # Priority dictionary
    priority_dict = dict([(p, dict()) for p in Priority])

    @classmethod
    def set_env(cls, env_arg):
        cls.env = env_arg
        cls.grid = cls.env.rail.grid

env = None
# legacy
def set_env(env_arg):
    global env
    GlobalContainer.set_env(env_arg)


class Utils(GlobalContainer):

    @staticmethod
    def _bits(i, value):
        """ Return direction dependent control bits. """
        return (value >> (3 - i) * 4) & 0xF

    def _all_control_bits(self, coordinate: Coordinate):
        """ Return list of control_bits for all directions. """
        return [self._bits(d, self.grid[coordinate.r][coordinate.c])
                for d in Direction]

    @staticmethod
    def _valid_directions(bits):
        """ Return indices of valid directions."""
        return [idx for idx, val in enumerate(bits) if val != 0]

    @staticmethod
    def _vertex(control_bits):
        """ Return true if control_bits are from a vertex. """
        return (bin(control_bits).count("1") > 1)

    def _vertex_directions(self, all_control_bits, valid_idxs):
        """ Return indices of vertices at current coordinate. """
        return [idx for idx in valid_idxs if self._vertex(all_control_bits[idx])  != 0]

    def _n_directions(self, coordinate: Coordinate):
        """Return the amount of valid directions that have transitions. """
        return len([vdi for vdi in self._valid_directions(
                    self._all_control_bits(coordinate)) if vdi != 0])

    @staticmethod
    def _n_controls(coordinate: Coordinate):
        """ Return amount of admitted controls. """
        return _n_directions(coordinate)

    @staticmethod
    def _is_intersection(coordinate: Coordinate):
        """ Return true if the current coordinate has more than 2 transitions. """
        return _n_directions(coordinate) > 2

    def _is_railway(self, state: State):
        """Return true if state has any transitions. """
        return (self.grid[state.r][state.c] >> (3 - state.d) * 4) & 0xF

    @staticmethod
    def _directions2controls(directions, direction_agent):
        """Transform a direction into a corresponding control for rail_env.  """
        ds_idxs = [int(i) for i in format(directions,'04b')]
        allowed = numpy.nonzero(ds_idxs)[0]
        da = direction_agent

        controls = [ControlDirection(control, Direction((da + o)%4)) for  (control, o)
                     in Tests if ((da + o)%4) in allowed]

        if not any(controls):
            raise RuntimeError()
        return controls

    def _controls(self, all_control_bits, valid_directions):
        """Return a list of lists for each valid direction. """
        controls = list()
        for d in valid_directions:
            directions = all_control_bits[d]
            controls += [self._directions2controls(directions, Direction(d))]
        return controls

    @staticmethod
    def _get_coordinate_type(valid_directions):
        if len(valid_directions) <= 2:
            return CoordinateType.NORMAL_RAILWAY
        return CoordinateType.INTERSECTION


class CoordinateContainer(Utils):
    """ Create a container of information collected in a coordinate.

        Note:
            Container to collect and extend information about railway 
            coordinates.

        Todo:
            Coordinate metrics should be accessible based on States
            -> e.g. direction based controls
    """
    def __init__(self, ID, coordinate, debug_is_enabled=True):
        all_control_bits = self._all_control_bits(coordinate)
        valid_directions = self._valid_directions(all_control_bits)
        vertex_directions = self._vertex_directions(all_control_bits, valid_directions)
        controls = self._controls(all_control_bits, valid_directions)

        self.id = ID
        self.type = self._get_coordinate_type(valid_directions)
        self.coordinate = coordinate

        # All states with nonzero n_controls with reference to their state_containers
        self.valid_states = dict()
        # Explicit controls indexed by states
        self.controls = dict()
        # Amount of controls for each direction
        self.n_controls = dict()
        # Direction to state mapping to recover same entry states for edge and their control
        self.direction2states = dict([(di, dict()) for di in Direction])

        if debug_is_enabled:
            print('\n##################################')
            print('##################################')
            print('Type: ', self.type)
            print('State:\n\t {}'.format(coordinate))
            print('Control bits:\n\t {}'.format(all_control_bits))
            print('Directions:\n\t {}'.format(valid_directions))
            print('Controls:\n\t {}'.format(controls))

        for d, controls in zip(valid_directions, controls):
            d = Direction(d)
            state = State(coordinate.r, coordinate.c, d)

            # Consider dead-ends TODO: debug
            valid_controls = controls
            for i, control in enumerate(controls):
                if not self._is_railway(Simulator(state, control)):
                    print('\t\t->Encountered dead-end (Flip!)')
                    valid_controls[i] = ApplyDeadendControl(control)

            self.controls[state] = valid_controls
            self.n_controls[state] = len(valid_controls)

            sc = StatesContainer(state, self)
            self.states[state] = sc
            self.valid_states[state] = sc
            for control in valid_controls:
                self.direction2states[control.direction][state] = control

            if d not in vertex_directions:
                if not self.type == CoordinateType.INTERSECTION:
                    continue
                self.intersections[state] = sc
                sc.type = StateType.INTERSECTION
            else:
                self.vertices[state] = sc
                sc.type = StateType.VERTEX
            self.nodes[state] = sc
        self.railway[coordinate] = self


class StatesContainer(object):
    """ Utility class to collect information about state.

        Used in global vertices and intersections.

        Note:
            Allows easy extension and access of state metrics.

        Todo:
            should directly be reference by agent
    """
    def __init__(self, state, coordinate_container):
        self.state = state
        self.coc = coordinate_container
        self.id = self.coc.id

        self.type = state_type
        self.n_controls = self.coc.n_controls[state]
        self.priority = Priority(self.n_controls - 1)
        self.controls = self.coc.controls[state]


class AgentContainer(GlobalContainer):
    """ Get subset of metrics from flatland environment. 
    
        Note:
            Defines agent interface to flatland agents.

        TODO:
            - architecture design
            - interfaces to simulator
    """
    def __init__(self, ID):
        self.id = ID
        a = self.env.agents[self.id]
        (r, c) = a.initial_position
        d = a.initial_direction
        self.state = State(r, c, d)
        self.soc = self.states[self.State]
        # get goal states
        # look up coordinate and get valid directions
        (r, c) = a.target_position
        self.target = self.railway[Coordinate(r,c)].states.keys()
        # TODO: compute fraction (steps for one transition)
        speed = a.speed_data['speed']
        # Amount of steps necessary for one grid traversal
        # TODO: DEV-stage: test pending
        import math; self.speed = math.ceil(1/speed)
        # List of edge_ids
        self.path = list()
        # Step based dictionary
        self.prediction = dict()
        self._initialise()

    def get_coc(self):
        return self.states[self.state]


class AgentTraverse(GlobalContainer):
    """ Utility class for maintaining overview of useful metrics. """

    def __init__(self, agent_id, edge):
        self.id = agent_id
        # Fetch agent container
        self.agent = self.agents[agent_id]
        # Traversal priority from edge
        self.priority = edge.priority
        self.edge = edge
        self.speed = self.agent.speed
        #TODO: self.eta = self._initialise_eta()

    def _initialise_eta(self):
        path_length = edge.path_length
        # get agent_speed
        # get path length 
        pass


class EdgeContainer(GlobalContainer):
    """ Edge related metrics and occupancy trackers.

        Todo:
            1.
                Register agents and their planned entry step
                -> If agent registers and no vote done -> force switch

            2.
                Add collision matrix with agent steps for
                global N prediction steps along path length M
                -> matrix &operator should yield zero for collision free
    """
    FORWARD = 1
    BACKWARD = -1

    def __init__(self, ID):
        self.id = ID
        self.vote = 0
        self._forward = dict()
        self._backward = dict()
        self._edge_dir = dict()
        self.active_agents = dict()

    def _reset_vote(self):
        """ Reset vote and allow all edge directions to be used. """
        self.vote = 0

    def _is_forward(self):
        return self.vote > 0

    def get_edges(self, voted=True):
        if voted:
            if self._is_forward:
                return self._forward.values()
            return self._backward.values()
        return dict(**self._forward, **self._backward).values()

    def add_forward_edge(self, edge):
        self._forward[edge.pair.vertex_1] = edge
        self._edge_direction[edge.pair.vertex_1] = EdgeContainer.FORWARD

    def add_reverse_edge(self, edge):
        self._backward[edge.pair.vertex_1] = edge
        self._edge_direction[edge_backward.pair.vertex_1] = EdgeContainer.BACKWARD

    def vote(self, state):
        """ Register interest to use an edge in certain direction. """
        self.vote += self._edge_direction[state]

    def register(self, agent_id, step):
        """ Register when an agent is expected to enter this edge.  """
        # TODO: if not voted, force switch
        self.vote += self._edge_direction[state]
        pass

    def enter(self, state, agent_id):
        """ Conduct entry procedure for agent from state.

            Note:
                The global priority_dict helps in selecting
                promising candidates for the rollout.
        """
        edge = self.edges[state]
        agent_container = AgentTraverse(agent_id=agent_id, edge=edge)
        # Store active agent and its prio from the selected edge
        self.active_agents[agent_id] = agent_container

        # Register agent globally to appropriate priority watchlist
        #   and reference to agent's traverse metrics
        self.priority_dict[edge.priority][agent_id] = agent_container

    def exit(self, agent_id):
        """ Remove agent from active_agents and priority_dict. """
        prio = self.active_agents[agent_id].priority
        self.priority_dict[prio].pop(agent_id, None)


class MyGraph(Utils):
    """Container for graph related actions.

        Note:
            Update active edges or compute shortest path.
    """
    def __init__(self, debug=False):

        self._verbose = debug
        self._show_transitions = False

        self.n_vertices = 0
        self.graph = networkx.Graph()
        self.initialise()

    def _is_explored(self, state, control):
        return StateControl(state, control) in self.edge_collection.keys()

    def _get_entry_states(self, coc, control):
        return coc.direction2states[control.direction]

    def _get_reverse_control_direction(self, path):
        """ Return control direction in reverse path direction.
            
            Note:
                Looks at one cell before exit_state and returns
                its control
        """
        return path[-2].control.direction

    def _initialise_railway(self):
        """ Defines all railway coordinates with unique railway ID. """
        env_railway = numpy.nonzero(self.grid)
        id_railway = -1
        for r, c in zip(*env_railway):
            id_railway += 1
            coordinate = Coordinate(r, c)
            CoordinateContainer(id_railway, coordinate)

    def _get_exit_coordinate_from_path(self, path):
        return self.states[path[-1].state]

    def _find_equal_traversible_states(self, state, control):
        """ Find all possible states that lead to the same edge. """
        return self.states[state].coc.direction2states[control.direction]

    def _find_edge_control(self, state, control):
        """ Return state controls pairs for each grid element along edge.

            Note:
                Returns list of (state,control) from state+1 cell until goal.

        """
        n_path = 1
        path = list()

        state = Simulate(state, controls)
        control = self.states[state].controls

        while self.states[state].type == StateType.NONE:
            n_path += 1
            control = control[0]
            state = Simulate(state, controls)
            path.append(StateControl(state, control))
            sc = self.states[state]
        return path

    def _define_edges_from_path(self, entry_states, path):
        edges = list()
        exit_state = path[-1].state
        for entry_state, control in entry_dict.items():
            if self._is_explored(entry_state, control):
                raise RuntimeError()
            edge_id = len(self.edge_collection) + 1
            edge_path = [StateControl(entry_state, control)] + path

            priority = exit_state.priority
            pair = Pair(entry_state, goal_state)
            edge = Edge(pair, priority, edge_path, n_path)

            self.edge_collection[StateControl(state, control)] = edge
            edges += edge

        return edges

    def _initialise_edges(self):
        for node in self.nodes.keys():
            controls = node.controls
            for control in controls:
                if self._is_explored(vertex.state, control):
                    continue
                edge_container_id = len(self.edges) + 1
                edge_container = EdgeContainer(edge_container_id)

                entry_coc = node.coc
                entry_states = self._find_equal_traversible_states(entry_coc,
                                                                   control)
                path = self._find_edge_control(entry_coc, control)
                edges = self._define_edges_from_path(entry_states, path)
                edge_container.add_forward_edges(edges_forward)

                exit_coc = self._get_exit_coc_from_path(path)
                reverse_direction = self._get_reverse_control_direction(path)
                # TODO: get_entry_states
                edges_backward = vertex.coc.direction2states[control.direction]
                edge_container.add_backward_edges(edges_backward)
                self.edges[edge_container_id] = edge_container

    def _connect_targets(self):
        """ To be done. """
        # get agent initial position
        # get agent min max speeds
        # get agent_id
        # add to self.agents[id]
        pass

    def initialise(self):
        self._initialise_railway()
        self._initialise_edges()
        self._connect_targets()



if __name__ == "__main__":
    print('Graph - Testbed')
    env.reset()
    env_renderer.reset()

    for step in range(500):
        env.step(dict((a,0) for a in range(env.get_num_agents())))
    env_renderer.render_env(show=True, show_predictions=False, show_observations=False)
    input('press to close')
