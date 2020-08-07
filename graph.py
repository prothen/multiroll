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


class AgentStatus(enum.IntEnum):
    NONE = 0
    INITIALISED = 1
    INFEASIBLE_PATH = 2
    FEASIBLE_PATH = 3


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


class GlobalContainer(object):
    # Define environment
    env = None
    # Define grid with uint16 transition bit encoding
    grid = None
    # All railway coordinates
    railway = dict()
    # Dict of agents with AgentContainer
    agents = dict()
    # Railway coordinates to IDs
    railway_ids = dict()
    # Dictionary of target coordinates
    targets = dict()
    # All States -> ControlDirection (control and physical direction)
    states = dict()
    # All nodes (vertices and intersections) indexed by state and state_container value
    nodes = dict()
    # All States that are vertices and their StateContainer
    vertices = dict()
    # All states that are intersections and their StateContainer
    intersections = dict()
    # All pairs of vertices (combine similar ones) and their ID as value
    pairs = dict()
    # All EdgeContainers indexed by a unique ID, partitioned in edges that share railway cells
    edges = dict()
    # A collection of all edges indexed by their StateControl and linked to the EdgeContainer
    edge_collection = dict()
    # Priority dictionary
    priority_dict = dict([(p, dict()) for p in Priority])
    # Global debug flag (overwritten in instances)
    debug_is_enabled = False

    @classmethod
    def set_env(cls, env_arg):
        cls.env = env_arg
        cls.grid = cls.env.rail.grid

env = None
def set_env(env_arg):
    """ Initialise this module with flatland interface.

        Note:
            This needs to be invoked before any other class usage.
    """
    global env
    GlobalContainer.set_env(env_arg)


class Utils(GlobalContainer):

    def switch_debug_mode(self, debug_is_enabled=None):
        """ Return global debug mode if no argument provided. """
        if debug_is_enabled is None:
            return
        self.debug_is_enabled = debug_is_enabled

    def debug(self, *args):
        """ Print message if debug_is_enabled is True. """
        if not self.debug_is_enabled:
            return
        import inspect
        frame = inspect.getouterframes(inspect.currentframe(), 2)
        print(str(self.__class__.__name__), ':\n\t', frame[1][3], ':\n\t\t',  *args)

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


class CoordinateContainer(Utils):
    """ Create a container of information collected in a coordinate.

        Note:
            Container to collect and extend information about railway 
            coordinates.

        Todo:
            Coordinate metrics should be accessible based on States
            -> e.g. direction based controls
    """
    def __init__(self, ID, coordinate, debug_is_enabled=None):
        self.id = ID
        self.coordinate = coordinate
        self.switch_debug_mode(debug_is_enabled)

        all_control_bits = self._all_control_bits(coordinate)
        valid_directions = self._valid_directions(all_control_bits)
        vertex_directions = self._vertex_directions(all_control_bits, valid_directions)
        controls = self._controls(all_control_bits, valid_directions)

        self.type = self._get_coordinate_type(valid_directions)

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

            sc = StateContainer(state, self)
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

    def _get_coordinate_type(self, valid_directions):
        if len(valid_directions) <= 2:
            if not self.coordinate in self.targets.keys():
                return CoordinateType.NORMAL_RAILWAY
        return CoordinateType.INTERSECTION

class StateContainer(object):
    """ Utility class to collect information about state.

        Used in global vertices and intersections.

        Note:
            Allows easy extension and access of state metrics.

        Todo:
            - should directly be reference by agent
    """
    def __init__(self, state, coordinate_container):
        self.state = state
        self.coc = coordinate_container
        self.id = self.coc.id

        self.type = StateType.NONE
        self.n_controls = self.coc.n_controls[state]
        self.priority = Priority(self.n_controls - 1)
        self.controls = self.coc.controls[state]
        self.direction2control = dict([(c.direction, c)
                                       for c in self.controls])
        # Store edge_id
        self.edges = list()

        # TODO: Store edges indexed by reachable states
        self.traverse = dict()


class AgentContainer(Utils):
    """ Get subset of metrics from flatland environment. 
    
        Note:
            Defines agent interface to flatland agents.

        TODO:
            - architecture design
            - interfaces to simulator
    """
    def __init__(self, ID, agent, debug_is_enabled=None):
        self.id = ID
        self._agent = agent
        self.switch_debug_mode(debug_is_enabled)

        a = self._agent
        (r, c) = a.initial_position
        d = Direction(a.initial_direction)
        self.state = State(r, c, d)
        self.status = AgentStatus.INITIALISED

        # Note: initialised in graph (locate_agents_in_graph)
        self.target = Coordinate(*agent.target)
        # To be documented: Usage and necessity?
        self.target_container = None
        # Add key for this agents target coordinate to global targets
        # NOTE: Used in decision of CoordinateContainer
        #       whether coordinate states become nodes
        self.targets[self.target] = None

        # Define possible search nodes for path computation
        self.target_nodes = None
        # Defines next node available for path decision
        self.current_node = None

        # TODO: when initialised?
        # Agent specific goal states that satisfy target coordinate
        self.tc : StateContainer = None
        self.target_edges = list()

        # speed = a.speed_data['speed']
        # import math; self.speed = math.ceil(1/speed)
        self.path = list()
        self.path_edge_containers = list()
        self.path_nodes = list()
        self.heuristic = dict()

    def initialise(self):
        """ Fetch current states and update targets.

            Note:
                Called after MyGraph has initialised railway.

        """
        #self.update()
        self.target_container = self.railway[self.target]
        self.target_nodes = self.target_container.valid_states
    
    def reset_path(self):
        self.heuristic = dict()
        self.path = list()
        self.path_edge_containers = list()
        self.path_nodes = list()
        self.heuristic = dict()
        self.status = AgentStatus.INFEASIBLE_PATH
        self.locate()

    def locate(self):
        """ Locate agent in graph and Find next search node.

            Note:
                If on edge use its path as heuristic.
        """
        state_container = self.states[self.state]
        if not state_container.type & StateType.NODE:
            edge_container_id = state_container.edges[0]
            edge_container = self.edges[edge_container_id]
            edge_direction = edge_container.state2direction[self.state]
            print('Edge direction initialised: {}'.format(edge_direction))
            print('On Edge container: {}'.format(edge_container_id))

            edge_container.force_vote(edge_direction, self)
            goal_state = edge_container.goal_state[edge_direction]
            self.heuristic.update(edge_container.path[edge_direction])
            self.current_node = edge_container.goal_state[edge_direction]
            self.path_edge_containers.append(edge_container)
            # NOTE: remove debug for production
            #print(self.id, ' Has heuristic since on EDGE')
            #print(self.edge_container_ids())
            #print(self.state)
            #print(self.path_edge_containers[0].state2progress[self.state])
            #eau
            return
        self.current_node = self.state

    def update_path(self, path):
        """ Use path with list of nodes to receive agent heuristics.

            Note:
                Path format defined from networkx with corresponding
                edge node entries.

            Todo:
                Debug the conversion to edge_ids
        """
        self.path_nodes = path
        for idx, node in enumerate(path):
            if idx == len(path)-1:
                continue
            print('from: ', node)
            print('to: ', path[idx+1])
            print('show all states accessible')
            print(self.states[node].traverse)
            control = self.states[node].traverse[path[idx+1]]
            edge = self.edge_collection[StateControl(node, control)]
            edge_path = edge.path
            self.path_edge_containers.append(self.edges[edge.container_id])
            self.heuristic.update(edge_path)
            self.heuristic.update([(node, control)])
            self.status = AgentStatus.FEASIBLE_PATH
            print('converted successfully!')
            # raise RuntimeError()

    def vote_edges(self):
        for edge_container_id, node in zip(self.edge_container_ids(),
                                           self.path_nodes):
            self.edges[edge_container_id].parse_agent_vote(node, self)

    def update(self):
        """ Update agent state with flatland environment state. """
        import flatland
        if not self._agent.status == flatland.envs.agent_utils.RailAgentStatus.ACTIVE:
            return
        a = self._agent
        d = a.direction
        (r, c) = a.position
        d = Direction(a.direction)
        self.state = State(r, c, d)
        # self.status = i.status

    # NOTE: move subsequent to __init__
    def edge_container_ids(self):
        return [e.id for e in self.path_edge_containers]
    
    # NOTE: Dev - remove for production
    def transition(self):
        self.path_edge_containers[0].eta

    def update_edge_progress(self):
        """ Return the amount of cells remaining after current state. """
        # TODO: trigger unregister event on edge_container and register event
        eta = self.path_edge_containers[0].update_agent_progress(self.id, self.state)
        if eta == 0:
            self.path_edge_containers.pop(0)
        return eta

    def set_control(self, controls):
        """ Update control dictionary and update active linked edge_containers"""
        # TODO: check if has a heuristic and path
        #       -> otherwise DO not departure
        import flatland
        if not self.status == AgentStatus.FEASIBLE_PATH:
            print('ID: ', self.id, ' TRIGGER emergency stop.')
            controls[self.id] = Control.S
            return
        print('Agent{}: '.format(self.id), '\nPath-IDs:{}'.format(self.edge_container_ids()))
        controls[self.id] = self.heuristic[self.state].control
        AgentType = flatland.envs.agent_utils.RailAgentStatus
        sc = self.states[self.state]
        # TODO: update current edge progress every step
        print('ETA:', self.update_edge_progress())
        print(AgentType(self._agent.status))
        #print(self.state)
        #print('Available control:')
        #print(sc.controls)
        return 
        controls[agent.id] = agent.heuristic[agent.state].control
        # print('FALIED', e)
        # controls[agent.id] = Control.F


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


class EdgeContainer(Utils):
    """ Edge related metrics and occupancy trackers.

        Note:
            Set default debug_is_enabled=None to fetch
            global debug_mode and set to True to activate
            debug for all EdgeContainers.

        Todo:
            1.
                Register agents and their planned entry step
                -> If agent registers and no vote done -> force switch

            2.
                Add collision matrix with agent steps for
                global N prediction steps along path length M
                -> matrix &operator should yield zero for collision free

            3. 
                AgentContainer -> get_control -> update current_edge -> move_agent
                    - If edge exit, select next edge_id from path_id
    """
    def __init__(self, ID, debug_is_enabled=True):
        self.id = ID
        self.vote = 0
        self.active_agents = dict()
        self.switch_debug_mode(debug_is_enabled)

        # EdgeDirection key with goal_state value
        self.goal_state = dict()
        # DirectionType key and common path values (2 entries)
        self.path = dict()
        # State keys and EdgeDirection values
        self.length = None

        self._forward = dict()
        self._backward = dict()
        # store forward and backward edges under EdgeDirection Key
        self._edge_registry = dict()

        self._edge_direction = dict()
        self._agent_registry = dict()

        self.state2progress = dict()
        # NOTE: dev dict: remove for production
        self.state2direction = dict()

    def _reset_vote(self):
        """ Reset vote and allow all edge directions to be used. """
        self.vote = 0
        self.voted = False

    def _is_forward(self):
        return self.vote >= 0

    def _get_direction(self, backward):
        """ Return the EdgeDirection for backward argument. """
        return (EdgeDirection.BACKWARD if backward
                else EdgeDirection.FORWARD)

    def on_direction(self, state):
        """ Return the direction of the edge for a given state. """
        return self.state2direction[state]

    def get_edges(self, voted=True):
        """ Return available edges under evaluated vote. """
        print('EC{}'.format(self.id), '(Vote{})'.format(self.vote))
        if voted:
            if self._is_forward():
                self.debug('EC-DEBUG: return forward edges')
                return self._forward.values()
            self.debug('EC-DEBUG: return backward edges')
            return self._backward.values()
        return self._edge_registry.values()
        # TODO: debug previous and remove subsequent
        #self.debug('EC-DEBUG: return all edges')
        #d = dict()
        #d.update(self._forward)
        #d.update(self._backward)
        #return d.values()
        #return dict(self._forwardself._backward).values()

    def add_edges(self, edges, backward=False):
        """ Add edges according to EdgeDirection to dict. """
        target_dict = (self._backward if backward else self._forward)
        edge_direction = self._get_direction(backward)
        for edge in edges:
            target_dict[edge.pair.vertex_1] = edge
            self._edge_direction[edge.pair.vertex_1] = edge_direction
            self._edge_registry[edge.pair.vertex_1] = edge

    def add_path(self, path, backward=False):
        """ Store common path in attribute according to EdgeDirection. """
        edge_direction = self._get_direction(backward)
        self.goal_state[edge_direction] = path[-1].state
        self.path[edge_direction] = path[:-1]
        for progress, StateControl in enumerate(path):
            print(self.id, ': ', StateControl.state)
            self.state2progress[StateControl.state] = progress
            # NOTE: dev dict -> to be removed for production
            self.state2direction[StateControl.state] = edge_direction
        self.length = len(path)

    def add_states(self, ingress_states, path, backward=False):
        """ Add direction encocding for state entries.

            Note:
                The ingress states are a dictionary of keys with states
                that share the same traversability direction
                and have values with ControlDirection.


            Note:
                - All obsolete: only required to add
                  ingress_states to state2direction
        """
        edge_direction = self._get_direction(backward)

        for state in ingress_states.keys():
            self.state2direction[state] = edge_direction
        for path_state_control in path:
            self.state2direction[path_state_control.state] = edge_direction

    def parse_agent_vote(self, state, agent):
        """ Register interest to use an edge in certain direction. """
        print('Agent edge ids:', agent.edge_container_ids())
        ids = agent.edge_container_ids()
        #print(agent.state)
        #print('Progress: {}'.format(self.edges[ids[1]].state2progress[state]))
        print('success??')
        # TODO: don't use edge_direction -> use PATH_STATES
        edge_direction = self.state2direction[state] #self._edge_direction[state]
        self.vote += edge_direction
        self._agent_registry[edge_direction] = agent.id
        self.voted = True

    def get_vote_affected_agents(self):
        """ Return all minority agent_ids from edge. 

            Note:
                If vote is positive (forward) or zero return all backward
                edges that are to be removed.
        """
        vote_affected = self._get_direction(backward=self.vote>=0)

        return self._agent_registry[vote_affected]

    def force_vote(self, edge_direction, agent):
        """ Enforce directional reservation for agents starting on edge. """
        self._reset_vote()
        self.vote += edge_direction
        self._agent_registry[edge_direction] = agent.id
        self.voted = True

    def register(self, agent_id, step):
        """ Register when an agent is expected to enter this edge.  """
        # TODO: if not voted, force switch
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

    # NOTE: Development: Test function
    def update_agent_progress(self, agent_id, state):
        """ Update agent_id edge progress and return eta in cell count. """
        progress = self.state2progress[state]
        return self.length - progress -1

    def exit(self, agent_id):
        """ Remove agent from active_agents and priority_dict. """
        prio = self.active_agents[agent_id].priority
        self.priority_dict[prio].pop(agent_id, None)
        # TODO: reset if last agent left -> indicates that edge can be revoted


class MyGraph(Utils):
    """Container for graph related actions.

        Note:
            Update active edges or compute shortest path.
    """
    def __init__(self, debug_is_enabled=None):
        self.switch_debug_mode(debug_is_enabled)
        self.visualisation_is_enabled = True  #NOTE: 'debug' in production stage

        self._graph = networkx.DiGraph()

        self._initialise_agents()
        self._initialise_graph()

    def _locate_agents_in_graph(self):
        for agent in self.agents.values():
            agent.initialise()
            agent.locate()

    def _is_explored(self, state, control):
        return StateControl(state, control) in self.edge_collection.keys()

    def _edge_ingress_states(self, state, direction):
        """ Find all states at coordinate that lead to the same edge. """
        return self.states[state].coc.direction2states[direction]

    def _reverse_edge_ingress_states(self, path):
        """ Return all states that led to edge from reversed direction. """
        goal_state = path[-1].state
        direction = FlipDirection[path[-2].control.direction]

        return self._edge_ingress_states(goal_state, direction)

    def _find_edge_path(self, ingress_states, edge_container_id):
        """ Return List of StateControl for state and control.

            Note:
                Returns list of (state,control) from state+1 cell until goal.

        """
        path = list()
        state, control = list(ingress_states.items())[0]
        path.append(StateControl(state, control))
        state = Simulator(state, control)

        while not self.states[state].type & StateType.NODE:
            self.states[state].edges.append(edge_container_id)
            control = self.states[state].controls[0]
            path.append(StateControl(state, control))
            state = Simulator(state, control)

        path.append(StateControl(state, ControlDirection(Control.S, None)))
        return path

    def _define_edges_from_path(self, ingress_states, path, 
                                edge_container_id):
        """ Parse entry_states and path into edges. """
        edges = list()
        goal_state = path[-1].state
        for ingress_state, control in ingress_states.items():
            if self._is_explored(ingress_state, control):
                raise RuntimeError()
            edge_id = len(self.edge_collection) + 1
            edge_path = [StateControl(ingress_state, control)] + path

            priority = self.states[goal_state].priority
            pair = Pair(ingress_state, goal_state)
            edge = Edge(pair, priority, edge_path, len(edge_path), 
                        edge_container_id)

            self.edge_collection[StateControl(ingress_state, control)] = edge
            edges.append(edge)
            if goal_state in self.states[ingress_state].traverse.keys():
                # Possible to have multiple edges from one state leading to another edge
                # This can lead to mismatched translation of networkx path tuples to
                # corresponding edge_container_ids
                print('duplicate edges encountered')
                raise RuntimeError()
            self.states[ingress_state].traverse[goal_state] = control

        return edges

    def _initialise_edges(self):
        """ Iterate over nodes and return edge containers. """
        for node in self.nodes.values():
            controls = node.controls
            for control in controls:
                if self._is_explored(node.state, control):
                    continue
                edge_container_id = len(self.edges) + 1
                edge_container = EdgeContainer(edge_container_id)

                # Forward edges
                direction = control.direction
                ingress_states = self._edge_ingress_states(node.state, direction)
                path = self._find_edge_path(ingress_states, edge_container_id)
                edges = self._define_edges_from_path(ingress_states, path,
                                                     edge_container_id)
                edge_container.add_path(path, backward=False)
                edge_container.add_states(ingress_states, path)
                edge_container.add_edges(edges, backward=False)

                # Backward edges
                ingress_states = self._reverse_edge_ingress_states(path)
                path = self._find_edge_path(ingress_states, edge_container_id)
                edges = self._define_edges_from_path(ingress_states, path,
                                                     edge_container_id)
                edge_container.add_path(path, backward=True)
                edge_container.add_states(ingress_states, path, backward=True)
                edge_container.add_edges(edges, backward=True)

                self.edges[edge_container_id] = edge_container
            node.edges.append(edge_container_id)

    def _initialise_railway(self):
        """ Defines all railway coordinates with unique railway ID. """
        env_railway = numpy.nonzero(self.grid)
        id_railway = -1
        for r, c in zip(*env_railway):
            id_railway += 1
            coordinate = Coordinate(r, c)
            CoordinateContainer(id_railway, coordinate)

    def _create_graph(self, direction_aware=True):
        """ Initialise networkx graph with all edges.

            Note:
                Recommended to first run vote edges and update edge_containers
                to remove deadlocks.

                Direction_aware flag allows to select only prioritised edges from
                each edge_container. (unidirectional section use)

            TODO:
                Remove edge_containers from voting, if active_agents on edge
                (e.g.: from init)
        """
        # NOTE: check computation time and profile graph update methods
        import time
        timestamp = time.time()
        self._graph.clear()
        for edge_container in self.edges.values():
            edges = edge_container.get_edges(voted=direction_aware)
            for edge in edges:
                self._graph.add_edge(*edge.pair, length=edge.length)
        print('Reset graph in {:.4}s'.format(time.time() - timestamp))
        print(self._graph.edges())

    def _update_agent_heuristics(self):
        """ Compute shortest path for each agent. """
        for agent in self.agents.values():
            agent.reset_path()
            self.shortest_path(agent.id)

    def _conduct_vote(self):
        """ Execute voting on all edges in agent's path. """
        for agent in self.agents.values():
            agent.vote_edges()

    def _update_graph(self):
        """ Update graph by removing outvoted graph. """
        # TODO: fix unvoted / irregularities etc.
        for edge_container in self.edges.values():
            edges = edge_container.get_edges()
            for edge in edges:
                self._graph.remove_edge(*edge.pair)

    def _initialise_graph(self):
        self._initialise_railway()
        self._initialise_edges()
        self._locate_agents_in_graph()
        self.debug('Initialise graph')
        self._create_graph(direction_aware=False)
        self._update_agent_heuristics()

        self._conduct_vote()
        self.debug('Recompute heuristics with direction constraints')
        self._create_graph(direction_aware=True)
        self._update_agent_heuristics()

    def _initialise_agents(self):
        """ Parse flatland metrics from agents. """
        for agent_id, agent in enumerate(self.env.agents):
            self.agents[agent_id] = AgentContainer(agent_id, agent)

    def _shortest_path(self, start, goal):
        """ Parse arguments to networkx implementation. """
        return networkx.shortest_path(self._graph, start, goal, 'length')

    def shortest_path(self, agent_id):
        """ Update heuristic for agent with agent_id. """
        import time
        agent = self.agents[agent_id]
        def agent_text():
            return 'Agent{0}: '.format(agent_id), agent.status,
        current = agent.current_node
        timestamp = time.time()
        for target in agent.target_nodes.keys():
            try:
                sp = self._shortest_path(current, target)
                agent.update_path(sp)
                debug_message = 'SUCCESS! '
                debug_message += '{} '.format(target)
                debug_message += '({:.4}s)'.format(time.time() - timestamp)
                self.debug(agent_text(), debug_message)
                return
            except (networkx.exception.NodeNotFound, networkx.NetworkXNoPath) as e:
                # TODO: consider that agent might be able to use 
                #       existing heuristic as path
                agent.status = AgentStatus.INFEASIBLE_PATH
                self.debug(agent_text(), ' ERROR! \n\t', e)
        debug_message = 'Total failure of path comptutation! '
        debug_message += '{} '.format(target)
        debug_message += '({:.4}s)'.format(time.time() - timestamp)
        self.debug(agent_text(), agent.status, debug_message)

    # NOTE: Final placement under rollout.py
    def controls(self):
        controls = dict()
        for agent in self.agents.values():
            agent.set_control(controls)
        return controls

    def update(self):
        """ Update each agent state with most recent flatland states. """
        for agent in self.agents.values():
            agent.update()

    def visualise(self, env_renderer):
        """ Call display utility methods and visualise metrics and states. """
        if self.visualisation_is_enabled:
            import display
            # define agent_ids to visualise
            # TODO: visualise agent path -> heuristic keys 
            display.show_agents(env_renderer, self.agents.values())



if __name__ == "__main__":
    print('Graph - Testbed')
    env.reset()
    env_renderer.reset()

    for step in range(500):
        env.step(dict((a,0) for a in range(env.get_num_agents())))

    env_renderer.render_env(show=True, show_predictions=False, show_observations=False)
    input('press to close')
