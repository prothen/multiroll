#!/usr/bin/env python
"""
    Multiagent rollout implemented on flatland environment.


    Note:
        This module expects to be initialised with

            graph.set_env(env)

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
StateControl = collections.namedtuple('State', ['state', 'control_direction')
# Store a pair of vertices
Pair = collections.namedtuple('Pair', ['vertex_1', 'vertex_2'])
# Define an edge with its corresponding feed_forward control (autopilot until next intersection)
Edge = collections.namedtuple('Edge', ['pair', 'priority', 'feed_forward', 'length'])


class Priority(enum.IntEnum):
    """ Priority definition for edges. """
    # Edge goal is intersection
    NONE = 0
    # Low amount of traversability 2 choices
    LOW = 1
    # High amount of traversability 3 choices
    HIGH = 2


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

    def __init__(self):
        self._update_env()

    @classmethod
    def _update_env(cls):
        global env
        cls.set_env(env)

    @classmethod
    def set_env(cls, env_arg):
        cls.env = env_arg
        cls.grid = cls.env.rail.grid

env = None
# legacy
def set_env(env_arg):
    global env
    env = env_arg
    #GlobalContainer.set_env(env_arg)


class Utils(GlobalContainer):

    @staticmethod
    def _bits(i, value):
        """ Return direction dependent control bits. """
        return (value >> (3 - i) * 4) & 0xF

    def _all_control_bits(self, coordinate: Coordinate):
        """ Return list of control_bits for all directions. """
        return [self._bits(d, self.grid[coordinate.r][coordinate.c])
                for d in range(Direction.COUNT)]

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
    def _is_intersection(coordinate: Coordinate):
        """ Return true if the current coordinate has more than 2 transitions. """
        return _n_directions(coordinate) > 2

    def _is_railway(self, state: State):
        """Return true if state has any transitions. """
        return (self.grid[state.r][state.c] >> (3 - state.d) * 4) & 0xF

    @staticmethod
    def _directions2controls(directions, direction_agent):
        """Transform a direction into a corresponding control for rail_env.  """
        ds = directions
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
    def __init__(self, ID, coordinate, debug_is_enabled=True):
        super().__init__()
        all_control_bits = self._all_control_bits(coordinate)
        valid_directions = self._valid_directions(all_control_bits)
        vertex_directions = self._vertex_directions(all_control_bits, valid_directions)
        controls = self._controls(all_control_bits, valid_directions)

        if debug_is_enabled:
            print('\n##################################')
            print('##################################')
            print('State:\n\t {}'.format(coordinate))
            print('Control bits:\n\t {}'.format(all_control_bits))
            print('Directions:\n\t {}'.format(valid_directions))
            print('Controls:\n\t {}'.format(controls))

        self.id = ID
        self.coordinate = coordinate

        self.valid_states = dict()
        self.controls = dict()
        self.n_directions = dict()

        for d, controls in zip(valid_directions, controls):
            self.n_directions[Direction(d)] = len(controls)
            state = State(coordinate.r, coordinate.c, Direction(d))
            valid_controls = controls
            for i, control in enumerate(controls):
                if not self._is_railway(Simulator(state, control)):
                    print('\t\t->Encountered dead-end (Flip!)')
                    valid_controls[i] = ApplyDeadendControl(control)
            self.controls[Direction(d)] = valid_controls
            sc = StatesContainer(state, self)
            self.states[state] = sc
            if Direction(d) in vertex_directions:
                self.vertices[state] = sc
            self.valid_states[state] = sc
        self.railway[coordinate] = self


class StatesContainer(object):
    """ Utility class to collect information about state.

        Used in global vertices and intersections.

        Note:
            Allows easy extension and access of state metrics.
    """
    def __init__(self, state, coordinate_container):
        self.state = state

        # Corresponding coordinate container
        self.coc = coordinate_container
        # Get common coordinate ID for collision prediction
        self.id = self.coc.id
        self.traversability = self.coc.n_directions[state.d]
        self.priority = Priority(self.traversability - 1)
        self.controls = self.coc.controls[state.d]


class AgentContainer(GlobalContainer):
    """ Get subset of metrics from flatland environment. """
    def __init__(self, ID):
        self.id = ID
        a = self.env.agents[self.id]
        (r, c) = a.initial_position
        d = a.initial_direction
        self.state = State(r, c, d)
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

    def initialise_prediction(self):
        self._find_initial_edge()
        self._find_shortest_path()
        self._populate_prediction()

    def _find_initial_edge(self):
        pass

    def _populate_prediction(self):
        """ Define global collision matrix for k iterations.

            railway_occupancy
            N_ID x M
        """
        edge_idx = 0
        edge_n = 0
        edge_id = self.path[edge_n]
        edge = self.edges[edge_id]
        for k in range(M):
            if k % self.speed:
                continue
            edge_idx += 1
            if edge.path_length >= edge_idx:
                edge_n += 1

            # TODO: check that feed_forward provides StateContainer reference
            state = edge.feed_forward[edge_idx]
            self.railway_occupancy[k+STEP][state.id] += 1
            # self.prediction[k+STEP] = 
            pass

    def step(self):
        """ tbd """
        # apply movement
        # move idx pointer
        # run enter or exit edge for priority queue


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
        super().__init__()

        self._verbose = debug
        self._show_transitions = False

        self.n_vertices = 0
        self.graph = networkx.Graph()
        self.initialise()

    def find_vertices(self):
        """ Defines all vertices with unique railway ID.

            Note
                self.vertices has railway id
                self.railway dict has ID as keys and vertices list with coordinate
                    --> each vertex has edges that need to be activated or deactivated
                    --> collect pairs
        """
        env_railway = numpy.nonzero(self.grid)
        id_railway = -1
        # FIND obvious state-based vertices for initial exploration
        for r, c in zip(*env_railway):
            id_railway += 1
            coordinate = Coordinate(r, c)
            CoordinateContainer(id_railway, coordinate)

    def _find_edge_entry_states(self, vertex, control):
        """ Find all possible states that lead to the same edge. """

        edge_entries = dict()
        entry_vertices[vertex] = control

        direction = Simulator(vertex, control).d

        coordinate_container = self.states[vertex].coc
        state_containers = coordinate_container.valid_states.values()

        for state_container in state_containers:
            state = state_container.state
            controls = state_container.controls
            for control in controls:
                if control.direction == direction:
                    entry_vertices[state] = control

                # Detect possible intersections on this occassion
                if state not in self.vertices.keys():
                    print('Found entry to edge via intersection')
                    if state not in self.intersections.keys():
                        print('Added entry to intersections dict!')
                        self.intersections[state] = self.states[state]
        return edge_entries

    def _find_edge_control(self, vertex, control):
        """ Return state controls pairs for each grid element along edge.

            Note:
                Returns list of (state,control) from vertex+1 cell until goal.

        """
        path_controls = list()
        state = vertex
        n_path = 1
        state = Simulate[controls.control](state, controls)
        controls = self.states[state]

        # test that the current coordinate is not an intersection and not a vertex
        while self._n_directions(state) <= 2 and len(controls) < 2:
            n_path += 1
            controls = controls[0]
            state = Simulate[controls.controls](state, controls)
            path.append((controls, state))
            sc = self.states[state]
            controls = sc.controls
        goal_state = state
        if goal_state not in self.vertices.keys():
            # Create artificial vertex for intersection
            self.intersections[state] = self.states[state]
        priority = self._edge_priority(goal_state)
        for state, control in edge_entry_states.items():
            state_control = StateControl(state, control)
            self.edge_collection[state_control] = edge_container_id 
            # define edge pairs priority
            pair = Pair(state, goal_state)
            self.edges = Edge(pair, priority, path, n_path)
        # get edge_container
        # add all edges 

    def _find_reverse_goal_state(self, edge_container):
        ec = edge_container

        # get edge
        # get path
        # get last element
        # extract direction
        # go through goal_coordinaten
        # -> create goal_coordinate
        # -> coc
        # -> go through all controls.items() 
        # -> save all states that provide direction to path element
        # -> update intersections if not added already
        # -> add StateControl hash to edges 
        pass

    def _define_edge_from_path(self, entry_states, path):
        # 
        goal_state = path[0][0])
        for entry_state, control in entry_states.items():
            state_control = StateControl(entry_state, control)
            pair = Pair(entry_state, goal_state)
            edge_id = len(self.edge_collection) + 1
            edge = Edge(pair, path, n_path)
        # define pair
        
        pass

    def _find_edges(self, vertex: State, intersections: dict):
        """ Return edges for a vertex to the next intersection or vertex.

            Note:
                The exploration is based on the deadend compliant state->control
                dictionary and creates an edge to itself.
        """
        edges = list()
        sc = self.vertices[vertex]
        controls = sc.controls
        print('found state controls: {}'.format(controls))
        for control in controls:
            # Skip already explored edges using the edge_collection
            if StateControl(sc.state, control) in self.edge_collection.keys():
                continue
            edge_container_id = len(self.edges) + 1
            edge_container = EdgeContainer(edge_container_id)
            print('use control{}'.format(controls_i))

            # GET FORWARD EDGES
            # get all entry states to the same railway section
            edge_entry_states = self._find_edge_entry_states(vertex, control)
            path = self._find_edge_control(vertex, control)
            edges = self._define_edge_from_path(edge_entry_states)
            for edge in edges:
                edge_container.add_forward_edge(edge)

            # GET BACKWARD EDGES
            goal_state_reversed = self._find_reverse_state(edge_forward)
            edge_entry_states = self._find_edge_entry_states(goal_state_reversed, control)
            edge_backward = self._find_edge_control(controls_i, goal_state)
            path = self._find_edge_control(goal_state_reversed, control_reversed)
            for edge in edges:
                edge_container.add_forward_edge(edge)

            # TODO: Shortest path -> Edge -> pair -> pairs -> other edges
            # Search corresponding direction
            # CREATING EDGECONTROL
            # TODO: define unique edge id

            # LEGACY:
            # find REVERSE
            # TODO: this is not the last path element but the goal_state direction!
            path_entry_direction = path[-1][1].d
            coordinate = Coordinate(state.r, state.c)
            c = self.railway[coordinate].controls
            ld = list()
            for cd in c:
                for cdi in cd:
                    if cdi.direction == path_entry_direction:
                        nd = (State(state.r, state.c, cdi.direction))
                        ld.append(nd)

            #TODO; define edge from all nd to vertex
            #       --> 
            # TODO: add valid_directions, all_control_bits
            #self.environment[coordinate].valid_directions
            # get last state before new_vertex 
            # get all controls 
            # test all controls / directions and collect all that yield previous state

            # for which states do we get in previous state
            vertex = state
            state = vertex
            n_path = 1
            state = Simulate[controls_i.control](state, controls_i)
            controls_i = self.states[state]
            # test that the current coordinate is not an intersection and not a vertex
            while self._n_directions(state) <= 2 and len(controls_i) < 2:
                n_path += 1
                controls_i = controls_i[0]
                state = Simulate[controls_i.control](state, controls_i)
                path.append(controls_i)
                controls_i = self.states[state]
            print('Self loop detected') if vertex == state else None
            if state not in self.vertices.keys():
                # Create artificial vertex for intersection
                # TODO: handle vertex collision
                # --> create lookup for vertex --> (m,N) --> same coordinate -> same m
                intersections[state] = None

            d = self.pairs.keys()
            condition_pair = pair in d
            condition_pair_mirror = pair_mirror in d
            # link each pair to same container
            def update_pairs(self):
                if not condition_pair:
                    if not condition_pair_mirror:
                        coordinate = Coordinate(state.r, state.c)
                        self.railay[coordinate].append_pair_id(self.n_pairs)
                        # create new pair ID
                        pair_container = PairContainer(ID=self.n_pairs,
                                                       pair=pair)
                        self.pairs[pair] = pair_container
                        self.pairs[pair].edges[edge_id] = edge
                        # ADD via external logic self.pairs[pair_mirror].append_edge(edge)
                        self.n_pairs += 1
                        print('Previously unencountered pair detected and added')
                        return
                    print('Pair: {} (linked to mirror)'.format(pair))
                    self.pairs[pair] = PairContainerMirror(self.pairs[pair_mirror])
                self.pairs[pair].append_edge(edge)
                print('Pair: {} (appended)'.format(pair))
            update_pairs(self)
            edges.append(edge)
        return edges

    def find_edges(self):
        intersections = dict()
        edge_list = list()
        START = True
        explore = dict(self.vertices)
        idx = 0
        # Explore all known vertices for intersections
        # -> then explore all intersections until no more intersections are detected
        while len(intersections) > 0 or START:
            START = False
            idx += 1
            intersections = dict()
            for vertex in explore.keys():
                edges = self._find_edges(vertex, intersections)
                self.vertices[vertex] = edges
                edge_list += edges
            self.vertices.update(intersections)
            # Add new intersections (fake-vertices) for exploration
            explore = dict(intersections)
            print('{} : found new'.format(idx))


    def initialise(self):
        self.find_vertices()
        self.find_edges()
        # collect collision elements
        # self.sort_edges()
        # connect agent targets
        # initialise agents
        # compute shortest path for agents

        print(list(railway.values())[-1])
        eoau
        # print amount of railway coordinates
        # --> use as occupancy indicator
        # --> from coordinate ID derive occupancy matrix
        # --> save coordinate ID for each vertex

        # get all vertices and find intersections
        # define all transitions
        print(env)
        for agent in env.agents:
            pass
            # state from agent
            # get goal coordinate
            # get valid directions of target)
            # find next vertices/intersections
            # create edges add to edges_list
            # find vertices (explore

        # for each agent find next vertex
        for agent in env.agents:
            pass
            # get initial position
            # get initial direction
            # create STATE
            # if a STATE is a vertex
            # optimise

        #for intersection in intersections.keys():
        # #   edges = find_vertices(intersection, more_intersections)
        #    pairs[intersection] = edges
        #    # print(edges)
        #    edge_list += edges

        #print('found even more intersections: {}'.format(more_intersections))
        # Find all pairs
        #for vertex, edges in pairs.items():
        #    for edge in edges:
        ##        v2 = edge.vertex_2
        # #       if v2 in vertices.keys():
        #            print('valid edge')
                #vertices[v2]
                #for edge in edges:
                #    if edge.vertex_2 == vertex:
                #        print('found fitting')
            

        # TODO: find shortest path for each agent and upvote edge direction with +1, if completed edge then remove own vote with -1
        #       -> update active paths consecutively and re-compute whenever the railway changed
        print('create graph now')


    def report_vertices(self):
        for vertex in vertices.keys():
            pass
            #print('\tVertex: \t{}'.format(vertex))
            #print('\tControl: \t{}'.format(self.states[vertex]))

    def show_vertices(self, env_renderer):
        #env_renderer.renderer.plot_single_agent((16,22), 1, 'r', target=(0,0),selected=True)
        #return
        v = list(vertices.keys())[0]
        controls = self.states[v]
        l = list()
        for v in vertices.keys():
            for control in controls:
                l += [Direction2Target[control.direction]]
                c = Transition2Color[control.direction]
                env_renderer.renderer.plot_single_agent((v.r, v.c), v.d, 'r',selected=True)

                if self._show_transitions:
                    env_renderer.renderer.plot_transition(
                            position_row_col=(v.r, v.c),
                            transition_row_col=l,
                            color=c
                            )

    def initialise_agents(self):
        """ To be done. """
        # get agent initial position
        # get agent min max speeds
        # get agent_id
        # add to self.agents[id]
        pass


if __name__ == "__main__":
    print('Graph - Testbed')
    env.reset()
    env_renderer.reset()

    for step in range(500):
        env.step(dict((a,0) for a in range(env.get_num_agents())))
    env_renderer.render_env(show=True, show_predictions=False, show_observations=False)
    input('press to close')
