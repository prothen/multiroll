#!/bin/env python

import numpy

from .constants import *

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
        # import inspect
        #frame = inspect.getouterframes(inspect.currentframe(), 2) - access with frame[1][3]
        print(str(self.__class__.__name__), ':\n\t',  ':\n\t\t',  *args)

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
