#!/bin/env python


from multiroll.scheduler import *
from .constants import *
from .framework import *


class CoordinateContainer(Utils, multiroll.coordinate.CoordinateContainer):
    """ Create a container of information collected in a coordinate.

        Note:
            Container to collect and extend information about railway 
            coordinates.

        Todo:
            Coordinate metrics should be accessible based on States
            -> e.g. direction based controls
    """
    pass

class StateContainer(Utils, multiroll.coordinate.StateContainer):
    """ Utility class to collect information about state.

        Used in global vertices and intersections.

        Note:
            Allows easy extension and access of state metrics.

        Todo:
            - should directly be reference by agent
    """
    pass
