#!/bin/env python

import enum
import collections

from multiroll.constants import *


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

