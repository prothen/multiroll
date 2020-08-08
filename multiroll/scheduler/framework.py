#!/bin/env python


import numpy


from multiroll.framework import *
from multiroll.scheduler import *
from .constants import *


class GlobalContainer(multiroll.framework.GlobalContainer):
    # Edge containers triggered for reactivation of unscheduled edges
    edge_reactivation = dict()


class Utils(GlobalContainer, multiroll.framework.Utils):
    pass

