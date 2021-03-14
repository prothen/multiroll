#!/usr/bin/env python
"""

    Author: Philipp Rothenhäusler, Stockholm 2020

"""

import numpy

from flatland.core.env_observation_builder import ObservationBuilder

class PlaceholderObs(ObservationBuilder):

    def reset(self):
        return

    def get(self, handle: int = 0):
        return numpy.empty((1))#numpy.ones((self.env.num_agents()))

