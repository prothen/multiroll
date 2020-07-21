#!/usr/bin/env python

import numpy

from flatland.core.env_observation_builder import ObservationBuilder

class Empty(ObservationBuilder):
    def __init__(self):
        super().__init__()
        print(self.__dict__)

    def reset(self):
        pass

    def get(self, handle: int = 0):
        return numpy.ones((self.env.num_agents()))
