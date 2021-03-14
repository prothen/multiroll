#!/bin/env python
"""

    Author: Philipp Rothenhäusler, Stockholm 2020

"""

import argparse

class Params:
    @classmethod
    def list(cls):
        return [(a, getattr(cls, a)) for a in dir(cls) if cls.__is_attribute(a)]

    @classmethod
    def keys(cls):
        return [a for a in dir(cls) if  cls.__is_attribute(a)]

    @classmethod
    def __is_attribute(cls, name):
        return cls.__not_function(name) and cls.__not_special(name)

    @classmethod
    def __not_function(cls, name):
        return not callable(getattr(cls, name))

    @staticmethod
    def __not_special(name):
        return not name.startswith('__')

class MapParamsDefault(Params):
    SEED = 14
    H = 40
    W = 60
    N_CONNECTIVITY = 4
    N_AGENTS = 10
    N_CITIES = 9
    REMOVE_AGENTS_AT_TARGET = True
    RECORD_STEPS = False


class VisualisationParamsDefault(Params):
    N_SIM_STEPS = 100
    PLOT_STEPS = 2


class RolloutParamsDefault(Params):
    N_PREDICT_STEPS = 10
    COST_NOT_FINISHED = 1
    COST_COLLISION = 10


class FlagsDefault(Params):
    DEBUG = False
    DISPLAY = True
    STEP = True
    ROLLOUT = False


class Config:
    """ Simple config wrapper to allow programmatic parameter setting.

        Supports subclassing and can be extended with JSON (yaml) parsing.

        Note:
            In order to customise settings in a script subclass the parameters
            and parse them as argument to the config initialisation.

        Numerical parameters are accessed through the dictionary self.params[name] and
        boolean flags are queried through ConfigInstance.Active(name)

    """
    def __init__(self,
                 flag_params=FlagsDefault,
                 visualisation_params=VisualisationParamsDefault,
                 map_params=MapParamsDefault,
                 rollout_params=RolloutParamsDefault):
        self._flag_params = flag_params
        self._visualisation_params = visualisation_params
        self._map_params = map_params
        self._rollout_params = rollout_params

        # Boolean flags
        self._flags = dict()
        # Numerical values and parameters
        self.params = dict()
        # Parser instance
        self._parser = None

        self._initialise_defaults()

    def active(self, flag):
        return self._flags[flag]

    def _initialise_params(self):
        # Define Visualisation defaults
        for entry, value in self._visualisation_params.list():
            self.params[entry] = value
        for entry, value in self._map_params.list():
            self.params[entry] = value
        for entry, value in self._rollout_params.list():
            self.params[entry] = value

    def _initialise_flags(self):
        for entry, value in FlagsDefault.list():
            self._flags[entry] = value

    def _initialise_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--display', type=str2bool,
                dest='DISPLAY', help='Enable visualisation',
                default=True)
        parser.add_argument('--step', type=str2bool,
                dest='STEP', help='Enable visualisation',
                default=True)
        parser.add_argument('--roll', type=str2bool,
                dest='ROLLOUT', help='Enable visualisation',
                default=True)
        self._parser = parser

    def _initialise_defaults(self):
        # define numerical parameters
        self._initialise_params()
        # define boolean flags
        self._initialise_flags()
        # define arg parser
        self._initialise_parser()

    def parse_args(self):
        args = self._parser.parse_args()
        for key in self._flag_params.keys():
            self._flags[key] = getattr(args,key, self._flags[key])

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

