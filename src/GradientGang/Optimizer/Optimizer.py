import abc
import types

class Optimizer (abc.ABC):
    @abc.abstractmethod
    def optimize(self, architecture_builder: types.FunctionType):
        pass