import abc
import types

class Optimizer (abc.ABC):
    @abc.abstractmethod
    def optimize(self, architecture_builder: types.FunctionType):
        """
        Optimize the architecture builder function.
        
        Args:
            architecture_builder (types.FunctionType): The function that builds the architecture.
        """
        pass