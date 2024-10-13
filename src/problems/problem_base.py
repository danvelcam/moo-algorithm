from abc import ABC, abstractmethod
import numpy as np

class ProblemBase(ABC):
    @abstractmethod
    def evaluate(self, individual: np.ndarray):
        """
        Method to evaluate an individual solution for the given problem.
        """
        pass

    @abstractmethod
    def get_bounds(self):
        """
        Defines the bounds for the problem's variables.
        """
        pass

    def generate_population(self,population_size: int) -> np.ndarray:
        """
        Method to generate a population compound by individuals to find a solution
        """
        bounds = self.get_bounds() 
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (population_size, bounds.shape[0]))
        return population