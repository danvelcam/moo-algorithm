from abc import ABC, abstractmethod
import numpy as np
import random 

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

    # def generate_population(self,population_size: int) -> np.ndarray:
    #     """
    #     Method to generate a population compound by individuals to find a solution
    #     """
    #     bounds = self.get_bounds() 
    #     population = np.random.uniform(bounds[:, 0], bounds[:, 1], (population_size, bounds.shape[0]))
    #     return population

    def generate_population(self, population_size: int) -> np.ndarray:
        """
        Method to generate a population compound by individuals to find a solution.
        This version generates the population using random.random(), without a specific distribution.
        """
        bounds = self.get_bounds()  # Obtener los límites para cada variable
        population = np.zeros((population_size, bounds.shape[0]))

        for i in range(population_size):
            for j in range(bounds.shape[0]):
                # Generar un valor aleatorio con random.random() y escalarlo a los límites
                population[i, j] = bounds[j, 0] + random.random() * (bounds[j, 1] - bounds[j, 0])

        return population