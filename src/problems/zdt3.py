from problems.problem_base import ProblemBase
import numpy as np
import math

class ZDT3(ProblemBase):
    def __init__(self, dimensions=30):
        self.dimensions = dimensions 
        self.pareto_front = 'src/zdt3_pf.dat'
        self.name = 'zdt3'


    def evaluate(self,individual: np.ndarray) -> np.ndarray:
        f1 = individual[0]
        g = 1 + (9 / (30 - 1)) * np.sum(individual[1:])
        h = 1 - math.sqrt(f1/g) - ((f1/g) * math.sin(10 * math.pi * f1 ))
        f2 = g * h
        return np.array([f1,f2])
    
    def get_bounds(self) -> np.ndarray:
        return np.array([[0,1]] * self.dimensions)
    
    def handle_boundary(self,y, handling):
        match handling:
            case "rebound":
                y = np.array([-xi if xi < 0 else (2 - xi if xi > 1 else xi) for xi in y]) 
            case "clip":
                y = np.array([0 if xi < 0 else ( 1 if xi > 1 else xi) for xi in y])
            case "wrapping":
                y = np.array([xi -1 if xi > 1 else (xi + 1 if xi < 0 else xi ) for xi in y ])
        return y

    
    

