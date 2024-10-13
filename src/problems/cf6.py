from problems.problem_base import ProblemBase
import numpy as np
import math

class CF6(ProblemBase):
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.pareto_front = 'src/cf6_pf.dat'
        self.name = 'cf6'

    def evaluate(self,individual: np.ndarray) -> np.ndarray:
        x1 = individual[0]
        J1 = [j for j in range(1, self.dimensions) if j % 2 != 0]  # Impares
        J2 = [j for j in range(1, self.dimensions) if j % 2 == 0]  # Pares

        y_j_J1 = np.array([individual[j] - 0.8 * x1 * np.cos(6 * np.pi * x1 + j * np.pi / self.dimensions) for j in J1])
        y_j_J2 = np.array([individual[j] - 0.8 * x1 * np.sin(6 * np.pi * x1 + j * np.pi / self.dimensions) for j in J2])
        
        f1 = x1 + np.sum(y_j_J1**2)
        f2 = (1-x1)**2 + np.sum(y_j_J2**2)
        
        return np.array([f1, f2])
    
    def constraints(self,individual: np.ndarray) -> np.ndarray:
        x1 = individual[0]
        x2 = individual[1]
        x4 = individual[3]

        c1 = x2 - 0.8 * x1 * np.sin(6 * np.pi * x1 + ((2*np.pi)/self.dimensions)) - np.sign(0.5 * (1-x1) - (1 - x2) ** 2)
        c1 = c1 * np.sqrt(np.abs(0.5 * (1 - x1) - (1-x2)**2))

        c2 = x4 - 0.8 * x1 * np.sin(6 * np.pi * x1 + ((4*np.pi)/self.dimensions)) - np.sign(0.25 * np.sqrt(1 - x1) - 0.5 * (1 - x1))
        c2 = c2 * np.sqrt(np.abs(0.25 - np.sqrt(1 - x1) - 0.5 * (1 - x1)))

        if c1 >= 0:
            c1 = 0
        if c2 >= 0:
            c2 = 0
        return np.array([c1,c2])
    
    def handle_boundary(self, y, handling):
        match handling:
            case "rebound":
                y[0] = -y[0] if y[0] < 0 else (2 - y[0] if y[0] > 1 else y[0])
                for i in range(1, len(y)):
                    y[i] = -2 if y[i] < -2 else (2 if y[i] > 2 else y[i])
            
            case "clip":
                y[0] = max(0, min(1, y[0]))
                for i in range(1, len(y)):
                    y[i] = max(-2, min(2, y[i]))
            
            case "wrapping":
                y[0] = y[0] % 1
                for i in range(1, len(y)):
                    if y[i] < -2:
                        y[i] += 4  
                    elif y[i] > 2:
                        y[i] -= 4  
        return y 
    

    def get_bounds(self) -> np.ndarray:
        return np.array([[0, 1]] + [[-2, 2]] * (self.dimensions - 1)) 
    

