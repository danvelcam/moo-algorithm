from problems.problem_base import ProblemBase
import numpy as np
import math

class CF6(ProblemBase):
    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.pareto_front = 'src/cf6_16.dat'
        self.name = 'cf6'
        self.max = 2 

    def evaluate(self,individual: np.ndarray) -> np.ndarray:
        x1 = individual[0]
        J1 = [j for j in range(2, self.dimensions+1) if j % 2 != 0]  # j impar y 2 <= j <= n
        J2 = [j for j in range(2, self.dimensions+1) if j % 2 == 0]  # j par y 2 <= j <= n

        # Inicializar las funciones objetivo
        f1 = x1
        f2 = (1 - x1)**2
        
        # Calcular y_j y sumar los términos correspondientes
        for j in J1:
            y_j = individual[j-1] - 0.8 * x1 * np.cos(6 * np.pi * x1 + (j * np.pi) / self.dimensions)
            f1 += y_j**2
            
        for j in J2:
            y_j = individual[j-1] - 0.8 * x1 * np.sin(6 * np.pi * x1 + (j * np.pi) / self.dimensions)
            f2 += y_j**2
        
        
        return np.array([f1, f2])
    
    def constraints(self,individual: np.ndarray) -> np.ndarray:
        x1 = individual[0]  # x1
        x2 = individual[1]  # x2
        x4 = individual[3]  # x4

        term1_r1 = x2 - 0.8 * x1 * np.sin(6 * np.pi * x1 + (2 * np.pi / self.dimensions))
        term2_r1 = np.sign(0.5 * (1 - x1) - (1 - x1)**2) * np.sqrt(abs(0.5 * (1 - x1) - (1 - x1)**2))
        restriccion1 = term1_r1 - term2_r1

        term1_r2 = x4 - 0.8 * x1 * np.sin(6 * np.pi * x1 + (4 * np.pi / self.dimensions))
        term2_r2 = np.sign(0.25 * np.sqrt(1 - x1) - 0.5 * (1 - x1)) * np.sqrt(abs(0.25 * np.sqrt(1 - x1) - 0.5 * (1 - x1)))
        restriccion2 = term1_r2 - term2_r2

        return np.array([restriccion1, restriccion2])

    
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
    

