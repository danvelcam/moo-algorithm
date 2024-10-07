import numpy as np
import math
import random
class MooAlgorithm():
    def __init__(self, population, generations, neighborhood, max, scale_factor=0.5, boundary_handling = "rebound"):
        if population  <= 0:
            raise ValueError(f"Population should be a positive number, but received {population}")
        if not  0 < neighborhood <= 1:
            raise ValueError(f"""Neighborhood should be a percentage 
                represented between 0 (not included) and 1 (included), but received {neighborhood}""")
        self.p = population
        self.g = generations
        self.ng_size = math.floor(neighborhood * population)
        self.max = max
        self.lambda_population = self.generate_lambda_population()
        self.euclidean_distance  = self.euclidean_distance_matrix()
        self.neighbors = self.closest_neighbors()
        self.xi = np.array([np.random.rand(30) for i in range(self.p)])
        self.evaluations = self.evaluate_population() 
        self.zi = np.array([np.min(self.evaluations[:,0]), np.min(self.evaluations[:,1] )])
        self.scale_factor = scale_factor
        self.boundary_handling = boundary_handling
        patata = self.cross(6)
        print(patata)
        print(self.mutation(patata))

    #Cuando generamos una nueva solucion se ha de verificar que se encuentre en el espacio de busqueda
    def generate_lambda_population(self):
        vectors = []
        for i in range(self.p):
            x1 = (i / self.p) * self.max
            x2 = self.max - x1
            vectors.append([x1, x2])
        return np.array(vectors)
    
    def euclidean_distance_matrix(self):
        matrix = []
        for i in range(self.p):
            row = []
            for j in range(self.p):
                vector_i = self.lambda_population[i]
                vector_j = self.lambda_population[j]
                row.append(np.linalg.norm(vector_i - vector_j))
            matrix.append(row)
        return np.array(matrix)

    def closest_neighbors(self):
        neighbors = []
        for i in range(self.p):
            closest_neighbors = np.argsort(self.euclidean_distance[i])[:self.ng_size]
            neighbors.append(closest_neighbors)
        return np.array(neighbors)
    
    def evaluate(self,individual):
        f1 = individual[0]
        g = 1 + (9 / (30 - 1)) * np.sum(individual[1:])
        h = 1 - math.sqrt(f1/g) - ((f1/g) * math.sin(10 * math.pi * f1 ))
        f2 = g * h
        return np.array([f1,f2])

    def evaluate_population(self):
        population_evaluated = []
        for i in range(self.p):
            population_evaluated.append(self.evaluate(self.xi[i]))
        return np.array(population_evaluated)
    
    def cross(self, individual_index):
        cross_individuals = random.choices(self.neighbors[individual_index], k=3)
        r1,r2,r3 = cross_individuals
        r1 = self.xi[r1]
        r2 = self.xi[r2]
        r3 = self.xi[r3]
        y = r1 + self.scale_factor * (r2 - r3)
        return self.handle_boundary(y)
       
    def handle_boundary(self,y):
        match self.boundary_handling:
            case "rebound":
                y = np.array([-xi if xi < 0 else (2 - xi if xi > 1 else xi) for xi in y]) 
            case "clip":
                y = np.array([0 if xi < 0 else ( 1 if xi > 1 else xi) for xi in y])
            case "wrapping":
                y = np.array([xi -1 if xi > 1 else (xi + 1 if xi < 0 else xi ) for xi in y ])
        return y

    #Assuming SIG is 20
    def mutation(self,individual):
        sigma = (self.max - 0) / 20 
        new_individual = individual + np.random.normal(0, sigma, size=30)
        return self.handle_boundary(new_individual)
    



MooAlgorithm(100,10,0.3,1.0,0.5)
