import numpy as np
import math
class MooAlgorithm():
    def __init__(self, population, generations, neighborhood, max):
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



MooAlgorithm(10,10,0.3,1.0)
