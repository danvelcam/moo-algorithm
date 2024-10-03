import numpy as np

class MooAlgorithm():
    def __init__(self,population,generations,neighbourhood_size, max):
        self.p = population
        self.g = generations
        self.ng_size = neighbourhood_size
        self.max = max
        self.lambda_population = self.generate_lambda_population()
        self.euclidean_distance  = self.euclidean_distance_matrix()
        
    #Cuando generamos una nueva solucion se ha de verificar que se encuentre en el espacio de busqueda
    def generate_lambda_population(self):
        if self.p  <= 0:
            raise ValueError(f"Population should be a positive number, but received {self.p}")
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
                      


MooAlgorithm(10,10,2,1.0)
