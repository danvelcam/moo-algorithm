import numpy as np

class MooAlgorithm():
    def __init__(self,population,generations,neighbourhood_size, max):
        self.p = population
        self.g = generations
        self.ng_size = neighbourhood_size
        self.max = max
        self.lambda_population = self.generate_lambda_population()
        

    def generate_lambda_population(self):
        vectors = []
        for i in range(self.p + 1):
            x1 = (i / self.p) * self.max
            x2 = self.max - x1
            vectors.append([x1, x2])
        return np.array(vectors)
    


MooAlgorithm(10,10,2,1.0)
