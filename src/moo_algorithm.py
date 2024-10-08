import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

class MooAlgorithm():
    def __init__(self, population, generations, neighborhood, max, scale_factor=0.5, boundary_handling = "rebound", cr=0.5):
        if population  <= 0:
            raise ValueError(f"Population should be a positive number, but received {population}")
        if not  0 < neighborhood <= 1:
            raise ValueError(f"""Neighborhood should be a percentage 
                represented between 0 (not included) and 1 (included), but received {neighborhood}""")
        self.p = population
        self.g = generations
        self.ng_size = math.floor(neighborhood * population)
        self.max = max
        self.scale_factor = scale_factor
        self.boundary_handling = boundary_handling
        self.cr = cr
        self.pr = 1 / self.p
        self.lambda_population = self.generate_lambda_population()
        self.euclidean_distance  = self.euclidean_distance_matrix()
        self.neighbors = self.closest_neighbors()
        self.xi = np.array([np.random.rand(30) for i in range(self.p)])
        self.evaluations = self.evaluate_population() 
        self.zi = np.array([np.min(self.evaluations[:,0]), np.min(self.evaluations[:,1] )])
       
    def generate_lambda_population(self):
        vectors = []
        for i in range(self.p):
            x1 = (i / (self.p - 1)) * self.max
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
        cross_individuals = np.random.choice(self.neighbors[individual_index], size=3,replace=False)
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
    
    def compare(self,x_index,y_evaluation):
        lambda_i = self.lambda_population[x_index]
        x_evaluation = self.evaluations[x_index]
        gx = np.max(lambda_i * np.abs(x_evaluation - self.zi))
        gy = np.max(lambda_i * np.abs(y_evaluation - self.zi))
        return gy <= gx
    
    def de_crossover(self,index, mutant_vector ):
        individual = self.xi[index]
        d = len(individual)
        j_rand = np.random.randint(0,d )
    
        u_trial = np.zeros(d)
    
        for j in range(d):
            if np.random.rand() <= self.cr or j == j_rand:
                u_trial[j] = mutant_vector[j]
            else:
                u_trial[j] = individual[j]
    
        return u_trial
    
    def run(self):
        for generation in range(self.g):
            for i in range(self.p):
                y = self.cross(i)
                y_mutated = self.de_crossover(i,y)
                if random.random() < self.pr:
                    y_mutated = self.mutation(y_mutated)
                y_evaluation = self.evaluate(y_mutated)
                self.zi = np.minimum(self.zi, y_evaluation)
                for neighbor_index in self.neighbors[i]:
                    if self.compare(neighbor_index, y_evaluation):
                        self.xi[neighbor_index] = y_mutated  # Actualizar vecino
                        self.evaluations[neighbor_index] = y_evaluation  # Actualizar evaluación del vecino
        return self.evaluations

    def plot(self):
        fig, ax = plt.subplots()
        x_pf, y_pf = self.read_data('src/PF.dat')
        pareto_plot = ax.scatter(x_pf, y_pf, color='green', label='Pareto front', marker='o')
        x, y = self.separate_coordinates()
        pop_plot = ax.scatter(x, y, color='blue', label='F(x)', marker='o')
        ax.set_xlabel("f1(x)")
        ax.set_ylabel("f2(x)")
        ax.legend()
        iteration_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')
        def update_frame(frame):
            self.run()
            x, y = self.separate_coordinates()
            pop_plot.set_offsets(np.c_[x, y])
            iteration_text.set_text(f"Iteración: {frame + 1}")
            return pop_plot, iteration_text
        anim = FuncAnimation(fig, update_frame, frames=self.g, interval=0.01, repeat=False)
        plt.show()
        return self.evaluations


    def read_data(self, file_name):
        x = []
        y = []
        with open(file_name, 'r') as file:
            for i in file:
                coordinates = i.split()
                x.append(float(coordinates[0]))
                y.append(float(coordinates[1]))
        return x, y
    
    def separate_coordinates(self):
        x = []
        y = []
        for i in self.evaluations:
            x.append(i[0])
            y.append(i[1])
        return x, y
    
#RECORDAR QUE PARA EL SOFTWARE DE MÉTRICAS LA 3 COLUMNA (RESTRICCIONES) SERÁ SIEMPRE 0 HA DE ESTAR INCLUIDA SI NO NO FUNCIONARÁ
#DEFINIR MEDIANTE _str__ el nombre de los ficheros para asi tener el nombre del fichero 
#DEJAR EL RUN FUERA DEL INIT 
#SEPERAR EL PLOT DEL RUN
# PARA ASI PODER LLAMAR DE MANERA CORRECTA Y AISLADA
#RECORDAR POSIBLE PROBLEMA DE LA COMA CON EL PUNTO
alg=MooAlgorithm(40,250,0.3,1.0,0.5)
alg.plot()
