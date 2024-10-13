import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from problems.zdt3 import ZDT3
from problems.cf6 import CF6
import os


class MooAlgorithm():
    def __init__(self, population, generations, neighborhood, problem,
            scale_factor=0.5, boundary_handling = "rebound", cr=0.5,  seed=None ):
        self._validate_parameters(population, neighborhood)
        self.p = population
        self.g = generations
        self.ng_size = math.floor(neighborhood * population)
        self.max = problem.max
        self.scale_factor = scale_factor
        self.seed = seed
        self._initialize_random_seed()
        self.problem = problem
        self.boundary_handling = boundary_handling
        self.cr = cr
        self.pr = 1 / self.p

        self.lambda_population = self.generate_lambda_population()
        self.euclidean_distance  = self.euclidean_distance_matrix()
        self.neighbors = self.closest_neighbors()
        self.xi = self.problem.generate_population(self.p)
        self.evaluations = self.evaluate_population()
        if isinstance(self.problem, CF6):
            self.constraints = np.array([self.problem.constraints(individual) for individual in self.xi])
        self.zi = np.array([np.min(self.evaluations[:,0]), np.min(self.evaluations[:,1] )])

        self.filename = f"allpop_{self.problem.name}_{self.problem.dimensions}d_{self.p}p_{self.g}g_seed{self.seed}.out"

    def _validate_parameters(self, population, neighborhood):
        if population <= 0:
            raise ValueError(f"Population should be a positive number, but received {population}")
        if not 0 < neighborhood <= 1:
            raise ValueError(f"""Neighborhood should be a percentage represented between 0 (not included) 
                                 and 1 (included), but received {neighborhood}""")
    
    def _initialize_random_seed(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
       
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
    
    
    def evaluate_population(self):
        return np.array([self.problem.evaluate(individual) for individual in self.xi])

    
    def cross(self, individual_index):
        cross_individuals = np.random.choice(self.neighbors[individual_index], size=3,replace=False)
        r1,r2,r3 = cross_individuals
        r1 = self.xi[r1]
        r2 = self.xi[r2]
        r3 = self.xi[r3]
        y = r1 + self.scale_factor * (r2 - r3)
        return self.problem.handle_boundary(y, self.boundary_handling)
    
    def mutation(self,individual):
        sigma = (self.max - 0) / 20 
        new_individual = individual + np.random.normal(0, sigma, size=self.problem.dimensions)
        return self.problem.handle_boundary(new_individual, self.boundary_handling)
    
  
    
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
    

    def compare(self,x_index,y_evaluation):
        lambda_i = self.lambda_population[x_index]
        x_evaluation = self.evaluations[x_index]
        gx = np.max(lambda_i * np.abs(x_evaluation - self.zi))
        gy = np.max(lambda_i * np.abs(y_evaluation - self.zi))
        return gy <= gx
    
    def compare_constraints(self, x_index,y_evaluation, y_constraints):
        constraints_x = self.constraints[x_index]
        x_violates = np.any(constraints_x < 0)
        y_violates = np.any(y_constraints < 0)

        if x_violates and y_violates:
            return np.sum(y_constraints) < np.sum(constraints_x)
        
        elif not x_violates and y_violates:
            return False
        
        elif x_violates and not y_violates:
            return True
        
        else:
            return self.compare(x_index,y_evaluation)
    
    def run(self):
        for generation in range(self.g):
            for i in range(self.p):
                y = self.cross(i)
                y_mutated = self.de_crossover(i,y)
                if random.random() < self.pr:
                    y_mutated = self.mutation(y_mutated)
                y_evaluation = self.problem.evaluate(y_mutated)
                self.zi = np.minimum(self.zi, y_evaluation)
                if isinstance(self.problem, ZDT3):
                    for neighbor_index in self.neighbors[i]:
                        if self.compare(neighbor_index, y_evaluation):
                            self.xi[neighbor_index] = y_mutated  
                            self.evaluations[neighbor_index] = y_evaluation 
                elif isinstance(self.problem, CF6):
                    y_constraints = self.problem.constraints(y_mutated)
                    for neighbor_index in self.neighbors[i]:
                        if self.compare_constraints(neighbor_index, y_evaluation, y_constraints):
                            self.xi[neighbor_index] = y_mutated  
                            self.evaluations[neighbor_index] = y_evaluation 
                            self.constraints[neighbor_index] = y_constraints
                    
            self.save_evaluations()
        return self.evaluations

    def plot(self):
        fig, ax = plt.subplots()
        x_pf, y_pf = self.read_data(self.problem.pareto_front)
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
    
    def save_evaluations(self):
        with open("./src/tests/"+self.filename, 'a') as file:
            for eval in self.evaluations:
                file.write(f"{eval[0]:.6f} {eval[1]:.6f} 0.00\n")


#problem = ZDT3()
#Cambios en el max porque segun problema ha de cambiar 
cf6 = CF6(4)
zdt3 = ZDT3()
alg = MooAlgorithm(population=40,generations=250,neighborhood=0.3,scale_factor=0.5,seed=121, problem=cf6)
alg.plot()
