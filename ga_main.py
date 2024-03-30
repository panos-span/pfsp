"""
PFSP Problem Solver

@Author: Panagiotis Spanakis
"""

from typing import Tuple
from copy import deepcopy
import pandas as pd
import numpy as np
from collections import deque


class DataLoader:
    """
    Load data from the input file
    """

    def __init__(self, data_path: str):
        """
        Load data from the input file into a pandas DataFrame
        :param data_path: the path to the input file
        """
        self.data = pd.read_csv(data_path, header=None)
        # Create columns J1 - J20 and name the rows M1 to M5
        self.data.columns = ['J' + str(i) for i in range(1, self.data.shape[1] + 1)]
        self.data.index = ['M' + str(i) for i in range(1, self.data.shape[0] + 1)]

    def data_to_numpy(self, data: pd.DataFrame) -> np.ndarray:
        """
        Convert the data to a numpy array
        :param data: the pandas DataFrame
        :return: A numpy array
        """
        self.data = data.to_numpy()
        return self.data


class GeneticAlgorithmHybrid:
    """
    Genetic Algorithm for the PFSP problem in a hybrid approach with Tabu Search
    """

    def __init__(self, data: np.ndarray):
        """
        Initialize the Genetic Algorithm Implementation with the data
        :param data: The data for the problem in a numpy array
        """
        self.data = data
        self.jobs = data.shape[1]
        self.machines = data.shape[0]

    def initialize_population(self, pop_size: int) -> np.ndarray:
        """
        Initialize the population with random permutations of the job count
        :param pop_size: the population size
        :param job_count: the number of jobs
        :return: the initialized population
        """
        return [np.random.permutation(self.jobs) + 1 for _ in range(pop_size)]

    def calculate_makespan(self, solution: np.ndarray) -> int:
        """
        Calculate the makespan of a solution.
        :param solution: The solution to evaluate.
        :return: The makespan of the solution.
        """
        # Initialize a matrix to store the completion times
        times = np.zeros((self.machines, self.jobs))

        for m in range(self.machines):
            for j in range(self.jobs):
                # Adjust job index for zero-based indexing
                job_index = solution[j] - 1

                if m == 0 and j == 0:
                    times[m][j] = self.data[m][job_index]
                elif m == 0:
                    times[m][j] = times[m][j - 1] + self.data[m][job_index]
                elif j == 0:
                    times[m][j] = times[m - 1][j] + self.data[m][job_index]
                else:
                    times[m][j] = max(times[m - 1][j], times[m][j - 1]) + self.data[m][job_index]
        return times[-1][-1]

    def tournament_selection(self, population: list, fitness, tournament_size: int) -> np.ndarray:
        selected_indices = np.random.choice(range(len(population)), tournament_size)
        selected_fitness = [fitness[i] for i in selected_indices]
        winner_index = selected_indices[np.argmin(selected_fitness)]
        return population[winner_index]

    def ordered_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        size = len(parent1)
        child = [None] * size
        start, end = sorted(np.random.choice(range(size), 2, replace=False))
        child[start:end + 1] = parent1[start:end + 1]
        fill_values = [item for item in parent2 if item not in child]
        fill_pos = [i for i in range(size) if child[i] is None]
        for i, value in zip(fill_pos, fill_values):
            child[i] = value
        return child

    def swap_mutation(self, sequence: np.ndarray) -> np.ndarray:
        idx1, idx2 = np.random.choice(range(len(sequence)), 2, replace=False)
        sequence[idx1], sequence[idx2] = sequence[idx2], sequence[idx1]
        return sequence

    def tabu_search(self, initial_solution: np.ndarray, initial_makespan: int, tabu_list: deque, tenure: int,
                    num_iterations: int = 50) -> Tuple[np.ndarray, int]:
        best_solution = initial_solution.copy()
        best_makespan = initial_makespan
        current_solution = initial_solution.copy()

        for iteration in range(num_iterations):
            neighborhood = []  # List to hold all neighbors (solutions) and their makespans
            for i in range(len(current_solution)):
                for j in range(i + 1, len(current_solution)):
                    if (i, j) not in tabu_list:  # Check if the move is not tabu
                        neighbor = current_solution.copy()
                        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]  # Swap jobs
                        neighbor_makespan = self.calculate_makespan(neighbor)
                        neighborhood.append((neighbor, neighbor_makespan, (i, j)))

            # If no non-tabu moves are available, break the loop
            if not neighborhood:
                break

            # Select the best move from the neighborhood
            neighborhood.sort(key=lambda x: x[1])
            next_solution, next_makespan, move = neighborhood[0]

            # Update current and best solutions
            current_solution, current_makespan = next_solution, next_makespan
            if next_makespan < best_makespan:
                best_solution, best_makespan = next_solution, next_makespan

                # Clear tabu list if a better solution is found
                tabu_list.clear()

            # Update the tabu list
            tabu_list.append(move)
            if len(tabu_list) > tenure:
                tabu_list.popleft()

        return best_solution, best_makespan

    def genetic_algorithm(self, pop_size: int = 100, generations: int = 100, mutation_rate: float = 0.01,
                          tournament_size: int = 3, tabu_tenure: int = 10, tabu_search_frequency: int = 2,
                          tabu_iterations: int = 50) -> Tuple[
        np.ndarray, int]:
        """
        The Genetic Algorithm for the PFSP problem
        :param pop_size: the population size
        :param generations: the number of generations
        :param mutation_rate: the mutation rate
        :param tournament_size: the size of the tournament selection
        :param tabu_tenure: the tabu tenure
        :return: the best solution and its makespan
        """
        population = self.initialize_population(pop_size)
        tabu_list = deque()

        best_sol_overall = None
        best_makespan_overall = float('inf')

        for generation in range(generations):
            fitness = [self.calculate_makespan(individual) for individual in population]

            # Tabu Search integration
            if generation % tabu_search_frequency == 0:
                for i in range(len(population)):
                    if np.random.rand() < 0.2:  # Apply TS to 20% of the population
                        individual_fitness = fitness[i]
                        improved_solution, improved_fitness = self.tabu_search(population[i], individual_fitness,
                                                                               tabu_list,
                                                                               tabu_tenure, tabu_iterations)
                        population[i] = improved_solution
                        fitness[i] = improved_fitness

            new_population = []
            for _ in range(pop_size):
                parent1 = self.tournament_selection(population, fitness, tournament_size)
                parent2 = self.tournament_selection(population, fitness, tournament_size)
                child = self.ordered_crossover(parent1, parent2)
                if np.random.rand() < mutation_rate:
                    child = self.swap_mutation(child)
                new_population.append(child)
            population = new_population

            # Updating best solution and fitness
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
            best_makespan = fitness[best_idx]
            print(best_solution)
            print(self.calculate_makespan(best_solution))

            if best_makespan < best_makespan_overall:
                best_sol_overall = best_solution
                best_makespan_overall = best_makespan
                # if best_makespan_overall == 1278:
                #    print("Found optimal solution!")
                #    break

            print(f"Generation {generation + 1}: Best Makespan = {best_makespan}")

        return best_sol_overall, best_makespan_overall


if __name__ == '__main__':
    data_path = 'input.csv'
    data_loader = DataLoader(data_path)
    data = data_loader.data_to_numpy(data_loader.data)
    ga = GeneticAlgorithmHybrid(data)

    np.random.seed(42)

    population_size = 50
    num_generations = 100
    tournament_size = 5
    mutation_rate = 0.2
    tabu_tenure = 5
    tabu_search_frequency = 2  # Apply TS every 2 generations

    best_solution, best_makespan = ga.genetic_algorithm(pop_size=50, generations=100, mutation_rate=0.2,
                                                        tournament_size=5, tabu_tenure=5, tabu_iterations=50,
                                                        tabu_search_frequency=2)
    print(f"Best Solution: {best_solution}")
    print(f"Best Makespan: {best_makespan}")
