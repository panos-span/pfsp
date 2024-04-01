"""
PFSP Problem Solver with a Hybrid Genetic Algorithm and Tabu Search

@Author: Panagiotis Spanakis
"""
# Import the necessary libraries
from typing import Tuple
import pandas as pd
import numpy as np
from collections import deque
from time import time
import json

# Define the optimal makespan for the problem
OPTIMAL_MAKESPAN = 1278
# Define the number of generations to run the Genetic Algorithm
NUM_GENERATIONS = 100


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

    def data_to_numpy(self) -> np.ndarray:
        """
        Convert the data to a numpy array
        :param data: the pandas DataFrame
        :return: A numpy array
        """
        self.data = self.data.to_numpy()
        return self.data


class GeneticAlgorithmHybrid:
    """
    Hybrid Genetic Algorithm for the PFSP problem with Tabu Search
    """

    def __init__(self, data: np.ndarray):
        """
        Initialize the Genetic Algorithm Implementation with the data
        :param data: the data in a numpy array
        """

        self.data = data
        self.jobs = data.shape[1]
        self.machines = data.shape[0]

    def initialize_population(self, pop_size: int) -> list:
        """
        Initialize the population with random permutations of the job count
        :param pop_size: The population size
        :return: A numpy array with the initialized population
        """
        return [np.random.permutation(self.jobs) + 1 for _ in range(pop_size)]

    def calculate_makespan(self, solution: np.ndarray) -> int:
        """
        Calculate the makespan of a solution
        :param solution: The solution to evaluate
        :return: The calculated makespan
        """
        # Initialize a matrix to store the completion times of each job on each machine
        times = np.zeros((self.machines, self.jobs))

        for m in range(self.machines):
            for j in range(self.jobs):
                # If we are in the first machine and in the first job
                # We just input the processing time of the job
                if m == 0 and j == 0:
                    times[m][j] = self.data[m][solution[j] - 1]
                # If we are in the first machine but not in the first job
                # We add the processing time of the job to the previous completion time
                elif m == 0:
                    times[m][j] = times[m][j - 1] + self.data[m][solution[j] - 1]
                # If we are in the first job but not in the first machine
                # We add the processing time of the job to the previous completion time
                elif j == 0:
                    times[m][j] = times[m - 1][j] + self.data[m][solution[j] - 1]
                # If we are not in the first job or the first machine
                # We add the processing time of the job to the maximum of the previous job on the machine
                # or the previous machine on the job
                else:
                    times[m][j] = max(times[m - 1][j], times[m][j - 1]) + self.data[m][solution[j] - 1]

        # The makespan is the completion time of the last job on the last machine
        # So the last element of the times matrix will be returned
        return times[-1][-1]

    def tournament_selection(self, population: list, fitness: list, tournament_size: int) -> np.ndarray:
        """
        Perform tournament selection on the population
        :param population: The population to select from
        :param fitness: The fitness values of the population
        :param tournament_size: The size of the tournament
        :return: The selected individual
        """
        # Choose random indices for the tournament
        selected_indices = np.random.choice(range(len(population)), tournament_size)
        # Get the makespan values of the selected individuals
        selected_fitness = [fitness[i] for i in selected_indices]
        # Select the winner of the tournament
        winner_index = selected_indices[np.argmin(selected_fitness)]
        # Return the winner of the tournament
        return population[winner_index]

    def ordered_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Perform ordered crossover on two parents
        :param parent1: The first parent
        :param parent2: The second parent
        :return: The children produced by the crossover
        """
        # Get the parent size
        size = len(parent1)
        # Initialize the child with the same size as the parents
        child = np.full(size, None, dtype=object)
        # Choose two random indices for the crossover
        start, end = sorted(np.random.choice(range(size), size=2, replace=False))
        # Copy the selected part of the first parent to the child
        child[start:end + 1] = parent1[start:end + 1]
        # Get the values that are not in the child from the second parent
        fill_values = [item for item in parent2 if item not in child]
        # Get the indices that need to be filled
        fill_pos = [i for i in range(size) if child[i] is None]
        # Fill the child with the values from the second parent
        for i, value in zip(fill_pos, fill_values):
            child[i] = value
        return child

    def swap_mutation(self, solution: np.ndarray) -> np.ndarray:
        """
        Perform swap mutation on a sequence
        :param solution: The provided solution
        :return: The mutated solution
        """
        # Choose two random indices to swap
        idx1, idx2 = np.random.choice(range(len(solution)), size=2, replace=False)
        # Swap the values at the indices
        solution[idx1], solution[idx2] = solution[idx2], solution[idx1]
        # Return the mutated solution
        return solution

    def tabu_search(self, initial_solution: np.ndarray, initial_makespan: int, tabu_list: deque, tenure: int,
                    num_iterations: int = 50) -> Tuple[np.ndarray, int]:
        """
        Perform tabu search on the initial solution
        :param initial_solution: The initial solution provided
        :param initial_makespan: The makespan of the initial solution
        :param tabu_list: The tabu list to use
        :param tenure: The tenure of the tabu list
        :param num_iterations: The number of iterations to perform
        :return: The Best solution and its makespan
        """
        # Initialize the best solution and its makespan
        best_solution = initial_solution.copy()
        best_makespan = initial_makespan
        current_solution = initial_solution.copy()

        # Perform the tabu search for the specified number of iterations
        for iteration in range(num_iterations):
            neighborhood = []  # List to hold all neighbors (solutions) and their makespans
            # Generate neighbors by swapping two jobs
            for i in range(self.jobs):
                for j in range(i + 1, self.jobs):
                    # Check if the move is not tabu else skip
                    if (i, j) not in tabu_list:
                        # Copy the current solution
                        neighbor = current_solution.copy()
                        # Swap the two jobs
                        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                        # Calculate the makespan of the neighbor
                        neighbor_makespan = self.calculate_makespan(neighbor)
                        # Save the move that was made
                        move = (i, j)
                        # Append the neighbor and its makespan to the neighborhood
                        neighborhood.append((neighbor, neighbor_makespan, move))

            # If no moves are available, break the loop
            if not neighborhood:
                break

            # Select the best move from the neighborhood
            neighborhood.sort(key=lambda x: x[1])
            # Get the best solution, its makespan and the move that was made
            current_solution, current_makespan, current_move = neighborhood[0]

            # Update the best solution and its makespan
            if current_makespan < best_makespan:
                best_solution, best_makespan = current_solution, current_makespan
                # Clear tabu list if a better solution is found
                tabu_list.clear()

                # Check if the optimal solution has been found
                if best_makespan == OPTIMAL_MAKESPAN:
                    print(f"Found optimal solution at iteration {iteration + 1}!")
                    return best_solution, best_makespan

            # Update the tabu list
            tabu_list.append(current_move)

            # If the tabu list is full, remove the oldest element
            if len(tabu_list) > tenure:
                tabu_list.popleft()

        return best_solution, best_makespan

    def genetic_algorithm(self, pop_size: int = 100, generations: int = 100, mutation_rate: float = 0.01,
                          tournament_size: int = 3, tabu_tenure: int = 10, tabu_search_frequency: int = 2,
                          tabu_iterations: int = 50, tabu_search_percentage: float = 0.1) \
            -> Tuple[np.ndarray, int]:
        """
        The Genetic Algorithm for the PFSP problem
        :param pop_size: the population size
        :param generations: the number of generations
        :param mutation_rate: the mutation rate
        :param tournament_size: the size of the tournament selection
        :param tabu_tenure: the tabu tenure
        :param tabu_search_frequency: the frequency to apply Tabu Search
        :param tabu_iterations: the number of iterations to perform Tabu Search
        :param tabu_search_percentage: the percentage of the population to apply Tabu Search
        :return: the best solution and its makespan
        """

        # Initialize the population
        population = self.initialize_population(pop_size)
        # Initialize the tabu list as a deque
        tabu_list = deque(maxlen=tabu_tenure)

        # Initialize the best solution and its makespan
        best_sol_overall = None
        best_makespan_overall = float('inf')

        # Begin the genetic algorithm loop
        for generation in range(generations):
            # Calculate the fitness (makespan) of each individual of the population
            fitness = [self.calculate_makespan(individual) for individual in population]

            # Tabu Search integration
            if generation % tabu_search_frequency == 0:
                for i in range(len(population)):
                    # Apply TS to (tabu_search_percentage) of the population
                    if np.random.rand() < tabu_search_percentage:
                        # Apply tabu search to the individual
                        improved_solution, improved_fitness = self.tabu_search(population[i], fitness[i],
                                                                               tabu_list,
                                                                               tabu_tenure, tabu_iterations)

                        # Check if we have found the optimal solution
                        if improved_fitness == OPTIMAL_MAKESPAN:
                            return improved_solution, improved_fitness

                        # Update the population and the fitness
                        population[i] = improved_solution.copy()
                        fitness[i] = improved_fitness

            # Initialize a new population
            new_population = []
            # Initialize the new fitness
            new_fitness = []

            # Generate the new population
            for _ in range(pop_size):
                # Perform tournament selection
                parent1 = self.tournament_selection(population, fitness, tournament_size)
                parent2 = self.tournament_selection(population, fitness, tournament_size)
                # Perform ordered crossover
                child = self.ordered_crossover(parent1, parent2)
                # Perform mutation if the mutation rate is met
                if np.random.rand() < mutation_rate:
                    child = self.swap_mutation(child)
                # Append the child to the new population
                new_population.append(child)
                # Calculate the fitness of the child
                new_fitness.append(self.calculate_makespan(child))

            # Update the population
            population = new_population.copy()
            # Update the fitness of the population
            fitness = new_fitness.copy()

            # Find the best solution and its makespan
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
            best_makespan = fitness[best_idx]
            print(best_solution)

            # Check if the best solution is better than the overall best solution
            if best_makespan < best_makespan_overall:
                best_sol_overall = best_solution.copy()
                best_makespan_overall = best_makespan
                # If the best solution is the optimal solution, break the loop
                if best_makespan_overall == OPTIMAL_MAKESPAN:
                    print("Found optimal solution!")
                    break

            print(f"Generation {generation + 1}: Best Makespan = {best_makespan}")

        return best_sol_overall, best_makespan_overall


if __name__ == '__main__':
    # Initialize the DataLoader
    data_path = 'input.csv'
    data_loader = DataLoader(data_path)
    # Convert the data to a numpy array
    data = data_loader.data_to_numpy()
    # Initialize the Genetic Algorithm Class with the numpy array
    ga = GeneticAlgorithmHybrid(data)

    # Set the parameters for the Genetic Algorithm from the optimal_parameters.json file
    best_params = json.load(open('optimal_params.json'))
    # Set the number of generations to 10, because the optimal parameters were found with 10 generations
    best_params['generations'] = NUM_GENERATIONS

    # Set a random seed for reproducibility
    np.random.seed(2)

    # Time the execution of the Genetic Algorithm
    start_time = time()

    # Run the Genetic Algorithm
    best_solution, best_makespan = ga.genetic_algorithm(**best_params)

    # Save the execution time
    execution_time = time() - start_time

    # Print the best solution and its makespan
    print(f"Best Solution: {best_solution}")
    print(f"Best Makespan: {best_makespan}")
    # Print the execution time
    print(f"Execution Time: {round(execution_time, 2)} seconds")

    # Write the best solution to a file and save it along with the execution time
    with open('solution.csv', 'w') as f:
        f.write("Optimal Solution:\n")
        f.write(', '.join(map(str, best_solution)))
        f.write('\n')
        f.write("Makespan:\n")
        f.write(str(best_makespan))
        f.write('\n')
        f.write("Execution Time:\n")
        f.write(str(round(execution_time, 2)) + ' seconds')
