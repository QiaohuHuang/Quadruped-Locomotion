import os  
import numpy as np
import random
import json
from quadruped_robot_env import QuadrupedRobotEnv
from copy import deepcopy

def softmax(fitness_scores, beta=0.1):
    """
    Compute softmax probabilities for fitness scores.
    :param fitness_scores: List of fitness scores.
    :param beta: Temperature parameter controlling selection pressure.
    :return: List of probabilities.
    """
    exps = np.exp(beta * fitness_scores)
    return exps / np.sum(exps)

def rank_based_selection(fitness_scores):
    """
    Generate probabilities based on rank.
    Higher-ranked individuals get higher probabilities.
    :param fitness_scores: List or numpy array of fitness scores.
    :return: List of probabilities.
    """
    sorted_indices = np.argsort(fitness_scores)
    ranks = np.argsort(sorted_indices)  # Assign ranks (0 = lowest fitness)
    probabilities = (ranks + 1) / np.sum(ranks + 1)  # Normalize
    return probabilities

class CPG:
    def __init__(self, frequency, amplitude, phase):
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase

    def generate_signal(self, t):
        return self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase)

class QuadrupedSimulator:
    def __init__(self, parameters):
        self.parameters = parameters
        self.cpgs = {joint: CPG(**params) for joint, params in parameters.items()}

    def simulate(self):
        env = QuadrupedRobotEnv(use_GUI=False, verbose=False)
        env.reset()
        fps = 240
        # Simulate over 10s
        max_time = 10
        total_reward = 0
        for i in range(fps * max_time):
            t = i / fps
            # Use the signals from 4 sine waves as action
            action = np.array([cpg.generate_signal(t) for joint, cpg in self.cpgs.items()])
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            if done:
                env.close()
                return total_reward
        env.close()
        return total_reward

def load_previous_generation(file_path):
    """
    Load the previous generation from a JSON file if it exists.
    :param file_path: Path to the JSON file.
    :return: Loaded parameters or None if the file doesn't exist.
    """
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return None

class EvolutionaryAlgorithm:
    def __init__(self, population_size, mutation_rate, last_generation_file="last_generation.json"):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.last_generation_file = last_generation_file

    def initialize_population(self):
        population = []
        shared_frequency = np.random.uniform(0.5, 2.0)  # Shared frequency for all joints
        for _ in range(self.population_size):
            parameters = {
                f"joint_{i}": {
                    "frequency": shared_frequency,
                    "amplitude": np.random.uniform(0.1, 1.0),
                    "phase": random.choice([0, np.pi]),
                }
                for i in range(4)  # 4 joints
            }
            population.append(parameters)
        return population

    def mutate(self, parameters):
        mutated = {}
        shared_frequency = max(
            1e-4, parameters["joint_0"]["frequency"] + np.random.uniform(-0.1, 0.1) * self.mutation_rate
        )  # Mutate shared frequency
        for joint, params in parameters.items():
            mutated[joint] = {
                "frequency": shared_frequency,
                "amplitude": min(max(1e-4, params["amplitude"] + np.random.uniform(-0.05, 0.05) * self.mutation_rate), 1),
                "phase": random.choice([0, np.pi]),
            }
        return mutated

    def crossover(self, parent1, parent2):
        child = {}
        for joint in parent1.keys():
            if np.random.rand() > 0.5:
                child[joint] = parent1[joint]
            else:
                child[joint] = parent2[joint]
        # Ensure shared frequency
        shared_frequency = child["joint_0"]["frequency"]
        for joint in child.keys():
            child[joint]["frequency"] = shared_frequency
        return child

    def evolve(self, generations, simulator, beta=1.0):
        # Load the previous best candidate if available
        previous_best_candidate = load_previous_generation(self.last_generation_file)

        # Initialize population and include the previous candidate if loaded
        population = self.initialize_population()
        if previous_best_candidate:
            print("Loaded previous best candidate from file.")
            population[0] = previous_best_candidate  # Replace the first candidate

        best_candidate = None
        best_fitness = float('-inf')  # Track the best fitness score

        for generation in range(generations):
            fitness_scores = []

            # Evaluate fitness of each candidate
            for candidate in population:
                fitness = simulator(candidate).simulate()
                fitness_scores.append(fitness)

            # Identify the best candidate in the current population
            current_best_idx = np.argmax(fitness_scores)
            current_best_candidate = deepcopy(population[current_best_idx])
            current_best_fitness = fitness_scores[current_best_idx]

            # Update the overall best candidate and fitness
            if current_best_fitness > best_fitness:
                best_candidate = deepcopy(current_best_candidate)
                best_fitness = current_best_fitness

            # Log progress
            print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")

            # Use rank-based selection probabilities
            fitness_scores = np.array(fitness_scores)
            selection_probs = rank_based_selection(fitness_scores)

            # Generate the next generation
            next_gen = [deepcopy(best_candidate)]  # Always preserve the best candidate
            while len(next_gen) < self.population_size:
                parent1_idx, parent2_idx = np.random.choice(
                    len(population), size=2, replace=False, p=selection_probs
                )
                parent1, parent2 = population[parent1_idx], population[parent2_idx]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_gen.append(child)

            population = next_gen

        # Save the final best candidate
        with open(self.last_generation_file, "w") as f:
            json.dump(best_candidate, f)

        return best_candidate

def run_ea():
    ea = EvolutionaryAlgorithm(population_size=50, mutation_rate=0.2)
    simulator = QuadrupedSimulator

    best_parameters = ea.evolve(generations=20, simulator=simulator)
    print("Best Parameters:", best_parameters)

run_ea()
