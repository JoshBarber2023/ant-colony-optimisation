import numpy as np
import random
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
from itertools import permutations

# Function to calculate distance between two points
def distance(city1, city2):
    return np.linalg.norm(np.array(city1) - np.array(city2))

# ACO Algorithm
class AntColony:
    def __init__(self, cities, num_ants, num_iterations, alpha, beta, rho, Q):
        self.cities = cities
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha  # pheromone importance
        self.beta = beta  # distance priority
        self.rho = rho  # pheromone evaporation rate
        self.Q = Q  # pheromone deposit factor
        self.num_cities = len(cities)
        self.pheromone = np.ones((self.num_cities, self.num_cities))  # Initialise pheromone matrix
        self.heuristic = np.zeros((self.num_cities, self.num_cities))  # Heuristic (1/distance)

        self.frames_dir = "aco_frames"
        os.makedirs(self.frames_dir, exist_ok=True)

        # Initialise heuristic matrix
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    self.heuristic[i, j] = 1 / distance(cities[i], cities[j])

    def choose_next_city(self, current_city, visited_cities):
        probabilities = []
        for next_city in range(self.num_cities):
            if next_city not in visited_cities:
                pheromone = self.pheromone[current_city][next_city] ** self.alpha
                heuristic = self.heuristic[current_city][next_city] ** self.beta
                probabilities.append(pheromone * heuristic)
            else:
                probabilities.append(0)

        total = sum(probabilities)
        if total == 0:
            return random.choice([i for i in range(self.num_cities) if i not in visited_cities])

        probabilities = [prob / total for prob in probabilities]
        return np.random.choice(range(self.num_cities), p=probabilities)

    def update_pheromone(self, ants_paths, ants_lengths):
        self.pheromone *= (1 - self.rho)

        for path, length in zip(ants_paths, ants_lengths):
            pheromone_deposit = self.Q / length
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i + 1]] += pheromone_deposit

    def find_optimal_path_brute_force(self, cities):
        """Only usable for small N! complexity (like ≤10 cities)."""
        min_length = float('inf')
        best_path = []
        for perm in permutations(range(len(cities))):
            length = sum(distance(cities[perm[i]], cities[perm[i + 1]]) for i in range(len(perm) - 1))
            if length < min_length:
                min_length = length
                best_path = perm
        return list(best_path), min_length

    def run(self):
        best_path = None
        best_length = float('inf')
        images = []

        # Determine optimal path for early stopping
        optimal_path, optimal_length = self.find_optimal_path_brute_force(self.cities)
        print(f"Known optimal path length (by brute force): {optimal_length:.2f}")

        plt.figure(figsize=(10, 8))

        for iteration in range(self.num_iterations):
            ants_paths = []
            ants_lengths = []

            for _ in range(self.num_ants):
                visited_cities = [random.randint(0, self.num_cities - 1)]
                for _ in range(self.num_cities - 1):
                    next_city = self.choose_next_city(visited_cities[-1], visited_cities)
                    visited_cities.append(next_city)

                path_length = sum(distance(self.cities[visited_cities[i]], self.cities[visited_cities[i + 1]])
                                  for i in range(self.num_cities - 1))
                ants_paths.append(visited_cities)
                ants_lengths.append(path_length)

                if path_length < best_length:
                    best_length = path_length
                    best_path = visited_cities

            self.update_pheromone(ants_paths, ants_lengths)

            filename = os.path.join(self.frames_dir, f"frame_{iteration:03d}.png")
            self.plot(iteration, ants_paths, ants_lengths, filename)

            # Check for early stopping
            if abs(best_length - optimal_length) < 1e-6:
                print(f"Early stopping at iteration {iteration + 1}: optimal path found.")
                break

        # Final plot for best path
        final_frame_path = os.path.join(self.frames_dir, f"frame_final.png")
        self.plot_best_path(best_path, best_length, final_frame_path)

        # Create a GIF from saved frames
        all_frames = sorted([os.path.join(self.frames_dir, f) for f in os.listdir(self.frames_dir) if f.endswith('.png')])
        images = [imageio.imread(frame) for frame in all_frames]
        imageio.mimsave("aco_animation.gif", images, duration=0.2)

        for f in all_frames:
            os.remove(f)
        os.rmdir(self.frames_dir)

        print("GIF saved as 'aco_animation.gif'")
        return best_path, best_length


    def plot(self, iteration, ants_paths, ants_lengths, save_path):
        plt.clf()

        cities_x = [city[0] for city in self.cities]
        cities_y = [city[1] for city in self.cities]
        plt.scatter(cities_x, cities_y, color='red', s=100, zorder=5)

        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                if self.pheromone[i][j] > 1:
                    x_values = [self.cities[i][0], self.cities[j][0]]
                    y_values = [self.cities[i][1], self.cities[j][1]]
                    alpha_value = min(self.pheromone[i][j] / 10, 1)
                    plt.plot(x_values, y_values, 'g-', alpha=alpha_value, linewidth=2)

        for path in ants_paths:
            for i in range(len(path) - 1):
                start_city = path[i]
                end_city = path[i + 1]
                x_values = [self.cities[start_city][0], self.cities[end_city][0]]
                y_values = [self.cities[start_city][1], self.cities[end_city][1]]
                plt.plot(x_values, y_values, 'b-', alpha=0.3, linewidth=1)

        plt.title(f"Iteration: {iteration + 1}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True)
        plt.savefig(save_path)

    def plot_best_path(self, best_path, best_length, save_path):
        plt.clf()

        cities_x = [city[0] for city in self.cities]
        cities_y = [city[1] for city in self.cities]
        plt.scatter(cities_x, cities_y, color='red', s=100, zorder=5)

        for i in range(len(best_path) - 1):
            start_city = best_path[i]
            end_city = best_path[i + 1]
            x_values = [self.cities[start_city][0], self.cities[end_city][0]]
            y_values = [self.cities[start_city][1], self.cities[end_city][1]]
            plt.plot(x_values, y_values, 'b-', alpha=1.0, linewidth=3)

        plt.title(f"Best Path (Length: {best_length})")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True)
        plt.savefig(save_path)


# Example cities (x, y) positions
cities = [(random.randint(0, 10), random.randint(0, 10)) for _ in range(10)]

# Initialise Ant Colony Optimisation
aco = AntColony(cities=cities, num_ants=10, num_iterations=100, alpha=1, beta=2, rho=0.1, Q=100)

# Run ACO to find the best path
best_path, best_length = aco.run()
print(f"Best path: {best_path}")
print(f"Best path length: {best_length}")
