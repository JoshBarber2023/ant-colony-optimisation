import numpy as np
import random
import matplotlib.pyplot as plt

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

        # Normalise the probabilities to sum to 1
        total = sum(probabilities)
        if total == 0:
            return random.choice([i for i in range(self.num_cities) if i not in visited_cities])
        
        probabilities = [prob / total for prob in probabilities]
        return np.random.choice(range(self.num_cities), p=probabilities)

    def update_pheromone(self, ants_paths, ants_lengths):
        # Evaporate pheromone
        self.pheromone *= (1 - self.rho)

        # Deposit pheromone based on ants' paths
        for path, length in zip(ants_paths, ants_lengths):
            pheromone_deposit = self.Q / length
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i + 1]] += pheromone_deposit

    def run(self):
        best_path = None
        best_length = float('inf')
        
        plt.figure(figsize=(10, 8))  # Create the figure outside the loop to maintain the same plot

        # Run for the specified number of iterations
        for iteration in range(self.num_iterations):
            ants_paths = []
            ants_lengths = []

            # Move each ant
            for _ in range(self.num_ants):
                visited_cities = [random.randint(0, self.num_cities - 1)]  # Start from random city
                for _ in range(self.num_cities - 1):
                    next_city = self.choose_next_city(visited_cities[-1], visited_cities)
                    visited_cities.append(next_city)

                # Compute the total length of the path
                path_length = sum(distance(self.cities[visited_cities[i]], self.cities[visited_cities[i + 1]]) 
                                  for i in range(self.num_cities - 1))
                ants_paths.append(visited_cities)
                ants_lengths.append(path_length)

                # Track the best path found
                if path_length < best_length:
                    best_length = path_length
                    best_path = visited_cities

            # Update pheromone based on all ants' paths
            self.update_pheromone(ants_paths, ants_lengths)

            # Plot the current progress: show paths, pheromone trails, and cities
            self.plot(iteration, ants_paths, ants_lengths)

        # After all iterations, plot the final best path in bold blue
        self.plot_best_path(best_path, best_length)

        return best_path, best_length

    def plot(self, iteration, ants_paths, ants_lengths):
        # Clear the plot to update it each iteration
        plt.clf()

        # Plot cities
        cities_x = [city[0] for city in self.cities]
        cities_y = [city[1] for city in self.cities]
        plt.scatter(cities_x, cities_y, color='red', s=100, zorder=5)

        # Plot pheromone trails
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                if self.pheromone[i][j] > 1:
                    x_values = [self.cities[i][0], self.cities[j][0]]
                    y_values = [self.cities[i][1], self.cities[j][1]]
                    
                    # Clamp alpha to the range [0, 1]
                    alpha_value = min(self.pheromone[i][j] / 10, 1)  # Ensure alpha is between 0 and 1
                    plt.plot(x_values, y_values, 'g-', alpha=alpha_value, linewidth=2)

        # Plot all ants' paths in light blue for progress tracking
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
        plt.show(block=False)
        plt.pause(0.1)

    def plot_best_path(self, best_path, best_length):
        # Clear the current axes and figure
        plt.clf()  # Clear the entire figure
        plt.cla()  # Clear the current axes

        # Plot cities
        cities_x = [city[0] for city in self.cities]
        cities_y = [city[1] for city in self.cities]
        plt.scatter(cities_x, cities_y, color='red', s=100, zorder=5)

        # Plot the final best path in bold blue
        for i in range(len(best_path) - 1):
            start_city = best_path[i]
            end_city = best_path[i + 1]
            x_values = [self.cities[start_city][0], self.cities[end_city][0]]
            y_values = [self.cities[start_city][1], self.cities[end_city][1]]
            plt.plot(x_values, y_values, 'b-', alpha=1.0, linewidth=3)  # Bold blue line for best path

        # Show the final plot with the best path length
        plt.title(f"Best Path (Length: {best_length})")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True)

        # Ensure the plot updates immediately
        plt.draw()
        plt.show()


# Example cities (x, y) positions
cities = [(random.randint(0, 10), random.randint(0, 10)) for _ in range(10)]

# Initialise Ant Colony Optimisation
aco = AntColony(cities=cities, num_ants=6, num_iterations=100, alpha=1, beta=2, rho=0.1, Q=100)

# Run ACO to find the best path
best_path, best_length = aco.run()
print(f"Best path: {best_path}")
print(f"Best path length: {best_length}")
