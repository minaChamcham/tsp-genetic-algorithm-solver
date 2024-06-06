# TSP Solver with Genetic Algorithm

This project provides a solution for the Traveling Salesman Problem (TSP) using Genetic Algorithm (GA). The project consists of three main source code files:

1. `tsp_ga.py`: Contains classes for implementing GA for TSP.
2. `tsp_ui.py`: Provides a user interface for interacting with the algorithm.
3. `tsp_test.py`: Conducts tests to evaluate the performance of the algorithm.

## Requirements
- numpy
- matplotlib
- PyQt5

## Usage

### Running the User Interface

To run the user interface, execute the `tsp_ui.py` file. This interface allows you to configure GA parameters and view the evolution of the TSP solution.

### Source Code Structure

#### `tsp_ga.py`

- `City`: Class representing a city in TSP.
  - `__init__(self, name, x, y)`: Initializes a city object with a name, x, and y coordinates.
  - `get_cities(cls, file_path)`: Class method to read a list of cities from a CSV file.
  - `calc_distance(cls, cities)`: Class method to calculate the total distance of a tour.

- `GA`: Class implementing the genetic algorithm for TSP.
  - `__init__(self, population_size, iterations_limit, mutation_rate, crossover_rate, crossover_type, mutation_type, target, tournament_selection_size)`: Initializes GA parameters.
  - `calc_fitness(self, state)`: Calculates the fitness of an individual.
  - `select_population(self, state)`: Selects the initial population.
  - `crossover(self, parent1, parent2)`: Performs crossover between two parent individuals.
  - `mutation(self, state)`: Performs mutation on an individual.
  - `evolve(self)`: Performs a generation of evolution.

#### `tsp_ui.py`

- `TSPApp`: Main class for the GUI application using PyQt5.
  - `initUI(self)`: Initializes the user interface.
  - `start_algorithm(self)`: Starts the GA.
  - `update_plot(self)`: Updates the plot and interface with the progress of the GA.
  - `stop_update(self)`: Pauses updates.
  - `continue_update(self)`: Resumes updates.
  - `plot_solution(self, solution)`: Plots the current solution route.
  - `plot_initial_cities(self)`: Plots the initial city positions.
  - `plot_fitness(self)`: Plots the fitness over generations.
  - `plot_avr(self)`: Plots the average fitness over generations.

#### `tsp_test.py`

- This file contains tests to evaluate the performance of the GA with different parameters. The results are written to the `tsp_ga_results.csv` file.

### City Data

The `TSP_10.csv` and the `TSP_35.csv` file contains data about the cities, with each line including the city name and its x and y coordinates.

Example:

```csv
CityName1, x1, y1
CityName2, x2, y2
...
```
