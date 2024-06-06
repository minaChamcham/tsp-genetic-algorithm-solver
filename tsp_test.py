import random
import time
import csv
from tsp_ga import GA, City


population_sizes = [100, 500, 1000]
crossover_rates = [0.7, 0.8, 0.9]
mutation_rates = [0.05, 0.1, 0.2]
crossover_types = ['OX', 'PMX', 'CX']
mutation_types = ['swap', 'inversion', 'scramble', 'insertion']

n_iterations = 150
cities = City.get_cities("TSP_35.csv")

with open('tsp_ga_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['Population Size', 'Mutation Rate', 'Crossover Rate', 'Crossover Type', 'Mutation Type', 'Fitness', 'Average Fitness', 'Time']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for _ in range(n_iterations):
        pop_size = random.choice(population_sizes)
        mu_rate = random.choice(mutation_rates)
        cross_rate = random.choice(crossover_rates)
        cross_type = random.choice(crossover_types)
        mu_type = random.choice(mutation_types)

        ga = GA(population_size=pop_size, mutation_rate=mu_rate, crossover_rate=cross_rate,
                crossover_type=cross_type, mutation_type=mu_type)

        ga.select_population(cities)

        start_time = time.time()
        for _ in range(ga.iterations_limit):
            ga.evolve()
        end_time = time.time()
        current_fitness = ga.fittest

        writer.writerow({
            'Population Size': pop_size,
            'Mutation Rate': mu_rate,
            'Crossover Rate': cross_rate,
            'Crossover Type': cross_type,
            'Mutation Type': mu_type,
            'Fitness': current_fitness,
            'Time': end_time - start_time,
            'Average Fitness': ga.calc_avr_fitness(ga.population)
        })
