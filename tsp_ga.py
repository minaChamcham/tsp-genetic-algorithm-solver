import csv
import random
import numpy as np


class City:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y

    def __str__(self):
        return f"{self.name}: ({self.x}, {self.y})"
    
    @classmethod
    def get_cities(cls, file_path):
        cities = []
        with open(file_path) as f:
            for city in csv.reader(f):
                cities.append(City(city[0], float(city[1]), float(city[2])))
        return cities
    
    @classmethod
    def calc_distance(cls, cities):
        coords = np.array([(city.x, city.y) for city in cities])
        total_distance = np.sum(np.linalg.norm(coords[1:] - coords[:-1], axis=1))
        total_distance += np.linalg.norm(coords[-1] - coords[0])
        return total_distance


class GA:
    def __init__(self, population_size=1000, iterations_limit=200, mutation_rate=0.1, crossover_rate=0.9, 
                 crossover_type='OX', mutation_type='swap', target=np.inf, tournament_selection_size=4):
        self.population_size = population_size
        self.iterations_limit = iterations_limit
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.target = target
        self.tournament_selection_size = tournament_selection_size
        self.gen_number = 0
        self.population = []
        self.fittest = 0

    def calc_fitness(self, state):
        # return ( 1 / (City.calc_distance(state) + 1 )) * pow(10, 6)
        return (- City.calc_distance(state))
    
    def calc_avr_fitness(self, population):
        return np.mean([indiv[0] for indiv in population])

    def select_population(self, state):
        self.population = []
        for _ in range(self.population_size):
            s = state.copy()
            random.shuffle(s)
            self.population.append([self.calc_fitness(s), s])
        self.fittest = max(self.population, key=lambda p: p[0])[0]

    def crossover(self, parent1, parent2):
        if self.crossover_type == 'PMX':
            return self.crossover_PMX(parent1, parent2)
        elif self.crossover_type == 'CX':
            return self.crossover_CX(parent1, parent2)
        elif self.crossover_type == 'OX':
            return self.crossover_OX(parent1, parent2)
        
    def crossover_PMX(self, parent1, parent2):
        # PMX - 2 điểm cắt

        # Chọn hai điểm cắt ngẫu nhiên đảm bảo điểm 1 nhỏ hơn điểm 2
        point1 = random.randint(0, len(parent1) - 1)
        point2 = random.randint(point1 + 1, len(parent1))
       
        # Tạo từ điển để tra cứu nhanh hơn
        parent1_map = {city: i for i, city in enumerate(parent1)}
        parent2_map = {city: i for i, city in enumerate(parent2)}

        # Tạo hai tập con
        child1 = [-1] * len(parent1)
        child2 = [-1] * len(parent2)

        # Sao chép đoạn giữa từ điểm 1 đến điểm 2 của tập tập cha mẹ
        for i in range(point1, point2):
            child1[i] = parent1[i]
            child2[i] = parent2[i]

        # Tạo hai set để tra cứu tránh việc trùng lặp
        mapped1 = set(child1[point1:point2])
        mapped2 = set(child2[point1:point2])

        # Điền các phần tử còn lại bằng thuật toán
        for i in range(point1, point2):
            if parent2[i] not in child1 and parent2[i] not in mapped1: # Kiểm tra tồn tại
                j = i
                while child1[j] != -1 and parent2[j] in parent1_map:
                    j = parent1_map[parent2[j]]
                child1[j] = parent2[i] # Thêm phần tử vào tập con
                mapped1.add(parent2[i]) # Thêm phần tử vào tập kiểm tra

            if parent1[i] not in child2 and parent1[i] not in mapped2: # Tương tự như trên
                j = i
                while child2[j] != -1 and parent1[j] in parent2_map:
                    j = parent2_map[parent1[j]]
                child2[j] = parent1[i]
                mapped2.add(parent1[i])

        # Điền phần tử nếu bị bỏ sót, sử dụng mapping
        for i in range(len(parent1)):
            if child1[i] == -1: # Kiểm tra còn phần tử nào chưa được điền
                city = next(c for c in parent2 if c not in child1) # Tìm phần tử đầu tiên trong parent thích hợp để điền vào tập con
                child1[i] = city
            if child2[i] == -1: # Tương tự
                city = next(c for c in parent1 if c not in child2)
                child2[i] = city

        return child1, child2
    
    def crossover_CX (self, parent1, parent2):
        # Cycle
        # Khởi tạo child1 và child2 là các list rỗng
        child1 = [-1] * len(parent1)
        child2 = [-1] * len(parent2)

        index = 0
        while True:
            child1[index] = parent1[index]
            index = parent2.index(parent1[index])
            if child1[index] != -1: break

        for i in range(len(parent1)):
            if child1[i] == -1:
                child1[i] = parent2[i]

        index = 0
        while True:
            child2[index] = parent2[index]
            index = parent1.index(parent2[index])
            if child2[index] != -1: break

        for i in range(len(parent2)):
            if child2[i] == -1:
                child2[i] = parent1[i]

        return child1, child2
    
    def crossover_OX(self, parent1, parent2):
        point = random.randint(0, len(parent1) - 1)

        child1 = parent1[0:point]
        for j in parent2:
            if (j in child1) == False:
                child1.append(j)

        child2 = parent2[0:point]
        for j in parent1:
            if (j in child2) == False:
                child2.append(j)

        return child1, child2

    def mutation(self, state):
        if self.mutation_type == 'swap':
            return self.swap_mutation(state)
        elif self.mutation_type == 'inversion':
            return self.inversion_mutation(state)
        elif self.mutation_type == 'scramble':
            return self.scramble_mutation(state)
        elif self.mutation_type == 'insertion':
            return self.insertion_mutation(state)
        
    def swap_mutation(self, state):
        point1, point2 = random.sample(range(len(state)), 2)
        state[point1], state[point2] = state[point2], state[point1]
        return state
    
    def inversion_mutation(self, state):
        point1, point2 = sorted(random.sample(range(len(state)), 2))
        state[point1:point2] = reversed(state[point1:point2])
        return state
    
    def scramble_mutation(self, state):
        point1, point2 = sorted(random.sample(range(len(state)), 2))
        subset = state[point1:point2]
        random.shuffle(subset)
        state[point1:point2] = subset
        return state
    
    def insertion_mutation(self, state):
        point1, point2 = random.sample(range(len(state)), 2)
        city = state.pop(point1)
        state.insert(point2, city)
        return state
    
    def tournament_selection(self):
        tournament = random.sample(self.population, self.tournament_selection_size)
        return max(tournament, key=lambda p: p[0])[1]

    def evolve(self):
        new_population = []
        # Tính số lượng cặp lai ghép cụ thể
        num_crossovers = int((self.crossover_rate * self.population_size) // 2)

        # Thực hiện lai ghép cho num_crossovers cặp cha mẹ
        for _ in range(num_crossovers):
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            child1, child2 = self.crossover(parent1, parent2)
            new_population.append(child1)
            new_population.append(child2)

        # Số còn lại được sao chép trực tiếp từ quần thể hiện tại
        for _ in range(self.population_size - (num_crossovers * 2)):
            child = self.tournament_selection()
            new_population.append(child)

        # Thực hiện đột biến
        for _ in range(round(self.mutation_rate * self.population_size)):
            random_state = random.choice(new_population)
            new_population[new_population.index(random_state)] = self.mutation(random_state)

        new_population = [[self.calc_fitness(state), state] for state in new_population]
        self.population = new_population
        self.gen_number += 1
        self.fittest = max(self.population, key=lambda p: p[0])[0]

        return sorted(self.population, key=lambda p: p[0])[-1], self.calc_avr_fitness(self.population)