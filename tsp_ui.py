import sys
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QPushButton, QLabel, QGridLayout, QComboBox, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import QTimer
from tsp_ga import City, GA


class TSPApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("TSP Solver with Genetic Algorithm")
        self.setGeometry(100, 100, 1200, 600)

        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)

        self.fitness_values = []
        self.avr_values = []
        
        layout = QGridLayout(centralWidget)

        # City plot
        self.city_widget = QWidget(self)
        self.city_layout = QVBoxLayout(self.city_widget)
        self.city_figure = plt.figure()
        self.city_canvas = FigureCanvas(self.city_figure)
        self.city_layout.addWidget(self.city_canvas)
        layout.addWidget(self.city_widget, 0, 0, 9, 6)  # (row, column, rowspan, colspan)

        # Fittest plot
        self.fitness_widget = QWidget(self)
        self.fitness_layout = QVBoxLayout(self.fitness_widget)
        self.fitness_figure, self.fitness_ax = plt.subplots()
        self.fitness_canvas = FigureCanvas(self.fitness_figure)
        self.fitness_layout.addWidget(self.fitness_canvas)
        layout.addWidget(self.fitness_widget, 0, 7, 4, 4)

        # Average Fitness plot
        self.avr_widget = QWidget(self)
        self.avr_layout = QVBoxLayout(self.avr_widget)
        self.avr_figure, self.avr_ax = plt.subplots()
        self.avr_canvas = FigureCanvas(self.avr_figure)
        self.avr_layout.addWidget(self.avr_canvas)
        layout.addWidget(self.avr_widget, 4, 7, 4, 4)

        # Input Form
        formLayout = QFormLayout()
        layout.addLayout(formLayout, 0, 12, 9, 2)
        self.populationInput = QSpinBox()
        self.populationInput.setRange(50, 100000) 
        self.populationInput.setValue(1000)
        self.iterationsInput = QSpinBox()
        self.iterationsInput.setRange(1, 100000) 
        self.iterationsInput.setValue(200)
        self.mutationInput = QDoubleSpinBox()
        self.mutationInput.setRange(0.0, 1.0) 
        self.mutationInput.setSingleStep(0.01)
        self.mutationInput.setValue(0.1)
        self.crossoverInput = QDoubleSpinBox()
        self.crossoverInput.setRange(0.0, 1.0) 
        self.crossoverInput.setSingleStep(0.01)
        self.crossoverInput.setValue(0.9)
        self.targetDistance = QDoubleSpinBox()
        self.targetDistance.setRange(0.0, 100000.0)  
        self.targetDistance.setSingleStep(1.0)
        self.targetDistance.setValue(0.0)
        self.crossoverType = QComboBox()
        self.crossoverType.addItems(
            ['OX', 'PMX', 'CX'])
        self.mutationType = QComboBox()
        self.mutationType.addItems(
            ['swap', 'inversion', 'scramble', 'insertion'])
        formLayout.addRow("Population Size:", self.populationInput)
        formLayout.addRow("Iterations Limit:", self.iterationsInput)
        formLayout.addRow("Crossover Rate:", self.crossoverInput)
        formLayout.addRow("Mutation Rate:", self.mutationInput)
        formLayout.addRow("Crossover Type:", self.crossoverType)
        formLayout.addRow("Mutation Type:", self.mutationType)
        formLayout.addRow("Target Distance:", self.targetDistance)

        self.startButton = QPushButton("Start")
        self.startButton.clicked.connect(self.start_algorithm)
        formLayout.addRow(self.startButton)
        self.stopButton = QPushButton("Stop")
        self.continueButton = QPushButton("Continue")
        self.stopButton.clicked.connect(self.stop_update)
        self.continueButton.clicked.connect(self.continue_update)
        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.stopButton)
        buttonLayout.addWidget(self.continueButton)
        formLayout.addRow(buttonLayout)

        self.genLabel = QLabel("Current Generation: 0")
        formLayout.addRow(self.genLabel)
        self.fittestLabel = QLabel("Fittest Value of CurGen: 0")
        formLayout.addRow(self.fittestLabel)
        self.distanceLabel = QLabel("Shortest Distance of CurGen: 0")
        formLayout.addRow(self.distanceLabel)
        self.timeLabel = QLabel("Elapsed Time: 0 seconds")
        formLayout.addRow(self.timeLabel)

        # Load cities
        self.cities = City.get_cities("TSP_35.csv")
        self.plot_initial_cities()
        self.plot_fitness()
        self.plot_avr()

        # Timer for animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.update_time = 20
        
        # Time for elapse
        self.start_time = None
        self.elapsed_time = 0

    def plot_initial_cities(self):
        self.city_figure.clear()
        ax = self.city_figure.add_subplot(111)
        city_xs = [city.x for city in self.cities]
        city_ys = [city.y for city in self.cities]
        ax.scatter(city_xs, city_ys, c='red', marker='o', label='Cities')
        for i, city in enumerate(self.cities):
            ax.annotate(city.name, (city.x + 0.05, city.y + 0.05))
        ax.set_aspect('equal', adjustable='box')
        self.city_canvas.draw()

    def plot_fitness(self):
        self.fitness_ax.clear()
        self.fitness_ax.plot(range(len(self.fitness_values)), self.fitness_values, color="red")
        self.fitness_ax.set_title("Fittest Value Over Generations")
        self.fitness_ax.set_xlabel("Generations")
        self.fitness_ax.set_ylabel("Fittest Value")
        self.fitness_ax.grid(True)
        # Setting fixed limits
        try: 
            self.fitness_ax.set_xlim(0, self.ga.iterations_limit)
        except AttributeError:
            self.fitness_ax.set_xlim(0, 100)
        # self.fitness_ax.set_ylim(0, max(self.fitness_values) + 1000 if self.fitness_values else 1)
        self.fitness_ax.set_ylim(min(self.fitness_values) if self.fitness_values else -1, 0)
        self.fitness_canvas.draw()

    def plot_avr(self):
        self.avr_ax.clear()
        self.avr_ax.plot(range(len(self.avr_values)), self.avr_values)
        self.avr_ax.set_title("Average Fitness Over Generations")
        self.avr_ax.set_xlabel("Generations")
        self.avr_ax.set_ylabel("Average Fitness")
        self.avr_ax.grid(True)
        # Setting fixed limits
        try: 
            self.avr_ax.set_xlim(0, self.ga.iterations_limit)
        except AttributeError:
            self.avr_ax.set_xlim(0, 100)
        # self.avr_ax.set_ylim(0, max(self.avr_values) + 1000 if self.avr_values else 1)
        self.avr_ax.set_ylim(min(self.avr_values) if self.avr_values else -1, 0)
        self.avr_canvas.draw()

    def start_algorithm(self):
        pop_size = self.populationInput.value()
        iter_limit = self.iterationsInput.value()
        mu_rate = self.mutationInput.value()
        cross_rate = self.crossoverInput.value()
        cross_type = self.crossoverType.currentText()
        mu_type = self.mutationType.currentText()
        target_distance = self.targetDistance.value()
        target_value = (- target_distance)
            
        self.ga = GA(population_size=pop_size, iterations_limit=iter_limit, mutation_rate=mu_rate, 
                     crossover_rate=cross_rate, crossover_type=cross_type, mutation_type=mu_type,
                     target=target_value)
        self.ga.select_population(self.cities)
        self.gen_number = 0
        self.fitness_values = []
        self.avr_values = []

        self.elapsed_time = 0  # Reset elapsed time
        self.start_time = time.time()  # Start the timer
        self.timer.start(self.update_time)  # Update plot every "update_time" ms
       
        QApplication.processEvents()

    def update_plot(self):
        solution, avr = self.ga.evolve()
        self.genLabel.setText("Current Generation: " + str(self.ga.gen_number))
        self.fittestLabel.setText(f"Fittest Value of CurGen: {self.ga.fittest:.4f}")
        self.plot_solution(solution[1])

        # Check if either the generation limit is reached or the target fitness is achieved
        if self.ga.gen_number >= self.ga.iterations_limit or self.ga.fittest >= self.ga.target:
            self.timer.stop()

        distance = City.calc_distance(solution[1])
        self.distanceLabel.setText(f"Shortest Distance of CurGen: {distance:.4f}") 

        self.fitness_values.append(self.ga.fittest)
        self.avr_values.append(avr)

        # Update elapsed time
        elapsed = self.elapsed_time + (time.time() - self.start_time)
        self.timeLabel.setText(f"Elapsed Time: {elapsed:.2f} seconds")

        self.plot_fitness()
        self.plot_avr()

    def stop_update(self):
        if self.start_time is not None:
            self.elapsed_time += time.time() - self.start_time  
        self.timer.stop()

    def continue_update(self):
        self.start_time = time.time()  # Restart the timer
        self.timer.start(self.update_time)  # Update plot every "update_time" ms

    def plot_solution(self, solution):
        self.city_figure.clear()
        ax = self.city_figure.add_subplot(111)
        city_xs = [city.x for city in self.cities]
        city_ys = [city.y for city in self.cities]
        ax.scatter(city_xs, city_ys, c='red', marker='o', label='Cities')
        for i, city in enumerate(self.cities):
            ax.annotate(city.name, (city.x + 0.05, city.y + 0.05)) 
        solution_xs = [city.x for city in solution]
        solution_ys = [city.y for city in solution]
        ax.plot(solution_xs + [solution_xs[0]], solution_ys + [solution_ys[0]], color='gray', linewidth=2, label='Solution Path')
        ax.set_aspect('equal', adjustable='box')
        self.city_canvas.draw()


def main():
    app = QApplication(sys.argv)
    ex = TSPApp()
    ex.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

