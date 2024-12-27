import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class InvertedPendulum:
    def __init__(self):
        # Parameter sistem
        self.g = 9.81  # gravitasi
        self.M = 0.5   # massa cart
        self.m = 0.2   # massa pendulum
        self.l = 0.3   # panjang pendulum
        self.b = 0.1   # koefisien gesek
        self.I = 0.006 # momen inersia
        self.dt = 0.02 # time step
        
        # State awal
        self.x = 0      # posisi cart
        self.theta = 0.1 # sudut pendulum (sedikit miring di awal)
        self.dx = 0     # kecepatan cart
        self.dtheta = 0 # kecepatan sudut pendulum
        
        # Parameter ABC
        self.food_sources = 20
        self.max_iter = 100
        self.limit = 30

    def update_state(self, u):
        # Prevent overflow in trigonometric functions
        if abs(self.theta) > np.pi/2:  # Limit theta to avoid extreme angles
            self.theta = np.sign(self.theta) * (np.pi/2)
            
        sin_theta = np.sin(self.theta)
        cos_theta = np.cos(self.theta)
        
        D = self.m * self.l * self.l * (self.M + self.m * (1 - cos_theta * cos_theta))
        if D == 0:
            D = 1e-10  # Avoid division by zero

        ddx = (1/D) * (self.m * self.m * self.l * self.l * self.g * cos_theta * sin_theta +
                       self.m * self.l * self.l * (self.m * self.l * self.dtheta * self.dtheta * sin_theta - self.b * self.dx + u))

        ddtheta = (1/D) * (-self.m * self.l * cos_theta * (self.m * self.l * self.dtheta * self.dtheta * sin_theta - self.b * self.dx + u) -
                           (self.M + self.m) * self.m * self.g * self.l * sin_theta)

        # Update state with integration
        self.x += self.dx * self.dt
        self.theta += self.dtheta * self.dt
        self.dx += ddx * self.dt
        self.dtheta += ddtheta * self.dt
        
        # Limit the velocities to prevent instability
        self.dx = np.clip(self.dx, -10, 10)
        self.dtheta = np.clip(self.dtheta, -10, 10)

class ABC:
    def __init__(self, pendulum):
        self.pendulum = pendulum
        self.solutions = []
        self.trials = np.zeros(pendulum.food_sources)
        self.best_solution = None
        self.best_fitness = float('inf')
        
        # Inisialisasi solusi acak
        for _ in range(pendulum.food_sources):
            solution = {
                'Kp': np.random.uniform(0, 100),
                'Ki': np.random.uniform(0, 100), 
                'Kd': np.random.uniform(0, 100)
            }
            self.solutions.append(solution)
    
    def calculate_fitness(self, solution):
        # Reset state
        self.pendulum.x = 0
        self.pendulum.theta = 0.1
        self.pendulum.dx = 0 
        self.pendulum.dtheta = 0
        
        total_error = 0
        # Simulasi sistem untuk periode tertentu
        for _ in range(100):
            # Hitung kontrol PID
            error = 0 - self.pendulum.theta
            u = solution['Kp'] * error + solution['Kd'] * (-self.pendulum.dtheta)
            u = np.clip(u, -10, 10)  # Limit control input
            
            # Update state sistem
            self.pendulum.update_state(u)
            
            # Akumulasi error absolut
            total_error += abs(self.pendulum.theta)
            
        return total_error
    
    def optimize(self):
        for iteration in range(self.pendulum.max_iter):
            # Employed Bee Phase
            for i in range(self.pendulum.food_sources):
                # Modifikasi solusi
                new_solution = self.solutions[i].copy()
                param = np.random.choice(['Kp', 'Ki', 'Kd'])
                new_solution[param] += (np.random.random() - 0.5) * 20
                
                # Evaluasi fitness
                old_fitness = self.calculate_fitness(self.solutions[i])
                new_fitness = self.calculate_fitness(new_solution)
                
                # Update solusi jika lebih baik
                if new_fitness < old_fitness:
                    self.solutions[i] = new_solution
                    self.trials[i] = 0
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_solution = new_solution.copy()
                else:
                    self.trials[i] += 1
            
            # Onlooker Bee Phase
            probabilities = []
            total_fitness = sum(1/(self.calculate_fitness(s) + 1e-10) for s in self.solutions)
            
            for solution in self.solutions:
                prob = (1/(self.calculate_fitness(solution) + 1e-10))/total_fitness
                probabilities.append(prob)
            
            t = 0
            i = 0
            while t < self.pendulum.food_sources:
                if np.random.random() < probabilities[i]:
                    # Modifikasi solusi
                    new_solution = self.solutions[i].copy()
                    param = np.random.choice(['Kp', 'Ki', 'Kd'])
                    new_solution[param] += (np.random.random() - 0.5) * 20
                    
                    # Evaluasi fitness
                    old_fitness = self.calculate_fitness(self.solutions[i])
                    new_fitness = self.calculate_fitness(new_solution)
                    
                    if new_fitness < old_fitness:
                        self.solutions[i] = new_solution
                        self.trials[i] = 0
                        if new_fitness < self.best_fitness:
                            self.best_fitness = new_fitness
                            self.best_solution = new_solution.copy()
                    else:
                        self.trials[i] += 1
                    t += 1
                i = (i + 1) % self.pendulum.food_sources
            
            # Scout Bee Phase
            for i in range(self.pendulum.food_sources):
                if self.trials[i] > self.pendulum.limit:
                    self.solutions[i] = {
                        'Kp': np.random.uniform(0, 100),
                        'Ki': np.random.uniform(0, 100),
                        'Kd': np.random.uniform(0, 100)
                    }
                    self.trials[i] = 0
            
            print(f"Iteration {iteration+1}, Best Fitness: {self.best_fitness}")

# Visualisasi
def animate_pendulum(pendulum, abc):
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    
    # Plot elemen
    cart, = ax.plot([], [], 'ks', markersize=20)
    rod, = ax.plot([], [], 'k-', lw=2)
    mass, = ax.plot([], [], 'bo', markersize=10)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    def init():
        cart.set_data([], [])
        rod.set_data([], [])
        mass.set_data([], [])
        time_text.set_text('')
        return cart, rod, mass, time_text
    
    def animate(i):
        # Hitung kontrol
        error = 0 - pendulum.theta
        u = (abc.best_solution['Kp'] * error + 
             abc.best_solution['Kd'] * (-pendulum.dtheta))
        u = np.clip(u, -10, 10)  # Limit control input
        
        # Update state
        pendulum.update_state(u)
        
        # Update plot
        cart.set_data([pendulum.x], [0])
        
        rod_x = [pendulum.x, pendulum.x + pendulum.l * np.sin(pendulum.theta)]
        rod_y = [0, -pendulum.l * np.cos(pendulum.theta)]
        rod.set_data(rod_x, rod_y)
        
        mass.set_data([pendulum.x + pendulum.l * np.sin(pendulum.theta)],
                     [-pendulum.l * np.cos(pendulum.theta)])
        
        time_text.set_text(f'time = {i * pendulum.dt:.1f}s')
        return cart, rod, mass, time_text
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=500,
                        interval=pendulum.dt * 1000, blit=True)
    plt.show()

# Jalankan simulasi
pendulum = InvertedPendulum()
abc = ABC(pendulum)
abc.optimize()
print("\nBest Solution:")
print(f"Kp: {abc.best_solution['Kp']:.2f}")
print(f"Ki: {abc.best_solution['Ki']:.2f}") 
print(f"Kd: {abc.best_solution['Kd']:.2f}")

# Animasi hasil
animate_pendulum(pendulum, abc)