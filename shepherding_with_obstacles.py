import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import astar


class Herding:
    def __init__(self, 
                 num_sheeps, 
                 num_dogs, 
                 grid_size, 
                 sheep_positions, 
                 dog_states, 
                 goal_position, 
                 goal_threshold, 
                 r0, max_iters, 
                 point_offset,
                 rectangle_obstacles=None):
        
        self.num_sheeps = num_sheeps
        self.num_dogs = num_dogs
        self.grid_size = grid_size
        self.goal_position = goal_position 
        self.goal_threshold = goal_threshold
        self.point_offset = point_offset
        self.rectangle_obstacles = rectangle_obstacles
        
        np.random.seed(0)

        self.sheep_positions = [[] for i in range(num_sheeps)]
        for i in range(num_sheeps):
            self.sheep_positions[i] = sheep_positions[i]
        self.sheep_mean = np.mean(sheep_positions, axis=0).reshape(1, 2)
        self.sheep_mean_dot = np.zeros((1, 2))
        self.dog_states = [[] for i in range(num_dogs)]
        for j in range(num_dogs):
            self.dog_states[j] = dog_states[j].reshape(1, 2)

        self.dt = 0.05
        self.r0 = r0
        self.r = r0
        self.phi = 0.0
        self.max_sheep_speed = 0.2
        self.max_dog_speed = 0.001
        self.max_dog_angular_speed = 4.5
        self.k = 10.0
        self.kd = 0.1
        self.min_dist = 1e-3
        self.max_iters = max_iters
        self.actual_iters = self.max_iters


    def steps(self):
        '''
        Run the herding simulation for a maximum number of iterations or until the goal is reached.
        '''
        for i in range(self.max_iters):
            s_dot = self.sheep_dynamics()

            for j in range(self.num_sheeps):
                new_sheep_pos = self.sheep_positions[j][-1] + s_dot[j] * self.dt
                self.sheep_positions[j] = np.vstack((self.sheep_positions[j], new_sheep_pos))

            self.sheep_mean = np.mean([pos[-1] for pos in self.sheep_positions], axis=0)
            self.sheep_mean_dot = np.mean(s_dot, axis=0)

            p_dot = self.controller_for_p_dot()
            ideal_heading, ideal_vel = self.get_ideal_heading_vel(p_dot)
            ideal_deltaj_star = self.ideal_delta_star(ideal_heading, ideal_vel)
            ideal_dog_position = self.get_ideal_dog_positions(ideal_deltaj_star, ideal_heading)
            r_dot = self.r_dot_controller(s_dot)
            dog_position_j_dot = self.tracking_controller(ideal_dog_position)
            self.r += r_dot * self.dt

            for j in range(self.num_dogs):
                new_dog_state = self.dog_states[j][-1] + dog_position_j_dot[j] * self.dt
                self.dog_states[j] = np.vstack((self.dog_states[j], new_dog_state))

            if np.linalg.norm(self.sheep_mean - self.goal_position) < self.goal_threshold:
                print(f"Goal reached in {i+1} iterations.")
                self.actual_iters = i + 1  
                return

        self.actual_iters = self.max_iters 


    def controller_for_p_dot(self):
        '''
        Compute the desired velocity for the point offset p.
        Following equation (15) from the paper: ṗ = -kp
        '''
        # Calculate point offset p (equation 19)
        s = self.sheep_mean
        qx = np.array([np.cos(self.phi), np.sin(self.phi)])
        p = s + self.point_offset * qx

        p_dot = -self.k * p
        
        return p_dot


    def get_ideal_heading_vel(self, p_dot):
        '''
        Compute the ideal heading and velocity based on desired point offset velocity.
        Following equations (3) and (22) from the paper.
        '''
        # The ideal heading is the direction of the desired velocity
        phi_ideal = np.arctan2(p_dot[1], p_dot[0])
        
        # The ideal velocity magnitude
        v_ideal_magnitude = np.linalg.norm(p_dot)
        v_ideal = v_ideal_magnitude * np.array([np.cos(phi_ideal), np.sin(phi_ideal)])
    
        if np.linalg.norm(v_ideal) > self.max_sheep_speed:
            v_ideal = (v_ideal / np.linalg.norm(v_ideal)) * self.max_sheep_speed
        
        self.phi = phi_ideal
        return phi_ideal, v_ideal


    def ideal_delta_star(self, phi_ideal, v_ideal):
        '''
        Compute the ideal angular positions (delta_j*) for the dogs around the sheep herd.
        Following the equations from Section III.B.2 of the paper.
        '''
        m = self.num_dogs
        r = self.r
        
        # Constants A and B for the sine arguments
        denominator_term = (2 - 2 * m)
        if m < 2 or denominator_term == 0:
            print("Error: Herding model requires at least 2 dogs (m >= 2).")
            return np.zeros(self.num_dogs)
            
        A = m / denominator_term
        B = 1 / denominator_term

        v_ideal_magnitude = np.linalg.norm(v_ideal)

        # Define the function f(Delta) whose root we seek (equation 11)
        def f(delta):
            if np.abs(np.sin(B * delta)) < 1e-8:
                return 1e9 * np.sign(A * delta)

            v_calc = np.sin(A * delta) / (r**2 * np.sin(B * delta))
            return v_calc - v_ideal_magnitude

        # Define the derivative of f(Delta) for Newton's method
        def f_prime(delta):
            sin_B_delta = np.sin(B * delta)
            cos_B_delta = np.cos(B * delta)
            sin_A_delta = np.sin(A * delta)
            cos_A_delta = np.cos(A * delta)
            
            if np.abs(sin_B_delta) < 1e-8:
                return 1e9
            
            numerator = (A * cos_A_delta * sin_B_delta - B * sin_A_delta * cos_B_delta)
            denominator = r**2 * sin_B_delta**2
            return numerator / denominator

        # Newton's Method
        delta = np.pi  # Initial guess
        
        for _ in range(20):
            f_val = f(delta)
            f_prime_val = f_prime(delta)
            
            if np.abs(f_val) < 1e-6:  # Fixed convergence criterion
                break
                
            if np.abs(f_prime_val) < 1e-10:
                delta = np.pi
                break
                
            delta_new = delta - f_val / f_prime_val
            delta_new = np.clip(delta_new, 0, 2*np.pi)
            
            if np.abs(delta_new - delta) < 1e-8:
                break
                
            delta = delta_new
                
        delta_star = delta
        
        # Calculate individual dog angles (equation 8)
        delta_j = np.zeros(self.num_dogs)
        for j in range(self.num_dogs):
            delta_j[j] = delta_star * (2*j - m + 1) / (2 * m - 2)

        return delta_j
    

    def get_ideal_dog_positions(self, ideal_deltaj_star, ideal_heading):
        '''
        Compute the ideal positions for the dogs around the sheep herd.
        Following equation (16) from the paper.
        '''
        ideal_dog_positions = np.zeros((self.num_dogs, 2))
        sheep_mean = self.sheep_mean
        
        for j in range(self.num_dogs):
            # Equation (16): d*_j = s + r[cos(φ* + Δ*_j), -sin(φ* + Δ*_j)]
            # Note: negative sign for positioning dogs behind the herd
            angle = ideal_heading + ideal_deltaj_star[j]
            ideal_dog_positions[j] = sheep_mean + self.r * np.array([
                -np.cos(angle), 
                -np.sin(angle)
            ])
            
        return ideal_dog_positions


    def sheep_dynamics(self):
        '''
        Compute the dynamics of the sheep based on repulsion from dogs and flocking behavior.
        Following equation (2) and (12) from the paper with added flocking dynamics.
        '''
        s_dot = np.zeros((self.num_sheeps, 2))
        cohesion_strength = 0.05
        separation_strength = 0.2
        cohesion_radius = 2.5
        separation_radius = 0.6

        for i in range(self.num_sheeps):
            vels = np.zeros(2)

            # Repulsion from dogs (equation 2)
            for j in range(self.num_dogs):
                diff = (self.sheep_positions[i][-1] - self.dog_states[j][-1, :2])
                dist = np.linalg.norm(diff)
                if dist < self.min_dist:
                    dist = self.min_dist
                repulsion = diff / (dist**3)
                vels += repulsion

            # Cohesion: attraction to nearby sheep
            cohesion = np.zeros(2)
            count_c = 0
            for k in range(self.num_sheeps):
                if k == i:
                    continue
                diff = self.sheep_positions[k][-1] - self.sheep_positions[i][-1]
                dist = np.linalg.norm(diff)
                if dist < cohesion_radius:
                    cohesion += diff
                    count_c += 1
            if count_c > 0:
                cohesion /= count_c
                vels += cohesion_strength * cohesion
    
            # Separation: repulsion from very close sheep
            separation = np.zeros(2)
            for k in range(self.num_sheeps):
                if k == i:
                    continue
                diff = self.sheep_positions[i][-1] - self.sheep_positions[k][-1]
                dist = np.linalg.norm(diff)
                if dist < separation_radius and dist > 1e-9:
                    separation += diff / (dist**2)
            vels += separation_strength * separation

            # Limit speed
            speed = np.linalg.norm(vels)
            if speed > 1e-9:
                vels = (vels / speed) * min(speed, self.max_sheep_speed)

            s_dot[i] = vels

        return s_dot


    def r_dot_controller(self, s_dot):
        '''
        Compute the rate of change of the radius (r_dot) for the sheep herd.
        Following equation (14) from the paper.
        '''
        r_dot = (self.r0 - self.r)
        for i in range(self.num_sheeps):
            r1 = 2 * (self.sheep_positions[i][-1] - self.sheep_mean)
            r2 = (s_dot[i] - self.sheep_mean_dot)
            r_dot += np.transpose(r1) @ r2
        r_dot = r_dot / self.num_sheeps
        return r_dot


    def tracking_controller(self, ideal_dog_position):
        '''
        Compute the tracking controller for the dogs to move towards their ideal positions.
        Following equation (17) from the paper.
        '''
        d_dot = np.zeros((self.num_dogs, 2))
        for j in range(self.num_dogs):
            d_dot[j] = self.kd * (ideal_dog_position[j] - self.dog_states[j][-1])
        return d_dot


    def animate(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-2, self.grid_size)
        ax.set_ylim(-2, self.grid_size)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        sheep_plots = [ax.plot([], [], 'bo', markersize=8, label='Sheep' if i==0 else '')[0] 
                      for i in range(self.num_sheeps)]
        dog_plots = [ax.plot([], [], 'rs', markersize=10, label='Dogs' if i==0 else '')[0] 
                    for i in range(self.num_dogs)]
        goal_plot = ax.plot(self.goal_position[0], self.goal_position[1], 'gx', 
                           markersize=15, markeredgewidth=3, label='Goal')[0]

        circle = plt.Circle((self.goal_position[0], self.goal_position[1]), 
                           self.goal_threshold, color='g', fill=False, 
                           linestyle='--', linewidth=2, label='Goal Region')
        ax.add_artist(circle)
        
        if self.rectangle_obstacles is not None:
            for pos, dims in self.rectangle_obstacles:
                rectangle = plt.Rectangle((pos[0], pos[1]), dims[0], dims[1], 
                                        color='gray', alpha=0.5)
                ax.add_artist(rectangle)

        ax.legend(loc='upper right')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Multi-Robot Herding Simulation')

        def init():
            for sheep_plot in sheep_plots:
                sheep_plot.set_data([], [])
            for dog_plot in dog_plots:
                dog_plot.set_data([], [])
            return sheep_plots + dog_plots + [goal_plot]

        def update(frame):
            for i, sheep_plot in enumerate(sheep_plots):
                x, y = self.sheep_positions[i][frame, 0], self.sheep_positions[i][frame, 1]
                sheep_plot.set_data([x], [y])

            for j, dog_plot in enumerate(dog_plots):
                x, y = self.dog_states[j][frame, 0], self.dog_states[j][frame, 1]
                dog_plot.set_data([x], [y]) 

            return sheep_plots + dog_plots + [goal_plot]

        ani = animation.FuncAnimation(fig, update, frames=self.actual_iters, 
                                     init_func=init, blit=True, interval=50)
        plt.show()
        ani.save('herding_simulation.gif', writer='pillow')


def main():
    num_sheep = 10
    num_dogs = 4
    max_iters = 2000
    grid_size = 30.0
    point_offset = 0.6

    rectangle_obstacle_dims = np.array([[10.0, 5.0], [5.0, 15.0]])
    rectangle_obstacle_positions = np.array([[10.0, 0.0], [12.0, 15.0]])
    rectangle_obstacles = [(pos, dims) for pos, dims in zip(rectangle_obstacle_positions, rectangle_obstacle_dims)]


    sheep_positions = np.random.normal(loc=(5.0, 20.0), scale=1.0, size=(num_sheep, 1, 2))

    dog_states = np.random.normal(loc=(22.0, 27.0), scale=0.5, size=(num_dogs, 1, 2))
    
    goal_position = np.array([0.0, 0.0])
    goal_threshold = 2.5
    r0 = 3.0

    herding_env = Herding(num_sheep, 
                          num_dogs, 
                          grid_size, 
                          sheep_positions, 
                          dog_states, 
                          goal_position, 
                          goal_threshold, 
                          r0, 
                          max_iters, 
                          point_offset,
                          rectangle_obstacles)
    
    print("Starting herding simulation...")
    print(f"Initial sheep mean position: {herding_env.sheep_mean}")
    print(f"Goal position: {goal_position}")
    print(f"Distance to goal: {np.linalg.norm(herding_env.sheep_mean - goal_position):.2f}")
    
    herding_env.steps()
    
    print(f"\nFinal sheep mean position: {herding_env.sheep_mean}")
    print(f"Final distance to goal: {np.linalg.norm(herding_env.sheep_mean - goal_position):.2f}")
    
    herding_env.animate()

if __name__ == "__main__":
    main()