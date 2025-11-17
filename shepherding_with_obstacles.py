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
                 intermediate_goals,
                 goal_threshold, 
                 r0, max_iters, 
                 point_offset,
                 rectangle_obstacles=None):
        
        self.num_sheeps = num_sheeps
        self.num_dogs = num_dogs
        self.grid_size = grid_size
        self.goal_position = np.array(goal_position, dtype=float)
        self.intermediate_goals = intermediate_goals
        self.goal_threshold = goal_threshold
        self.point_offset = point_offset
        self.rectangle_obstacles = rectangle_obstacles if rectangle_obstacles is not None else []

        np.random.seed(0)

        self.sheep_positions = [None] * num_sheeps
        for i in range(num_sheeps):
            self.sheep_positions[i] = np.array(sheep_positions[i], dtype=float).reshape(-1, 2)

        self.sheep_mean = np.mean([pos[-1] for pos in self.sheep_positions], axis=0).astype(float)
        self.sheep_mean_dot = np.zeros(2, dtype=float)

        self.dog_states = [None] * num_dogs
        for j in range(num_dogs):
            self.dog_states[j] = np.array(dog_states[j], dtype=float).reshape(-1, 2)

        self.dt = 0.05
        self.r0 = r0
        self.r = r0
        self.phi = 0.0
        self.max_sheep_speed = 0.2
        self.max_dog_speed = 0.5 
        self.max_dog_angular_speed = 4.5
        self.k = 10.0
        self.kd = 0.5
        self.min_dist = 1e-3
        self.max_iters = max_iters
        self.actual_iters = self.max_iters

        self.obstacle_safety_radius = 1.0
        self.obstacle_repulsion_gain = 10.0


    def steps(self):
        """
        Steps: move through intermediate waypoints first (if any), then final goal.
        Local obstacle avoidance is applied (repulsive force fields).
        """
        goals = []
        if self.intermediate_goals is not None and len(self.intermediate_goals) > 0:
            goals = [np.array(g, dtype=float) for g in self.intermediate_goals]
        goals.append(np.array(self.goal_position, dtype=float))

        current_goal_index = 0
        current_goal = goals[current_goal_index]

        for i in range(self.max_iters):
            self.sheep_mean = np.mean([pos[-1] for pos in self.sheep_positions], axis=0)

            if np.linalg.norm(self.sheep_mean - current_goal) < self.goal_threshold:
                print(f"Reached goal {current_goal_index + 1}/{len(goals)} at iteration {i+1}")
                current_goal_index += 1
                if current_goal_index >= len(goals):
                    print(f"Final goal reached in {i+1} iterations.")
                    self.actual_iters = i + 1
                    return
                current_goal = goals[current_goal_index]

            self.goal_position = current_goal


            s_dot = self.sheep_dynamics()

            for j in range(self.num_sheeps):
                new_sheep_pos = self.sheep_positions[j][-1] + s_dot[j] * self.dt
                self.sheep_positions[j] = np.vstack((self.sheep_positions[j], new_sheep_pos))

            new_sheep_mean = np.mean([pos[-1] for pos in self.sheep_positions], axis=0)
            self.sheep_mean_dot = (new_sheep_mean - self.sheep_mean) / max(self.dt, 1e-9)
            self.sheep_mean = new_sheep_mean

            p_dot = self.controller_for_p_dot()
            ideal_heading, ideal_vel = self.get_ideal_heading_vel(p_dot)
            ideal_deltaj_star = self.ideal_delta_star(ideal_heading, ideal_vel)
            ideal_dog_position = self.get_ideal_dog_positions(ideal_deltaj_star, ideal_heading)
            r_dot = self.r_dot_controller(s_dot)
            dog_position_j_dot = self.tracking_controller(ideal_dog_position)
            self.r += r_dot * self.dt

            for j in range(self.num_dogs):
                vel = dog_position_j_dot[j]
                speed = np.linalg.norm(vel)
                if speed > self.max_dog_speed:
                    vel = (vel / speed) * self.max_dog_speed
                new_dog_state = self.dog_states[j][-1] + vel * self.dt
                self.dog_states[j] = np.vstack((self.dog_states[j], new_dog_state))

        self.actual_iters = self.max_iters
        print("Max iterations reached, stopping.")


    def controller_for_p_dot(self):
        '''
        Compute the desired velocity for the point offset p.
        Following equation (15) from the paper: ṗ = -kp.
        For intermediate goals, p -> p - current goal_position.
        '''
        s = self.sheep_mean
        qx = np.array([np.cos(self.phi), np.sin(self.phi)])
        p = s + self.point_offset * qx

        p_dot = -self.k * (p - self.goal_position)
        return p_dot


    def get_ideal_heading_vel(self, p_dot):
        '''
        Compute the ideal heading and velocity based on desired point offset velocity.
        '''
        phi_ideal = np.arctan2(p_dot[1], p_dot[0])
        v_ideal_magnitude = np.linalg.norm(p_dot)
        v_ideal = v_ideal_magnitude * np.array([np.cos(phi_ideal), np.sin(phi_ideal)])

        if np.linalg.norm(v_ideal) > self.max_sheep_speed:
            v_ideal = (v_ideal / np.linalg.norm(v_ideal)) * self.max_sheep_speed

        self.phi = phi_ideal
        return phi_ideal, v_ideal


    def ideal_delta_star(self, phi_ideal, v_ideal):
        '''
        Compute the ideal angular positions (delta_j*) for the dogs around the sheep herd.
        '''
        m = self.num_dogs
        r = self.r

        denominator_term = (2 - 2 * m)
        if m < 2 or denominator_term == 0:
            print("Error: Herding model requires at least 2 dogs (m >= 2).")
            return np.zeros(self.num_dogs)

        A = m / denominator_term
        B = 1 / denominator_term

        v_ideal_magnitude = np.linalg.norm(v_ideal)

        def f(delta):
            if np.abs(np.sin(B * delta)) < 1e-8:
                return 1e9 * np.sign(A * delta)
            v_calc = np.sin(A * delta) / (r**2 * np.sin(B * delta))
            return v_calc - v_ideal_magnitude

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

        delta = np.pi
        for _ in range(50):
            f_val = f(delta)
            f_prime_val = f_prime(delta)
            if np.abs(f_val) < 1e-6:
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
        delta_j = np.zeros(self.num_dogs)
        for j in range(self.num_dogs):
            delta_j[j] = delta_star * (2*j - m + 1) / (2 * m - 2)

        return delta_j


    def get_ideal_dog_positions(self, ideal_deltaj_star, ideal_heading):
        '''
        Compute the ideal positions for the dogs around the sheep herd.
        '''
        ideal_dog_positions = np.zeros((self.num_dogs, 2))
        sheep_mean = self.sheep_mean

        for j in range(self.num_dogs):
            angle = ideal_heading + ideal_deltaj_star[j]
            ideal_dog_positions[j] = sheep_mean + self.r * np.array([
                -np.cos(angle), 
                -np.sin(angle)
            ])
        return ideal_dog_positions


    def sheep_dynamics(self):
        '''
        Compute the dynamics of the sheep based on repulsion from dogs and flocking behavior,
        plus obstacle avoidance as a repulsive field.
        '''
        s_dot = np.zeros((self.num_sheeps, 2))
        cohesion_strength = 0.05
        separation_strength = 0.2
        cohesion_radius = 2.5
        separation_radius = 0.6

        for i in range(self.num_sheeps):
            vels = np.zeros(2)

            # Repulsion from dogs
            for j in range(self.num_dogs):
                diff = (self.sheep_positions[i][-1] - self.dog_states[j][-1])
                dist = np.linalg.norm(diff)
                if dist < self.min_dist:
                    dist = self.min_dist
                repulsion = diff / (dist**3)
                vels += repulsion

            # Cohesion
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

            # Separation
            separation = np.zeros(2)
            for k in range(self.num_sheeps):
                if k == i:
                    continue
                diff = self.sheep_positions[i][-1] - self.sheep_positions[k][-1]
                dist = np.linalg.norm(diff)
                if dist < separation_radius and dist > 1e-9:
                    separation += diff / (dist**2)
            vels += separation_strength * separation

            # Obstacle avoidance
            vels += self.obstacle_repulsion(self.sheep_positions[i][-1])

            speed = np.linalg.norm(vels)
            if speed > 1e-9:
                vels = (vels / speed) * min(speed, self.max_sheep_speed)

            s_dot[i] = vels

        return s_dot


    def obstacle_repulsion(self, point):
        """
        Smooth repulsive velocity from axis-aligned rectangular obstacles.
        Returns a 2D vector pushing the agent away from obstacles when inside safety radius.
        """
        total = np.zeros(2, dtype=float)
        for (pos, dims) in self.rectangle_obstacles:
            pos = np.array(pos, dtype=float)
            dims = np.array(dims, dtype=float)

            closest = np.array([
                np.clip(point[0], pos[0], pos[0] + dims[0]),
                np.clip(point[1], pos[1], pos[1] + dims[1])
            ])
            diff = point - closest
            dist = np.linalg.norm(diff)

            if dist < 1e-6:
                corner = pos 
                corners = [pos, pos + np.array([dims[0], 0.0]), pos + np.array([0.0, dims[1]]), pos + dims]
                dists = [np.linalg.norm(point - c) for c in corners]
                corner = corners[int(np.argmax(dists))]
                diff = point - corner
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    continue

            if dist < self.obstacle_safety_radius:
                force_mag = self.obstacle_repulsion_gain * (1.0 / max(dist, 1e-6) - 1.0 / self.obstacle_safety_radius)
                total += (force_mag * diff / (dist**2))

        return total


    def r_dot_controller(self, s_dot):
        '''
        Compute the rate of change of the radius (r_dot) for the sheep herd.
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
        Dogs try to track their ideal positions but also avoid obstacles (local repulsion).
        '''
        d_dot = np.zeros((self.num_dogs, 2))
        for j in range(self.num_dogs):
            d_dot[j] = self.kd * (ideal_dog_position[j] - self.dog_states[j][-1])

        for j in range(self.num_dogs):
            d_dot[j] += self.obstacle_repulsion(self.dog_states[j][-1])

        return d_dot


    def animate(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-2, self.grid_size)
        ax.set_ylim(-2, self.grid_size)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        sheep_plots = [ax.plot([], [], 'bo', markersize=6, label='Sheep' if i==0 else '')[0] 
                      for i in range(self.num_sheeps)]
        dog_plots = [ax.plot([], [], 'rs', markersize=8, label='Dogs' if i==0 else '')[0] 
                    for i in range(self.num_dogs)]
        goal_plot = ax.plot(self.goal_position[0], self.goal_position[1], 'gx', 
                           markersize=12, markeredgewidth=3, label='Goal')[0]

        circle = plt.Circle((self.goal_position[0], self.goal_position[1]), 
                           self.goal_threshold, color='g', fill=False, 
                           linestyle='--', linewidth=2, label='Goal Region')
        ax.add_artist(circle)
        
        # draw rectangle obstacles
        if self.rectangle_obstacles is not None:
            for pos, dims in self.rectangle_obstacles:
                rectangle = plt.Rectangle((pos[0], pos[1]), dims[0], dims[1], 
                                        color='gray', alpha=0.5)
                ax.add_artist(rectangle)
                pad = self.obstacle_safety_radius
                rect_pad = plt.Rectangle((pos[0] - pad, pos[1] - pad),
                                         dims[0] + 2*pad, dims[1] + 2*pad,
                                         fill=False, linestyle=':', linewidth=1)
                ax.add_artist(rect_pad)

        plotted_flag = False
        if self.intermediate_goals is not None:
            for g in self.intermediate_goals:
                ax.plot(g[0], g[1], 'yo', markersize=7, label='Intermediate Goal' if not plotted_flag else '')
                plotted_flag = True

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
            frame = min(frame, self.actual_iters - 1)
            for i, sheep_plot in enumerate(sheep_plots):
                idx = min(frame, self.sheep_positions[i].shape[0] - 1)
                x, y = self.sheep_positions[i][idx, 0], self.sheep_positions[i][idx, 1]
                sheep_plot.set_data([x], [y])

            for j, dog_plot in enumerate(dog_plots):
                idx = min(frame, self.dog_states[j].shape[0] - 1)
                x, y = self.dog_states[j][idx, 0], self.dog_states[j][idx, 1]
                dog_plot.set_data([x], [y]) 

            return sheep_plots + dog_plots + [goal_plot]

        ani = animation.FuncAnimation(fig, update, frames=self.actual_iters, 
                                     init_func=init, blit=True, interval=50)
        plt.show()
        try:
            ani.save('herding_simulation.gif', writer='pillow')
            print("Saved animation to herding_simulation.gif")
        except Exception as e:
            print("Could not save gif:", e)



def is_in_obstacle(point, rectangle_obstacles, safety_margin):
    for (pos, dims) in rectangle_obstacles:
        if (pos[0] - safety_margin <= point[0] <= pos[0] + dims[0] + safety_margin and
            pos[1] - safety_margin <= point[1] <= pos[1] + dims[1] + safety_margin):
            return True
    return False


def get_neighbors(rectangle_obstacles, resolution, safety_margin):
    def neighbors_fnct(node):
        if hasattr(node, "data"):
            node = node.data
        node = np.array(node, dtype=float)

        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for direction in directions:
            neighbor = node + np.array(direction, dtype=float) * resolution
            neighbor = tuple(neighbor)

            if not is_in_obstacle(neighbor, rectangle_obstacles, safety_margin):
                neighbors.append(neighbor)

        return neighbors
    return neighbors_fnct


def heuristic_cost_estimate():
    def heuristic_fnct(node, goal):
        node = node.data if hasattr(node, "data") else node
        goal = goal.data if hasattr(goal, "data") else goal
        return 100*np.linalg.norm(np.array(node) - np.array(goal))
    return heuristic_fnct


def distance_between():
    def distance_fnct(node1, node2):
        node1 = node1.data if hasattr(node1, "data") else node1
        node2 = node2.data if hasattr(node2, "data") else node2
        return np.linalg.norm(np.array(node1) - np.array(node2))
    return distance_fnct


def is_goal_reached():
    def goal_reached_fnct(node, goal):
        node = node.data if hasattr(node, "data") else node
        goal = goal.data if hasattr(goal, "data") else goal
        return np.linalg.norm(np.array(node) - np.array(goal)) < 1.0
    return goal_reached_fnct


def generate_intermediate_goals(dog_states, sheep_positions, rectangle_obstacles, grid_size, resolution, safety_margin):
    start_sheep_mean = np.mean(sheep_positions, axis=0).reshape(1, 2)
    start = tuple(start_sheep_mean[0])

    goal = (0.0, 0.0)
    print(f"Computing A* path from start {start} to goal {goal}...")
    path = astar.find_path(start=start,
                           goal=goal,
                           neighbors_fnct=get_neighbors(rectangle_obstacles, resolution, safety_margin),
                           heuristic_cost_estimate_fnct=heuristic_cost_estimate(),
                           distance_between_fnct=distance_between(),
                           is_goal_reached_fnct=is_goal_reached(),
                           )
    if path is None:
        return []
    waypoints = []
    for n in path:
        if hasattr(n, "data"):
            waypoints.append(tuple(map(float, n.data)))
        else:
            waypoints.append(tuple(map(float, n)))
    return waypoints


def main():
    num_sheep = 10
    num_dogs = 4
    max_iters = 10000
    grid_size = 30.0
    point_offset = 0.6

    rectangle_obstacle_dims = np.array([[10.0, 5.0], [5.0, 15.0]])
    rectangle_obstacle_positions = np.array([[10.0, 0.0], [12.0, 15.0]])
    rectangle_obstacles = [(pos, dims) for pos, dims in zip(rectangle_obstacle_positions, rectangle_obstacle_dims)]

    # initial sheep positions: (num_sheep, 1, 2)
    sheep_positions = np.random.normal(loc=(25.0, 25.0), scale=0.5, size=(num_sheep, 1, 2))

    # initial dog states: (num_dogs, 1, 2)
    dog_states = np.random.normal(loc=(25.0, 30.0), scale=0.5, size=(num_dogs, 1, 2))

    resolution = 5.0
    safety_margin = 4.0

    intermediate_goals = generate_intermediate_goals(dog_states, sheep_positions, rectangle_obstacles, grid_size, resolution, safety_margin)
    
    if intermediate_goals:
        print("Intermediate goals from A*:")
        for idx, goal in enumerate(intermediate_goals):
            print(f"  {idx+1}: {goal}")
    else:
        print("No intermediate goals returned by A* (path empty or blocked).")

    goal_position = np.array([0.0, 0.0])
    goal_threshold = 2.5
    r0 = 3.0

    herding_env = Herding(num_sheep, 
                          num_dogs, 
                          grid_size, 
                          sheep_positions, 
                          dog_states, 
                          goal_position,
                          intermediate_goals, 
                          goal_threshold, 
                          r0, 
                          max_iters, 
                          point_offset,
                          rectangle_obstacles)
    
    print(f"Initial sheep mean position: {herding_env.sheep_mean}")
    print(f"Goal position: {goal_position}")
    
    herding_env.steps()
    
    print(f"\nFinal sheep mean position: {herding_env.sheep_mean}")
    
    herding_env.animate()


if __name__ == "__main__":
    main()
