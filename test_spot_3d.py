import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from icecream import ic
import random
import math
from custom_envs.spot_3dmodule.spot_3dmodule import Map
from matplotlib.cm import Greys
from matplotlib.colors import Normalize
  
class Env_test(Map):
    def __init__(
        self,
        init_x = None,
        init_y = None,
        num_wall = 1,
        wall_length = 4,
        grid_length = 20,
        num_stairs = 2,
        stairs_weight_rand = 1,
        stairs_weight_bias = 3,
        floor_width_rand = 1,
        floor_width_bias = 9,
        grid_length_height = 2,
        grid_density = 0.3,
        wall_thickness = 0.6,
        seed = 18,
        spot_lidar_angle = 10,
        agent_size = 0.24,
        four_sign_distance = 1.5,
        random_map = True

    ) -> None:
        super().__init__(
            num_wall = num_wall,
            wall_length = wall_length,
            grid_length = grid_length,
            num_stairs = num_stairs,
            stairs_weight_rand = stairs_weight_rand,
            stairs_weight_bias = stairs_weight_bias,
            floor_width_rand = floor_width_rand,
            floor_width_bias = floor_width_bias,
            grid_length_height = grid_length_height,
            grid_density = grid_density,
            wall_thickness = wall_thickness,
            seed = seed,
            spot_lidar_angle = spot_lidar_angle,
            random_map = random_map
        )
        self.generate_map()
        
        self.map_image = plt.figure(figsize=(10, 10))
        self.ax = self.map_image.add_subplot(projection='3d')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.grid(False)

        self.ax.set_axis_off()
        self.revese_env = 1 # nomal env
        revese_map_z = self.map_z * self.revese_env # 獲取z軸的值
        norm = Normalize(vmin= -1 * self.grid_length_height * 2, vmax=self.grid_length_height * 1.5)
        colors = Greys(1 - norm(revese_map_z))
        self.ax.plot_surface(self.map_x, self.map_y, revese_map_z, facecolors=colors, rstride=1, cstride=1, alpha=0.2)
        rate =  2 * self.grid_length_height / self.grid_length
        self.ax.set_box_aspect([1, 1, rate])

        self.angular_grid = plt.figure()
        self.ag_ax = self.angular_grid.add_subplot(111, polar=True)
        self.ag_ax.set_theta_offset(np.pi / 2)
        angular_grid_len = self.num_angle_lidar

        self.theta = np.linspace(0, 2 * np.pi, angular_grid_len, endpoint=False)
        self.angular_grid_data = np.zeros(angular_grid_len)
        self.ag_ax.plot(self.theta, self.angular_grid_data)

        if init_x is not None: xpos = int(init_x)
        else:  xpos = int(self.grid_length * random.random())
        if init_y is not None: ypos = int(init_y)
        else: ypos = int(self.grid_length * random.random())
        self.agent_x = min(xpos, self.map_max_x)
        self.agent_y = min(ypos, self.map_max_y)
        self.agent_size = agent_size
        self.goal_x = min(int(self.grid_length * random.random()), self.map_max_x)
        self.goal_y = min(int(self.grid_length * random.random()), self.map_max_y)
        self.goal_z = self.map_z[int(self.goal_y / self.grid_density) , int(self.goal_x / self.grid_density)] * self.revese_env
        self.agent_z = self.map_z[int(self.agent_y / self.grid_density) , int(self.agent_x / self.grid_density)] * self.revese_env
        self.agent_dir = 0
        self.four_sign_distance = four_sign_distance
        self.create_agent_in_map()
        self.generate_keybord()

    def create_agent_in_map(self):
        self.agent_plt_object, = self.ax.plot([self.agent_x], [self.agent_y], [self.agent_z], 'o', markersize=7, color='yellow')
        self.agent_plt_object.set_zorder(10)

        self.agent_dir_plt_object, = self.ax.plot([self.agent_x], [self.agent_y], [self.agent_z], 'o', markersize=3.5, color='deepskyblue')
        self.agent_dir_plt_object.set_zorder(10)

        self.front_short, = self.ax.plot([self.agent_x], [self.agent_y], [self.agent_z], 'o', markersize=3.5, color='deepskyblue')
        self.front_short.set_zorder(10)
        self.front_far, = self.ax.plot([self.agent_x], [self.agent_y], [self.agent_z], 'o', markersize=3.5, color='deepskyblue')
        self.front_far.set_zorder(10)
        self.front_right, = self.ax.plot([self.agent_x], [self.agent_y], [self.agent_z], 'o', markersize=3.5, color='deepskyblue')
        self.front_right.set_zorder(10)
        self.front_left, = self.ax.plot([self.agent_x], [self.agent_y], [self.agent_z], 'o', markersize=3.5, color='deepskyblue')
        self.front_left.set_zorder(10)
        """ 
        self.back_short, = self.ax.plot([self.agent_x], [self.agent_y], [self.agent_z], 'o', markersize=3.5, color='deepskyblue')
        self.back_short.set_zorder(10)
        self.back_far, = self.ax.plot([self.agent_x], [self.agent_y], [self.agent_z], 'o', markersize=3.5, color='deepskyblue')
        self.back_far.set_zorder(10)
        self.back_right, = self.ax.plot([self.agent_x], [self.agent_y], [self.agent_z], 'o', markersize=3.5, color='deepskyblue')
        self.back_right.set_zorder(10)
        self.back_left, = self.ax.plot([self.agent_x], [self.agent_y], [self.agent_z], 'o', markersize=3.5, color='deepskyblue')
        self.back_left.set_zorder(10)
        """
        goal, = self.ax.plot([self.goal_x], [self.goal_y], [self.goal_z], 'o', markersize=7, color='fuchsia')
        goal.set_zorder(10)
        return 0
    
    def generate_keybord(self):
        self.map_image.canvas.mpl_connect('key_press_event', self.on_press)
        plt.show()

    def on_press(self, event):
        """
            RL 中會放在 update_state 中
        """
        if event.key == '8':
            self.agent_x = self.agent_x + 0.4 * math.cos(self.agent_dir)
            self.agent_y = self.agent_y + 0.4 * math.sin(self.agent_dir)
        if event.key == '2':
            self.agent_x = self.agent_x - 0.4 * math.cos(self.agent_dir)
            self.agent_y = self.agent_y - 0.4 * math.sin(self.agent_dir)
        if event.key == '4':
            self.agent_x = self.agent_x - 0.4 * math.sin(self.agent_dir)
            self.agent_y = self.agent_y + 0.4 * math.cos(self.agent_dir)
        if event.key == '6':
            self.agent_x = self.agent_x + 0.4 * math.sin(self.agent_dir)
            self.agent_y = self.agent_y - 0.4 * math.cos(self.agent_dir)
        if event.key == '9':
            self.agent_dir = self.agent_dir - 0.2
        if event.key == '7':
            self.agent_dir = self.agent_dir + 0.2
        #ic(self.agent_dir,  math.cos(self.agent_dir))
        self.display_observation()
        
    def display_observation(self):
        # angular grid
        point_x = min(int(self.agent_x / self.grid_density + 0.5), self.map_max_x - 1)
        point_y = min(int(self.agent_y / self.grid_density + 0.5), self.map_max_y - 1)
        left_point_x = min(int(self.agent_x / self.grid_density), self.map_max_x - 1)
        down_point_y = min(int(self.agent_y / self.grid_density), self.map_max_y - 1)
        right_point_x = min(int(self.agent_x / self.grid_density + 1), self.map_max_x - 1)
        up_point_y = min(int(self.agent_y / self.grid_density + 1), self.map_max_y - 1)
        precise_new_agent_grid_x = min(self.agent_x / self.grid_density, self.map_max_x - 1) - left_point_x
        precise_new_agent_grid_y = min(self.agent_y / self.grid_density, self.map_max_y - 1) - down_point_y
        self.agent_plt_object.set_color("yellow")
        for i in range(self.num_angle_lidar):
            angle_ = self.agent_dir * 360 / 2.0 / np.pi / self.spot_lidar_angle
            real_angle = int(i + angle_ + 0.5)
            if real_angle < 0: real_angle += self.num_angle_lidar
            elif real_angle >= self.num_angle_lidar: real_angle -= self.num_angle_lidar
            left_down_angle_distance = self.different_point_angle_distance[self.num_angle_lidar * self.map_max_x * down_point_y + self.num_angle_lidar * left_point_x + real_angle] 
            left_up_angle_distance = self.different_point_angle_distance[self.num_angle_lidar * self.map_max_x * up_point_y + self.num_angle_lidar * left_point_x + real_angle] 
            right_down_angle_distance = self.different_point_angle_distance[self.num_angle_lidar * self.map_max_x * down_point_y + self.num_angle_lidar * right_point_x + real_angle] 
            right_up_angle_distance = self.different_point_angle_distance[self.num_angle_lidar * self.map_max_x * up_point_y + self.num_angle_lidar * right_point_x + real_angle] 
            nomal_angle_distance = self.different_point_angle_distance[self.num_angle_lidar * self.map_max_x * point_y + self.num_angle_lidar * point_x + real_angle] 
            precise_angle_distance =  precise_new_agent_grid_x * precise_new_agent_grid_y * right_up_angle_distance + (1 - precise_new_agent_grid_x) * precise_new_agent_grid_y * left_up_angle_distance + precise_new_agent_grid_x * (1 - precise_new_agent_grid_y) * right_down_angle_distance + (1 - precise_new_agent_grid_x) * (1 - precise_new_agent_grid_y) * left_down_angle_distance
            max_angle_distance = max(max(left_down_angle_distance, left_up_angle_distance), max(right_down_angle_distance, right_up_angle_distance))
            min_angle_distance = min(min(left_down_angle_distance, left_up_angle_distance), min(right_down_angle_distance, right_up_angle_distance))
            smallest_distance = self.grid_density *  min(min(precise_new_agent_grid_x, 1 - precise_new_agent_grid_x), min(precise_new_agent_grid_y, 1 - precise_new_agent_grid_y))
            if (max_angle_distance - min_angle_distance > 1.41421 * self.grid_density):
              self.angular_grid_data[i] = nomal_angle_distance - smallest_distance
              if nomal_angle_distance - smallest_distance <= self.agent_size:  
                  self.agent_plt_object.set_color("#666666")
            else:
                self.angular_grid_data[i] = precise_angle_distance
                if precise_angle_distance <= self.agent_size: 
                   self.agent_plt_object.set_color("#666666")

        self.ag_ax.clear()
        self.ag_ax.plot(self.theta, self.angular_grid_data)
        self.angular_grid.canvas.draw()

        # agent position
        self.agent_x = max(min(self.agent_x , self.grid_length - 0.01), 0)
        self.agent_y = max(min(self.agent_y , self.grid_length - 0.01), 0)
        if self.agent_dir < 0:  self.agent_dir += np.pi * 2
        elif self.agent_dir > np.pi * 2: self.agent_dir -= np.pi * 2
        self.agent_z = self.map_z[int(self.agent_y / self.grid_density) , int(self.agent_x / self.grid_density)] * self.revese_env

        self.agent_plt_object.set_data(
            [self.agent_x],
            [self.agent_y],
        )
        self.agent_plt_object.set_3d_properties([self.agent_z])
        dir_agent_x = math.cos(self.agent_dir) * 0.2 + self.agent_x
        dir_agent_y = math.sin(self.agent_dir) * 0.2 + self.agent_y
        self.agent_dir_plt_object.set_data(
            [dir_agent_x],
            [dir_agent_y],
        )
        self.agent_dir_plt_object.set_3d_properties([self.agent_z])
        
        
        # goal position
        goal_x_ = self.goal_x - self.agent_x
        goal_y_ = self.goal_y - self.agent_y
        goal_spot_x = goal_x_ * math.cos(self.agent_dir) + goal_y_ * math.sin(self.agent_dir)
        goal_spot_y = -1 * goal_x_ * math.sin(self.agent_dir) + goal_y_ * math.cos(self.agent_dir)
        #ic(goal_spot_x, goal_spot_y)

        # 8 sign gradient
        short_distance = 1.5
        sigh_ = 1.0
        sigh_angle = 30.0
        sigh_f = sigh_ * math.cos(sigh_angle * np.pi / 180.0)
        sigh_l = sigh_ * math.sin(sigh_angle * np.pi / 180.0)
        far_distance = 3.0
        front_short_grid_x = self.agent_x + math.cos(self.agent_dir) * short_distance
        front_short_grid_y = self.agent_y + math.sin(self.agent_dir) * short_distance
        front_far_grid_x = self.agent_x + math.cos(self.agent_dir) * far_distance
        front_far_grid_y = self.agent_y + math.sin(self.agent_dir) * far_distance
        front_right_grid_x = self.agent_x + math.cos(self.agent_dir) * sigh_f + math.sin(self.agent_dir) * sigh_l
        front_right_grid_y = self.agent_y + math.sin(self.agent_dir) * sigh_f - math.cos(self.agent_dir) * sigh_l
        front_left_grid_x = self.agent_x + math.cos(self.agent_dir) * sigh_f - math.sin(self.agent_dir) * sigh_l
        front_left_grid_y = self.agent_y + math.sin(self.agent_dir) * sigh_f + math.cos(self.agent_dir) * sigh_l
        
        #front_agent_x = math.cos(self.agent_dir) * self.four_sign_distance + self.agent_x
        #front_agent_y = math.sin(self.agent_dir) * self.four_sign_distance + self.agent_y
        front_short_z = self.map_z[int(front_short_grid_y / self.grid_density) , int(front_short_grid_x / self.grid_density)] * self.revese_env
        front_far_z = self.map_z[int(front_far_grid_y / self.grid_density) , int(front_far_grid_x / self.grid_density)] * self.revese_env
        front_right_z = self.map_z[int(front_right_grid_y / self.grid_density) , int(front_right_grid_x / self.grid_density)] * self.revese_env
        front_left_z = self.map_z[int(front_left_grid_y / self.grid_density) , int(front_left_grid_x / self.grid_density)] * self.revese_env

        self.front_short.set_data(
            [front_short_grid_x],
            [front_short_grid_y],
        )
        self.front_far.set_data(
            [front_far_grid_x],
            [front_far_grid_y],
        )
        self.front_right.set_data(
            [front_right_grid_x],
            [front_right_grid_y],
        )
        self.front_left.set_data(
            [front_left_grid_x],
            [front_left_grid_y],
        )
        self.front_short.set_3d_properties([front_short_z])
        self.front_far.set_3d_properties([front_far_z])
        self.front_right.set_3d_properties([front_right_z])
        self.front_left.set_3d_properties([front_left_z])
        front_distance = front_short_z - self.agent_z
        ic(front_distance)
        grid_x = int(self.agent_x / self.grid_density + 0.5)
        grid_y = int(self.agent_y / self.grid_density + 0.5)
        agent_pitch = self.map_dx(grid_x, grid_y) * math.cos(self.agent_dir) + self.map_dy(grid_x, grid_y) * math.sin(self.agent_dir)
        if agent_pitch < 0:
            # wrong dir on stair
            self.front_short.set_color("#666666")
        else: self.front_short.set_color("deepskyblue")
        self.map_image.canvas.draw()

    def map_dx(self, map_x, map_y):
        if map_x < 1: return -2 * self.revese_env
        elif map_x >= self.map_max_x - 2: return 2 * self.revese_env
        elif map_y >= 0 and map_y < self.map_max_y - 1: return 0.5 * (self.map_z[map_y, map_x + 1] - self.map_z[map_y, map_x - 1]) * self.revese_env
        else: return 0

    def map_dy(self, map_x, map_y):
        if map_y < 1: return -2 * self.revese_env
        elif map_y >= self.map_max_y - 2: return 2 * self.revese_env
        elif map_x >= 0 and map_x < self.map_max_x - 1: return 0.5 * (self.map_z[map_y + 1, map_x] - self.map_z[map_y - 1, map_x]) * self.revese_env
        else: return 0

    def test_launch(self):
        for dir_ in range(360):
            dir = np.pi * 2 * dir_ / 360  
            self.agent_dir = dir

if __name__ == "__main__":
    test_env = Env_test()
    test_env.test_launch()

