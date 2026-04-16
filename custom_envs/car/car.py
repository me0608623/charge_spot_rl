# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import copy
import heapq
import random
#import math
import numpy as np
from gym import spaces
from gym.utils import seeding

from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed
from warp_drive.utils.gpu_environment_context import CUDAEnvironmentContext

_OBSERVATIONS = Constants.OBSERVATIONS
_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS
_LOC_X = "loc_x"
_LOC_Y = "loc_y"
_SP_X = "speed_x"
_SP_Y = "speed_y"
_DIR = "direction"
_ACC_X = "acceleration_x"
_ACC_Y = "acceleration_y"
_SIG = "still_in_the_game"
def point_on_stair(point, mode, origin_r, mid_l, floor_r1, floor_l2, stair_lenght, grid_length_height):
    reg1_ = (np.array(origin_r) - np.array(point)) * (np.array(point) - np.array(mid_l)) # > 0 代表點在裡面 x y 分開

    reg3_ = (np.array(floor_r1) - np.array(point)) * (np.array(point) - np.array(floor_l2))
    zz = 0
    if reg1_[0] > 0 and reg1_[1] > 0:
        if abs(abs(origin_r[0] - mid_l[0]) - stair_lenght) < 0.0001 :
            zz = abs(grid_length_height * (point[0] - origin_r[0]) / stair_lenght)
        elif abs(abs(origin_r[1] - mid_l[1]) - stair_lenght) < 0.0001 :
            zz = abs(grid_length_height * (point[1] - origin_r[1]) / stair_lenght)
        else:
            print("stair array error")
        if mode == 0:
            zz = -1 * zz
    elif reg3_[0] > 0 and reg3_[1] > 0:
        if mode == 0:
            zz = -grid_length_height
        else:
            zz = grid_length_height

    return zz

def point_on_wall(point, segment_start, segment_end):
    """
    判斷一個點是否在一個線段上。
    """
    vec_line = np.array(segment_end) - np.array(segment_start)
    vec1 = np.array(point) - np.array(segment_start)
    vec2 = np.array(point) - np.array(segment_end)
    dot_product = np.dot(vec1, vec_line) / (np.linalg.norm(vec_line)) ** 2
    if dot_product > 0 and dot_product<1:
        dis = np.sqrt((np.linalg.norm(vec1)) ** 2 -(dot_product * np.linalg.norm(vec_line)) ** 2 )
    elif dot_product >= 1:
        dis = np.linalg.norm(vec2)
    else:
        dis = np.linalg.norm(vec1)
    # dis = 牆壁 和 point 之間的最短距離
    if dis < 0.3:
        return True
    return False

def point_on_any_wall(point, segments):
    """
    判斷一個點是否在任何一個線段上。
    """
    for segment in segments:

        if point_on_wall(point, segment[0:2], segment[2:4]):
            return True
    return False

def point_on_any_stair(point, segments, grid_length_height):
    z_ = 0
    for segment in segments:
        z_ = point_on_stair(
            point,
            segment[0],
            segment[1:3],
            segment[3:5],
            segment[5:7],
            segment[7:9],
            segment[9],
            grid_length_height
        )
        if z_ > 0.0 or z_ < 0.0:
            return True, z_
    return False, z_

class Car(CUDAEnvironmentContext):
    """
        real car info 
        fsp = 10
        acc_max: 1.0 m/s fit
        acc_min: -0.5 m/s fit
        vel_tolerance: 0.07 m/s
        turn_max: pi/2.618 radians/sec fit
        omega_tolerance: 0.05 radians/s
        max_speed: 0.6 m/s fit
        min_speed: 0.0 m/s fit
    """
    def __init__(
        self,
        Training = False,
        use_cuda = False,
        PHASE = 1,
        num_spot = 1,
        num_goals = 10,
        num_obstacle = 10,
        num_wall = 3,
        wall_length_half_rand = 3,
        num_stairs = 1,
        stairs_weight_rand = 1,
        stairs_weight_bias = 2,
        floor_width_rand = 2,
        floor_width_bias = 2,
        episode_length = 100,
        grid_length_bias = 10,
        grid_length_rand = 5.0,
        grid_length_height = 2.0,
        grid_density = 0.3,
        seed = None,
        num_obs_see = 10,
        spot_lidar_angle = 4,
        spot_max_speed_x = 0.8, 
        spot_min_speed_x = -0.2,
        spot_max_acceleration_x = 0.2,
        spot_min_acceleration_x = -0.1,
        spot_max_speed_y = 0.3,
        spot_max_acceleration_y = 0.05,
        spot_max_turn_pi_ = 15.0, # max_turn = np.pi / 15
        spot_reward_get_goal = 20.0,
        spot_floor_direction_reward = 0.05,
        base_floor_direction_reward = 0.05,
        spot_cost_operate = 0.05,
        spot_cost_speed_y_rate_with_acc_x  = 0.4,
        spot_cost_turn_rate_with_acc_x = 0.6,
        spot_penalty_hit = -15.0,
        spot_action_levels = 20,
        spot_size_bias = 0.15,
        spot_size_rand = 0.2,
        reward_on_floor = 0.1,
        goals_reward_step = 0.1,
        goals_reward_speed = 0.1,
        goal_size = 0.1,
        goal_speed_rate_with_spot = 0.5,
        obstacle_speed_rate_with_spot = 1,
        Testing_type = 1,
        starting_location_x = None,
        starting_location_y = None,
        starting_directions = None
    ):
        """

        """
        super().__init__()
        self.float_dtype = np.float32
        self.int_dtype = np.int32
        self.eps = self.float_dtype(1e-10)
        self.PHASE = self.float_dtype(PHASE)
        
        self.Training = Training
        self.Training_ = self.float_dtype(0)

        if self.Training == True:
            self.Training_ = self.float_dtype(1)
            assert num_spot > 0
            assert num_obstacle > 0
            assert num_goals > 0
        elif Testing_type == 1:
            if num_spot != 1 or num_goals != 1 :
                print("testing number spot and goal = 1 ")
            num_spot = 1
            num_goals = 1
        elif Testing_type == 2:
            episode_length = 10000
            num_spot = 1
            num_goals = 1
        self.Testing_type = self.float_dtype(Testing_type)
        if 360 % spot_lidar_angle != 0:
            print("spot_lidar_angle slould be 360 common factor .")
        assert 360 % spot_lidar_angle == 0
        self.spot_lidar_angle = self.int_dtype(spot_lidar_angle)
        self.spot_lidar_angle_f = self.float_dtype(spot_lidar_angle)

        self.num_spot = num_spot  
        self.num_obstacle = num_obstacle
        self.num_goals = num_goals
        self.num_agents = self.num_spot + self.num_obstacle + self.num_goals

        assert episode_length > 0
        self.episode_length = episode_length

        # Seeding
        self.np_random = np.random
        if seed is not None:
            self.seed(seed)

        agent_size = self.np_random.random(self.num_agents) * spot_size_rand + spot_size_bias
        grid_length = int(grid_length_rand * random.random()) + grid_length_bias

        # Angent Setting
        no_spot_ = self.np_random.choice(
            np.arange(self.num_agents), self.num_obstacle + self.num_goals, replace = False
        )
        obstacle_ = self.np_random.choice(
            no_spot_, self.num_obstacle, replace = False
        )

        self.agent_type = {}
        self.spots = {}
        self.obstacles = {}
        self.goals = {}

        for agent_id in range(self.num_agents):
            if agent_id in set(no_spot_):
                if agent_id in set(obstacle_):
                    self.agent_type[agent_id] = 2  # obstacles
                    self.obstacles[agent_id] = True
                else:
                    self.agent_type[agent_id] = 0  # goals
                    self.goals[agent_id] = True
                    agent_size[agent_id] = goal_size
            else:
                self.agent_type[agent_id] = 1  # spots
                self.spots[agent_id] = True

        # Square 2D grid
        assert grid_length > 0
        self.grid_length = self.float_dtype(grid_length)
        self.grid_diagonal = self.grid_length * np.sqrt(2)
        self.grid_length_height = self.float_dtype(grid_length_height)
        print("grid_length = ",grid_length,"\n")
        self.agent_size = self.float_dtype(agent_size)

        if starting_location_x is None:
            assert starting_location_y is None

            starting_location_x = self.grid_length * self.np_random.random(
                self.num_agents
            )
            starting_location_y = self.grid_length * self.np_random.random(
                self.num_agents
            )
        else:
            assert len(starting_location_x) == self.num_agents
            assert len(starting_location_y) == self.num_agents
 
        self.starting_location_x = starting_location_x
        self.starting_location_y = starting_location_y

        if starting_directions is None:
            starting_directions = self.np_random.choice(
                [0, np.pi / 2, np.pi, np.pi * 3 / 2], self.num_agents, replace=True
            )
        else:
            assert len(starting_directions) == self.num_agents
        self.starting_directions = starting_directions

        # Set the speed limit
        self.spot_max_speed_x = self.float_dtype(spot_max_speed_x)
        self.spot_min_speed_x = self.float_dtype(spot_min_speed_x)
        self.spot_max_speed_y = self.float_dtype(spot_max_speed_y)
        self.spot_min_speed_y = self.float_dtype(spot_max_speed_y * -1)

        # Set the acceleration limit
        self.spot_max_acceleration_x = self.float_dtype(spot_max_acceleration_x)
        self.spot_min_acceleration_x = self.float_dtype(spot_min_acceleration_x)
        self.spot_max_acceleration_y = self.float_dtype(spot_max_acceleration_y)
        self.spot_min_acceleration_y = self.float_dtype(spot_max_acceleration_y * -1)

        # Set the turn limit
        self.spot_max_turn = self.float_dtype(np.pi / spot_max_turn_pi_)
        self.spot_min_turn = self.float_dtype(-1 * np.pi / spot_max_turn_pi_)

        # Set the start speed and acceleration
        self.starting_speeds_x = np.zeros(self.num_agents, dtype=self.float_dtype)
        self.starting_accelerations_x = np.zeros(self.num_agents, dtype=self.float_dtype)
        self.starting_speeds_y = np.zeros(self.num_agents, dtype=self.float_dtype)
        self.starting_accelerations_y = np.zeros(self.num_agents, dtype=self.float_dtype)

        # Set Action
        assert spot_action_levels >= 0
        self.spot_action_levels = spot_action_levels

        # Acceleration actions
        self.acceleration_actions_x = np.linspace(
            self.spot_min_acceleration_x, self.spot_max_acceleration_x, self.spot_action_levels
        )
        self.acceleration_actions_y = np.linspace(
            self.spot_min_acceleration_y, self.spot_max_acceleration_y, self.spot_action_levels
        )
        # Add acceleration_actions = 0
        self.acceleration_actions_x = np.insert(self.acceleration_actions_x, 0, 0).astype(
            self.float_dtype
        )
        self.acceleration_actions_y = np.insert(self.acceleration_actions_y, 0, 0).astype(
            self.float_dtype
        )
        self.skill_levels = [self.float_dtype(goal_speed_rate_with_spot) if self.agent_type[agent_id] == 0 else self.float_dtype(1.0) if self.agent_type[agent_id] == 1 else self.float_dtype(obstacle_speed_rate_with_spot) for agent_id in range(self.num_agents)]

        # Turn actions
        self.turn_actions = np.linspace(
            self.spot_min_turn, self.spot_max_turn, self.spot_action_levels
        )
        # Add turn_actions = 0
        self.turn_actions = np.insert(self.turn_actions, 0, 0).astype(self.float_dtype)

        # These will be set during reset (see below)
        self.timestep = None
        self.global_state = None

        # Defining observation and action spaces
        self.observation_space = None  # Note: this will be set via the env_wrapper
        self.action_space = {
            agent_id: spaces.MultiDiscrete(
                (
                    len(self.acceleration_actions_x),
                    len(self.acceleration_actions_y),
                    len(self.turn_actions)
                )
            )
            for agent_id in range(self.num_agents)
        }

        self.init_obs = None  # Will be set later in generate_observation()

        assert num_obs_see <= self.num_agents
        self.num_obs_see = num_obs_see
        self.num_obs_see_ = self.float_dtype(num_obs_see)

        # Rewards and penalties
        """
        display_reward = 
            goal(5) dynamic_Obstacle Obstacle SPOT Step Move Floor
            spot(3) dynamic_Obstacle Obstacle Goal dynamic floor
            一半紀錄
        """
        display_reward = np.zeros(13 * self.num_agents, dtype=self.float_dtype)
        self.display_reward = self.float_dtype(display_reward)

        # Spot
        self.spot_penalty_hit = self.float_dtype(spot_penalty_hit)
        self.spot_reward_get_goal = self.float_dtype(spot_reward_get_goal)
        self.spot_floor_direction_reward = self.float_dtype(spot_floor_direction_reward)
        self.base_floor_direction_reward = self.float_dtype(base_floor_direction_reward)
        
        self.spot_cost_operate = self.float_dtype(spot_cost_operate)
        self.spot_cost_speed_y_rate_with_acc_x = self.float_dtype(spot_cost_speed_y_rate_with_acc_x)
        self.spot_cost_turn_rate_with_acc_x = self.float_dtype(spot_cost_turn_rate_with_acc_x)

        # Goal
        self.goals_reward_step = self.float_dtype(goals_reward_step)
        self.goals_reward_speed = self.float_dtype(goals_reward_speed)
        self.reward_on_floor = self.float_dtype(reward_on_floor)
        
        self.still_in_the_game = None
        self.use_cuda = use_cuda
        self.goal_at_reset = copy.deepcopy(self.goals)
        self.spots_at_reset = copy.deepcopy(self.spots)

        # 3D Map
        wall_reg_ = np.random.rand(num_wall)
        wall_angle = np.arccos(2 * wall_reg_ - 1) * 180 / np.pi # 0 - 180
        wall_length_half = np.random.rand(num_wall) * wall_length_half_rand
        wall_distance_from_origin_x = np.random.rand(num_wall) * (grid_length - 2 * wall_length_half) + wall_length_half
        wall_distance_from_origin_y = np.random.rand(num_wall) * (grid_length - 2 * wall_length_half) + wall_length_half

        # one map one stairs

        stairs_weight = stairs_weight_bias + random.random() * stairs_weight_rand
        stairs_length = self.grid_length_height * 1.732 + np.random.rand(num_stairs)
        floor_width = floor_width_bias * np.ones(num_stairs) + floor_width_rand * random.random()
        stairs_distance_from_origin_x = np.random.rand(num_stairs) * (grid_length - 2 * stairs_length) + stairs_length
        stairs_distance_from_origin_y = np.random.rand(num_stairs) * (grid_length - 2 * stairs_length) + stairs_length
        stairs_up_or_down = ((8 * np.random.rand(num_stairs)).astype(int)) # mod(2)= down , up . mod(4)=angle_ up , down , left , right

        wall_start_x = wall_distance_from_origin_x + wall_length_half * np.cos(wall_angle)
        wall_start_y = wall_distance_from_origin_y + wall_length_half * np.sin(wall_angle)
        wall_end_x = wall_distance_from_origin_x - wall_length_half * np.cos(wall_angle)
        wall_end_y = wall_distance_from_origin_y - wall_length_half * np.sin(wall_angle)

        stairs_mode = stairs_up_or_down % 2

        reg_stair_ = stairs_up_or_down // 2
        reg_stair = reg_stair_ % 2
        right_left_stair = reg_stair_ // 2
        up_down_stair = [1 if x==1 else -1 for x in reg_stair]


        stairs_origin_r_x = stairs_distance_from_origin_x - 0.5 * stairs_weight * right_left_stair
        stairs_origin_r_y = stairs_distance_from_origin_y - 0.5 * stairs_weight * (1 - right_left_stair)
        stairs_mid_l_x = stairs_distance_from_origin_x + 0.5 * stairs_weight * right_left_stair + stairs_length * up_down_stair * (1 - right_left_stair)
        stairs_mid_l_y = stairs_distance_from_origin_y + 0.5 * stairs_weight * (1 - right_left_stair) + stairs_length * up_down_stair * right_left_stair

        stairs_floor_r1_x = stairs_distance_from_origin_x - 0.5 * floor_width * right_left_stair + stairs_length * up_down_stair * (1 - right_left_stair)
        stairs_floor_r1_y = stairs_distance_from_origin_y - 0.5 * floor_width * (1 - right_left_stair) + stairs_length * up_down_stair * right_left_stair
        stairs_floor_l2_x = stairs_distance_from_origin_x + 0.5 * floor_width * right_left_stair + (stairs_length + floor_width) * up_down_stair * (1 - right_left_stair)
        stairs_floor_l2_y = stairs_distance_from_origin_y + 0.5 * floor_width * (1 - right_left_stair) + (stairs_length + floor_width) * up_down_stair * right_left_stair

        wall_list = [] # [ wall_start_x, wall_start_y, wall_end_x, wall_end_y ]
        stair_list = [] # [stairs_mode, 6 point x, y]

        for num_wall in  range(num_wall):
            wall_sub_list = []
            wall_sub_list.append(wall_start_x[num_wall])
            wall_sub_list.append(wall_start_y[num_wall])
            wall_sub_list.append(wall_end_x[num_wall])
            wall_sub_list.append(wall_end_y[num_wall])
            wall_list.append(wall_sub_list)

        for num_stair in  range(num_stairs):
            stair_sub_list = []
            stair_sub_list.append(stairs_mode[num_stair])
            stair_sub_list.append(stairs_origin_r_x[num_stair])
            stair_sub_list.append(stairs_origin_r_y[num_stair])
            stair_sub_list.append(stairs_mid_l_x[num_stair])
            stair_sub_list.append(stairs_mid_l_y[num_stair])
            stair_sub_list.append(stairs_floor_r1_x[num_stair])
            stair_sub_list.append(stairs_floor_r1_y[num_stair])
            stair_sub_list.append(stairs_floor_l2_x[num_stair])
            stair_sub_list.append(stairs_floor_l2_y[num_stair])
            stair_sub_list.append(stairs_length[num_stair])
            stair_list.append(stair_sub_list)

        # 產生 3D 地圖圖層
        x_range = [0, grid_length]
        y_range = [0, grid_length]
        x = np.arange(x_range[0], x_range[1] + 1, grid_density)
        y = np.arange(y_range[0], y_range[1] + 1, grid_density)
        xx, yy = np.meshgrid(x, y)
        len_x = len(x)
        len_y = len(y)
        self.xx = xx
        self.yy = yy
        self.grid_map_z = np.zeros_like(xx)
        self.grid_map_on_stair = np.zeros((len_x * len_y), dtype=self.float_dtype)
        self.grid_density = grid_density
        self.max_i = len_x
        self.max_j = len_y
        
        # grip map
        for i in range(len(x)):
            for j in range(len(y)):
                point = np.array([i*grid_density , j*grid_density])
                on_stair ,z_ = point_on_any_stair(point, stair_list, grid_length_height)
                if on_stair:
                    self.grid_map_z[i,j] = self.float_dtype(z_)
                elif point_on_any_wall(point, wall_list):
                    self.grid_map_z[i,j] = self.float_dtype(grid_length_height)
        
        self.grid_map_z_ = self.grid_map_z.reshape((len_x * len_y))

        # 計算 3D 地圖每個點每個角度看出去的距離
        self.different_point_angle_distance = np.zeros((int(len_x * len_y * 360 / self.spot_lidar_angle)), dtype=self.float_dtype)
        for i in range(len_x):
            for j in range(len_y):
                this_point_origin_z = self.grid_map_z[i,j]
                self.cheak_point_distance(this_point_origin_z, i, j)
                diff_i = abs(self.grid_map_z[min(i + 1, len_x - 1), j] - self.grid_map_z[i, j])
                diff_j = abs(self.grid_map_z[i, min(j + 1, len_y - 1)] - self.grid_map_z[i, j])
                if (diff_i > 0 and diff_i < 1) or (diff_j > 0 and diff_j < 1):
                    self.grid_map_on_stair[self.max_j * i + j] = self.float_dtype(1.0)
                else:
                    self.grid_map_on_stair[self.max_j * i + j] = self.float_dtype(0.0)
        num_angle_lidar = int(360 / self.spot_lidar_angle)
        last_different_point_angle_distance_id = int(num_angle_lidar * self.max_j * (self.max_j - 1) + num_angle_lidar * (self.max_j - 1) + num_angle_lidar - 1)
        #self.different_point_angle_distance[last_different_point_angle_distance_id] = self.different_point_angle_distance[last_different_point_angle_distance_id - 100] 
        """
            output self.different_point_angle_distance run about 2 min
        """
    name = "Car"

    def map_dx(self, map_x, map_y):
        if map_x < 1: return -1 * self.grid_length_height
        elif map_x >= self.max_j - 1: return self.grid_length_height
        elif map_y > 0 and map_y < self.max_j: return 0.5 * (self.grid_map_z[map_x + 1, map_y] - self.grid_map_z[map_x - 1, map_y])
        else: return 0

    def map_dy(self, map_x, map_y):
        if map_y < 1: return -1 * self.grid_length_height
        elif map_y >= self.max_j - 1: return self.grid_length_height
        elif map_x > 0 and map_x < self.max_j: return 0.5 * (self.grid_map_z[map_x, map_y + 1] - self.grid_map_z[map_x, map_y - 1])
        else: return 0

    def cheak_point_distance(self, this_point_origin_z, point_i, point_j):
        num_angle_lidar = int(360 / self.spot_lidar_angle)
        len_grid_precision = 0.2
        max_distance = int(self.grid_length / len_grid_precision + 1)
        for k in range(num_angle_lidar):
            len = 0
            on_stair_len = 0
            last_z = this_point_origin_z
            for len_ in range(max_distance):
                len = len_ * len_grid_precision + len_grid_precision
                x_ = int(point_i + np.cos(np.radians(k * self.spot_lidar_angle)) * len)
                y_ = int(point_j + np.sin(np.radians(k * self.spot_lidar_angle)) * len)
                if x_ > self.max_i - 1 or x_ < 0 or y_ > self.max_j - 1 or y_ < 0:
                    break
                if abs(last_z - self.grid_map_z[x_,y_]) > 0.3:
                    break
                #if abs(last_z - this_point_origin_z) > 0.3:
                    # 不同樓層的距離比較短
                    #on_stair_len += len_grid_precision
                last_z = self.grid_map_z[x_,y_]
            self.different_point_angle_distance[num_angle_lidar * self.max_j * point_i + num_angle_lidar * point_j + k] =  self.float_dtype((len - len_grid_precision) * self.grid_density - 0.5 * (on_stair_len) * self.grid_density)
        return 0

    def seed(self, seed=None):
        """
        Seeding the environment with a desired seed
        Note: this uses the code in
        https://github.com/openai/gym/blob/master/gym/utils/seeding.py
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_global_state(self, key=None, value=None, t=None, dtype=None):
        """
        Set the global state for a specified key, value and timestep.
        Note: for a new key, initialize global state to all zeros.
        """
        assert key is not None
        if dtype is None:
            dtype = self.float_dtype

        # If no values are passed, set everything to zeros.
        if key not in self.global_state:
            self.global_state[key] = np.zeros(
                (self.episode_length + 1, self.num_agents), dtype=dtype
            )

        if t is not None and value is not None:
            assert isinstance(value, np.ndarray)
            assert value.shape[0] == self.global_state[key].shape[1]

            self.global_state[key][t] = value

    def update_state(self, delta_accelerations_x, delta_accelerations_y, delta_turns):
        """
        """
        loc_x_prev_t = self.global_state[_LOC_X][self.timestep - 1]
        loc_y_prev_t = self.global_state[_LOC_Y][self.timestep - 1]
        speed_prev_t_x = self.global_state[_SP_X][self.timestep - 1]
        speed_prev_t_y = self.global_state[_SP_Y][self.timestep - 1]
        dir_prev_t = self.global_state[_DIR][self.timestep - 1]
        acc_prev_t_x = self.global_state[_ACC_X][self.timestep - 1]
        acc_prev_t_y = self.global_state[_ACC_Y][self.timestep - 1]
        still_in_game_prev_t = self.global_state[_SIG][self.timestep - 1]
        # Update direction and acceleration
        # Do not update location if agent is out of the game !
        #print(delta_accelerations)
        dir_curr_t = (
            (dir_prev_t + delta_turns) % (2 * np.pi) * self.still_in_the_game
        ).astype(self.float_dtype)

        acc_curr_t_x = acc_prev_t_x + delta_accelerations_x
        acc_curr_t_y = acc_prev_t_y + delta_accelerations_y

        # 0 <= speed <= max_speed (multiplied by the skill levels).
        # Reset acceleration to 0 when speed is outside this range
        max_speed_x = self.spot_max_speed_x * np.array(self.skill_levels)
        min_speed_x = self.spot_min_speed_x * np.array(self.skill_levels)
        max_speed_y = self.spot_max_speed_y * np.array(self.skill_levels)
        min_speed_y = self.spot_min_speed_y * np.array(self.skill_levels)

        # x dim speed and acceleration 
        speed_curr_t_x = self.float_dtype(
            np.clip(speed_prev_t_x + acc_curr_t_x, min_speed_x, max_speed_x) * self.still_in_the_game
        )
        speed_curr_t_y = self.float_dtype(
            np.clip(speed_prev_t_y + acc_curr_t_y, min_speed_y, max_speed_y) * self.still_in_the_game
        )
        acc_curr_t_x = acc_curr_t_x * (speed_curr_t_x > min_speed_x) * (speed_curr_t_x < max_speed_x)

        # y dim speed and acceleration 
        for agent_id in range(self.num_agents):
            if self.agent_type[agent_id] == 1:
                speed_curr_t_y[agent_id] = self.float_dtype(
                    np.clip(speed_prev_t_y[agent_id] + acc_curr_t_y[agent_id], min_speed_y[agent_id], max_speed_y[agent_id]) * self.still_in_the_game[agent_id]
                )
                acc_curr_t_y[agent_id] = acc_curr_t_y[agent_id] * (speed_curr_t_y[agent_id] > min_speed_y[agent_id]) * (speed_curr_t_y[agent_id] < max_speed_y[agent_id])
            else:
                speed_curr_t_y[agent_id] = self.float_dtype(0.0)
                acc_curr_t_y[agent_id] = self.float_dtype(0.0)

        loc_x_curr_t = self.float_dtype(
            loc_x_prev_t + speed_curr_t_x * np.cos(dir_curr_t) - speed_curr_t_y * np.sin(dir_curr_t)
        )
        loc_y_curr_t = self.float_dtype(
            loc_y_prev_t + speed_curr_t_x * np.sin(dir_curr_t) + speed_curr_t_y * np.cos(dir_curr_t)
        )

        if self.Training == False:
            for agent_id in range(self.num_agents):
                if self.agent_type[agent_id] == 1:
                    loc_x_curr_t[agent_id] = self.float_dtype(loc_x_prev_t[agent_id])
                    loc_y_curr_t[agent_id] = self.float_dtype(loc_y_prev_t[agent_id])
                    if self.Testing_type == 1:
                        if still_in_game_prev_t[agent_id] == 0:
                            still_in_game_prev_t[agent_id] == 1
                            loc_x_curr_t[agent_id] = self.float_dtype(self.grid_length * random.random())
                            loc_y_curr_t[agent_id] = self.float_dtype(self.grid_length * random.random())

        # Clip x and y if agent has crossed edge
        clipped_loc_x_curr_t = self.float_dtype(
            np.clip(loc_x_curr_t, 0.0, self.grid_length)
        )

        clipped_loc_y_curr_t = self.float_dtype(
            np.clip(loc_y_curr_t, 0.0, self.grid_length)
        )

        # Set global states
        self.set_global_state(key = _LOC_X, value = clipped_loc_x_curr_t, t=self.timestep)
        self.set_global_state(key = _LOC_Y, value = clipped_loc_y_curr_t, t=self.timestep)
        self.set_global_state(key = _SP_X, value = speed_curr_t_x, t=self.timestep)
        self.set_global_state(key = _SP_Y, value = speed_curr_t_y, t=self.timestep)
        self.set_global_state(key = _DIR, value = dir_curr_t, t=self.timestep)
        self.set_global_state(key = _ACC_X, value = acc_curr_t_x, t=self.timestep)
        self.set_global_state(key = _ACC_Y, value = acc_curr_t_y, t=self.timestep)
 
    def compute_distance(self, agent1, agent2):
        """
        Note: 'compute_distance' is only used when running on CPU step() only.
        When using the CUDA step function, this Python method (compute_distance)
        is also part of the step() function!
        """
        return np.sqrt(
            (
                self.global_state[_LOC_X][self.timestep, agent1]
                - self.global_state[_LOC_X][self.timestep, agent2]
            )
            ** 2
            + (
                self.global_state[_LOC_Y][self.timestep, agent1]
                - self.global_state[_LOC_Y][self.timestep, agent2]
            )
            ** 2
            - self.agent_size[agent1] - self.agent_size[agent2]
        ).astype(self.float_dtype)
    
    def compute_hit_wall(self, agent_x, agent_y, agent_size):
        """
        沒寫上樓梯的角度
        """

        agent_grid_x = agent_x / self.grid_density
        agent_grid_y = agent_y / self.grid_density
        for grid_map_x in range(len(self.grid_map_z)):
            for grid_map_y in range(len(self.grid_map_z[grid_map_x])):
                if abs(np.sqrt((agent_grid_x - grid_map_x) ** 2 + (agent_grid_y - grid_map_y) ** 2) - agent_size / self.grid_density) < 0.5:
                    if abs(self.grid_map_z[grid_map_x, agent_grid_y] - self.grid_map_z[grid_map_x + 1, agent_grid_y]) > self.grid_density * 1.2 / 1.732:  return 1
                    if abs(self.grid_map_z[grid_map_x, agent_grid_y] - self.grid_map_z[grid_map_x, agent_grid_y + 1]) > self.grid_density * 1.2 / 1.732:  return 1
        return 0

    
    def compute_hit_wall_angle_base(self, agent_x, agent_y, agent_size):
        """
        observation 看不出是否上樓梯，所以加一個是否上在樓梯
        """

        left_new_agent_grid_x = int(agent_x / self.grid_density)
        down_new_agent_grid_y = int(agent_y / self.grid_density)
        right_new_agent_grid_x = min(left_new_agent_grid_x + 1, self.max_j - 1)
        up_new_agent_grid_y = min(down_new_agent_grid_y + 1, self.max_j - 1)
        precise_new_agent_grid_x = float(agent_x / self.grid_density) - left_new_agent_grid_x
        precise_new_agent_grid_y = float(agent_y / self.grid_density) - down_new_agent_grid_y
        num_angle_lidar = 360 / self.spot_lidar_angle
        for angle in range(num_angle_lidar):
            left_down_angle_distance = self.different_point_angle_distance[num_angle_lidar * self.max_j * left_new_agent_grid_x + num_angle_lidar * down_new_agent_grid_y + angle]
            left_up_angle_distance = self.different_point_angle_distance[num_angle_lidar * self.max_j * left_new_agent_grid_x + num_angle_lidar * up_new_agent_grid_y + angle]
            right_down_angle_distance = self.different_point_angle_distance[num_angle_lidar * self.max_j * right_new_agent_grid_x + num_angle_lidar * down_new_agent_grid_y + angle]
            right_up_angle_distance = self.different_point_angle_distance[num_angle_lidar * self.max_j * right_new_agent_grid_x + num_angle_lidar * up_new_agent_grid_y + angle]
            precise_angle_distance = precise_new_agent_grid_x * precise_new_agent_grid_y * right_up_angle_distance + (1 - precise_new_agent_grid_x) * precise_new_agent_grid_y * left_up_angle_distance + precise_new_agent_grid_x * (1 - precise_new_agent_grid_y) * right_down_angle_distance + (1 - precise_new_agent_grid_x) * (1 - precise_new_agent_grid_y) * left_down_angle_distance
            max_angle_distance = max(max(left_down_angle_distance, left_up_angle_distance), max(right_down_angle_distance, right_up_angle_distance))
            min_angle_distance = min(min(left_down_angle_distance, left_up_angle_distance), min(right_down_angle_distance, right_up_angle_distance))
            smallest_distance = min(min(precise_new_agent_grid_x, 1 - precise_new_agent_grid_x), min(precise_new_agent_grid_y, 1 - precise_new_agent_grid_y))
            if max_angle_distance -  min_angle_distance > 1.415 * self.grid_density:
                if min_angle_distance - smallest_distance < agent_size:
                    return 1
            elif precise_angle_distance < agent_size:
                return 1
        return 0

    def k_nearest_neighbors(self, agent_id, k):
        """
        Note: 'k_nearest_neighbors' is only used when running on CPU step() only.
        When using the CUDA step function, this Python method (k_nearest_neighbors)
        is also part of the step() function!
        """
        agent_ids_and_distances = []

        for ag_id in range(self.num_agents):
            if (ag_id != agent_id) and (self.still_in_the_game[ag_id]):
                agent_ids_and_distances += [
                    (ag_id, self.compute_distance(agent_id, ag_id))
                ]
        k_nearest_neighbor_ids_and_distances = heapq.nsmallest(
            k, agent_ids_and_distances, key=lambda x: x[1]
        )

        return [
            item[0]
            for item in k_nearest_neighbor_ids_and_distances[
                : self.num_obs_see
            ]
        ]

    def generate_observation(self):
        """
        Generate and return the observations for every agent.
        """
        obs = {}

        normalized_global_obs = None
        for feature in [
            (_LOC_X, 1),
            (_LOC_Y, 1),
            (_SP_X, self.spot_max_speed_x + self.eps),
            (_SP_Y, self.spot_max_speed_y + self.eps),
            (_DIR, 2 * np.pi),
        ]:
            if normalized_global_obs is None:
                normalized_global_obs = (
                    self.global_state[feature[0]][self.timestep] / feature[1]
                )
            else:
                normalized_global_obs = np.vstack(
                    (
                        normalized_global_obs,
                        self.global_state[feature[0]][self.timestep] / feature[1],
                    )
                )
        agent_types = np.array(
            [self.agent_type[agent_id] for agent_id in range(self.num_agents)]
        )
        time = np.array([float(self.timestep) / self.episode_length])

        for agent_id in range(self.num_agents):
            if self.timestep == 0:
                # Set obs to all zeros
                obs_global_states = np.zeros(
                    (
                        normalized_global_obs.shape[0],
                        self.num_obs_see,
                    )
                )
                obs_agent_types = np.zeros(self.num_obs_see, dtype=self.float_dtype)
                obs_still_in_the_game = np.zeros(self.num_obs_see, dtype=self.float_dtype)
                obs_agent_size = np.zeros(self.num_obs_see, dtype=self.float_dtype)
                obs_local_angle = np.zeros(int(360 / self.spot_lidar_angle), dtype=self.float_dtype)
                self.init_obs = np.concatenate(
                    [
                        np.array([0.0]),  # acceleration x
                        np.array([0.0]),  # acceleration y
                        np.array([0.0]),  # speed x
                        np.array([0.0]),  # speed y
                        np.array([0.0]),  # dir
                        np.array([0.0]),  # size
                        np.array([0.0]),  # goal x
                        np.array([0.0]),  # goal y
                        np.array([0.0]),  # goal z
                        np.array([0.0]),  # agent x
                        np.array([0.0]),  # agent y
                        np.array([0.0]),  # front_pitch
                        np.array([0.0]),  # front_row
                        np.array([0.0]),  # back_pitch
                        np.array([0.0]),  # back_row
                        np.array([0.0]),  # left_pitch
                        np.array([0.0]),  # left_row
                        np.array([0.0]),  # right_pitch
                        np.array([0.0]),  # right_row
                        np.array([0.0]),  # time
                        np.vstack(
                            (
                                obs_global_states,
                                obs_agent_types,
                                obs_still_in_the_game,
                                obs_agent_size
                            )
                        ).reshape(-1),
                        obs_local_angle.reshape(-1),
                    ]
                )


            # Initialize obs to all zeros
            obs[agent_id] = self.init_obs
            
            # Set obs for agents still in the game
            if self.still_in_the_game[agent_id]:
                nearest_neighbor_ids = self.k_nearest_neighbors(
                    agent_id, k = self.num_obs_see
                )
                normalized_global_obs_ = copy.deepcopy(normalized_global_obs)
                normalized_global_obs_[0,nearest_neighbor_ids] = normalized_global_obs[0, nearest_neighbor_ids] - int(normalized_global_obs[0, agent_id])
                normalized_global_obs_[1,nearest_neighbor_ids] = normalized_global_obs[1, nearest_neighbor_ids] - int(normalized_global_obs[1, agent_id])

                obs_global_states = np.hstack(
                    (
                        normalized_global_obs_[:, nearest_neighbor_ids],
                        np.zeros(
                            (
                                normalized_global_obs.shape[0],
                                self.num_obs_see
                                - len(nearest_neighbor_ids),
                            )
                        ),
                    )
                )


                obs_type_agent =  np.array([0 if agent_types[k] == 0 else 1 for k in nearest_neighbor_ids])
                obs_size_agent = np.array([self.agent_size[k] + self.agent_size[agent_id] for k in nearest_neighbor_ids])

                num_angle_lidar = int(360 / self.spot_lidar_angle)
                angle_ = int(0.5 + self.global_state[_DIR][self.timestep][agent_id] * 360 / (2 * np.pi) / self.spot_lidar_angle)
                left_new_agent_grid_x = int(self.global_state[_LOC_X][self.timestep][agent_id] / self.grid_density)
                down_new_agent_grid_y = int(self.global_state[_LOC_Y][self.timestep][agent_id] / self.grid_density)
                right_new_agent_grid_x = min(left_new_agent_grid_x + 1, self.max_j - 1)
                up_new_agent_grid_y = min(down_new_agent_grid_y + 1, self.max_j - 1)
                precise_new_agent_grid_x = float(self.global_state[_LOC_X][self.timestep][agent_id] / self.grid_density) - left_new_agent_grid_x
                precise_new_agent_grid_y = float(self.global_state[_LOC_Y][self.timestep][agent_id] / self.grid_density) - down_new_agent_grid_y
                obs_local_angle = np.zeros(int(360 / self.spot_lidar_angle), dtype=self.float_dtype)
                for k in range(num_angle_lidar):
                    real_angle = k + angle_
                    if real_angle >= num_angle_lidar: real_angle = real_angle - num_angle_lidar
                    if real_angle < 0: real_angle = real_angle + num_angle_lidar
                    left_down_angle_distance = self.different_point_angle_distance[num_angle_lidar * self.max_j * left_new_agent_grid_x + num_angle_lidar * down_new_agent_grid_y + real_angle]
                    left_up_angle_distance = self.different_point_angle_distance[num_angle_lidar * self.max_j * left_new_agent_grid_x + num_angle_lidar * up_new_agent_grid_y + real_angle]
                    right_down_angle_distance = self.different_point_angle_distance[num_angle_lidar * self.max_j * right_new_agent_grid_x + num_angle_lidar * down_new_agent_grid_y + real_angle]
                    right_up_angle_distance = self.different_point_angle_distance[num_angle_lidar * self.max_j * right_new_agent_grid_x + num_angle_lidar * up_new_agent_grid_y + real_angle]
                    precise_angle_distance = precise_new_agent_grid_x * precise_new_agent_grid_y * right_up_angle_distance + (1 - precise_new_agent_grid_x) * precise_new_agent_grid_y * left_up_angle_distance + precise_new_agent_grid_x * (1 - precise_new_agent_grid_y) * right_down_angle_distance + (1 - precise_new_agent_grid_x) * (1 - precise_new_agent_grid_y) * left_down_angle_distance
                    max_angle_distance = max(max(left_down_angle_distance, left_up_angle_distance), max(right_down_angle_distance, right_up_angle_distance))
                    min_angle_distance = min(min(left_down_angle_distance, left_up_angle_distance), min(right_down_angle_distance, right_up_angle_distance))
                    smallest_distance = min(min(precise_new_agent_grid_x, 1 - precise_new_agent_grid_x), min(precise_new_agent_grid_y, 1 - precise_new_agent_grid_y))
                    if (max_angle_distance - min_angle_distance > 1.415 * self.grid_density): obs_local_angle[k] = min_angle_distance - smallest_distance
                    else: obs_local_angle[k] = precise_angle_distance

                still_in_the_game_ = np.array([self.still_in_the_game[k] for k in nearest_neighbor_ids])
                dir_agent = self.global_state[_DIR][self.timestep][agent_id]
                dir_dx = np.cos(dir_agent) 
                dir_dy = np.sin(dir_agent)
                front_agent_grid_x = int((self.global_state[_LOC_X][self.timestep][agent_id] + dir_dx) / self.grid_density + 0.5)
                front_agent_grid_y = int((self.global_state[_LOC_Y][self.timestep][agent_id] + dir_dy) / self.grid_density + 0.5) 
                back_agent_grid_x = int((self.global_state[_LOC_X][self.timestep][agent_id] - dir_dx) / self.grid_density + 0.5)
                back_agent_grid_y = int((self.global_state[_LOC_Y][self.timestep][agent_id] - dir_dy) / self.grid_density + 0.5) 
                
                left_agent_grid_x = int((self.global_state[_LOC_X][self.timestep][agent_id] - dir_dy) / self.grid_density + 0.5)
                left_agent_grid_y = int((self.global_state[_LOC_Y][self.timestep][agent_id] + dir_dx) / self.grid_density + 0.5)
                right_agent_grid_x = int((self.global_state[_LOC_X][self.timestep][agent_id] + dir_dy) / self.grid_density + 0.5)
                right_agent_grid_y = int((self.global_state[_LOC_Y][self.timestep][agent_id] - dir_dx) / self.grid_density + 0.5)

                front_pitch = self.map_dx(front_agent_grid_x, front_agent_grid_y) * dir_dx + self.map_dy(front_agent_grid_x, front_agent_grid_y) * dir_dy
                front_row = self.map_dx(front_agent_grid_x, front_agent_grid_y) * dir_dy - self.map_dy(front_agent_grid_x, front_agent_grid_y) * dir_dx
                back_pitch = - self.map_dx(back_agent_grid_x, back_agent_grid_y) * dir_dx - self.map_dy(back_agent_grid_x, back_agent_grid_y) * dir_dy
                back_row = - self.map_dx(back_agent_grid_x, back_agent_grid_y) * dir_dy + self.map_dy(back_agent_grid_x, back_agent_grid_y) * dir_dx
                
                left_pitch = self.map_dy(left_agent_grid_x, left_agent_grid_y) * dir_dx - self.map_dx(left_agent_grid_x, left_agent_grid_y) * dir_dy
                left_row = self.map_dy(left_agent_grid_x, left_agent_grid_y) * dir_dy + self.map_dx(left_agent_grid_x, left_agent_grid_y) * dir_dx
                right_pitch = - self.map_dy(right_agent_grid_x, right_agent_grid_y) * dir_dx + self.map_dx(right_agent_grid_x, right_agent_grid_y) * dir_dy
                right_row = - self.map_dy(right_agent_grid_x, right_agent_grid_y) * dir_dy - self.map_dx(right_agent_grid_x, right_agent_grid_y) * dir_dx
                
                if agent_types[agent_id] != 1 :
                    fix_goal_x = np.array([0.0])
                    fix_goal_y = np.array([0.0])
                    fix_goal_z = np.array([0.0])
                else:
                    agent_map_x = int(self.global_state[_LOC_X][self.timestep][agent_id] / self.grid_density + 0.5)
                    agent_map_y = int(self.global_state[_LOC_Y][self.timestep][agent_id] / self.grid_density + 0.5)
                    agent_map_z = self.grid_map_z[agent_map_x, agent_map_y]
                    for k in nearest_neighbor_ids:
                        if agent_types[k] == 0:
                            goal_x_ = self.global_state[_LOC_X][self.timestep][k] - self.global_state[_LOC_X][self.timestep][agent_id]
                            goal_y_ = self.global_state[_LOC_Y][self.timestep][k] - self.global_state[_LOC_Y][self.timestep][agent_id]
                            other_map_x = int(self.global_state[_LOC_X][self.timestep][k] / self.grid_density + 0.5)
                            other_map_y = int(self.global_state[_LOC_Y][self.timestep][k] / self.grid_density + 0.5)
                            other_map_z = self.grid_map_z[other_map_x, other_map_y]
                            goal_z_ = other_map_z - agent_map_z
                            fix_goal_x = np.array([goal_x_])
                            fix_goal_y = np.array([goal_y_])
                            fix_goal_z = np.array([goal_z_])
                            break
                        elif k == nearest_neighbor_ids[len(nearest_neighbor_ids) - 1]:
                            fix_goal_x = np.array([0.0])
                            fix_goal_y = np.array([0.0])
                            fix_goal_z = np.array([0.0])

                obs_agent_types = np.hstack(    
                    (
                        obs_type_agent,
                        np.zeros(
                            (
                                self.num_obs_see
                                - len(nearest_neighbor_ids)
                            )
                        ),
                    )
                )
                obs_still_in_the_game = (
                    np.hstack(
                        (
                            still_in_the_game_,
                            np.zeros(
                                (
                                    self.num_obs_see
                                    - len(nearest_neighbor_ids)
                                )
                            ),
                        )
                    ),
                )
                obs_agent_size = (
                    np.hstack(
                        (
                            obs_size_agent,
                            np.zeros(
                                (
                                    self.num_obs_see
                                    - len(nearest_neighbor_ids)
                                )
                            ),
                        )
                    ),
                )

                # Form the observation
                obs[agent_id] = np.concatenate(
                    [
                        np.array([self.global_state[_ACC_X][self.timestep][agent_id] / (self.spot_max_acceleration_x + self.eps)]),
                        np.array([self.global_state[_ACC_Y][self.timestep][agent_id] / (self.spot_max_acceleration_y + self.eps)]),
                        np.array([self.global_state[_SP_X][self.timestep][agent_id] / (self.spot_max_speed_x + self.eps)]),
                        np.array([self.global_state[_SP_Y][self.timestep][agent_id] / (self.spot_max_speed_y + self.eps)]),
                        np.array([self.global_state[_DIR][self.timestep][agent_id] / (2 * np.pi)]),
                        np.array([self.agent_size[agent_id]]),
                        fix_goal_x,
                        fix_goal_y,
                        fix_goal_z,
                        np.array([self.global_state[_LOC_X][self.timestep][agent_id] / self.grid_length]),
                        np.array([self.global_state[_LOC_Y][self.timestep][agent_id] / self.grid_length]),
                        np.array([front_pitch]),
                        np.array([front_row]),
                        np.array([back_pitch]),
                        np.array([back_row]),
                        np.array([left_pitch]),
                        np.array([left_row]),
                        np.array([right_pitch]),
                        np.array([right_row]),
                        time,
                        np.vstack(
                            (
                                obs_global_states,
                                obs_agent_types,
                                obs_still_in_the_game,
                                obs_agent_size
                            )
                        ).reshape(-1),
                        obs_local_angle.reshape(-1),
                    ]
                )

        return obs

    def compute_reward(self):

        # Initialize rewards
        rew = {agent_id: 0.0 for agent_id in range(self.num_agents)}

        spots_list = sorted(self.spots)
        obstacles_list = sorted(self.obstacles)
        goals_list = sorted(self.goals)

        # At least one wall present
        obstacles_locations_x = self.global_state[_LOC_X][self.timestep][obstacles_list]
        goals_locations_x = self.global_state[_LOC_X][self.timestep][goals_list]
        spots_locations_x = self.global_state[_LOC_X][self.timestep][spots_list]

        obstacles_locations_y = self.global_state[_LOC_Y][self.timestep][obstacles_list]
        goals_locations_y = self.global_state[_LOC_Y][self.timestep][goals_list]
        spots_locations_y = self.global_state[_LOC_Y][self.timestep][spots_list]

        spots_to_obstacles_distances = np.sqrt(
            (
                np.repeat(spots_locations_x, self.num_obstacle)
                - np.tile(obstacles_locations_x, self.num_spot)
            )
            ** 2
            + (
                np.repeat(spots_locations_y, self.num_obstacle)
                - np.tile(obstacles_locations_y, self.num_spot)
            )
            ** 2
        ).reshape(self.num_spot, self.num_obstacle)

        goals_to_obstacles_distances = np.sqrt(
            (
                np.repeat(goals_locations_x, self.num_obstacle)
                - np.tile(obstacles_locations_x, self.num_goals)
            )
            ** 2
            + (
                np.repeat(goals_locations_y, self.num_obstacle)
                - np.tile(obstacles_locations_y, self.num_goals)
            )
            ** 2
        ).reshape(self.num_goals, self.num_obstacle)

        goals_to_spots_distances = np.sqrt(
            (
                np.repeat(goals_locations_x, self.num_spot)
                - np.tile(spots_locations_x, self.num_goals)
            )
            ** 2
            + (
                np.repeat(goals_locations_y, self.num_spot)
                - np.tile(spots_locations_y, self.num_goals)
            )
            ** 2
        ).reshape(self.num_goals, self.num_spot)

        min_spots_to_obstacles_distances = np.min(
            spots_to_obstacles_distances, axis=1
        )
        argmin_spots_to_obstacles_distances = np.argmin(
            spots_to_obstacles_distances, axis=1
        )
        nearest_obstacles_spots_ids = [
            obstacles_list[idx] for idx in argmin_spots_to_obstacles_distances
        ]

        min_goals_to_obstacles_distances = np.min(
            goals_to_obstacles_distances, axis=1
        )
        argmin_goals_to_obstacles_distances = np.argmin(
            goals_to_obstacles_distances, axis=1
        )
        nearest_obstacles_goals_ids = [
            obstacles_list[idx] for idx in argmin_goals_to_obstacles_distances
        ]

        min_goals_to_spots_distances = np.min(
            goals_to_spots_distances, axis=1
        )
        argmin_goals_to_spots_distances = np.argmin(
            goals_to_spots_distances, axis=1
        )
        nearest_spots_goals_ids = [
            spots_list[idx] for idx in argmin_goals_to_spots_distances
        ]

        for idx, spots_id in enumerate(spots_list):
            # 動態障礙物
            if min_spots_to_obstacles_distances[idx] < self.agent_size[idx] + self.agent_size[nearest_obstacles_spots_ids[idx]]:
                rew[spots_id] += float(self.spot_penalty_hit)
                self.still_in_the_game[spots_id] = 0
                self.num_hit_obstacle += 1
                del self.spots[spots_id]
                self.num_spot -= 1

            # 靜態障礙物   
            compute_hit_wall_ = self.compute_hit_wall_angle_base(
                self.global_state[_LOC_X][self.timestep][spots_id],
                self.global_state[_LOC_Y][self.timestep][spots_id],
                self.agent_size[spots_id]
            )
            if compute_hit_wall_ == 1:
                rew[spots_id] += float(self.spot_penalty_hit)
                self.still_in_the_game[spots_id] = 0
                self.num_hit_obstacle += 1
                del self.spots[spots_id]
                self.num_spot -= 1

            # 運動模型
            if self.PHASE > 1:
                nomal_acc_x = abs(self.global_state[_ACC_X][self.timestep][spots_id]) / max(self.spot_max_acceleration_x, abs(self.spot_min_acceleration_x))
                nomal_speed_y = abs(self.global_state[_SP_Y][self.timestep][spots_id]) / self.spot_max_speed_y
                nomal_turn_ = abs(self.global_state[_DIR][self.timestep][spots_id] - self.global_state[_DIR][self.timestep - 1][spots_id] ) / self.spot_max_turn
                rew[spots_id] += self.spot_cost_operate * (1 - nomal_acc_x) ** 2
                rew[spots_id] += self.spot_cost_operate * self.spot_cost_speed_y_rate_with_acc_x * (1 - nomal_speed_y) 
                rew[spots_id] += self.spot_cost_operate * self.spot_cost_turn_rate_with_acc_x * (1 - nomal_turn_) ** 2
            else:
                if self.global_state[_SP_X][self.timestep][spots_id] > 0:
                    rew[spots_id] += self.spot_cost_operate * 20

            # Floor reward
            agent_grid_x = int(self.global_state[_LOC_X][self.timestep][goal_id] / self.grid_density + 0.5)
            agent_grid_y = int(self.global_state[_LOC_Y][self.timestep][goal_id] / self.grid_density + 0.5)
            self_agent_id = self.max_j * agent_grid_x + agent_grid_y
            if abs(self.grid_map_z_[self_agent_id]) > 0 and abs(self.grid_map_z_[self_agent_id]) < self.grid_length_height:
                dir_agent = self.global_state[_DIR][self.timestep][spots_id]
                dir_dx = np.cos(dir_agent) 
                dir_dy = np.sin(dir_agent)
                if self.map_dx(agent_grid_x, agent_grid_y) * dir_dx + self.map_dy(agent_grid_x, agent_grid_y) * dir_dy > 0:
                    rew[spots_id] += self.spot_floor_direction_reward
                else:
                    rew[spots_id] -= self.base_floor_direction_reward

        for idx, goal_id in enumerate(goals_list):
            # 動態障礙物
            if min_goals_to_obstacles_distances < self.agent_size[idx] + self.agent_size[nearest_obstacles_goals_ids[idx]]:
                rew[goal_id] += float(self.spot_penalty_hit)
                
            # 靜態障礙物
            compute_hit_wall_ = self.compute_hit_wall_angle_base(
                self.global_state[_LOC_X][self.timestep][goal_id],
                self.global_state[_LOC_Y][self.timestep][goal_id],
                self.agent_size[goal_id]
            )
            if compute_hit_wall_ == 1:
                rew[goal_id] += float(self.spot_penalty_hit)

            # SPOT reward
            if min_goals_to_spots_distances[idx] < self.agent_size[idx] + self.agent_size[nearest_spots_goals_ids[idx]]:
                rew[goal_id] += float(self.spot_penalty_hit)
                rew[nearest_spots_goals_ids[idx]] += self.spot_reward_get_goal
                self.still_in_the_game[goal_id] = 0
                self.global_state[_SIG][self.timestep :, goal_id] = 0
                self.num_get_goal += 1
                if self.Training:
                    del self.goals[goal_id]
                    self.num_goals -= 1

            # Step reward
            rew[goal_id] += self.goals_reward_step

            # Move reward
            goal_max_speed = self.skill_levels[goal_id] * self.spot_max_speed_x
            goal_max_acceleration = self.skill_levels[goal_id] * self.spot_max_acceleration_x
            nomal_speed_x = abs(self.global_state[_SP_X][self.timestep][goal_id]) / goal_max_speed
            nomal_acc_x = abs(self.global_state[_ACC_X][self.timestep][goal_id]) / goal_max_acceleration
            goal_move_reward = 0.5 * self.goals_reward_speed * (nomal_speed_x + 1 - nomal_acc_x)
            rew[goal_id] += goal_move_reward

            # Floor reward
            if self.PHASE > 2:
                agent_grid_x = int(self.global_state[_LOC_X][self.timestep][goal_id] / self.grid_density + 0.5)
                agent_grid_y = int(self.global_state[_LOC_Y][self.timestep][goal_id] / self.grid_density + 0.5)

                self_agent_id = self.max_j * agent_grid_x + agent_grid_y
                if abs(self.grid_map_z_[self_agent_id]) > 0 and abs(self.grid_map_z_[self_agent_id]) < self.grid_length_height:
                    rew[goal_id] += self.reward_on_floor
        return rew

    def get_data_dictionary(self):
        """
        Create a dictionary of data to push to the device
        """
        self.still_in_the_game =[1 for agent_id in range(self.num_agents)]
        data_dict = DataFeed()
        for feature in [_LOC_X, _LOC_Y, _SP_X, _SP_Y, _DIR, _ACC_X, _ACC_Y]:
            data_dict.add_data(
                name=feature,
                data=self.global_state[feature][0],
                save_copy_and_apply_at_reset=True
            )
        
        data_dict.add_data(
            name="display_reward",
            data=self.display_reward,
            save_copy_and_apply_at_reset=True
        )
        data_dict.add_data(name="turn_actions", data=self.turn_actions)
        data_dict.add_data(name="acceleration_actions_x", data=self.acceleration_actions_x)
        data_dict.add_data(name="acceleration_actions_y", data=self.acceleration_actions_y)
        data_dict.add_data(
            name="neighbor_distances",
            data=np.zeros((self.num_agents, self.num_agents - 1), dtype=self.float_dtype),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(name="agent_size",data=self.agent_size)
        data_dict.add_data(name="skill_levels", data=self.skill_levels)
        data_dict.add_data(name="different_point_angle_distance", data=self.different_point_angle_distance)
        data_dict.add_data(name="grid_map_z", data=self.grid_map_z_)
        data_dict.add_data(name="Testing_type", data=self.Testing_type)
        data_dict.add_data(name="Training", data=self.Training_)
        data_dict.add_data(name="Training_Phase", data=self.PHASE)
        data_dict.add_data(name="grid_length", data=self.grid_length)
        data_dict.add_data(name="grid_density", data=self.grid_density)
        data_dict.add_data(name="num_obs_see", data=self.num_obs_see_)
        data_dict.add_data(name="spot_lidar_angle", data=self.spot_lidar_angle_f)
        data_dict.add_data(name="spot_max_turn", data=self.spot_max_turn)
        data_dict.add_data(name="spot_max_speed_x", data=self.spot_max_speed_x)
        data_dict.add_data(name="spot_min_speed_x", data=self.spot_min_speed_x)
        data_dict.add_data(name="spot_max_speed_y", data=self.spot_max_speed_y)
        data_dict.add_data(name="spot_min_speed_y", data=self.spot_min_speed_y)
        data_dict.add_data(name="spot_max_acceleration_x", data=self.spot_max_acceleration_x)
        data_dict.add_data(name="spot_min_acceleration_x", data=self.spot_min_acceleration_x)
        data_dict.add_data(name="spot_max_acceleration_y", data=self.spot_max_acceleration_y)
        data_dict.add_data(name="spot_min_acceleration_y", data=self.spot_min_acceleration_y)
        data_dict.add_data(name="spot_reward_get_goal", data=self.spot_reward_get_goal)
        data_dict.add_data(name="spot_floor_direction_reward", data=self.spot_floor_direction_reward)
        data_dict.add_data(name="base_floor_direction_reward", data=self.base_floor_direction_reward)
        data_dict.add_data(name="spot_penalty_hit", data=self.spot_penalty_hit)
        data_dict.add_data(name="spot_cost_operate", data=self.spot_cost_operate)
        data_dict.add_data(name="spot_cost_speed_y_rate_with_acc_x", data=self.spot_cost_speed_y_rate_with_acc_x)
        data_dict.add_data(name="spot_cost_turn_rate_with_acc_x", data=self.spot_cost_turn_rate_with_acc_x)
        data_dict.add_data(name="goals_reward_step", data=self.goals_reward_step)
        data_dict.add_data(name="goals_reward_speed", data=self.goals_reward_speed)
        data_dict.add_data(name="reward_on_floor", data=self.reward_on_floor)
        grid_length_max_y = self.float_dtype(self.max_j)
        data_dict.add_data(name="grid_length_max_y", data=grid_length_max_y)
        data_dict.add_data(name="grid_length_height", data=self.grid_length_height)
        
        data_dict.add_data(
            name="still_in_the_game",
            data=self.still_in_the_game,
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="neighbor_ids_sorted_by_distance",
            data=np.zeros((self.num_agents, self.num_agents - 1), dtype=np.int32),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="nearest_neighbor_ids",
            data=np.zeros(
                (self.num_agents, self.num_obs_see), dtype=np.int32
            ),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(name="agent_types", 
            data=[self.agent_type[agent_id] for agent_id in range(self.num_agents)],
        )
        data_dict.add_data(
            name = "num_spot",
            data = self.num_spot,
            save_copy_and_apply_at_reset=True
        )
        self.num_hit_obstacle = 0
        data_dict.add_data(
            name="num_hit_obstacle",
            data=self.num_hit_obstacle,
            save_copy_and_apply_at_reset=True,
        )
        self.num_get_goal = 0
        data_dict.add_data(
            name = "num_get_goal",
            data = self.num_get_goal,
            save_copy_and_apply_at_reset=True,
        )

        return data_dict

    def get_tensor_dictionary(self):
        tensor_dict = DataFeed()
        return tensor_dict

    def reset(self):

        self.timestep = 0

        self.global_state = {}

        # cuda init
        self.set_global_state(
            key=_LOC_X, value=self.starting_location_x, t=self.timestep
        )
        self.set_global_state(
            key=_LOC_Y, value=self.starting_location_y, t=self.timestep
        )
        self.set_global_state(key=_SP_X, value=self.starting_speeds_x, t=self.timestep)
        self.set_global_state(key=_SP_Y, value=self.starting_speeds_y, t=self.timestep)
        self.set_global_state(key=_DIR, value=self.starting_directions, t=self.timestep)
        self.set_global_state(key=_ACC_X, value=self.starting_accelerations_x, t=self.timestep)
        self.set_global_state(key=_ACC_Y, value=self.starting_accelerations_y, t=self.timestep)

        # Array to keep track of the agents that are still in play
        self.still_in_the_game = np.ones(self.num_agents, dtype=self.int_dtype)

        # Initialize global state for "still_in_the_game" to all ones
        self.global_state[_SIG] = np.ones(
            (self.episode_length + 1, self.num_agents), dtype=self.int_dtype
        )
        self.num_hit_obstacle = 0
        self.num_get_goal = 0
        self.goals = copy.deepcopy(self.goal_at_reset)
        self.num_goals = len(self.goals)
        self.spots = copy.deepcopy(self.spots_at_reset)
        self.num_spot = len(self.spots)

        return self.generate_observation()

    def step(self, actions=None):
        """
        Env step() - The GPU version calls the corresponding CUDA kernels
        """
        self.timestep += 1
        
        if self.use_cuda:
            # self.timestep = now num_episodes 看起來像訓練次數
            args = [
                "grid_map_z",
                "different_point_angle_distance",
                _LOC_X,
                _LOC_Y,
                _SP_X,
                _SP_Y,
                _DIR,
                _ACC_X,
                _ACC_Y,
                _OBSERVATIONS,
                _REWARDS,
                "turn_actions",
                "acceleration_actions_x",
                "acceleration_actions_y",
                "neighbor_distances",
                "agent_size",
                "skill_levels",
                "display_reward",
                "Testing_type",
                "Training",
                "Training_Phase",
                "grid_length",
                "grid_density",
                "num_obs_see",
                "spot_lidar_angle",
                "spot_max_turn",
                "spot_max_speed_x",
                "spot_min_speed_x",
                "spot_max_speed_y",
                "spot_min_speed_y",
                "spot_max_acceleration_x",
                "spot_min_acceleration_x",
                "spot_max_acceleration_y",
                "spot_min_acceleration_y",
                "spot_reward_get_goal",
                "spot_floor_direction_reward",
                "base_floor_direction_reward",
                "spot_penalty_hit",
                "spot_cost_operate",
                "spot_cost_speed_y_rate_with_acc_x",
                "spot_cost_turn_rate_with_acc_x",
                "goals_reward_step",
                "goals_reward_speed",
                "reward_on_floor",
                "grid_length_max_y",
                "grid_length_height",
                ("n_agents", "meta"),
                ("episode_length", "meta"),
                _ACTIONS,
                "still_in_the_game",
                "neighbor_ids_sorted_by_distance",
                "nearest_neighbor_ids",
                "agent_types",
                "_done_",
                "_timestep_",
                "num_spot",
                "num_hit_obstacle",
                "num_get_goal"
            ]
            self.cuda_step(
                *self.cuda_step_function_feed(args),
                block = self.cuda_function_manager.block,
                grid = self.cuda_function_manager.grid,
            )
            result = None  # do not return anything
        else:
            assert isinstance(actions, dict)
            assert len(actions) == self.num_agents

            acceleration_x_action_ids = [
                actions[agent_id][0] for agent_id in range(self.num_agents)
            ]

            acceleration_y_action_ids = [
                actions[agent_id][1] for agent_id in range(self.num_agents)
            ]

            turn_action_ids = [
                actions[agent_id][2] for agent_id in range(self.num_agents)
            ]

            assert all(0 <= acc_x <= self.spot_action_levels for acc_x in acceleration_x_action_ids)
            assert all(0 <= acc_y <= self.spot_action_levels for acc_y in acceleration_y_action_ids)
            assert all(0 <= turn <= self.spot_action_levels for turn in turn_action_ids)

            delta_accelerations_x = self.acceleration_actions_x[acceleration_x_action_ids]
            delta_accelerations_y = self.acceleration_actions_y[acceleration_y_action_ids]
            delta_turns = self.turn_actions[turn_action_ids]

            # Update state and generate observation
            self.update_state(delta_accelerations_x, delta_accelerations_y, delta_turns)
            if not self.use_cuda:
                obs = self.generate_observation()

            # Compute rewards and done
            rew = self.compute_reward()

            done = {
                "__all__": (self.timestep >= self.episode_length)
            }
            info = {}

            result = obs, rew, done, info
        return result
