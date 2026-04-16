"""
    這個環境需要設定成 goal 的任務和 spot 完全不相關。
    goal 不會有spot reward，被 spot 抓到會消失，可是可以吃到 reward，消失的目的是為了spot
    還沒有建立新環境 Spot_3dIn
"""

import copy
import heapq
import random
#import math
import numpy as np
from gym import spaces
from gym.utils import seeding
from icecream import ic
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

class Map():
    def __init__(
            self,
            num_wall = 2,
            wall_length = 4,
            grid_length = 20,
            num_stairs = 1,
            spot_lidar_angle = 30,
            stairs_weight_rand = 1,
            stairs_weight_bias = 2,
            floor_width_rand = 2,
            floor_width_bias = 2,
            grid_length_height = 2,
            grid_density = 0.3,
            wall_thickness = 0.3,
            seed = None,
            random_map = False 
        ) -> None:
        super().__init__()
        if random_map == False:  np.random.seed(seed + 23)
        else: np.random.seed(int(random.random() * 10000))
        self.num_wall = num_wall
        self.wall_length = wall_length
        self.grid_length = grid_length
        self.num_stairs = min(int(np.random.rand() * num_stairs) + 1, num_stairs)
        self.stairs_weight_rand = stairs_weight_rand
        self.stairs_weight_bias = stairs_weight_bias
        self.floor_width_rand = floor_width_rand
        self.floor_width_bias = floor_width_bias
        self.grid_length_height = grid_length_height
        self.grid_density = grid_density
        self.wall_thickness = wall_thickness
        self.spot_lidar_angle = spot_lidar_angle
        self.num_angle_lidar = int(360 / self.spot_lidar_angle)
        
        self.x = np.arange(0, self.grid_length, self.grid_density)
        self.y = np.arange(0, self.grid_length, self.grid_density)
        self.map_x, self.map_y = np.meshgrid(self.y, self.x)
        self.map_z = np.zeros_like(self.map_x)
        self.map_limit = np.zeros_like(self.map_x)
        self.map_max_x = len(self.x)
        self.map_max_y = len(self.y)
        
    def point_on_stair(self, point, mode, origin_r, mid_l, floor_r1, floor_l2, stair_lenght, grid_length_height):
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

    def check_point_on_line(self, point, line_position1,  line_position2):
        vec_line = np.array(line_position2) - np.array(line_position1)
        vec1 = np.array(point) - np.array(line_position1)
        vec2 = np.array(point) - np.array(line_position2)
        dot_product = np.dot(vec1, vec_line) / (np.linalg.norm(vec_line)) ** 2
        if dot_product > 0 and dot_product < 1:
            dis = np.sqrt((np.linalg.norm(vec1)) ** 2 -(dot_product * np.linalg.norm(vec_line)) ** 2 )
        elif dot_product >= 1:
            dis = np.linalg.norm(vec2)
        else:
            dis = np.linalg.norm(vec1)
        # dis = 牆壁 和 point 之間的最短距離
        if dis < self.wall_thickness:
            return True
        return False
    
    def calculate_height(self, point, right_or_left, right, far):
        z = 0
        if right_or_left != 0:
            z = abs((point[1] - right[1]) / (far[1] - right[1]))
        else:
            z = abs((point[0] - right[0]) / (far[0] - right[0]))
        return z
    
    def check_point_in_area(self, point, position1, position2):
        small_x = min(position1[0], position2[0])
        big_x = max(position1[0], position2[0])
        small_y = min(position1[1], position2[1])
        big_y = max(position1[1], position2[1])
        if point[0] > small_x and point[0] < big_x and point[1] > small_y and point[1] < big_y:
            return True
        return False

    def check_point_on_wall(self, point, wall_list):
        for wall in wall_list:
            if self.check_point_on_line(point, wall['start'], wall['end']):
               return True
        return False 
    
    def point_on_any_stair(self, point, stair_list):
        z_ = 0
        on_stair_list = []
        on_floor_list = []
        on_wall = False
        # if point on stair or floor, point will on stair 
        for stair in stair_list:
            on_stair = self.check_point_in_area(point, stair['origin_r'], stair['mid_l'])
            on_floor = self.check_point_in_area(point, stair['floor_r1'], stair['floor_l2'])
            if on_stair: on_stair_list.append(stair)
            if on_floor: on_floor_list.append(stair)

        # check point on stair, and cheak origin piont is zero floor
        for stair in on_stair_list:
            z_ = self.calculate_height(point, right_or_left = stair['right_or_left'], right = stair["origin_r"], far = stair['mid_l']) * self.grid_length_height
            if stair['down']: z_ = z_ * -1
            orign_x = stair["origin_r"][0]
            orign_y = stair["origin_r"][1]
            far_x = stair["mid_l"][0]
            far_y = stair["mid_l"][1]
            on_wall = False
            if stair['right_or_left']:
                if self.check_point_on_line(point, [orign_x, far_y], stair["origin_r"]): on_wall = True
                if self.check_point_on_line(point, [far_x, orign_y], stair["mid_l"]): on_wall = True
                if self.check_point_on_line(point, [far_x, orign_y], stair["origin_r"]):
                    if len(on_stair_list) > 1 or len(on_floor_list) > 0:
                        on_wall = True
                if self.check_point_on_line(point, [orign_x, far_y], stair["mid_l"]):
                    if len(on_stair_list) > 1:  on_wall = True
            else:
                if self.check_point_on_line(point, [far_x, orign_y] , stair["origin_r"]): on_wall = True
                if self.check_point_on_line(point, [orign_x, far_y] , stair["mid_l"]): on_wall = True
                if self.check_point_on_line(point, [orign_x, far_y], stair["origin_r"]):
                    if len(on_stair_list) > 1 or len(on_floor_list) > 0:
                        on_wall = True
                if self.check_point_on_line(point, [far_x, orign_y], stair["mid_l"]):
                    if len(on_stair_list) > 1:  on_wall = True
            return on_wall, z_
            
        # check point on double floor
        if len(on_floor_list) >= 2:
            if on_floor_list[0]['down'] == on_floor_list[1]['down']:
                z_ = self.grid_length_height
                if on_floor_list[0]['down']: z_ = z_ * -1
                return False, z_

        # double floor diffent high, high choose first one
        if len(on_floor_list) > 0:
            stair = on_floor_list[0]
            z_ = self.grid_length_height
            if stair['down']: z_ = z_ * -1
            floor_x1 = stair['floor_r1'][0]
            floor_x2 = stair['floor_l2'][0]
            floor_y1 = stair['floor_r1'][1]
            floor_y2 = stair['floor_l2'][1]
            if self.check_point_on_line(point, [floor_x1, floor_y2] , stair['floor_l2']): on_wall = True
            if self.check_point_on_line(point, [floor_x2, floor_y1] , stair['floor_l2']): on_wall = True
            if stair['right_or_left']:
                # floor_r1_x = floor_x
                if self.check_point_on_line(point, stair['floor_r1'],  [floor_x1, floor_y2]): on_wall = True
                if self.check_point_on_line(point, stair['floor_r1'], [stair["origin_r"][0], floor_y1]): on_wall = True
                if self.check_point_on_line(point, [floor_x2, floor_y1], [stair["mid_l"][0], floor_y1]): on_wall = True
            else:
                # floor_r1_y = floor_y
                if self.check_point_on_line(point, stair['floor_r1'], [floor_x2, floor_y1]): on_wall = True
                if self.check_point_on_line(point, stair['floor_r1'], [floor_x1, stair["origin_r"][1]]): on_wall = True
                if self.check_point_on_line(point, [floor_x1, floor_y2], [floor_x1, stair["mid_l"][1]]): on_wall = True
    
        return on_wall, z_
    
    def cheak_point_distance(self,this_point_origin_z, point_y, point_x):
        
        len_grid_precision = 1.0 / 3.0
        max_distance = int(self.grid_length / len_grid_precision + 1)
        for k in range(self.num_angle_lidar):
            len = 0
            for len_ in range(max_distance):
                len = len_ * len_grid_precision + len_grid_precision
                x_ = int(point_x + np.cos(np.radians(k * self.spot_lidar_angle)) * len + 0.5)
                y_ = int(point_y + np.sin(np.radians(k * self.spot_lidar_angle)) * len + 0.5)
                if x_ > self.map_max_x - 1 or x_ < 0 or y_ > self.map_max_y - 1 or y_ < 0:
                    break
                if self.map_limit[y_, x_] == 1: break
            self.different_point_angle_distance[self.num_angle_lidar * self.map_max_x * point_y + self.num_angle_lidar * point_x + k] =  (len - len_grid_precision) * self.grid_density 
        return 0

    def generate_map(self):
        wall_list = [
            {"start":[0, 0], "end":[0, self.grid_length]},
            {"start":[0, 0], "end":[self.grid_length, 0]},
            {"start":[self.grid_length, self.grid_length], "end":[0, self.grid_length]},
            {"start":[self.grid_length, self.grid_length], "end":[self.grid_length, 0]}
        ]
        stair_list = []
        grid_length_half = self.grid_length * 0.5
        wall_length_half = self.wall_length * 0.5
        for wall_idx in  range(self.num_wall):
            wall_angle = np.random.rand() * 2.0 * np.pi
            origin_x =  np.random.rand() * self.grid_length
            origin_y =  np.random.rand() * self.grid_length
            wall_info = {
                "start": [
                    origin_x + wall_length_half * np.cos(wall_angle),
                    origin_y + wall_length_half * np.sin(wall_angle)
                ],
                "end":[
                    origin_x - wall_length_half * np.cos(wall_angle),
                    origin_y - wall_length_half * np.sin(wall_angle)
                ]
            }
            wall_list.append(wall_info)

        stairs_length = self.grid_length_height * 1.7
        for stair_idx in  range(self.num_stairs):
            stairs_weight = np.random.rand() * self.stairs_weight_rand + self.stairs_weight_bias
            floor_width = np.random.rand() * self.floor_width_rand + self.floor_width_bias
            half_stairs_weight = stairs_weight * 0.5
            half_floor_width = floor_width * 0.5
            origin_x =  (np.random.rand() - 0.5) * self.grid_length * 0.8 + grid_length_half
            origin_y =  (np.random.rand() - 0.5) * self.grid_length * 0.8 + grid_length_half
            up_or_down_z = True if random.random() > 0.5 else False
            right_or_left = 0 if np.random.rand() > 0.5 else 1
            up_or_down_y = -1 if np.random.rand() > 0.5 else 1
            stair_info = {
                "right_or_left": right_or_left,
                "origin_r":[
                    origin_x - half_stairs_weight * right_or_left,
                    origin_y - half_stairs_weight * (1 - right_or_left)
                ],
                "mid_l":[
                    origin_x + half_stairs_weight * right_or_left + stairs_length * up_or_down_y * (1 - right_or_left),
                    origin_y + half_stairs_weight * (1 - right_or_left) + stairs_length * up_or_down_y * right_or_left
                ],
                "floor_r1":[
                    origin_x - half_floor_width * right_or_left + stairs_length * up_or_down_y * (1 - right_or_left),
                    origin_y - half_floor_width * (1 - right_or_left) + stairs_length * up_or_down_y * right_or_left
                ],
                "floor_l2":[
                    origin_x + half_floor_width * right_or_left + (stairs_length + floor_width) * up_or_down_y * (1 - right_or_left),
                    origin_y + half_floor_width * (1 - right_or_left) + (stairs_length + floor_width) * up_or_down_y * right_or_left
                ],
                "up": up_or_down_z,
                "down": not up_or_down_z
            }
            stair_list.append(stair_info)

        # grip map
        for i in range(self.map_max_y):
            for j in range(self.map_max_x):
                point = [
                    j * self.grid_density,
                    i * self.grid_density
                ]
                on_stair, z_ = self.point_on_any_stair(point, stair_list)
                on_wall = self.check_point_on_wall(point, wall_list)
                if i == 0 or i == self.map_max_y - 1 or j == 0 or j == self.map_max_x - 1: on_wall = True
                self.map_z[i,j] = z_
                if on_wall and z_ == 0:
                    self.map_z[i,j] = self.grid_length_height
                if on_stair or (on_wall and z_ == 0):
                    self.map_limit[i, j] = 1
        

        self.different_point_angle_distance = np.zeros((int(self.map_max_x * self.map_max_y * 360 / self.spot_lidar_angle)), dtype=np.float32)
        for i in range(self.map_max_y):
            for j in range(self.map_max_x):
                this_point_origin_z = self.map_limit[i, j]
                self.cheak_point_distance(this_point_origin_z, i, j)

class Spot_3d(CUDAEnvironmentContext, Map):
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
        rl_fps = 2,
        num_spot = 1,
        num_goals = 10,
        num_obstacle = 10,
        num_wall = 3,
        wall_length = 3,
        wall_thickness = 0.3,
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
        seed_s = None,
        seed_g = None,
        random_agent = False,
        random_map = False,
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
        spot_reward_on_floor = 20.0,
        spot_cost_operate = 0.05,
        spot_cost_speed_y_rate_with_acc_x  = 0.4,
        spot_cost_turn_rate_with_acc_x = 0.6,
        spot_penalty_hit = -15.0,
        goal_penalty_hit = -15.0,
        spot_action_levels = 20,
        spot_size_bias = 0.15,
        spot_head_distance = 0.6,
        spot_ass_distance = 0.42,
        obs_size_rand = 0.2,
        reward_on_floor = 0.1,
        goals_reward_step = 0.1,
        goals_reward_speed = 0.1,
        goal_size = 0.1,
        goal_speed_rate_with_spot = 0.5,
        obstacle_speed_rate_with_spot = 1,
        four_sign_distance = 1.5,
        Testing_type = 1,
        starting_location_x = None,
        starting_location_y = None,
        starting_directions = None
    ):
        """

        """
        Map.__init__(
            self,
            num_wall=num_wall,
            wall_length=wall_length,
            grid_length=grid_length_bias,
            num_stairs=num_stairs,
            stairs_weight_rand=stairs_weight_rand,
            stairs_weight_bias=stairs_weight_bias,
            floor_width_rand=floor_width_rand,
            floor_width_bias=floor_width_bias,
            grid_length_height=grid_length_height,
            grid_density=grid_density,
            wall_thickness=wall_thickness,
            seed=seed,
            spot_lidar_angle=spot_lidar_angle,
            random_map = random_map
        )

        self.float_dtype = np.float32
        self.int_dtype = np.int32
        self.eps = self.float_dtype(1e-10)
        self.PHASE = self.float_dtype(PHASE)
        self.rl_fps = self.float_dtype(rl_fps)
        
        self.Training = Training
        self.Training_ = self.float_dtype(0)

        if self.Training == True:
            self.Training_ = self.float_dtype(1)
            assert num_spot > 0
            assert num_obstacle > 0
            assert num_goals > 0
        if Testing_type == 1:
            self.Training_ = self.float_dtype(1)
            assert num_spot > 0
            assert num_obstacle > 0
            assert num_goals > 0
        elif Testing_type == 2:
            if num_spot != 1 or num_goals != 1 :
                print("testing number spot and goal = 1 ")
            num_spot = 1
            num_goals = 1
        elif Testing_type == 3:
            episode_length = 10000
            num_spot = 1
            num_goals = 1
        self.Testing_type = self.float_dtype(Testing_type)
        if 360 % spot_lidar_angle != 0:
            print("spot_lidar_angle slould be 360 common factor .")
        assert 360 % spot_lidar_angle == 0
        self.spot_lidar_angle = self.int_dtype(spot_lidar_angle)
        self.spot_lidar_angle_f = self.float_dtype(spot_lidar_angle)

        assert episode_length > 0
        self.episode_length = episode_length

        # Seeding
        self.np_random = np.random

        self.random_agent = random_agent
        if seed is not None: np.random.seed(seed)
        if self.random_agent: np.random.seed(int(random.random() * 10000))

        self.num_spot = num_spot  
        self.num_obstacle = int(num_obstacle * self.np_random.rand() + 1)
        self.num_goals = num_goals
        self.num_agents = self.num_spot + self.num_obstacle + self.num_goals

        #ic(num_spot, num_goals, num_obstacle)
        agent_size = self.np_random.random(self.num_agents) * obs_size_rand + spot_size_bias - obs_size_rand / 2.0
        grid_length = grid_length_bias
        

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
                agent_size[agent_id] = spot_size_bias

        # Square 2D grid
        assert grid_length > 0
        self.grid_length = self.float_dtype(grid_length)
        self.grid_diagonal = self.grid_length * np.sqrt(2)
        self.grid_length_height = self.float_dtype(grid_length_height)
        self.spot_head_distance = self.float_dtype(spot_head_distance)
        self.spot_ass_distance = self.float_dtype(spot_ass_distance)
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
            starting_directions = 2 * np.pi * self.np_random.random(self.num_agents)
        else:
            assert len(starting_directions) == self.num_agents
        self.starting_directions = starting_directions
        self.random_agent_seed = self.float_dtype(self.np_random.rand())

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

        #print("acceleration_actions_x = ", self.acceleration_actions_x)
        #print("acceleration_actions_y = ", self.acceleration_actions_y)

        self.skill_levels = [self.float_dtype(goal_speed_rate_with_spot) if self.agent_type[agent_id] == 0 else self.float_dtype(1.0) if self.agent_type[agent_id] == 1 else (self.float_dtype(obstacle_speed_rate_with_spot - 0.3) * random.random() + 0.3)for agent_id in range(self.num_agents)]
        
        # Set the start speed and acceleration
        self.starting_speeds_x = (self.np_random.random(self.num_agents) * (self.spot_max_speed_x - self.spot_min_speed_x) + self.spot_min_speed_x) * np.array(self.skill_levels)
        self.starting_accelerations_x = np.zeros(self.num_agents, dtype=self.float_dtype)
        self.starting_speeds_y = (self.np_random.random(self.num_agents) * (self.spot_max_speed_y - self.spot_min_speed_y) + self.spot_min_speed_y) * np.array(self.skill_levels)
        self.starting_accelerations_y = np.zeros(self.num_agents, dtype=self.float_dtype)
        speed_zero = self.np_random.random(self.num_agents)
        self.starting_speeds_x = np.where(speed_zero < 0.2, 0.0, self.starting_speeds_x)
        self.starting_speeds_y = np.where(speed_zero < 0.2, 0.0, self.starting_speeds_y)
        
        # Turn actions
        self.turn_actions = np.linspace(
            self.spot_min_turn, self.spot_max_turn, self.spot_action_levels
        )

        #print("turn_actions = ", self.turn_actions)

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

        #assert num_obs_see <= self.num_agents
        self.num_obs_see = num_obs_see
        self.num_obs_see_ = self.float_dtype(num_obs_see)

        # Rewards and penalties
        """
        display_reward = 
            goal(5) dynamic_Obstacle Obstacle SPOT Step Move Floor
            spot(3) dynamic_Obstacle Obstacle Goal dynamic floor
            一半紀錄
        """
        display_reward = np.zeros(14 * self.num_agents, dtype=self.float_dtype)
        self.display_reward = self.float_dtype(display_reward)
        self.four_sign_distance = self.float_dtype(four_sign_distance)

        # Spot
        self.spot_penalty_hit = self.float_dtype(spot_penalty_hit)
        self.goal_penalty_hit = self.float_dtype(goal_penalty_hit)
        self.spot_reward_get_goal = self.float_dtype(spot_reward_get_goal)
        self.spot_reward_on_floor = self.float_dtype(spot_reward_on_floor)
        
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

        # caulate 3d map Map class
        np.random.seed(seed + 17)
        self.generate_map()
        self.grid_map_z_ = self.map_z.reshape((self.map_max_x * self.map_max_y))

        spots_list = sorted(self.spots)
        for idx, spots_id in enumerate(spots_list):
            compute_hit_wall_ = self.compute_hit_wall(
                self.starting_location_x[spots_id],
                self.starting_location_y[spots_id],
                self.agent_size[spots_id] + 0.1
            )
            if compute_hit_wall_ == False:
                agent_grid_x = min(int(self.starting_location_x[spots_id] / self.grid_density + 0.5), self.map_max_x - 1)
                agent_grid_y = min(int(self.starting_location_y[spots_id] / self.grid_density + 0.5), self.map_max_x - 1)
                self_agent_id = self.map_max_x * agent_grid_y + agent_grid_x
                if abs(self.grid_map_z_[self_agent_id]) > 0 and abs(self.grid_map_z_[self_agent_id]) < self.grid_length_height:
                    dir_agent = self.starting_directions[spots_id]
                    dir_dx = np.cos(dir_agent) 
                    dir_dy = np.sin(dir_agent)
                    if self.map_dx(agent_grid_x, agent_grid_y) * dir_dx + self.map_dy(agent_grid_x, agent_grid_y) * dir_dy > 0:
                        compute_hit_wall_ = True

            dir_agent = self.starting_directions[spots_id]
            dir_dx = np.cos(dir_agent) 
            dir_dy = np.sin(dir_agent)
            point_x = self.starting_location_x[spots_id] + dir_dx * self.spot_head_distance
            point_y = self.starting_location_y[spots_id] + dir_dy * self.spot_head_distance
            if compute_hit_wall_ == False: 
                compute_hit_wall_ = self.compute_hit_wall(
                    point_x,
                    point_y,
                    self.agent_size[spots_id] + 0.1
                )
            if compute_hit_wall_ == False:
                agent_grid_x = max(min(int(point_x / self.grid_density + 0.5), self.map_max_x - 1), 0)
                agent_grid_y = max(min(int(point_y / self.grid_density + 0.5), self.map_max_x - 1), 0)
                self_agent_id = self.map_max_x * agent_grid_y + agent_grid_x
                if abs(self.grid_map_z_[self_agent_id]) > 0 and abs(self.grid_map_z_[self_agent_id]) < self.grid_length_height:
                    dir_agent = self.starting_directions[spots_id]
                    dir_dx = np.cos(dir_agent) 
                    dir_dy = np.sin(dir_agent)
                    if self.map_dx(agent_grid_x, agent_grid_y) * dir_dx + self.map_dy(agent_grid_x, agent_grid_y) * dir_dy > 0:
                        compute_hit_wall_ = True

            point_x = self.starting_location_x[spots_id] - dir_dx * self.spot_ass_distance
            point_y = self.starting_location_y[spots_id] - dir_dy * self.spot_ass_distance
            if compute_hit_wall_ == False: 
                compute_hit_wall_ = self.compute_hit_wall(
                    point_x,
                    point_y,
                    self.agent_size[spots_id] + 0.1
                )
            if compute_hit_wall_ == False:
                agent_grid_x = max(min(int(point_x / self.grid_density + 0.5), self.map_max_x - 1), 0)
                agent_grid_y = max(min(int(point_y / self.grid_density + 0.5), self.map_max_x - 1), 0)
                self_agent_id = self.map_max_x * agent_grid_y + agent_grid_x
                if abs(self.grid_map_z_[self_agent_id]) > 0 and abs(self.grid_map_z_[self_agent_id]) < self.grid_length_height:
                    dir_agent = self.starting_directions[spots_id]
                    dir_dx = np.cos(dir_agent) 
                    dir_dy = np.sin(dir_agent)
                    if self.map_dx(agent_grid_x, agent_grid_y) * dir_dx + self.map_dy(agent_grid_x, agent_grid_y) * dir_dy > 0:
                        compute_hit_wall_ = True

            while(compute_hit_wall_):
                self.starting_location_x[spots_id] = self.grid_length * self.np_random.rand()
                self.starting_location_y[spots_id] = self.grid_length * self.np_random.rand()
                compute_hit_wall_ = self.compute_hit_wall(
                    self.starting_location_x[spots_id],
                    self.starting_location_y[spots_id],
                    self.agent_size[spots_id] + 0.1
                )
                if compute_hit_wall_ == False:
                    agent_grid_x = min(int(self.starting_location_x[spots_id] / self.grid_density + 0.5), self.map_max_x - 1)
                    agent_grid_y = min(int(self.starting_location_y[spots_id] / self.grid_density + 0.5), self.map_max_x - 1)
                    self_agent_id = self.map_max_x * agent_grid_y + agent_grid_x
                    if abs(self.grid_map_z_[self_agent_id]) > 0 and abs(self.grid_map_z_[self_agent_id]) < self.grid_length_height:
                        dir_agent = self.starting_directions[spots_id]
                        dir_dx = np.cos(dir_agent) 
                        dir_dy = np.sin(dir_agent)
                        if self.map_dx(agent_grid_x, agent_grid_y) * dir_dx + self.map_dy(agent_grid_x, agent_grid_y) * dir_dy > 0:
                            compute_hit_wall_ = True

                dir_agent = self.starting_directions[spots_id]
                dir_dx = np.cos(dir_agent) 
                dir_dy = np.sin(dir_agent)
                point_x = self.starting_location_x[spots_id] + dir_dx * self.spot_head_distance
                point_y = self.starting_location_y[spots_id] + dir_dy * self.spot_head_distance
                if compute_hit_wall_ == False: 
                    compute_hit_wall_ = self.compute_hit_wall(
                        point_x,
                        point_y,
                        self.agent_size[spots_id] + 0.1
                    )
                if compute_hit_wall_ == False:
                    agent_grid_x = max(min(int(point_x / self.grid_density + 0.5), self.map_max_x - 1), 0)
                    agent_grid_y = max(min(int(point_y / self.grid_density + 0.5), self.map_max_x - 1), 0)
                    self_agent_id = self.map_max_x * agent_grid_y + agent_grid_x
                    if abs(self.grid_map_z_[self_agent_id]) > 0 and abs(self.grid_map_z_[self_agent_id]) < self.grid_length_height:
                        dir_agent = self.starting_directions[spots_id]
                        dir_dx = np.cos(dir_agent) 
                        dir_dy = np.sin(dir_agent)
                        if self.map_dx(agent_grid_x, agent_grid_y) * dir_dx + self.map_dy(agent_grid_x, agent_grid_y) * dir_dy > 0:
                            compute_hit_wall_ = True

                point_x = self.starting_location_x[spots_id] - dir_dx * self.spot_ass_distance
                point_y = self.starting_location_y[spots_id] - dir_dy * self.spot_ass_distance
                if compute_hit_wall_ == False: 
                    compute_hit_wall_ = self.compute_hit_wall(
                        point_x,
                        point_y,
                        self.agent_size[spots_id] + 0.1
                    )
                if compute_hit_wall_ == False:
                    agent_grid_x = max(min(int(point_x / self.grid_density + 0.5), self.map_max_x - 1), 0)
                    agent_grid_y = max(min(int(point_y / self.grid_density + 0.5), self.map_max_x - 1), 0)
                    self_agent_id = self.map_max_x * agent_grid_y + agent_grid_x
                    if abs(self.grid_map_z_[self_agent_id]) > 0 and abs(self.grid_map_z_[self_agent_id]) < self.grid_length_height:
                        dir_agent = self.starting_directions[spots_id]
                        dir_dx = np.cos(dir_agent) 
                        dir_dy = np.sin(dir_agent)
                        if self.map_dx(agent_grid_x, agent_grid_y) * dir_dx + self.map_dy(agent_grid_x, agent_grid_y) * dir_dy > 0:
                            compute_hit_wall_ = True

        goal_list = sorted(self.goals)
        for idx, goal_id in enumerate(goal_list):
            compute_hit_wall_ = self.compute_hit_wall(
                self.starting_location_x[goal_id],
                self.starting_location_y[goal_id],
                self.agent_size[goal_id]
            )

            while(compute_hit_wall_):
                self.starting_location_x[goal_id] = self.grid_length * self.np_random.rand()
                self.starting_location_y[goal_id] = self.grid_length * self.np_random.rand()
                compute_hit_wall_ = self.compute_hit_wall(
                    self.starting_location_x[goal_id],
                    self.starting_location_y[goal_id],
                    self.agent_size[goal_id]
                )

    name = "Spot_3dIn"

    def map_dx(self, map_x, map_y):
        if map_x < 1: return -2
        elif map_x >= self.map_max_x - 2: return 2
        elif map_y >= 0 and map_y < self.map_max_y: return 0.5 * (self.map_z[map_y, map_x + 1] - self.map_z[map_y, map_x - 1])
        else: return 0

    def map_dy(self, map_x, map_y):
        if map_y < 1: return -2
        elif map_y >= self.map_max_y - 2: return 2
        elif map_x >= 0 and map_x < self.map_max_x: return 0.5 * (self.map_z[map_y + 1, map_x] - self.map_z[map_y - 1, map_x])
        else: return 0

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
        still_in_game_prev_t = self.global_state[_SIG][self.timestep - 1]
        # Update direction and acceleration
        # Do not update location if agent is out of the game !
        #print(delta_accelerations)
        delta_turns = delta_turns * np.array(self.skill_levels)
        dir_curr_t = (
            (dir_prev_t + delta_turns) % (2 * np.pi) * self.still_in_the_game
        ).astype(self.float_dtype)

        acc_curr_t_x = delta_accelerations_x * self.skill_levels
        acc_curr_t_y = delta_accelerations_y * self.skill_levels

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
        acc_curr_t_x = acc_curr_t_x  * (speed_curr_t_x > min_speed_x) * (speed_curr_t_x < max_speed_x)

        # y dim speed and acceleration 
        for agent_id in range(self.num_agents):
            if self.agent_type[agent_id] == 1:
                speed_curr_t_y[agent_id] = self.float_dtype(
                    np.clip(speed_prev_t_y[agent_id] + acc_curr_t_y[agent_id], min_speed_y[agent_id], max_speed_y[agent_id]) * self.still_in_the_game[agent_id]
                )
                
                acc_curr_t_y[agent_id] = acc_curr_t_y[agent_id] * (speed_curr_t_y[agent_id] > min_speed_y[agent_id]) * (speed_curr_t_y[agent_id] < max_speed_y[agent_id])
                total_speed = pow(pow(speed_curr_t_x[agent_id], 2) + pow(speed_curr_t_y[agent_id], 2), 0.5)
                if total_speed > max_speed_x:
                    speed_curr_t_x[agent_id] = speed_curr_t_x[agent_id] / total_speed
                    speed_curr_t_y[agent_id] = speed_curr_t_y[agent_id] / total_speed
            else:
                speed_curr_t_y[agent_id] = self.float_dtype(0.0)
                acc_curr_t_y[agent_id] = self.float_dtype(0.0)
        
        frame_speed_x = speed_curr_t_x / self.rl_fps
        frame_speed_y = speed_curr_t_y / self.rl_fps
        loc_x_curr_t = self.float_dtype(
            loc_x_prev_t + frame_speed_x * np.cos(dir_curr_t) - frame_speed_y * np.sin(dir_curr_t)
        )
        loc_y_curr_t = self.float_dtype(
            loc_y_prev_t + frame_speed_x * np.sin(dir_curr_t) + frame_speed_y * np.cos(dir_curr_t)
        )

        if self.Training == False:
            for agent_id in range(self.num_agents):
                if self.agent_type[agent_id] == 1:
                    loc_x_curr_t[agent_id] = self.float_dtype(loc_x_prev_t[agent_id])
                    loc_y_curr_t[agent_id] = self.float_dtype(loc_y_prev_t[agent_id])
                    if self.Testing_type > 1:
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
        point_x = min(int(agent_x / self.grid_density + 0.5), self.map_max_x - 1)
        point_y = min(int(agent_y / self.grid_density + 0.5), self.map_max_y - 1)
        left_x = min(int(agent_x / self.grid_density), self.map_max_x - 1)
        down_y = min(int(agent_y / self.grid_density), self.map_max_y - 1)
        right_x = min(int(agent_x / self.grid_density + 1), self.map_max_x - 1)
        up_y = min(int(agent_y / self.grid_density + 1), self.map_max_y - 1)
        dx = min(agent_x / self.grid_density, self.map_max_x - 1) - left_x
        dy = min(agent_y / self.grid_density, self.map_max_y - 1) - down_y
        for i in range(self.num_angle_lidar):
            ld_distance = self.different_point_angle_distance[self.num_angle_lidar * self.map_max_x * down_y + self.num_angle_lidar * left_x + i] 
            lu_distance = self.different_point_angle_distance[self.num_angle_lidar * self.map_max_x * up_y + self.num_angle_lidar * left_x + i] 
            rd_distance = self.different_point_angle_distance[self.num_angle_lidar * self.map_max_x * down_y + self.num_angle_lidar * right_x + i] 
            ru_distance = self.different_point_angle_distance[self.num_angle_lidar * self.map_max_x * up_y + self.num_angle_lidar * right_x + i] 
            nomal_angle_distance = self.different_point_angle_distance[self.num_angle_lidar * self.map_max_x * point_y + self.num_angle_lidar * point_x + i] 
            precise_angle_distance =  dx * dy * ru_distance + (1 - dx) * dy * lu_distance + dx * (1 - dy) * rd_distance + (1 - dx) * (1 - dy) * ld_distance
            max_distance = max(max(ld_distance, lu_distance), max(rd_distance, ru_distance))
            min_distance = min(min(ld_distance, lu_distance), min(rd_distance, ru_distance))
            smallest_distance = self.grid_density *  min(min(dx, 1 - dx), min(dy, 1 - dy))
            if (max_distance - min_distance > 1.41421 * self.grid_density):
              if nomal_angle_distance - smallest_distance <= agent_size:  return True
            else:
                if precise_angle_distance <= agent_size: return True
        return False

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

    def k_nearest_neighbors_obs(self, agent_id, k):
        """
        Note: 'k_nearest_neighbors_obs' is only used when running on CPU step() only.
        When using the CUDA step function, this Python method (k_nearest_neighbors_obs)
        is also part of the step() function!
        """
        agent_ids_and_distances = []

        for ag_id in range(self.num_agents):
            if (ag_id != agent_id) and (self.still_in_the_game[ag_id]) and (self.agent_type[ag_id] == 2):
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

    def k_nearest_neighbors_goal(self, agent_id, k):
        """
        Note: 'k_nearest_neighbors_obs' is only used when running on CPU step() only.
        When using the CUDA step function, this Python method (k_nearest_neighbors_obs)
        is also part of the step() function!
        """
        agent_ids_and_distances = []

        for ag_id in range(self.num_agents):
            if (ag_id != agent_id) and (self.still_in_the_game[ag_id]) and (self.agent_type[ag_id] == 0):
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
                obs_global_states[:,:2] = 20.0
                obs_agent_size = np.zeros(self.num_obs_see, dtype=self.float_dtype)
                obs_local_angle = np.zeros(int(360 / self.spot_lidar_angle), dtype=self.float_dtype)
                self.init_obs = np.concatenate(
                    [
                        np.array([0.0]),  # speed x
                        np.array([0.0]),  # speed y
                        np.array([20.0]), # goal x
                        np.array([20.0]), # goal y
                        np.array([2.0]),  # goal z
                        np.array([0.0]),  # goal dir
                        np.array([1.0]),  # goal speed
                        np.array([0.0]),  # time
                        np.array([0.0]),  # in_game
                        np.array([0.0]),  # front_pitch
                        np.array([0.0]),  # back_pitch
                        np.array([0.0]),  # left_pitch
                        np.array([0.0]),  # right_pitch
                        np.vstack(
                            (
                                obs_global_states,
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
                nearest_neighbor_ids = self.k_nearest_neighbors_obs( agent_id, k = self.num_obs_see )
                agent_view_global_obs_ = copy.deepcopy(normalized_global_obs)
                dir_agent = self.global_state[_DIR][self.timestep][agent_id]
                diff_x = normalized_global_obs[0, nearest_neighbor_ids] - normalized_global_obs[0, agent_id]
                diff_y = normalized_global_obs[1, nearest_neighbor_ids] - normalized_global_obs[1, agent_id]
                agent_view_global_obs_[0,nearest_neighbor_ids] = diff_x * np.cos(dir_agent) + diff_y * np.sin(dir_agent)
                agent_view_global_obs_[1,nearest_neighbor_ids] = -1 * diff_x * np.sin(dir_agent) + diff_y * np.cos(dir_agent)
                agent_view_global_obs_[4,nearest_neighbor_ids] = normalized_global_obs[4, nearest_neighbor_ids] - normalized_global_obs[4, agent_id]
                obs_global_states = np.hstack(
                    (
                        agent_view_global_obs_[:, nearest_neighbor_ids],
                        np.zeros(
                            (
                                normalized_global_obs.shape[0],
                                self.num_obs_see - len(nearest_neighbor_ids),
                            )
                        ),
                    )
                )
                obs_size_agent = np.array([self.agent_size[k] + self.agent_size[agent_id] for k in nearest_neighbor_ids])
                still_in_the_game_ = np.array([self.still_in_the_game[k] for k in nearest_neighbor_ids])

                angle_ = int(0.5 + self.global_state[_DIR][self.timestep][agent_id] * 360 / (2 * np.pi) / self.spot_lidar_angle)
                left_x = min(int(self.global_state[_LOC_X][self.timestep][agent_id] / self.grid_density), self.map_max_x - 1)
                down_y = min(int(self.global_state[_LOC_Y][self.timestep][agent_id] / self.grid_density), self.map_max_y - 1)
                right_x = min(left_x + 1, self.map_max_x - 1)
                up_y = min(down_y + 1, self.map_max_y - 1)
                dx = float(self.global_state[_LOC_X][self.timestep][agent_id] / self.grid_density) - left_x
                dy = float(self.global_state[_LOC_Y][self.timestep][agent_id] / self.grid_density) - down_y
                obs_local_angle = np.zeros(self.num_angle_lidar, dtype=self.float_dtype)
                for k in range(self.num_angle_lidar):
                    real_angle = k + angle_
                    if real_angle >= self.num_angle_lidar: real_angle = real_angle - self.num_angle_lidar
                    if real_angle < 0: real_angle = real_angle + self.num_angle_lidar
                    ld_distance = self.different_point_angle_distance[self.num_angle_lidar * self.map_max_x * down_y + self.num_angle_lidar * left_x + real_angle]
                    lu_distance = self.different_point_angle_distance[self.num_angle_lidar * self.map_max_x * up_y + self.num_angle_lidar * left_x + real_angle]
                    rd_distance = self.different_point_angle_distance[self.num_angle_lidar * self.map_max_x * down_y + self.num_angle_lidar * right_x + real_angle]
                    ru_distance = self.different_point_angle_distance[self.num_angle_lidar * self.map_max_x * up_y + self.num_angle_lidar * right_x + real_angle]
                    precise_angle_distance = dx * dy * ru_distance + (1 - dx) * dy * lu_distance + dx * (1 - dy) * rd_distance + (1 - dx) * (1 - dy) * ld_distance
                    max_distance = max(max(ld_distance, lu_distance), max(rd_distance, ru_distance))
                    min_distance = min(min(ld_distance, lu_distance), min(rd_distance, ru_distance))
                    smallest_distance = min(min(dx, 1 - dx), min(dy, 1 - dy))
                    if (max_distance - min_distance > 1.415 * self.grid_density): obs_local_angle[k] = min_distance - smallest_distance
                    else: obs_local_angle[k] = precise_angle_distance
                agent_x = self.global_state[_LOC_X][self.timestep][agent_id]
                agent_y =self.global_state[_LOC_Y][self.timestep][agent_id]
                cos_dir = np.cos(dir_agent) * self.four_sign_distance
                sin_dir = np.sin(dir_agent) * self.four_sign_distance
                point_x = min(int(agent_x / self.grid_density + 0.5), self.map_max_x - 1)
                point_y = min(int(agent_y / self.grid_density + 0.5), self.map_max_y - 1)
                front_x = max(min(int((agent_x + cos_dir) / self.grid_density + 0.5), self.map_max_x - 1), 0)
                front_y = max(min(int((agent_y + sin_dir) / self.grid_density + 0.5), self.map_max_y - 1), 0)
                back_x = max(min(int((agent_x - cos_dir) / self.grid_density + 0.5), self.map_max_x - 1), 0)
                back_y = max(min(int((agent_y - sin_dir) / self.grid_density + 0.5), self.map_max_y - 1), 0)
                left_x = max(min(int((agent_x - sin_dir) / self.grid_density + 0.5), self.map_max_x - 1), 0)
                left_y = max(min(int((agent_y + cos_dir) / self.grid_density + 0.5), self.map_max_y - 1), 0)
                right_x = max(min(int((agent_x + sin_dir) / self.grid_density + 0.5), self.map_max_x - 1), 0)
                right_y = max(min(int((agent_y - cos_dir) / self.grid_density + 0.5), self.map_max_y - 1), 0)
                agent_z = self.map_z[point_y, point_x] 
                front_agent_z = self.map_z[front_y, front_x]
                back_agent_z = self.map_z[back_y, back_x]
                left_agent_z = self.map_z[left_y, left_x]
                right_agent_z = self.map_z[right_y, right_x]

                cos_dir = np.cos(dir_agent) * self.four_sign_distance * 0.15
                sin_dir = np.sin(dir_agent) * self.four_sign_distance * 0.15
                front_x = max(min(int((agent_x + cos_dir) / self.grid_density + 0.5), self.map_max_x - 1), 0)
                front_y = max(min(int((agent_y + sin_dir) / self.grid_density + 0.5), self.map_max_y - 1), 0)
                back_x = max(min(int((agent_x - cos_dir) / self.grid_density + 0.5), self.map_max_x - 1), 0)
                back_y = max(min(int((agent_y - sin_dir) / self.grid_density + 0.5), self.map_max_y - 1), 0)
                left_x = max(min(int((agent_x - sin_dir) / self.grid_density + 0.5), self.map_max_x - 1), 0)
                left_y = max(min(int((agent_y + cos_dir) / self.grid_density + 0.5), self.map_max_y - 1), 0)
                right_x = max(min(int((agent_x + sin_dir) / self.grid_density + 0.5), self.map_max_x - 1), 0)
                right_y = max(min(int((agent_y - cos_dir) / self.grid_density + 0.5), self.map_max_y - 1), 0)
                close_front_agent_z = self.map_z[front_y, front_x]
                close_back_agent_z = self.map_z[back_y, back_x]
                close_left_agent_z = self.map_z[left_y, left_x]
                close_right_agent_z = self.map_z[right_y, right_x]

                obs_0 = np.array([0.0])
                obs_1 = np.array([0.0])
                obs_2 = np.array([0.0])
                obs_3 = np.array([0.0])
                obs_4 = np.array([0.0])
                obs_5 = np.array([0.0])
                obs_6 = np.array([0.0])
                obs_7 = np.array([close_front_agent_z - agent_z])
                obs_8 = np.array([close_back_agent_z - agent_z])
                obs_9 = np.array([close_left_agent_z - agent_z])
                obs_10 = np.array([close_right_agent_z - agent_z])
                
                if agent_types[agent_id] != 0 : # spot
                    obs_0 = np.array([self.global_state[_SP_X][self.timestep][agent_id] / (self.spot_max_speed_x + self.eps)])
                    obs_1 = np.array([self.global_state[_SP_Y][self.timestep][agent_id] / (self.spot_max_speed_y + self.eps)])
                    agent_map_x = min(int(self.global_state[_LOC_X][self.timestep][agent_id] / self.grid_density + 0.5), self.map_max_x - 1)
                    agent_map_y = min(int(self.global_state[_LOC_Y][self.timestep][agent_id] / self.grid_density + 0.5), self.map_max_y - 1)
                    agent_map_z = self.map_z[agent_map_y, agent_map_x]
                    nearest_goal = self.k_nearest_neighbors_goal( agent_id, k = 1)
                    for k in nearest_goal:
                        diff_goal_x_ = self.global_state[_LOC_X][self.timestep][k] - self.global_state[_LOC_X][self.timestep][agent_id]
                        diff_goal_y_ = self.global_state[_LOC_Y][self.timestep][k] - self.global_state[_LOC_Y][self.timestep][agent_id]
                        goal_map_x = min(int(self.global_state[_LOC_X][self.timestep][k] / self.grid_density + 0.5), self.map_max_x - 1)
                        goal_map_y = min(int(self.global_state[_LOC_Y][self.timestep][k] / self.grid_density + 0.5), self.map_max_y- 1)
                        goal_map_z = self.map_z[goal_map_y, goal_map_x]
                        diff_dir = self.global_state[_DIR][self.timestep][k] - self.global_state[_DIR][self.timestep][agent_id]
                        agent_view_x = diff_goal_x_ * np.cos(dir_agent) + diff_goal_y_ * np.sin(dir_agent)
                        agent_view_y = -1 * diff_goal_x_ * np.sin(dir_agent) + diff_goal_y_ * np.cos(dir_agent)
                        obs_2 = np.array([agent_view_x]) # goal x
                        obs_3 = np.array([agent_view_y]) # goal y
                        obs_4 = np.array([goal_map_z - agent_map_z]) # goal z
                        obs_5 = np.array([diff_dir]) / (2 * np.pi) # goal dir
                        obs_6 = np.array([self.global_state[_SP_X][self.timestep][k] / (self.spot_max_speed_x + self.eps)]) # goal speed

                
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
                        obs_0,
                        obs_1,
                        obs_2,
                        obs_3,
                        obs_4,
                        obs_5,
                        obs_6,
                        obs_7,
                        obs_8,
                        obs_9,
                        obs_10,
                        time,
                        np.array([1.0]),
                        np.array([front_agent_z - agent_z]),
                        np.array([back_agent_z - agent_z]),
                        np.array([left_agent_z - agent_z]),
                        np.array([right_agent_z - agent_z]),
                        np.vstack(
                            (
                                obs_global_states,
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
        spots_head_locations_x = self.global_state[_LOC_X][self.timestep][spots_list] + self.spot_head_distance * np.cos(self.global_state[_DIR][self.timestep][spots_list])
        spots_ass_locations_x = self.global_state[_LOC_X][self.timestep][spots_list] - self.spot_ass_distance * np.cos(self.global_state[_DIR][self.timestep][spots_list])


        obstacles_locations_y = self.global_state[_LOC_Y][self.timestep][obstacles_list]
        goals_locations_y = self.global_state[_LOC_Y][self.timestep][goals_list]
        spots_locations_y = self.global_state[_LOC_Y][self.timestep][spots_list]
        spots_head_locations_y = self.global_state[_LOC_Y][self.timestep][spots_list] + self.spot_head_distance * np.sin(self.global_state[_DIR][self.timestep][spots_list])
        spots_ass_locations_y = self.global_state[_LOC_Y][self.timestep][spots_list] - self.spot_ass_distance * np.sin(self.global_state[_DIR][self.timestep][spots_list])
        

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

        spots_head_to_obstacles_distances = np.sqrt(
            (
                np.repeat(spots_head_locations_x, self.num_obstacle)
                - np.tile(obstacles_locations_x, self.num_spot)
            )
            ** 2
            + (
                np.repeat(spots_head_locations_y, self.num_obstacle)
                - np.tile(obstacles_locations_y, self.num_spot)
            )
            ** 2
        ).reshape(self.num_spot, self.num_obstacle)

        spots_ass_to_obstacles_distances = np.sqrt(
            (
                np.repeat(spots_ass_locations_x, self.num_obstacle)
                - np.tile(obstacles_locations_x, self.num_spot)
            )
            ** 2
            + (
                np.repeat(spots_ass_locations_y, self.num_obstacle)
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
        
        min_spots_head_to_obstacles_distances = np.min(
            spots_head_to_obstacles_distances, axis=1
        )
        argmin_spots_head_to_obstacles_distances = np.argmin(
            spots_head_to_obstacles_distances, axis=1
        )
        nearest_obstacles_spots_head_ids = [
            obstacles_list[idx] for idx in argmin_spots_head_to_obstacles_distances
        ]

        min_spots_ass_to_obstacles_distances = np.min(
            spots_ass_to_obstacles_distances, axis=1
        )
        argmin_spots_ass_to_obstacles_distances = np.argmin(
            spots_ass_to_obstacles_distances, axis=1
        )
        nearest_obstacles_spots_ass_ids = [
            obstacles_list[idx] for idx in argmin_spots_ass_to_obstacles_distances
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
            if min_spots_to_obstacles_distances[idx] < self.agent_size[idx] + self.agent_size[nearest_obstacles_spots_ids[idx]] or min_spots_head_to_obstacles_distances[idx] < self.agent_size[idx] + self.agent_size[nearest_obstacles_spots_head_ids[idx]] or min_spots_ass_to_obstacles_distances[idx] < self.agent_size[idx] + self.agent_size[nearest_obstacles_spots_ass_ids[idx]]:
                rew[spots_id] += float(self.spot_penalty_hit)
                self.still_in_the_game[spots_id] = 0
                self.num_hit_obstacle += 1
                del self.spots[spots_id]
                self.num_spot -= 1

            # 靜態障礙物   
            compute_hit_wall_ = self.compute_hit_wall(
                self.global_state[_LOC_X][self.timestep][spots_id],
                self.global_state[_LOC_Y][self.timestep][spots_id],
                self.agent_size[spots_id]
            )
            if compute_hit_wall_ == False:
                compute_hit_wall_ = self.compute_hit_wall(
                    self.global_state[_LOC_X][self.timestep][spots_id] + self.spot_head_distance * np.cos(self.global_state[_DIR][self.timestep][spots_id]),
                    self.global_state[_LOC_Y][self.timestep][spots_id] + self.spot_head_distance * np.sin(self.global_state[_DIR][self.timestep][spots_id]), 
                    self.agent_size[spots_id]
                )
            if compute_hit_wall_ == False:
                compute_hit_wall_ = self.compute_hit_wall(
                    self.global_state[_LOC_X][self.timestep][spots_id] - self.spot_ass_distance * np.cos(self.global_state[_DIR][self.timestep][spots_id]),
                    self.global_state[_LOC_Y][self.timestep][spots_id] - self.spot_ass_distance * np.sin(self.global_state[_DIR][self.timestep][spots_id]), 
                    self.agent_size[spots_id]
                )
            if compute_hit_wall_:
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
                    rew[spots_id] += self.spot_cost_operate

            # Floor reward
            agent_grid_x = min(int(self.global_state[_LOC_X][self.timestep][spots_id] / self.grid_density + 0.5), self.map_max_x - 1)
            agent_grid_y = min(int(self.global_state[_LOC_Y][self.timestep][spots_id] / self.grid_density + 0.5), self.map_max_x - 1)
            self_agent_id = self.map_max_x * agent_grid_y + agent_grid_x
            nomal_speed_x = abs(self.global_state[_SP_X][self.timestep][spots_id]) / self.spot_max_speed_x
            if abs(self.grid_map_z_[self_agent_id]) > 0 and abs(self.grid_map_z_[self_agent_id]) < self.grid_length_height:
                dir_agent = self.global_state[_DIR][self.timestep][spots_id]
                dir_dx = np.cos(dir_agent) 
                dir_dy = np.sin(dir_agent)
                if self.map_dx(agent_grid_x, agent_grid_y) * dir_dx + self.map_dy(agent_grid_x, agent_grid_y) * dir_dy > 0:
                    rew[spots_id] += self.spot_reward_on_floor * nomal_speed_x
                else:
                    rew[spots_id] += float(self.spot_penalty_hit)

        for idx, goal_id in enumerate(goals_list):
            hit_goal = False
            # 動態障礙物
            if min_goals_to_obstacles_distances < self.agent_size[idx] + self.agent_size[nearest_obstacles_goals_ids[idx]]:
                rew[goal_id] += float(self.goal_penalty_hit)
                hit_goal = True
                
            # 靜態障礙物
            compute_hit_wall_ = self.compute_hit_wall(
                self.global_state[_LOC_X][self.timestep][goal_id],
                self.global_state[_LOC_Y][self.timestep][goal_id],
                self.agent_size[goal_id]
            )
            if compute_hit_wall_:
                #rew[goal_id] += float(self.goal_penalty_hit)
                hit_goal = True

            # SPOT reward
            if min_goals_to_spots_distances[idx] < self.agent_size[idx] + self.agent_size[nearest_spots_goals_ids[idx]]:
                #rew[goal_id] += 0 #float(self.goal_penalty_hit) # goal 在 observation 看不到 spot
                rew[nearest_spots_goals_ids[idx]] += self.spot_reward_get_goal
                #self.still_in_the_game[goal_id] = 0
                self.global_state[_SIG][self.timestep :, goal_id] = 0
                self.num_get_goal += 1
                hit_goal = True
                if self.Training:
                    del self.goals[goal_id]
                    self.num_goals -= 1

            # Step reward
            rew[goal_id] += self.goals_reward_step

            # Move reward
            nomal_speed_x = abs(self.global_state[_SP_X][self.timestep][goal_id]) / self.spot_max_speed_x  # goal acc need to x skill
            nomal_acc_x = abs(self.global_state[_ACC_X][self.timestep][goal_id]) / self.spot_max_acceleration_x
            goal_move_reward = 2 * self.goals_reward_speed * (nomal_speed_x - 0.3 * nomal_acc_x)
            rew[goal_id] += goal_move_reward

            # Floor reward
            agent_grid_x = min(int(self.global_state[_LOC_X][self.timestep][goal_id] / self.grid_density + 0.5), self.map_max_x - 1)
            agent_grid_y = min(int(self.global_state[_LOC_Y][self.timestep][goal_id] / self.grid_density + 0.5), self.map_max_y - 1)

            self_agent_id = self.map_max_x * agent_grid_y + agent_grid_x
            if abs(self.grid_map_z_[self_agent_id]) > 0 and abs(self.grid_map_z_[self_agent_id]) < self.grid_length_height and hit_goal == False:
                rew[goal_id] += self.reward_on_floor * nomal_speed_x
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
        data_dict.add_data(name="rl_fps", data=self.rl_fps)
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
        data_dict.add_data(name="spot_reward_on_floor", data=self.spot_reward_on_floor)
        data_dict.add_data(name="spot_penalty_hit", data=self.spot_penalty_hit)
        data_dict.add_data(name="goal_penalty_hit", data=self.goal_penalty_hit)
        data_dict.add_data(name="spot_cost_operate", data=self.spot_cost_operate)
        data_dict.add_data(name="spot_cost_speed_y_rate_with_acc_x", data=self.spot_cost_speed_y_rate_with_acc_x)
        data_dict.add_data(name="spot_cost_turn_rate_with_acc_x", data=self.spot_cost_turn_rate_with_acc_x)
        data_dict.add_data(name="goals_reward_step", data=self.goals_reward_step)
        data_dict.add_data(name="goals_reward_speed", data=self.goals_reward_speed)
        data_dict.add_data(name="reward_on_floor", data=self.reward_on_floor)
        grid_length_max_y = self.float_dtype(self.map_max_x)
        data_dict.add_data(name="grid_length_max_y", data=grid_length_max_y)
        data_dict.add_data(name="grid_length_height", data=self.grid_length_height)
        data_dict.add_data(name="four_sign_distance", data = self.four_sign_distance)
        data_dict.add_data(name="spot_head_distance", data = self.spot_head_distance)
        data_dict.add_data(name="spot_ass_distance", data = self.spot_ass_distance)
        data_dict.add_data(name="random_agent_seed", data=self.random_agent_seed)
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
        self.set_global_state( key=_LOC_X, value=self.starting_location_x, t=self.timestep )
        self.set_global_state( key=_LOC_Y, value=self.starting_location_y, t=self.timestep )
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
                "_done_",
                "_timestep_",
                "num_spot",
                "num_hit_obstacle",
                "num_get_goal",
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
                "grid_map_z",
                "different_point_angle_distance",
                "Testing_type",
                "Training",
                "Training_Phase",
                "grid_length",
                "grid_density",
                "num_obs_see",
                "rl_fps",
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
                "spot_reward_on_floor",
                "spot_penalty_hit",
                "goal_penalty_hit",
                "spot_cost_operate",
                "spot_cost_speed_y_rate_with_acc_x",
                "spot_cost_turn_rate_with_acc_x",
                "goals_reward_step",
                "goals_reward_speed",
                "reward_on_floor",
                "grid_length_max_y",
                "grid_length_height",
                "four_sign_distance",
                "spot_head_distance",
                "spot_ass_distance",
                "random_agent_seed",
                ("n_agents", "meta"),
                ("episode_length", "meta"),
                _ACTIONS,
                "still_in_the_game",
                "neighbor_ids_sorted_by_distance",
                "nearest_neighbor_ids",
                "agent_types"
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
