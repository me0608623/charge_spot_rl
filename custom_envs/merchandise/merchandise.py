"""
    這個環境需需要訓練 agent 的交易能力，此環境採用股票交易方式，
    訓練 agent 在面對不同的 市場狀態 和 自身需求 的情況下該更和價錢進行買賣才能獲得更高的效益
    agent 輸入
        市場 在之前的定價，單位
        agent 估計的未來時間 定價，單位(標準差和平均)
        自身條件 
            物品數量
            現有資金
            未來產出
            未來需求
            定價 - 實際值
    agent 輸出
        買 
            前一時刻定價 * n 要買
            購買單位 k
        賣
            前一時刻定價 * n 要賣
            賣出單位 k
        n 為
            [0.17, 0.3, 0.45, 0.63, 0.77, 0.87, 0.94, 0.97, 0.99, 1.0, 1.01, 1.03, 1.07, 1.15, 1.3, 1.6, 2.2, 3.4, 5.8] 
        k 為
            [0.01, 0.015, 0.023, 0.035, 0.053, 0.08, 0.12, 0.18, 0.27, 0.4, 0.6, 0.9, 1.35, 2, 3] x 上一時刻成交單位
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
from custom_envs.merchandise.electricity_price import Env


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


class Merchandise_Env(CUDAEnvironmentContext):
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
        num_investor_t1 = 5,
        num_investor_t2 = 0,
        num_investor_t3 = 0,
        num_investor_t4 = 0,
        num_investor_t5 = 0,
        num_fool = 2,
        agent_demand = None,
        agent_produce_ability = None,
        agent_num_goods = None,
        agent_max_num_goods = None,
        random_need_goods = 1,  # agent 平均需求(純量)
        random_produce_ability = 0.7,  # agent 平均需求(純量)
        random_num_goods = 0.2,
        random_max_num_goods = 3.0,
    ):
        self.float_dtype = np.float32
        self.int_dtype = np.int32
        self.eps = self.float_dtype(1e-10)
        self.PHASE = self.float_dtype(PHASE)

        self.num_fool = num_fool
        self.num_investor_t1 = num_investor_t1
        self.num_investor_t2 = num_investor_t2
        self.num_investor_t3 = num_investor_t3
        self.num_investor_t4 = num_investor_t4
        self.num_investor_t5 = num_investor_t5

        self.num_agents = self.num_fool + self.num_investor_t1 + self.num_investor_t2 + self.num_investor_t3 + self.num_investor_t4 + self.num_investor_t5

        # Angent Setting
        investors_ = self.np_random.choice(
            np.arange(self.num_agents), self.num_agents - self.num_fool, replace = False
        )


        self.agent_type = {}
        self.fools = {}
        self.investors = {}

        for agent_id in range(self.num_agents):
            if agent_id < self.num_fool:
                self.agent_type[agent_id] = 0  # fool
                self.fools[agent_id] = True
            else:
                self.investors[agent_id] = True 
                if agent_id < self.num_fool + self.num_investor_t1:  self.agent_type[agent_id] = 1  # investor_t1
                elif agent_id < self.num_fool + self.num_investor_t1 + self.num_investor_t2:  self.agent_type[agent_id] = 2  # investor_t2
                elif agent_id < self.num_fool + self.num_investor_t1 + self.num_investor_t2 + self.num_investor_t3:  self.agent_type[agent_id] = 3  # investor_t3
                elif agent_id < self.num_fool + self.num_investor_t1 + self.num_investor_t2 + self.num_investor_t3 + self.num_investor_t4:  self.agent_type[agent_id] = 4  # investor_t4
                else:  self.agent_type[agent_id] = 5  # investor_t5

        # 隨機生成交易需求 (產出，需求，物件數量，存量上限) 單位都是(純量)
        
        if all(x is None for x in [agent_average_demand, agent_produce_ability, agent_num_goods, agent_max_num_goods])
            agent_demand  = random_need_goods * self.np_random.random(self.num_agents)
            agent_produce_ability = random_produce_ability * self.np_random.random(self.num_agents)
            agent_num_goods = random_num_goods * self.np_random.random(self.num_agents)
            agent_max_num_goods = random_max_num_goods * self.np_random.random(self.num_agents)
            if agent_max_num_goods > agent_produce_ability: agent_max_num_goods = agent_produce_ability
            if agent_num_goods > agent_max_num_goods: agent_num_goods = agent_max_num_goods

        self.agent_average_demand = agent_average_demand  # 存量
        self.agent_produce_ability = agent_produce_ability
        self.agent_num_goods = agent_num_goods  # init goods
        self.agent_max_num_goods = agent_max_num_goods

        # Set Action
        assert action_price_levels >= 0 
        assert action_num_levels >= 0 
        self.action_price_levels = action_price_levels
        self.action_num_levels = action_num_levels

        # Acceleration actions
        self.action_buy_price = np.array([0.17, 0.3, 0.45, 0.63, 0.77, 0.87, 0.94, 0.97, 0.99, 1.0, 1.01, 1.03, 1.07, 1.15, 1.3, 1.6, 2.2, 3.4, 5.8])
        self.action_sell_price = np.array([0.17, 0.3, 0.45, 0.63, 0.77, 0.87, 0.94, 0.97, 0.99, 1.0, 1.01, 1.03, 1.07, 1.15, 1.3, 1.6, 2.2, 3.4, 5.8])
        self.action_buy_num = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # 基於 agent_max_num_goods
        self.action_sell_num = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # 基於 agent_max_num_goods

        # These will be set during reset (see below)
        self.timestep = None
        self.global_state = None

        # Defining observation and action spaces
        self.observation_space = None  # Note: this will be set via the env_wrapper
        self.action_space = {
            agent_id: spaces.MultiDiscrete(
                (
                    len(self.action_buy_price),
                    len(self.action_sell_price),
                    len(self.action_buy_num),
                    len(self.action_sell_num),
                )
            )
            for agent_id in range(self.num_agents)
        }

        display_reward = np.zeros(14 * self.num_agents, dtype=self.float_dtype)
        self.display_reward = self.float_dtype(display_reward)

    name = "Merchandise_Env"


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
            每個 Agent 看到的自身資訊應該基於自身上限
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
                        np.array([0.0]),  # goal z
                        np.array([0.0]),  # goal dir
                        np.array([1.0]),  # goal speed
                        np.array([0.0]),  # front short
                        np.array([0.0]),  # front far
                        np.array([0.0]),  # front right
                        np.array([0.0]),  # front left
                        np.array([0.0]),  # back short
                        np.array([0.0]),  # back far
                        np.array([0.0]),  # back right
                        np.array([0.0]),  # back left
                        np.array([0.0]),  # time
                        np.array([0.0]),  # in game
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
                        obs_11,
                        obs_12,
                        obs_13,
                        obs_14,
                        time,
                        np.array([1.0]),
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
        data_dict.add_data(name="stair_position", data=self.stair_position)
        num_stair = self.float_dtype(self.num_stairs)
        data_dict.add_data(name="num_stairs", data=num_stair)
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
                "stair_position",
                "num_stairs",
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
