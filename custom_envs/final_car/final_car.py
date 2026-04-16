# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import copy
import heapq
import random
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
_SP = "speed"
_DIR = "direction"
_ACC = "acceleration"
_SIG = "still_in_the_game"


class FinalCar(CUDAEnvironmentContext):
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
        Phase = 1,
        Training = False,
        num_taggers = 1,
        num_walls = 10,
        num_goals = 10,
        num_other_car = 0,
        grid_length = 15.0,
        episode_length = 100,
        starting_location_x = None,
        starting_location_y = None,
        starting_directions = None,
        agent_size = None,
        seed = None,
        max_speed = 0.12, 
        min_speed = -0.0,
        skill_level_wall = 1.0,
        skill_level_goal = 0.0,
        skill_level_tagger = 1.0,
        max_acceleration = 0.2,  # m/s
        min_acceleration = -0.1, # m/s
        max_turn = np.pi / 15,  # np.pi /12
        min_turn = -np.pi / 15,  # np.pi /12
        num_acceleration_levels = 20,
        num_turn_levels = 20,
        num_other_agents_observed = 10,
        tagging_distance = 0.1,
        reward_tagger_get_goal = 20.0,
        reward_car_end_the_game = 0.0,
        reward_limit_acceleration = 0.05,
        reward_limit_turn = 0.05,
        penalty_agent_hit = -15.0,
        penalty_goal_be_catch = -0.0,
        step_reward_for_tagger = 0.1,
        step_reward_for_wall = 0.0,
        step_reward_for_goal = 0.0,
        use_cuda = False,
        Testing_type = 1
    ):
        """

        """
        super().__init__()
        self.Training = Training
        self.float_dtype = np.float32
        self.int_dtype = np.int32
        self.eps = self.float_dtype(1e-10)

        self.distance_eff_resize = self.float_dtype(1.0)
        self.testing_acceleration_rw_eff = self.float_dtype(1)    # 0 :close the acceleration and trun reward
        self.testing_observation_eff = self.float_dtype(1)  # 0 : look agent types  1:look can catch
        self.give_car_one_fix_goal = self.float_dtype(1)
        #reward_car_end_the_game = 5 * episode_length / 100
        if self.Training == True:
            assert num_taggers > 0
            assert num_walls > 0
            assert num_goals > 0
        elif Testing_type == 1:
            if num_taggers != 1 or num_goals != 1 :
                print("testing number tagger and goal = 1 ")
            num_taggers = 1
            num_goals = 1
        elif Testing_type == 2:
            episode_length = 10000
            num_taggers = 1
            num_goals = 1
        self.Testing_type = Testing_type

        self.num_taggers = num_taggers  
        self.num_walls = num_walls
        self.num_goals = num_goals
        self.num_other_car = num_other_car

        self.num_agents = self.num_taggers + self.num_walls + self.num_goals + self.num_other_car

        assert episode_length > 0
        self.episode_length = episode_length

        # Seeding
        self.np_random = np.random
        if seed is not None:
            self.seed(seed)

        assert 0 <= tagging_distance <= 1
        self.distance_margin = tagging_distance

        Phase_ = Phase
        Phase = int((Phase / random.random()) % 20) 

        penalty_agent_hit = penalty_agent_hit #- 3 * int(Phase_ / 269) 
        # self.distance_margin = 0.1
        # (b - a) * random_sample() + a
        agent_size = (self.np_random.rand(self.num_agents) * 0.6 + 0.8)
        grid_length = Phase + 10

        self.phase = 2


        # Starting taggers
        walls = self.np_random.choice(
            np.arange(self.num_agents), self.num_walls + self.num_other_car + self.num_goals, replace = False
        )
        walls1 = self.np_random.choice(
            walls, self.num_walls + self.num_other_car, replace = False
        )
        walls2 = self.np_random.choice(
            walls1, self.num_other_car, replace = False
        )

        self.agent_type = {}
        self.taggers = {}
        self.walls = {}
        self.other_cars = {}
        self.goals = {}

        for agent_id in range(self.num_agents):
            if agent_id in set(walls):
                if agent_id in set(walls1):
                    self.agent_type[agent_id] = 0  # wall
                    if agent_id in set(walls2):
                        self.other_cars[agent_id] = True
                        print("error ")
                    else:
                        self.walls[agent_id] = True
                else:
                    self.agent_type[agent_id] = 1  # goal
                    self.goals[agent_id] = True
                    # agent_size[agent_id] = self.distance_margin * self.distance_eff_resize
                    agent_size[agent_id] = 0.15
            else:
                self.agent_type[agent_id] = 2  # Tagger
                self.taggers[agent_id] = True
                agent_size[agent_id] = 0.35

        # Square 2D grid
        assert grid_length > 0
        self.grid_length = self.float_dtype(grid_length)
        self.grid_diagonal = self.grid_length * np.sqrt(2)

        self.agent_size = self.float_dtype(agent_size)

        if starting_location_x is None:
            assert starting_location_y is None

            starting_location_x = self.grid_length * self.np_random.rand(
                self.num_agents
            )
            starting_location_y = self.grid_length * self.np_random.rand(
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

        # Set the max speed level
        self.max_speed = self.float_dtype(max_speed)
        self.min_speed = self.float_dtype(min_speed)

        # All agents start with 0 speed and acceleration
        self.starting_speeds = np.zeros(self.num_agents, dtype=self.float_dtype)
        self.starting_accelerations = np.zeros(self.num_agents, dtype=self.float_dtype)

        assert num_acceleration_levels >= 0
        assert num_turn_levels >= 0

        self.num_acceleration_levels = num_acceleration_levels
        self.num_turn_levels = num_turn_levels
        self.max_acceleration = self.float_dtype(max_acceleration)
        self.min_acceleration = self.float_dtype(min_acceleration)

        self.max_turn = self.float_dtype(max_turn)
        self.min_turn = self.float_dtype(min_turn)
        self.goal_x = 0
        self.goal_y = 0
        # Acceleration actions
        self.acceleration_actions = np.linspace(
            self.min_acceleration, self.max_acceleration, self.num_acceleration_levels
        )
        # Add action 0 - this will be the no-op, or 0 acceleration
        self.acceleration_actions = np.insert(self.acceleration_actions, 0, 0).astype(
            self.float_dtype
        )
        # Turn actions
        self.turn_actions = np.linspace(
            self.min_turn, self.max_turn, self.num_turn_levels
        )
        # Add action 0 - this will be the no-op, or 0 turn
        self.turn_actions = np.insert(self.turn_actions, 0, 0).astype(self.float_dtype)
        # Tagger and wall agent skill levels.
        # Skill levels multiply on top of the acceleration levels
 
        self.skill_levels = [self.float_dtype(skill_level_wall) if self.agent_type[agent_id] == 0 else self.float_dtype(skill_level_goal) if self.agent_type[agent_id] == 1 else self.float_dtype(skill_level_tagger) for agent_id in range(self.num_agents)]

        # These will be set during reset (see below)
        self.timestep = None
        self.global_state = None

        # Defining observation and action spaces
        self.observation_space = None  # Note: this will be set via the env_wrapper
        self.action_space = {
            agent_id: spaces.MultiDiscrete(
                (len(self.acceleration_actions), len(self.turn_actions))
            )
            for agent_id in range(self.num_agents)
        }

        self.init_obs = None  # Will be set later in generate_observation()

        assert num_other_agents_observed <= self.num_agents
        self.num_other_agents_observed = num_other_agents_observed

        # Rewards and penalties
        assert reward_tagger_get_goal >= 0
        assert reward_car_end_the_game >= 0
        assert penalty_agent_hit <= 0
        assert penalty_goal_be_catch <= 0
        assert step_reward_for_wall >= 0
        assert step_reward_for_goal >= 0
        assert step_reward_for_tagger >= 0

        self.reward_tagger_get_goal = self.float_dtype(reward_tagger_get_goal)
        self.penalty_agent_hit = self.float_dtype(penalty_agent_hit)

        self.reward_car_end_the_game = self.float_dtype(reward_car_end_the_game)
        self.penalty_goal_be_catch = self.float_dtype(penalty_goal_be_catch)

        self.step_reward_for_wall = self.float_dtype(step_reward_for_wall)
        self.step_reward_for_goal = self.float_dtype(step_reward_for_goal)
        self.step_reward_for_tagger = self.float_dtype(step_reward_for_tagger)
        self.step_rewards = [self.step_reward_for_tagger if self.agent_type[agent_id] == 2 else self.step_reward_for_wall if self.agent_type[agent_id] == 0 else self.step_reward_for_goal for agent_id in range(self.num_agents)]
        
        self.reward_limit_acceleration = self.float_dtype(
            reward_limit_acceleration
        )
        self.reward_limit_turn = self.float_dtype(
            reward_limit_turn
        )

        self.still_in_the_game = None
        self.use_cuda = use_cuda
        self.goal_at_reset = copy.deepcopy(self.goals)
        self.taggers_at_reset = copy.deepcopy(self.taggers)

    name = "FinalCar"

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

    def update_state(self, delta_accelerations, delta_turns):
        """
        """
        loc_x_prev_t = self.global_state[_LOC_X][self.timestep - 1]
        loc_y_prev_t = self.global_state[_LOC_Y][self.timestep - 1]
        speed_prev_t = self.global_state[_SP][self.timestep - 1]
        dir_prev_t = self.global_state[_DIR][self.timestep - 1]
        acc_prev_t = self.global_state[_ACC][self.timestep - 1]
        still_in_game_prev_t = self.global_state[_SIG][self.timestep - 1]
        # Update direction and acceleration
        # Do not update location if agent is out of the game !
        #print(delta_accelerations)
        dir_curr_t = (
            (dir_prev_t + delta_turns) % (2 * np.pi) * self.still_in_the_game
        ).astype(self.float_dtype)
        acc_curr_t = acc_prev_t + delta_accelerations

        # 0 <= speed <= max_speed (multiplied by the skill levels).
        # Reset acceleration to 0 when speed is outside this range
        max_speed = self.max_speed * np.array(self.skill_levels)
        min_speed = self.min_speed * np.array(self.skill_levels)
        speed_curr_t = self.float_dtype(
            np.clip(speed_prev_t + acc_curr_t, min_speed, max_speed) * self.still_in_the_game
        )
        
        acc_curr_t = acc_curr_t * (speed_curr_t > min_speed) * (speed_curr_t < max_speed)
        loc_x_curr_t = self.float_dtype(
            loc_x_prev_t + speed_curr_t * np.cos(dir_curr_t)
        )
        loc_y_curr_t = self.float_dtype(
            loc_y_prev_t + speed_curr_t * np.sin(dir_curr_t)
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
        self.set_global_state(key = _SP, value = speed_curr_t, t=self.timestep)
        self.set_global_state(key = _DIR, value = dir_curr_t, t=self.timestep)
        self.set_global_state(key = _ACC, value = acc_curr_t, t=self.timestep)
 
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
                : self.num_other_agents_observed
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
            (_SP, self.max_speed + self.eps),
            #(_ACC, self.max_acceleration + self.eps),
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
                        self.num_other_agents_observed,
                    )
                )
                obs_agent_types = np.zeros(self.num_other_agents_observed)
                obs_still_in_the_game = np.zeros(self.num_other_agents_observed)
                obs_agent_size = np.zeros(self.num_other_agents_observed)

                # Form the observation
                self.init_obs = np.concatenate(
                    [
                        np.array([0.0]),  # acceleration
                        np.array([0.0]),  # speed
                        np.array([0.0]),  # dir
                        np.array([0.0]),  # size
                        np.array([0.0]),  # goal x
                        np.array([0.0]),  # goal y
                        np.array([0.0]),  # agent x
                        np.array([0.0]),  # agent y
                        np.array([0.0]),  # time
                        np.vstack(
                            (
                                obs_global_states,
                                obs_agent_types,
                                obs_still_in_the_game,
                                obs_agent_size
                            )
                        ).reshape(-1),
                    ]
                )

            # Initialize obs to all zeros
            obs[agent_id] = self.init_obs
            
            # Set obs for agents still in the game
            if self.still_in_the_game[agent_id]:
                nearest_neighbor_ids = self.k_nearest_neighbors(
                    agent_id, k=self.num_other_agents_observed
                )
                normalized_global_obs_ = copy.deepcopy(normalized_global_obs)
                normalized_global_obs_[0,nearest_neighbor_ids] = normalized_global_obs[0, nearest_neighbor_ids] - int(normalized_global_obs[0, agent_id])
                normalized_global_obs_[1,nearest_neighbor_ids] = normalized_global_obs[1, nearest_neighbor_ids] - int(normalized_global_obs[1, agent_id])

                # For the case when the number of remaining agent ids is fewer
                # than self.num_other_agents_observed (because agents have exited
                # the game), we also need to pad obs wih zeros
                obs_global_states = np.hstack(
                    (
                        normalized_global_obs_[:, nearest_neighbor_ids],
                        np.zeros(
                            (
                                normalized_global_obs.shape[0],
                                self.num_other_agents_observed
                                - len(nearest_neighbor_ids),
                            )
                        ),
                    )
                )

                obs_type_agent =  np.array([0 if agent_types[k] == 1 else 1 for k in nearest_neighbor_ids])   
                obs_size_agent = np.array([self.agent_size[k]  for k in nearest_neighbor_ids])

                
                still_in_the_game_ = np.array([self.still_in_the_game[k] for k in nearest_neighbor_ids])
                
                if agent_types[agent_id] != 2 or self.give_car_one_fix_goal == 0:
                    fix_goal_x = np.array([0.0])
                    fix_goal_y = np.array([0.0])
                else:
                    for k in nearest_neighbor_ids:
                        if agent_types[k] == 1:
                            goal_x_ = self.global_state[_LOC_X][self.timestep][k] - self.global_state[_LOC_X][self.timestep][agent_id]
                            goal_y_ = self.global_state[_LOC_Y][self.timestep][k] - self.global_state[_LOC_Y][self.timestep][agent_id]
                            fix_goal_x = np.array([goal_x_])
                            fix_goal_y = np.array([goal_y_])
                            break
                        elif k == nearest_neighbor_ids[len(nearest_neighbor_ids)-1]:
                            fix_goal_x = np.array([0.0])
                            fix_goal_y = np.array([0.0])

                obs_agent_types = np.hstack(    
                    (
                        obs_type_agent,
                        np.zeros(
                            (
                                self.num_other_agents_observed
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
                                    self.num_other_agents_observed
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
                                    self.num_other_agents_observed
                                    - len(nearest_neighbor_ids)
                                )
                            ),
                        )
                    ),
                )
                # Form the observation
                obs[agent_id] = np.concatenate(
                    [
                        np.array([self.global_state[_ACC][self.timestep][agent_id] / (self.max_acceleration + self.eps)]),
                        np.array([self.global_state[_SP][self.timestep][agent_id] / (self.max_speed + self.eps)]),
                        np.array([self.global_state[_DIR][self.timestep][agent_id] / (2 * np.pi)]),
                        np.array([self.agent_size[agent_id]]),
                        fix_goal_x,
                        fix_goal_y,
                        np.array([self.global_state[_LOC_X][self.timestep][agent_id] / self.grid_length]),
                        np.array([self.global_state[_LOC_Y][self.timestep][agent_id] / self.grid_length]),
                        time,
                        np.vstack(
                            (
                                obs_global_states,
                                obs_agent_types,
                                obs_still_in_the_game,
                                obs_agent_size
                            )
                        ).reshape(-1)
                    ]
                )
        return obs

    def compute_reward(self):
        """
            Compute and return the rewards for each agent.
            the reward didn't set car hit car penalty,because use under way to
            write the small distance will find the self by agent, so distance = 0.
            but it ok,write in cuda code is find.
        """
        # Initialize rewards
        rew = {agent_id: 0.0 for agent_id in range(self.num_agents)}

        taggers_list = sorted(self.taggers)
        walls_list = sorted(self.walls + self.other_cars)
        goals_list = sorted(self.goals)

        # At least one wall present
        wall_locations_x = self.global_state[_LOC_X][self.timestep][walls_list]
        goal_locations_x = self.global_state[_LOC_X][self.timestep][goals_list]
        tagger_locations_x = self.global_state[_LOC_X][self.timestep][taggers_list]

        wall_locations_y = self.global_state[_LOC_Y][self.timestep][walls_list]
        goal_locations_y = self.global_state[_LOC_Y][self.timestep][goals_list]
        tagger_locations_y = self.global_state[_LOC_Y][self.timestep][taggers_list]

        taggers_to_walls_distances = np.sqrt(
            (
                np.repeat(tagger_locations_x, self.num_walls)
                - np.tile(wall_locations_x, self.num_taggers)
            )
            ** 2
            + (
                np.repeat(tagger_locations_y, self.num_walls)
                - np.tile(wall_locations_y, self.num_taggers)
            )
            ** 2
        ).reshape(self.num_taggers, self.num_walls)

        goals_to_walls_distances = np.sqrt(
            (
                np.repeat(goal_locations_x, self.num_walls)
                - np.tile(wall_locations_x, self.num_goals)
            )
            ** 2
            + (
                np.repeat(goal_locations_y, self.num_walls)
                - np.tile(wall_locations_y, self.num_goals)
            )
            ** 2
        ).reshape(self.num_goals, self.num_walls)

        goal_to_taggers_distances = np.sqrt(
            (
                np.repeat(goal_locations_x, self.num_taggers)
                - np.tile(tagger_locations_x, self.num_goals)
            )
            ** 2
            + (
                np.repeat(goal_locations_y, self.num_taggers)
                - np.tile(tagger_locations_y, self.num_goals)
            )
            ** 2
        ).reshape(self.num_goals, self.num_taggers)

        min_taggers_to_walls_distances = np.min(
            taggers_to_walls_distances, axis=1
        )
        argmin_taggers_to_walls_distances = np.argmin(
            taggers_to_walls_distances, axis=1
        )
        nearest_wall_taggers_ids = [
            walls_list[idx] for idx in argmin_taggers_to_walls_distances
        ]

        min_goals_to_walls_distances = np.min(
            goals_to_walls_distances, axis=1
        )
        argmin_goals_to_walls_distances = np.argmin(
            goals_to_walls_distances, axis=1
        )
        nearest_wall_goal_ids2 = [
            walls_list[idx] for idx in argmin_goals_to_walls_distances
        ]

        min_goal_to_taggers_distances = np.min(
            goal_to_taggers_distances, axis=1
        )
        argmin_goal_to_taggers_distances = np.argmin(
            goal_to_taggers_distances, axis=1
        )
        nearest_tagger_ids2 = [
            taggers_list[idx] for idx in argmin_goal_to_taggers_distances
        ]
        
        for agent_id in range(self.num_agents):
            if self.still_in_the_game[agent_id]:
                rew[agent_id] += self.step_rewards[agent_id]
                nomal_acc_ = abs(self.global_state[_ACC][self.timestep][agent_id])/ max(self.max_acceleration, abs(self.min_acceleration))
                nomal_turn_ = abs(self.global_state[_DIR][self.timestep][agent_id] - self.global_state[_DIR][self.timestep - 1][agent_id] ) / max_turn
                rew[agent_id] += reward_limit_acceleration * (1 - nomal_acc_) ** 2
                rew[agent_id] += reward_limit_turn * (1 - nomal_turn_) ** 2

        for idx, taggers_id in enumerate(taggers_list):
            if min_taggers_to_walls_distances[idx] < self.agent_size[idx] + self.agent_size[nearest_wall_taggers_ids[idx]]:
                rew[taggers_id] += float(self.penalty_agent_hit)
                self.still_in_the_game[taggers_id] = 0
                self.num_hit_wall += 1
                del self.taggers[taggers_id]
                self.num_taggers -= 1

        for idx, goal_id in enumerate(goals_list):
            if min_goal_to_taggers_distances[idx] < self.agent_size[idx] + self.agent_size[nearest_tagger_ids2[idx]]:
                rew[goal_id] += self.penalty_goal_be_catch
                rew[nearest_tagger_ids2[idx]] += self.reward_tagger_get_goal
                self.still_in_the_game[goal_id] = 0
                self.global_state[_SIG][self.timestep :, goal_id] = 0
                self.num_get_goal += 1
                if self.Training:
                    del self.goals[goal_id]
                    self.num_goals -= 1
            if min_goals_to_walls_distances < self.agent_size[idx] + self.agent_size[nearest_wall_taggers_ids[idx]]:
                rew[agent_id] += float(self.penalty_agent_hit)

        if self.timestep == self.episode_length:
            for tagger_id in self.taggers:
                rew[tagger_id] += self.reward_car_end_the_game
        return rew

    def get_data_dictionary(self):
        """
        Create a dictionary of data to push to the device
        """
        self.still_in_the_game =[1 for agent_id in range(self.num_agents)]
        data_dict = DataFeed()
        for feature in [_LOC_X, _LOC_Y, _SP, _DIR, _ACC]:
            data_dict.add_data(
                name=feature,
                data=self.global_state[feature][0],
                save_copy_and_apply_at_reset=True,
            )
        data_dict.add_data(name="Training", data=self.Training)
        data_dict.add_data(name="testing_acceleration_rw_eff", data=self.testing_acceleration_rw_eff)
        data_dict.add_data(name="testing_observation_eff", data=self.testing_observation_eff)
        data_dict.add_data(
            name="give_car_one_fix_goal",
            data= self.give_car_one_fix_goal,
            ave_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(name="Testing_type", data=self.Testing_type)
        data_dict.add_data(name="agent_types", 
            data=[self.agent_type[agent_id] for agent_id in range(self.num_agents)],
        )
        data_dict.add_data(name="agent_size",data=self.agent_size)
        data_dict.add_data(name="grid_length", data=self.grid_length)
        data_dict.add_data(name="acceleration_actions", data=self.acceleration_actions)
        data_dict.add_data(name="turn_actions", data=self.turn_actions)
        data_dict.add_data(name="max_speed", data=self.max_speed)
        data_dict.add_data(name="min_speed", data=self.min_speed)
        data_dict.add_data(
            name="num_other_agents_observed", data=self.num_other_agents_observed
        )
        data_dict.add_data(name="skill_levels", data=self.skill_levels)
        data_dict.add_data(
            name="still_in_the_game",
            data=self.still_in_the_game,
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="neighbor_distances",
            data=np.zeros((self.num_agents, self.num_agents - 1), dtype=np.int32),
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
                (self.num_agents, self.num_other_agents_observed), dtype=np.int32
            ),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="step_rewards",
            data=self.step_rewards,
        )
        data_dict.add_data(
            name="distance_eff_resize",
            data = self.distance_eff_resize,
        )
        data_dict.add_data(
            name="reward_tagger_get_goal", data=self.reward_tagger_get_goal
        )
        
        data_dict.add_data(
            name="penalty_agent_hit", data=self.penalty_agent_hit
        )
        data_dict.add_data(
            name="penalty_goal_be_catch", data=self.penalty_goal_be_catch
        )
        data_dict.add_data(
            name="reward_car_end_the_game",
            data=self.reward_car_end_the_game,
        )
        data_dict.add_data(
            name="reward_limit_acceleration",
            data=self.reward_limit_acceleration,
        )
        data_dict.add_data(
            name="reward_limit_turn",
            data=self.reward_limit_turn,
        )
        data_dict.add_data(
            name="max_acceleration",
            data=self.max_acceleration,
        )
        data_dict.add_data(
            name="min_acceleration",
            data=self.min_acceleration,
        )
        data_dict.add_data(
            name="max_turn",
            data=self.max_turn,
        )
        self.num_hit_wall = 0
        data_dict.add_data(
            name="num_hit_wall",
            data=self.num_hit_wall,
            save_copy_and_apply_at_reset=True,
        )
        self.num_get_goal = 0
        data_dict.add_data(
            name = "num_get_goal",
            data = self.num_get_goal,
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name = "num_taggers",
            data = self.num_taggers,
            save_copy_and_apply_at_reset=True,
        )

        return data_dict

    def get_tensor_dictionary(self):
        tensor_dict = DataFeed()
        return tensor_dict

    def reset(self):
        """
        Env reset().
        """
        # Reset time to the beginning
        self.timestep = 0

        # Re-initialize the global state
        self.global_state = {}
        self.set_global_state(
            key=_LOC_X, value=self.starting_location_x, t=self.timestep
        )
        self.set_global_state(
            key=_LOC_Y, value=self.starting_location_y, t=self.timestep
        )
        self.set_global_state(key=_SP, value=self.starting_speeds, t=self.timestep)
        self.set_global_state(key=_DIR, value=self.starting_directions, t=self.timestep)
        self.set_global_state(
            key=_ACC, value=self.starting_accelerations, t=self.timestep
        )

        # Array to keep track of the agents that are still in play
        self.still_in_the_game = np.ones(self.num_agents, dtype=self.int_dtype)

        # Initialize global state for "still_in_the_game" to all ones
        self.global_state[_SIG] = np.ones(
            (self.episode_length + 1, self.num_agents), dtype=self.int_dtype
        )
        self.num_hit_wall = 0
        self.num_get_goal = 0
        self.goals = copy.deepcopy(self.goal_at_reset)
        self.num_goals = len(self.goals)
        self.taggers = copy.deepcopy(self.taggers_at_reset)
        self.num_taggers = len(self.taggers)

        return self.generate_observation()

    def step(self, actions=None):
        """
        Env step() - The GPU version calls the corresponding CUDA kernels
        """
        self.timestep += 1
        
        if self.use_cuda:
            # run train_batch_size/ num_envs
            args = [
                _LOC_X,
                _LOC_Y,
                _SP,
                _DIR,
                _ACC,
                "Training",
                "testing_acceleration_rw_eff",
                "testing_observation_eff",
                "give_car_one_fix_goal",
                "Testing_type",
                "agent_types",
                "agent_size",
                "grid_length",
                "acceleration_actions",
                "turn_actions",
                "max_speed",
                "min_speed",
                "num_other_agents_observed",
                "skill_levels",
                "still_in_the_game",
                _OBSERVATIONS,
                _ACTIONS,
                "neighbor_distances",
                "neighbor_ids_sorted_by_distance",
                "nearest_neighbor_ids",
                _REWARDS,
                "step_rewards",
                "distance_eff_resize",
                "reward_tagger_get_goal",
                "penalty_agent_hit",
                "penalty_goal_be_catch",
                "reward_car_end_the_game",
                "reward_limit_acceleration",
                "reward_limit_turn",
                "max_acceleration",
                "min_acceleration",
                "max_turn",
                "num_hit_wall",
                "num_get_goal",
                "num_taggers",
                "_done_",
                "_timestep_",
                ("n_agents", "meta"),
                ("episode_length", "meta")
            ]
            self.cuda_step(
                *self.cuda_step_function_feed(args),
                block=self.cuda_function_manager.block,
                grid=self.cuda_function_manager.grid,
            )
            result = None  # do not return anything
        else:
            assert isinstance(actions, dict)
            assert len(actions) == self.num_agents

            acceleration_action_ids = [
                actions[agent_id][0] for agent_id in range(self.num_agents)
            ]

            turn_action_ids = [
                actions[agent_id][1] for agent_id in range(self.num_agents)
            ]

            assert all(
                0 <= acc <= self.num_acceleration_levels
                for acc in acceleration_action_ids
            )
            assert all(0 <= turn <= self.num_turn_levels for turn in turn_action_ids)

            delta_accelerations = self.acceleration_actions[acceleration_action_ids]
            delta_turns = self.turn_actions[turn_action_ids]

            # Update state and generate observation
            self.update_state(delta_accelerations, delta_turns)
            if not self.use_cuda:
                obs = self.generate_observation()

            # Compute rewards and done
            rew = self.compute_reward()
            done = {
                "__all__": (self.timestep >= self.episode_length) 
                or (self.num_taggers == 0)
            }
            info = {}

            result = obs, rew, done, info
        return result
