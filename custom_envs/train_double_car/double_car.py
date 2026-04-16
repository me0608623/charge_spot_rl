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


class DoubleCar(CUDAEnvironmentContext):
    """
        this is a env from ken, in this env taggers(car) will be 
        segmentation with two group, it can chase the other group taggers,
        but can't touch the same group taggers and the wall.

        to design like this is because the goal move is better then don't move
        and i think the goal policy is no necessary, the taggers policy can do 
        same thing even better, and also train faster at taggers.
    """
    def __init__(
        self,
        Phase = 1,
        num_taggers = 4,
        num_walls = 10,
        testing_num_goals = 1,
        grid_length = 10.0,
        episode_length = 100,
        starting_location_x = None,
        starting_location_y = None,
        starting_directions = None,
        agent_size = None,
        seed = None,
        max_speed = 1.0,
        skill_level_wall = 0.2,
        skill_level_tagger = 1.0,
        max_acceleration = 0.5,
        min_acceleration = -0.5,
        max_turn = np.pi / 2,
        min_turn = -np.pi / 2,
        num_acceleration_levels = 10,
        num_turn_levels = 10,
        step_penalty_for_tagger = -0.02,
        step_reward_for_wall = 0.0,
        step_reward_for_goal = 0.01,
        num_other_agents_observed = 2,
        tagging_distance = 0.01,
        tag_reward_for_tagger = 5.0,
        tagger_hit_penalty = -1.0,
        limit_acceleration_reward = 0.02,
        limit_turn_reward = 0.02,
        use_cuda = False,
        Training = True
    ):

        super().__init__()
        self.Training = Training
        self.float_dtype = np.float32
        self.int_dtype = np.int32
        # small number to prevent indeterminate cases
        self.eps = self.float_dtype(1e-10)

        assert num_walls > 0
        self.num_walls = num_walls

        assert episode_length > 0
        self.episode_length = episode_length

        if num_taggers % 2 == 1 and self.Training == True:
            print("num_taggers = 2 x int")
            assert num_taggers % 2 == 0 
        elif self.Training == False:
            if num_taggers != 1:
                print("testing number tagger = 1 ")
            num_taggers = 1

            self.testing_num_goals = testing_num_goals
        else:
            self.testing_num_goals = 1

        assert num_taggers > 0
        self.num_taggers = num_taggers
        self.num_agents = self.num_taggers + self.num_walls + self.testing_num_goals

        #PHASE setting
        """
            PHASE1 setting
                taggers = 5 
                walls   = 10
                goals   = 40
                grid_length = 10
                skill_level_wall = 0.2
                targeting reward = 1

            PHASE2 setting
                taggers = 10 
                walls   = 10
                goals   = 40
                grid_length = 40
                targeting reward = 10
                goal model train = False
                skill_level_wall = 0.35
                can look agent size

            PHASE3 setting
                taggers = 10 
                walls   = 40
                goals   = 40
                goal model speeds
                grid_length = 60
                skill_level_goal = 0.2
                skill_level_wall = 0.4


        if Phase==1:
            #all need to change 
            hit_tagger_for_tagger_penalty = -1
            grid_length = 10
            if num_goals < num_walls * 2:
                print("--------------------------")
                print("num_goals >= num_walls * 2")
                print("--------------------------")
            if num_taggers > grid_length*grid_length / 10:
                print("--------------------------")
                print("Too many num_taggers (can be cut in half)")
                print("--------------------------")    
            assert num_goals >= num_walls * 2
            assert num_taggers < grid_length*grid_length /10
        elif Phase==2:
            hit_tagger_for_tagger_penalty = -1
            tag_reward_for_tagger = 10
            grid_length = 40
            skill_level_wall = 0.35
            if num_goals < num_walls:
                print("--------------------------")
                print("num_goals >= num_walls")
                print("--------------------------")
            assert num_goals >= num_walls
        else:
            hit_tagger_for_tagger_penalty = -1
            grid_length = 10
            print("--------------------------")
            print("Please set phase(1 or 2)")
            print("--------------------------")
        
        """
        self.phase = Phase
        # Seeding
        self.np_random = np.random
        if seed is not None:
            self.seed(seed)

        no_walls = self.np_random.choice(
            np.arange(self.num_agents), self.num_taggers + self.testing_num_goals, replace=False
        )
        cars = self.np_random.choice(
            no_walls, self.num_taggers, replace=False
        )
        if self.Training == True:
            cars_group1 = self.np_random.choice(
                cars, int(self.num_taggers / 2), replace=False
            )

        self.agent_type = {}
        self.taggers = {}
        self.car_group1 ={}
        self.car_group2 ={}
        self.walls = {}
        self.goals = {}

        if Phase==1:
            agent_size = [0 for agent_id in range(self.num_agents)]
        else:
            agent_size = self.np_random.rand(self.num_agents)  #PHASE2
        
        for agent_id in range(self.num_agents):                
            if agent_id in set(no_walls):
                if agent_id in set(cars):
                    self.taggers[agent_id] = True
                    self.agent_type[agent_id] = 0
                    if self.Training == True:
                        if agent_id in set(cars_group1):
                            # cars group1 agent_type = 0
                            self.car_group1[agent_id] = True
                        else:
                            self.agent_type[agent_id] = 1  # cars group2
                            self.car_group2[agent_id] = True
                    else:
                        self.car_group1[agent_id] = True
                else:
                    self.agent_type[agent_id] = 3  # goal
                    self.goals[agent_id] = True
                    agent_size[agent_id] = 0
            else:
                self.agent_type[agent_id] = 2  # wall
                self.walls[agent_id] = True

        # Square 2D grid
        assert grid_length > 0
        self.grid_length = self.float_dtype(grid_length)
        self.grid_diagonal = self.grid_length * np.sqrt(2)

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

        # Set the max speed level
        self.max_speed = self.float_dtype(max_speed)

        # All agents start with 0 speed and acceleration
        self.starting_speeds = np.zeros(self.num_agents, dtype=self.float_dtype)
        self.starting_accelerations = np.zeros(self.num_agents, dtype=self.float_dtype)

        assert num_acceleration_levels >= 0
        assert num_turn_levels >= 0
        display_reward = np.zeros(14 * self.num_agents, dtype=self.float_dtype)
        self.display_reward = self.float_dtype(display_reward)

        # The num_acceleration and num_turn levels refer to the number of
        # uniformly-spaced levels between (min_acceleration and max_acceleration)
        # and (min_turn and max_turn), respectively.
        self.num_acceleration_levels = num_acceleration_levels
        self.num_turn_levels = num_turn_levels
        self.max_acceleration = self.float_dtype(max_acceleration)
        self.min_acceleration = self.float_dtype(min_acceleration)

        self.max_turn = self.float_dtype(max_turn)
        self.min_turn = self.float_dtype(min_turn)

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
 
        self.skill_levels = [self.float_dtype(skill_level_wall * random.random()) if self.agent_type[agent_id] == 2 else self.float_dtype(skill_level_tagger) for agent_id in range(self.num_agents)]

        # These will be set during reset (see below)
        self.timestep = None
        self.global_state = None

        # Defining observation and action spaces
        self.observation_space = None  # Note: this will be set via the env_wrapper
        self.action_space = {
            agent_id: spaces.MultiDiscrete(
                (
                    len(self.acceleration_actions), 
                    len(self.acceleration_actions), 
                    len(self.turn_actions),
                )
            )
            for agent_id in range(self.num_agents)
        }
        # Used in generate_observation()
        # When use_full_observation is True, then all the agents will have info of
        # all the other agents, otherwise, each agent will only have info of
        # its k-nearest agents (k = num_other_agents_observed)
        self.init_obs = None  # Will be set later in generate_observation()

        assert num_other_agents_observed <= self.num_agents
        self.num_other_agents_observed = num_other_agents_observed

        # Distance margin between agents for non-zero rewards
        # If a tagger is closer than this to a wall, the tagger
        # gets a positive reward, and the wall a negative reward
        assert 0 <= tagging_distance <= 1
        self.distance_margin = (tagging_distance * self.grid_length).astype(self.float_dtype)

        # Rewards and penalties
        assert tag_reward_for_tagger >= 0
        assert tagger_hit_penalty <= 0
        assert step_penalty_for_tagger <= 0
        self.tag_reward_for_tagger = self.float_dtype(tag_reward_for_tagger)
        self.tagger_hit_penalty = self.float_dtype(tagger_hit_penalty)
        self.step_penalty_for_tagger = self.float_dtype(step_penalty_for_tagger)
        self.step_reward_for_goal = self.float_dtype(step_reward_for_goal)

        assert step_reward_for_wall >= 0
        self.step_reward_for_wall = self.float_dtype(step_reward_for_wall)

        self.step_rewards =[self.step_reward_for_wall if self.agent_type[agent_id] == 2 else self.step_penalty_for_tagger if self.agent_type[agent_id] != 3 else self.step_reward_for_goal for agent_id in range(self.num_agents)]

        self.limit_acceleration_reward = self.float_dtype(limit_acceleration_reward)
        self.limit_turn_reward = self.float_dtype(limit_turn_reward)

        self.still_in_the_game = None

        # These will also be set via the env_wrapper
        # use_cuda will be set to True (by the env_wrapper), if needed
        # to be simulated on the GPU
        self.use_cuda = use_cuda
        self.taggers_at_reset = copy.deepcopy(self.taggers)
        self.goal_at_reset = copy.deepcopy(self.goals)

    name = "DoubleCar"

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
        Note: 'update_state' is only used when running on CPU step() only.
        When using the CUDA step function, this Python method (update_state)
        is part of the step() function!

        The logic below mirrors (part of) the step function in CUDA.
        """
        loc_x_prev_t = self.global_state[_LOC_X][self.timestep - 1]
        loc_y_prev_t = self.global_state[_LOC_Y][self.timestep - 1]
        speed_prev_t = self.global_state[_SP][self.timestep - 1]
        dir_prev_t = self.global_state[_DIR][self.timestep - 1]
        acc_prev_t = self.global_state[_ACC][self.timestep - 1]

        # Update direction and acceleration
        # Do not update location if agent is out of the game !
        
        dir_curr_t = (
            (dir_prev_t + delta_turns) % (2 * np.pi) * self.still_in_the_game
        ).astype(self.float_dtype)
        acc_curr_t = acc_prev_t + delta_accelerations

        # 0 <= speed <= max_speed (multiplied by the skill levels).
        # Reset acceleration to 0 when speed is outside this range
        max_speed = self.max_speed * np.array(self.skill_levels)
        speed_curr_t = self.float_dtype(
            np.clip(speed_prev_t + acc_curr_t, 0.0, max_speed) * self.still_in_the_game
        )
        acc_curr_t = acc_curr_t * (speed_curr_t > 0) * (speed_curr_t < max_speed)

        loc_x_curr_t = self.float_dtype(
            loc_x_prev_t + speed_curr_t * np.cos(dir_curr_t)
        )
        loc_y_curr_t = self.float_dtype(
            loc_y_prev_t + speed_curr_t * np.sin(dir_curr_t)
        )

        # Clip x and y if agent has crossed edge
        clipped_loc_x_curr_t = self.float_dtype(
            np.clip(loc_x_curr_t, 0.0, self.grid_length)
        )

        clipped_loc_y_curr_t = self.float_dtype(
            np.clip(loc_y_curr_t, 0.0, self.grid_length)
        )

        # Set global states
        self.set_global_state(key=_LOC_X, value=clipped_loc_x_curr_t, t=self.timestep)
        self.set_global_state(key=_LOC_Y, value=clipped_loc_y_curr_t, t=self.timestep)
        self.set_global_state(key=_SP, value=speed_curr_t, t=self.timestep)
        self.set_global_state(key=_DIR, value=dir_curr_t, t=self.timestep)
        self.set_global_state(key=_ACC, value=acc_curr_t, t=self.timestep)

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
            (_LOC_X, self.grid_diagonal),
            (_LOC_Y, self.grid_diagonal),
            (_SP, self.max_speed + self.eps),
            (_ACC, self.max_speed + self.eps),
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
                        np.array([0.0]),  # time
                        np.array([0.0]),  # agent size
                        np.array([0.0]),
                        np.array([0.0]),
                        np.array([0.0]),
                        np.array([0.0]),
                        np.array([0.0]),
                        np.array([0.0]),
                        np.array([0.0]),
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
                # For the case when the number of remaining agent ids is fewer
                # than self.num_other_agents_observed (because agents have exited
                # the game), we also need to pad obs wih zeros
                obs_global_states = np.hstack(
                    (
                        normalized_global_obs[:, nearest_neighbor_ids]
                        - normalized_global_obs[:, agent_id].reshape(-1, 1),
                        np.zeros(
                            (
                                normalized_global_obs.shape[0],
                                self.num_other_agents_observed
                                - len(nearest_neighbor_ids),
                            )
                        ),
                    )
                )

                obs_type_agent =  np.array([ 1 if agent_types[agent_id] == 2 or agent_types[k] == agent_types[agent_id] or agent_types[k] == 2 else 0 for k in nearest_neighbor_ids])
                obs_size_agent = np.array([ 0. if agent_types[agent_id] == 2 else self.agent_size[k] + self.agent_size[agent_id] if agent_types[k] == agent_types[agent_id] or agent_types[k] == 2 else 0. for k in nearest_neighbor_ids])
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
                            self.still_in_the_game[nearest_neighbor_ids],
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
                        time,
                        np.array([self.agent_size[agent_id]]),
                        np.array([0.0]),
                        np.array([0.0]),
                        np.array([0.0]),
                        np.array([0.0]),
                        np.array([0.0]),
                        np.array([0.0]),
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
        """
        # Initialize rewards
        rew = {agent_id: 0.0 for agent_id in range(self.num_agents)}

        car_list1 = sorted(self.car_group1)
        car_list2 = sorted(self.car_group2)
        goals_list = sorted(self.goals)
        # At least one wall present
        if self.num_taggers > 0:
            walls_list = sorted(self.walls)
            wall_locations_x = self.global_state[_LOC_X][self.timestep][walls_list]
            car_list1_x = self.global_state[_LOC_X][self.timestep][car_list1]
            car_list2_x = self.global_state[_LOC_X][self.timestep][car_list2]
            goal_list_x = self.global_state[_LOC_X][self.timestep][goals_list]

            wall_locations_y = self.global_state[_LOC_Y][self.timestep][walls_list]
            car_list1_y = self.global_state[_LOC_Y][self.timestep][car_list1]
            car_list2_y = self.global_state[_LOC_Y][self.timestep][car_list2]
            goals_list_y = self.global_state[_LOC_Y][self.timestep][goals_list]

            if self.Training == True:
                num_car1 = self.num_taggers / 2
                num_car2 = self.num_taggers / 2
            else:
                num_car1 = self.num_taggers
                num_car2 = 0


            walls_to_car1_distances = np.sqrt(
                (
                    np.repeat(wall_locations_x, num_car1 )
                    - np.tile(car_list1_x, self.num_walls)
                )
                ** 2
                + (
                    np.repeat(wall_locations_y, num_car1 )
                    - np.tile(car_list1_y, self.num_walls)
                )
                ** 2
            ).reshape(self.num_walls, num_car1)

            walls_to_car2_distances = np.sqrt(
                (
                    np.repeat(wall_locations_x, num_car2 )
                    - np.tile(car_list2_x, self.num_walls)
                )
                ** 2
                + (
                    np.repeat(wall_locations_y, num_car2 )
                    - np.tile(car_list2_y, self.num_walls)
                )
                ** 2
            ).reshape(self.num_walls, num_car2)

            car1_to_car2_distances = np.sqrt(
                (
                    np.repeat(car_list1_x, num_car2)
                    - np.tile(car_list2_x, num_car1)
                )
                ** 2
                + (
                    np.repeat(car_list1_y, num_car2)
                    - np.tile(car_list2_y,num_car1)
                )
                ** 2
            ).reshape(num_car1, num_car2 )

            car1_to_car1_distances = np.sqrt(
                (
                    np.repeat(car_list1_x, num_car1)
                    - np.tile(car_list1_x, num_car1)
                )
                ** 2
                + (
                    np.repeat(car_list1_y, num_car1)
                    - np.tile(car_list1_y, num_car1)
                )
                ** 2
            ).reshape(num_car1, num_car1)

            car2_to_car2_distances = np.sqrt(
                (
                    np.repeat(car_list2_x, num_car2)
                    - np.tile(car_list2_x, num_car2)
                )
                ** 2
                + (
                    np.repeat(car_list2_y, num_car2)
                    - np.tile(car_list2_y, num_car2)
                )
                ** 2
            ).reshape(num_car2 , num_car2)

            car1_to_goal_distances = np.sqrt(
                (
                    np.repeat(car_list1_x, self.testing_num_goals)
                    - np.tile(goal_list_x, num_car1)
                )
                ** 2
                + (
                    np.repeat(car_list1_y, self.testing_num_goals)
                    - np.tile(goal_list_y, num_car1)
                )
                ** 2
            ).reshape(num_car1, self.testing_num_goals)

            car2_to_goal_distances = np.sqrt(
                (
                    np.repeat(car_list2_x, self.testing_num_goals)
                    - np.tile(goal_list_x, num_car2)
                )
                ** 2
                + (
                    np.repeat(car_list2_y, self.testing_num_goals)
                    - np.tile(goal_list_y, num_car2)
                )
                ** 2
            ).reshape(num_car2, self.testing_num_goals)

            min_walls_to_car1_distances = np.min(
                walls_to_car1_distances, axis=1
            )
            argmin_walls_to_car1_distances = np.argmin(
                walls_to_car1_distances, axis=1
            )
            nearest_car1_ids = [
                taggers_list[idx] for idx in argmin_walls_to_car1_distances
            ]

            min_walls_to_car2_distances = np.min(
                walls_to_car2_distances, axis=1
            )
            argmin_walls_to_car2_distances = np.argmin(
                walls_to_car2_distances, axis=1
            )
            nearest_car2_ids = [
                taggers_list[idx] for idx in argmin_walls_to_car2_distances
            ]

            min_car1_to_car2_distances = np.min(
                car1_to_car2_distances, axis=1
            )
            argmin_car1_to_car2_distances = np.argmin(
                car1_to_car2_distances, axis=1
            )
            nearest_others_car_ids = [
                taggers_list[idx] for idx in argmin_car1_to_car2_distances
            ]

            min_car1_to_car1_distances = np.min(
                car1_to_car1_distances, axis=1
            )
            argmin_car1_to_car1_distances = np.argmin(
                car1_to_car1_distances, axis=1
            )
            nearest_car1_car1_ids = [
                taggers_list[idx] for idx in argmin_car1_to_car1_distances
            ]

            min_car2_to_car2_distances = np.min(
                car2_to_car2_distances, axis=1
            )
            argmin_car2_to_car2_distances = np.argmin(
                car2_to_car2_distances, axis=1
            )
            nearest_car2_car2_ids = [
                taggers_list[idx] for idx in argmin_car2_to_car2_distances
            ]

            min_car1_to_goal_distances = np.min(
                car1_to_goal_distances, axis=1
            )
            argmin_car1_to_goal_distances = np.argmin(
                car1_to_goal_distances, axis=1
            )
            nearest_car1_goal_ids = [
                taggers_list[idx] for idx in argmin_car1_to_goal_distances
            ]

            min_car2_to_goal_distances = np.min(
                car2_to_goal_distances, axis=1
            )
            argmin_car2_to_goal_distances = np.argmin(
                car2_to_goal_distances, axis=1
            )
            nearest_car2_goal_ids = [
                taggers_list[idx] for idx in argmin_car2_to_goal_distances
            ]

        # Rewards
        # Add edge hit reward penalty and the step rewards/ penalties
        for agent_id in range(self.num_agents):
            if self.still_in_the_game[agent_id]:
                rew[agent_id] += self.step_rewards[agent_id]

        for idx, wall_id in enumerate(walls_list):
            if min_walls_to_car1_distances[idx] < self.distance_margin + self.agent_size[idx] + self.agent_size[nearest_car1_ids[idx]]:
                rew[nearest_car1_ids[idx]] += self.tagger_hit_penalty
            if min_walls_to_car2_distances[idx] < self.distance_margin + self.agent_size[idx] + self.agent_size[nearest_car2_ids[idx]]:
                rew[nearest_car2_ids[idx]] += self.tagger_hit_penalty

        for idx, car_id in enumerate(car_list1):
            if min_car1_to_car2_distances[idx] < self.distance_margin:
                rew[car_id] += self.tag_reward_for_tagger
                rew[nearest_others_car_ids[idx]] += self.tag_reward_for_tagger

                self.still_in_the_game[car_id] = 0
                self.still_in_the_game[nearest_others_car_ids[idx]] = 0
                del self.taggers[car_id]
                del self.taggers[nearest_others_car_ids[idx]]
                self.num_taggers -= 2
                self.global_state[_SIG][self.timestep :, car_id] = 0
                self.global_state[_SIG][self.timestep :, nearest_others_car_ids[idx]] = 0
            if min_car1_to_goal_distances[idx] < self.distance_margin:
                rew[car_id] += self.tag_reward_for_tagger
                self.still_in_the_game[nearest_car1_goal_ids[idx]] = 0
                del self.goals[nearest_car1_goal_ids[idx]]
                self.testing_num_goals -= 1
                self.global_state[_SIG][self.timestep :, nearest_car1_goal_ids[idx]] = 0
            if min_car1_to_car1_distances[idx] < self.distance_margin + self.agent_size[idx] + self.agent_size[nearest_car1_car1_ids[idx]]:
                rew[car_id] += self.tagger_hit_penalty

        for idx, car_id in enumerate(car_list2):
            if min_car2_to_car2_distances[idx] < self.distance_margin + self.agent_size[idx] + self.agent_size[nearest_car2_car2_ids[idx]]:
                rew[car_id] += self.tagger_hit_penalty
            if min_car2_to_goal_distances[idx] < self.distance_margin:
                rew[car_id] += self.tag_reward_for_tagger
                self.still_in_the_game[nearest_car2_goal_ids[idx]] = 0
                del self.goals[nearest_car2_goal_ids[idx]]
                self.testing_num_goals -= 1
                self.global_state[_SIG][self.timestep :, nearest_car2_goal_ids[idx]] = 0
            
        return rew

    def get_data_dictionary(self):
        """
        Create a dictionary of data to push to the device
        """
        self.still_in_the_game = [1 for agent_id in range(self.num_agents)]
        data_dict = DataFeed()
        for feature in [_LOC_X, _LOC_Y, _SP, _DIR, _ACC]:
            data_dict.add_data(
                name=feature,
                data=self.global_state[feature][0],
                save_copy_and_apply_at_reset=True,
            )
        data_dict.add_data(
            name="agent_types",
            data=[self.agent_type[agent_id] for agent_id in range(self.num_agents)],
        )
        data_dict.add_data(
            name="agent_size",
            data=self.agent_size,
        )
        data_dict.add_data(name="grid_length", data = self.grid_length)
        data_dict.add_data(name="acceleration_actions", data=self.acceleration_actions)
        data_dict.add_data(name="turn_actions", data=self.turn_actions)
        data_dict.add_data(name="max_speed", data=self.max_speed)
        data_dict.add_data(name="num_other_agents_observed", data=self.num_other_agents_observed)
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
        data_dict.add_data(name="step_rewards", data = self.step_rewards)
        data_dict.add_data(name="distance_margin", data=self.distance_margin)
        data_dict.add_data(
            name="tag_reward_for_tagger", data=self.tag_reward_for_tagger
        )
        data_dict.add_data(
            name="tagger_hit_penalty", data=self.tagger_hit_penalty
        )
        data_dict.add_data(
            name="limit_acceleration_reward",
            data=self.limit_acceleration_reward,
        )
        data_dict.add_data(
            name="limit_turn_reward",
            data=self.limit_turn_reward,
        )
        data_dict.add_data(
            name="display_reward",
            data=self.display_reward,
            save_copy_and_apply_at_reset=True
        )
        data_dict.add_data(
            name="max_acceleration",
            data=self.max_acceleration,
        )
        data_dict.add_data(
            name="max_turn",
            data=self.max_turn,
        )
        data_dict.add_data(name="training", data = self.Training)
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

        # Reinitialize some variables that may have changed during previous episode
        self.taggers = copy.deepcopy(self.taggers_at_reset)
        self.num_taggers = len(self.taggers)
        self.goals = copy.deepcopy(self.goal_at_reset)
        self.testing_num_goals = len(self.goals)

        return self.generate_observation()

    def step(self, actions=None):
        """
        Env step() - The GPU version calls the corresponding CUDA kernels
        """
        self.timestep += 1
        if self.use_cuda:
            # CUDA version of step()
            # This subsumes update_state(), generate_observation(),
            # and compute_reward()

            args = [
                _LOC_X,
                _LOC_Y,
                _SP,
                _DIR,
                _ACC,
                "agent_types",
                "agent_size",
                "grid_length",
                "acceleration_actions",
                "turn_actions",
                "max_speed",
                "num_other_agents_observed",
                "skill_levels",
                "display_reward",
                "still_in_the_game",
                _OBSERVATIONS,
                _ACTIONS,
                "neighbor_distances",
                "neighbor_ids_sorted_by_distance",
                "nearest_neighbor_ids",
                _REWARDS,
                "step_rewards",
                "distance_margin",
                "tag_reward_for_tagger",
                "tagger_hit_penalty",
                "limit_acceleration_reward",
                "limit_turn_reward",
                "max_acceleration",
                "max_turn",
                "training",
                "_done_",
                "_timestep_",
                ("n_agents", "meta"),
                ("episode_length", "meta"),
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
                actions[agent_id][2] for agent_id in range(self.num_agents)
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
