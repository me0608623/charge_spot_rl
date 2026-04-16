# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import copy
import heapq
from random import sample

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


class CarContinuous(CUDAEnvironmentContext):
    """
    The game of tag on a continuous circular 2D space.
    There are some taggers trying to tag several runners.
    The taggers want to get as close as possible to the runner, while the runner
    wants to get as far away from them as possible.
    Once a runner is tagged, he exits the game if runner_exits_game_after_tagged is True
    otherwise he continues to run around (and the tagger can catch him again)
    """

    def __init__(
        self,
        num_car=10,
        num_goal=10,
        num_wall=5,
        num_people=10,
        episode_length=100,
        grid_length=40.0,		#map size
        starting_location_x=None,
        starting_location_y=None,
        starting_directions=None,
        seed=None,
        max_speed=1.0,
        skill_level_car=1.0,
        skill_level_people=0.3,
        max_acceleration=0.1,
        min_acceleration=-0.1,
        max_turn=np.pi / 2,
        min_turn=-np.pi / 2,
        num_acceleration_levels=20,
        num_turn_levels=20,
        use_cuda=False,
        tagging_distance=0.01,
        edge_hit_penalty=-0.2,
        reward_car_collision = -10.0,
        reward_car_goal = 10.0,
        reward_car_right_goal = 30.0,
        reward_car_step_penalty = -0.1,
        reward_people_walk = 0.1,
    ):
        """
        Args:
            num_taggers (int, optional): [number of taggers in the environment].
                Defaults to 1.
            num_runners (int, optional): [number of taggers in the environment].
                Defaults to 10.
            grid_length (float, optional): [length of the square grid]. Defaults to 10.0
            episode_length (int, optional): [episode length]. Defaults to 100.
            starting_location_x ([ndarray], optional): [starting x locations of the
                agents]. Defaults to None.
            starting_location_y ([ndarray], optional): [starting y locations of the
            agents]. Defaults to None.
            starting_directions ([ndarray], optional): starting orientations
                in [0, 2*pi]. Defaults to None.
            seed ([type], optional): [seeding parameter]. Defaults to None.
            max_speed (float, optional): [max speed of the agents]. Defaults to 1.0
            skill_level_runner (float, optional): [runner skill level;
                this essentially is a multiplier to the max_speed].
                Defaults to 1.0
            skill_level_tagger (float, optional): [tagger skill level]. Defaults to 1.0
            max_acceleration (float, optional): [the max acceleration]. Defaults to 1.0.
            min_acceleration (float, optional): [description]. Defaults to -1.0
            max_turn ([type], optional): [description]. Defaults to np.pi/2.
            min_turn ([type], optional): [description]. Defaults to -np.pi/2.
            num_acceleration_levels (int, optional): [number of acceleration actions
                uniformly spaced between max and min acceleration]. Defaults to 10.
            num_turn_levels (int, optional): [number of turn actions uniformly spaced
                between max and min turns]. Defaults to 10.
            edge_hit_penalty (float, optional): [penalty for hitting the edge (wall)].
                Defaults to -0.0.
            use_full_observation (bool, optional): [boolean indicating whether to
                include all the agents' data in the observation or just the nearest
                neighbors]. Defaults to True.
            num_other_agents_observed (int, optional): [number of nearest neighbors
                in the obs (only takes effect when use_full_observation is False)].
                Defaults to 2.
            tagging_distance (float, optional): [margin between a
                tagger and runner to consider the runner as 'tagged'. This multiplies
                on top of the grid length]. Defaults to 0.01.
            tag_reward_for_tagger (float, optional): [positive reward for the tagger
                upon tagging a runner]. Defaults to 1.0
            step_penalty_for_tagger (float, optional): [penalty for every step
                the game goes on]. Defaults to -0.0.
            tag_penalty_for_runner (float, optional): [negative reward for getting
                tagged]. Defaults to -1.0
            step_reward_for_runner (float, optional): [reward for every step the
                runner isn't tagged]. Defaults to 0.0.
            end_of_game_reward_for_runner (float, optional): [reward at the end of
                the game for a runner that isn't tagged]. Defaults to 1.0.
            runner_exits_game_after_tagged (bool, optional): [boolean indicating
                whether runners exit the game after getting tagged or can remain in and
                continue to get tagged]. Defaults to True.
            use_cuda (bool, optional): [boolean to indicate whether to use the CPU
                or the GPU. (cuda) for stepping through the environment].
                Defaults to False.
        """
        super().__init__()

        self.float_dtype = np.float32
        self.int_dtype = np.int32
        # small number to prevent indeterminate cases
        self.eps = self.float_dtype(1e-10)

        assert num_car > 0
        assert num_goal > 0
        assert num_wall >= 0
        assert num_people >= 0        
        
        self.num_car = num_car
        self.num_goal = num_goal
        self.num_wall = num_wall
        self.num_people = num_people
        

        self.num_agents = self.num_car + self.num_goal + self.num_wall + self.num_people

        assert episode_length > 0
        self.episode_length = episode_length

        # Square 2D grid
        assert grid_length > 0
        self.grid_length = self.float_dtype(grid_length)
        self.grid_diagonal = self.grid_length * np.sqrt(2)



        # Seeding
        self.np_random = np.random
        if seed is not None:
            self.seed(seed)

        # Starting cars
        reg1 = self.np_random.choice(
            np.arange(self.num_agents), self.num_goal + self.num_wall + self.num_people, replace=False
        )
        cars =set(np.arange(self.num_agents)) - set(reg1)
        
        reg2 = self.np_random.choice(
            reg1, self.num_wall + self.num_people, replace=False
        )
        goal = set(reg1) - set(reg2)
        
        people= self.np_random.choice(
            reg2, self.num_people, replace=False
        )
        wall = set(reg2) - set(people)

        self.agent_type = {}
        self.cars = {}
        self.goals = {}
        self.walls = {}
        self.peoples = {}
        
        for agent_id in range(self.num_agents):
            if agent_id in set(cars):
                self.agent_type[agent_id] = 0  # car
                self.cars[agent_id] = True
            elif agent_id in set(goal):
                self.agent_type[agent_id] = 1  # goal
                self.goals[agent_id] = True
            elif agent_id in set(wall):
                self.agent_type[agent_id] = 2  # wall
                self.walls[agent_id] = True
            else:
                self.agent_type[agent_id] = 3  # people
                self.peoples[agent_id] = True

        self.cars_right_goal = [sample(goal, 1)[0] if agent_id in set(cars) else 0 for agent_id in range(self.num_agents)]
        
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

        # All agents start with 0 speed and acceleration
        self.starting_speeds = np.zeros(self.num_agents, dtype=self.float_dtype)
        self.starting_accelerations = np.zeros(self.num_agents, dtype=self.float_dtype)

        assert num_acceleration_levels >= 0
        assert num_turn_levels >= 0

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

        # Tagger and runner agent skill levels.
        # Skill levels multiply on top of the acceleration levels

        self.skill_levels = [ self.float_dtype(skill_level_car) if self.agent_type[agent_id] ==0 else self.float_dtype(skill_level_people) if self.agent_type[agent_id]==3 else 0 for agent_id in range(self.num_agents) ]

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
        # Used in generate_observation()
        # When use_full_observation is True, then all the agents will have info of
        # all the other agents, otherwise, each agent will only have info of
        # its k-nearest agents (k = num_other_agents_observed)

        # Distance margin between agents for non-zero rewards
        # If a tagger is closer than this to a runner, the tagger
        # gets a positive reward, and the runner a negative reward
        assert 0 <= tagging_distance <= 1
        self.distance_margin_for_reward = (tagging_distance * self.grid_length).astype(
            self.float_dtype
        )

        # Rewards and penalties
        assert reward_car_goal >= 0
        self.reward_car_goal = self.float_dtype(reward_car_goal)
        assert reward_car_goal < reward_car_right_goal
        self.reward_car_right_goal =self.float_dtype(reward_car_right_goal)
        assert reward_car_step_penalty <= 0
        self.reward_car_step_penalty = self.float_dtype(reward_car_step_penalty)
        assert reward_car_collision <= 0
        self.reward_car_collision = self.float_dtype(reward_car_collision)
        assert edge_hit_penalty <= 0
        self.edge_hit_penalty = self.float_dtype(edge_hit_penalty)

        assert reward_people_walk >= 0
        self.reward_people_walk = self.float_dtype(reward_people_walk)

        # Note: These will be set later
        self.edge_hit_reward_penalty = None
        self.still_in_the_game = None
        self.step_rewards = [
            self.agent_type[agent_id] *self.reward_car_step_penalty 
            for agent_id in range(self.num_agents)
        ]
        # These will also be set via the env_wrapper
        # use_cuda will be set to True (by the env_wrapper), if needed
        # to be simulated on the GPU
        self.use_cuda = use_cuda

        # Copy runners dict for applying at reset
        self.goals_at_reset = copy.deepcopy(self.goals)

    name = "CarContinuous"

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
        
        loc_x_curr_t = [self.float_dtype(loc_x_prev_t[agent_id] + speed_curr_t[agent_id] * np.cos(dir_curr_t[agent_id])) if self.agent_type[agent_id] == 0 or self.agent_type[agent_id] == 3 else self.float_dtype(loc_x_prev_t) for agent_id in range(self.num_agents)]
        loc_y_curr_t = [self.float_dtype(loc_y_prev_t[agent_id] + speed_curr_t[agent_id] * np.sin(dir_curr_t[agent_id])) if self.agent_type[agent_id] == 0 or self.agent_type[agent_id] == 3 else self.float_dtype(loc_y_prev_t) for agent_id in range(self.num_agents)]
        self.reward_people_walk =[self.float_dtype(speed_curr_t[agent_id] / self.max_speed ) if self.agent_type[agent_id] == 3 else 0 for agent_id in range(self.num_agents)]

        # Crossing the edge
        has_crossed_edge = ~(
            (loc_x_curr_t >= 0)
            & (loc_x_curr_t <= self.grid_length)
            & (loc_y_curr_t >= 0)
            & (loc_y_curr_t <= self.grid_length)
        )

        # Clip x and y if agent has crossed edge
        clipped_loc_x_curr_t = self.float_dtype(
            np.clip(loc_x_curr_t, 0.0, self.grid_length)
        )

        clipped_loc_y_curr_t = self.float_dtype(
            np.clip(loc_y_curr_t, 0.0, self.grid_length)
        )

        # Penalize reward if agents hit the walls
        self.edge_hit_reward_penalty = self.edge_hit_penalty * has_crossed_edge

        # Set global states
        self.set_global_state(key=_LOC_X, value=clipped_loc_x_curr_t, t=self.timestep)
        self.set_global_state(key=_LOC_Y, value=clipped_loc_y_curr_t, t=self.timestep)
        self.set_global_state(key=_SP, value=speed_curr_t, t=self.timestep)
        self.set_global_state(key=_DIR, value=dir_curr_t, t=self.timestep)
        self.set_global_state(key=_ACC, value=acc_curr_t, t=self.timestep)

    def generate_observation(self):
        """
        Generate and return the observations for every agent.
        know athor agent reletive loc but dont know self sp , acc , dir, 
        know self type
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
        normalized_global_goal_x_obs = [normalized_global_obs[0][idx]- normalized_global_obs[0][self.cars_right_goal[idx]] for idx in range(self.num_agents)]
        normalized_global_goal_y_obs = [normalized_global_obs[1][idx]- normalized_global_obs[0][self.cars_right_goal[idx]] for idx in range(self.num_agents)]
        
        for agent_id in range(self.num_agents):
            # Set obs for agents still in the game
            obs[agent_id] = np.concatenate(
                [
                    np.vstack(
                        (
                            np.zeros_like(normalized_global_obs),
                            agent_types,
                            self.still_in_the_game,
                        )
                    )[
                        :,
                        [idx for idx in range(self.num_agents) if idx != agent_id],
                    ].reshape(
                        -1
                    ),  # filter out the obs for the current agent,but i think it need current agent
                    np.array([0.0]),
                    np.array([0.0]),
                    np.array([agent_types[agent_id]] ),
                    np.array([0.0]),
                ]
            )    
            if self.still_in_the_game[agent_id]:
                obs[agent_id] = np.concatenate(
                    [
                        np.vstack(
                            (
                                normalized_global_obs
                                - normalized_global_obs[:, agent_id].reshape(-1, 1),
                                agent_types,
                                self.still_in_the_game,
                            )
                        )[
                            :,
                            [idx for idx in range(self.num_agents)if idx != agent_id],
                        ].reshape(
                            -1
                        ),  # filter out the obs for the current agent
                        np.array([normalized_global_goal_x_obs[agent_id]]),  #goal point
                        np.array([normalized_global_goal_y_obs[agent_id]]),  #goal point
                        np.array([agent_types[agent_id]] ),
                        time,
                    ]
                )
        return obs

    def compute_reward(self):
        """
        Compute and return the rewards for each agent.
        """
        # Initialize rewards
        rew = {agent_id: 0.0 for agent_id in range(self.num_agents)}

        cars_list = sorted(self.cars)
        walls_list = sorted(self.walls)
        peoples_list = sorted(self.peoples)

        # At least one runner present
        if self.num_goal > 0:
            goals_list = sorted(self.goals)
            car_locations_x = self.global_state[_LOC_X][self.timestep][cars_list]
            wall_locations_x = self.global_state[_LOC_X][self.timestep][walls_list]
            goal_locations_x = self.global_state[_LOC_X][self.timestep][goals_list]
            people_locations_x = self.global_state[_LOC_X][self.timestep][peoples_list]

            car_locations_y = self.global_state[_LOC_Y][self.timestep][cars_list]
            wall_locations_y = self.global_state[_LOC_Y][self.timestep][walls_list]
            goal_locations_y = self.global_state[_LOC_Y][self.timestep][goals_list]
            people_locations_y = self.global_state[_LOC_Y][self.timestep][peoples_list]

            right_goal_x = self.global_state[_LOC_X][self.timestep][self.cars_right_goal]
            right_goal_y = self.global_state[_LOC_Y][self.timestep][self.cars_right_goal]


            cars_to_goals_distances = np.sqrt(
                (
                    np.repeat(car_locations_x, self.num_goal)
                    - np.tile(goal_locations_x, self.num_car)
                )
                ** 2
                + (
                    np.repeat(car_locations_y, self.num_goal)
                    - np.tile(goal_locations_y, self.num_car)
                )
                ** 2
            ).reshape(self.num_car, self.num_goal)

            cars_to_right_goals_distances = np.sqrt(
                (np.array(right_goal_x )- np.array(car_locations_x))** 2
                    + (np.array(right_goal_y )- np.array(car_locations_y)) ** 2              
            )

            cars_to_cars_distances = np.sqrt(
                (
                    np.repeat(car_locations_x, self.num_car)
                    - np.tile(car_locations_x, self.num_car)
                )
                ** 2
                + (
                    np.repeat(car_locations_y, self.num_car)
                    - np.tile(car_locations_y, self.num_car)
                )
                ** 2
            ).reshape(self.num_car, self.num_car)

            cars_to_walls_distances = np.sqrt(
                (
                    np.repeat(car_locations_x, self.num_wall)
                    - np.tile(wall_locations_x, self.num_car)
                )
                ** 2
                + (
                    np.repeat(car_locations_y, self.num_wall)
                    - np.tile(wall_locations_y, self.num_car)
                )
                ** 2
            ).reshape(self.num_car, self.num_wall)
                        
            cars_to_peoples_distances = np.sqrt(
                (
                    np.repeat(car_locations_x, self.num_people)
                    - np.tile(people_locations_x, self.num_car)
                )
                ** 2
                + (
                    np.repeat(car_locations_y, self.num_people)
                    - np.tile(people_locations_y, self.num_car)
                )
                ** 2
            ).reshape(self.num_car, self.num_people)

            min_cars_to_goals_distances = np.min(
                cars_to_goals_distances, axis=1
            )
            argmin_cars_to_goals_distances = np.argmin(
                cars_to_goals_distances, axis=1
            )
            nearest_car_goal_ids = [
                goals_list[idx] for idx in argmin_cars_to_goals_distances
            ]

            min_cars_to_cars_distances = np.min(
                cars_to_cars_distances, axis=1
            )
            argmin_cars_to_cars_distances = np.argmin(
                cars_to_cars_distances, axis=1
            )

            min_cars_to_walls_distances = np.min(
                cars_to_walls_distances, axis=1
            )
            argmin_cars_to_walls_distances = np.argmin(
                cars_to_walls_distances, axis=1
            )

            min_cars_to_peoples_distances = np.min(
                cars_to_peoples_distances, axis=1
            )
            argmin_cars_to_peoples_distances = np.argmin(
                cars_to_peoples_distances, axis=1
            )


        # Rewards
        # Add edge hit reward penalty and the step rewards/ penalties
        for idx in enumerate(self.cars_list):
            rew[idx] += self.step_rewards[idx]
            rew[idx] += self.edge_hit_reward_penalty[idx]
            if min_cars_to_goals_distances[idx] < self.distance_margin_for_reward:
                if cars_to_right_goals_distances[idx] < self.distance_margin_for_reward:
                    rew[idx] += self.reward_car_right_goal
                else:
                    rew[idx] += self.reward_car_goal
                self.still_in_the_game[nearest_car_goal_ids[idx]] = 0
                del self.goals[nearest_car_goal_ids[idx]]
                self.num_goal -=1
                self.global_state[_SIG][self.timestep :, nearest_car_goal_ids[idx]] = 0

            if min_cars_to_cars_distances[idx] < self.distance_margin_for_reward:
                rew[idx] += self.reward_car_collision
            if min_cars_to_walls_distances[idx] < self.distance_margin_for_reward or min_cars_to_peoples_distances < self.distance_margin_for_reward:
                rew[idx] += self.reward_car_collision


        for idx in enumerate(self.peoples_list):
            rew[idx] += self.reward_people_walk[idx] 
            rew[idx] += self.edge_hit_reward_penalty[idx]

        return rew
    
    def get_data_dictionary(self):
        """
        Create a dictionary of data to push to the device
        """
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
            name="edge_hit_reward_penalty",
            data=self.edge_hit_reward_penalty,
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(name="edge_hit_penalty", data=self.edge_hit_penalty)
        data_dict.add_data(name="grid_length", data=self.grid_length)
        data_dict.add_data(name="acceleration_actions", data=self.acceleration_actions)
        data_dict.add_data(name="turn_actions", data=self.turn_actions)
        data_dict.add_data(name="max_speed", data=self.max_speed)
        data_dict.add_data(name="skill_levels", data=self.skill_levels)
        data_dict.add_data(
            name="still_in_the_game",
            data=self.still_in_the_game,
            save_copy_and_apply_at_reset=True,
        )

        data_dict.add_data(
            name="step_rewards",
            data=self.step_rewards,
        )
        data_dict.add_data(
            name="distance_margin_for_reward", data=self.distance_margin_for_reward
        )

        data_dict.add_data(
            name="reward_people_walk",
            data=self.reward_people_walk,
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(name="reward_car_step_penalty", data=self.reward_car_step_penalty)
        data_dict.add_data(
            name="reward_car_collision",
            data=self.reward_car_collision
        )
        data_dict.add_data(
            name="reward_car_goal",
            data=self.reward_car_goal
        )
        data_dict.add_data(
            name="reward_car_right_goal",
            data=self.reward_car_right_goal
        )
        data_dict.add_data(
            name="right_goal",
            data=self.cars_right_goal,
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

        # Penalty for hitting the edges
        self.edge_hit_reward_penalty = np.zeros(self.num_agents, dtype=self.float_dtype)

        # Reinitialize some variables that may have changed during previous episode

        self.goals = copy.deepcopy(self.goals_at_reset)
        self.num_goal = len(self.goals)
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
                "_done_",
                "_timestep_",
                "agent_types",
                "right_goal",
                "still_in_the_game",
                _ACTIONS,
                _OBSERVATIONS,
                _REWARDS,
                _LOC_X,
                _LOC_Y,
                _SP,
                _DIR,
                _ACC,
                "step_rewards",
                "acceleration_actions",
                "turn_actions",
                "reward_people_walk",
                "skill_levels",
                "edge_hit_reward_penalty",
                "edge_hit_penalty",
                "grid_length",
                "max_speed",
                "distance_margin_for_reward",
                "reward_car_step_penalty",
                "reward_car_collision",
                "reward_car_goal",
                "reward_car_right_goal",
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
                "__all__": ((self.timestep >= self.episode_length)
                or (self.num_goal == 0))
            }
            info = {}

            result = obs, rew, done, info
        return result
