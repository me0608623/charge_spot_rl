# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause
#
"""
The Fully Connected Network class
"""

import numpy as np
import torch
import torch.nn.functional as func
from gym.spaces import Box, Dict, Discrete, MultiDiscrete
from torch import nn
from custom_envs.models import Mix_Process_Model
from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed
import csv
import json
import wandb
from icecream import ic
import copy

_OBSERVATIONS = Constants.OBSERVATIONS
_PROCESSED_OBSERVATIONS = Constants.PROCESSED_OBSERVATIONS
_ACTION_MASK = Constants.ACTION_MASK

_LARGE_NEG_NUM = -1e20


def apply_logit_mask(logits, mask=None):
    """
    Mask values of 1 are valid actions.
    Add huge negative values to logits with 0 mask values.
    """
    if mask is None:
        return logits

    logit_mask = torch.ones_like(logits) * _LARGE_NEG_NUM
    logit_mask = logit_mask * (1 - mask)
    return logits + logit_mask


# Policy networks
# ---------------
class CustomModelConnected(nn.Module):
    """
    Fully connected network implementation in Pytorch
    """

    name = "torch_fully_connected"

    def __init__(
        self,
        env,
        policy,
        policy_tag_to_agent_id_map,
        create_separate_placeholders_for_each_policy = False,
        obs_dim_corresponding_to_num_agents = "first",
        training = 'train',
        model_type = None,
        memory_dim = 10,
    ):
        super().__init__()

        self.env = env
        self.training = training
        self.model_type = model_type
        self.action_x_dataset = []
        self.action_y_dataset = []
        self.action_turn_dataset = []
        self.speed = []
        self.acc_distribute = []

        self.action_idx = 0
        self.start_game = False
        
        
        #assert isinstance(layers_dim, list)
        #num_fc_layers = len(layers_dim)
        self.policy = policy
        self.policy_tag_to_agent_id_map = policy_tag_to_agent_id_map
        self.create_separate_placeholders_for_each_policy = (
            create_separate_placeholders_for_each_policy
        )
        assert obs_dim_corresponding_to_num_agents in ["first", "last"]
        self.obs_dim_corresponding_to_num_agents = obs_dim_corresponding_to_num_agents

        sample_agent_id = self.policy_tag_to_agent_id_map[self.policy][0]
        self.memory_dim = memory_dim
        # Flatten obs space
        self.observation_space = self.env.env.observation_space[sample_agent_id]
        flattened_obs_size = self.get_flattened_obs_size()
        self.flattened_obs_size = flattened_obs_size

        if isinstance(self.env.env.action_space[sample_agent_id], Discrete):
            action_space = [self.env.env.action_space[sample_agent_id].n]
        elif isinstance(self.env.env.action_space[sample_agent_id], MultiDiscrete):
            action_space = self.env.env.action_space[sample_agent_id].nvec
        else:
            raise NotImplementedError
        hidden_dim = 10
        mix_output_dim = 256
        model_config = {
            'action_space': action_space.tolist(),
            'input_dim': flattened_obs_size,
            'mix_output_dim': mix_output_dim,
            'hidden_dim': hidden_dim
        }
        file_path = "custom_model/Spot_3D_NEW/model_config.json"
        with open(file_path, "w") as json_file:
            json.dump(model_config, json_file, indent=4)
        self.custom_model = Mix_Process_Model(
            action_space, 
            flattened_obs_size, 
            mix_output_dim, 
            hidden_dim, 
            self.model_type,
            self.memory_dim
        )
        self.action_mask = None
        self.memory = None
        print("self.custom_model = ",self.custom_model)

    def get_flattened_obs_size(self):
        """Get the total size of the observations after flattening"""
        if isinstance(self.observation_space, Box):
            obs_size = np.prod(self.observation_space.shape)
        elif isinstance(self.observation_space, Dict):
            obs_size = 0
            for key in self.observation_space:
                if key == _ACTION_MASK:
                    pass
                else:
                    obs_size += np.prod(self.observation_space[key].shape)
        else:
            raise NotImplementedError("Observation space must be of Box or Dict type")
        return int(obs_size)

    def reshape_and_flatten_obs(self, obs):
        """
        # Note: WarpDrive assumes that all the observation are shaped
        # (num_agents, *feature_dim), i.e., the observation dimension
        # corresponding to 'num_agents' is the first one. If the observation
        # dimension corresponding to num_agents is last, we will need to
        # permute the axes to align with WarpDrive's assumption.
        """
        num_envs = obs.shape[0]
        if self.create_separate_placeholders_for_each_policy:
            num_agents = len(self.policy_tag_to_agent_id_map[self.policy])
        else:
            num_agents = self.env.n_agents

        if self.obs_dim_corresponding_to_num_agents == "first":
            pass
        elif self.obs_dim_corresponding_to_num_agents == "last":
            shape_len = len(obs.shape)
            if shape_len == 1:
                obs = obs.reshape(-1, num_agents)  # valid only when num_agents = 1
            obs = obs.permute(0, -1, *range(1, shape_len - 1))
        else:
            raise ValueError(
                "num_agents can only be the first "
                "or the last dimension in the observations."
            )
        return obs.reshape(num_envs, num_agents, -1)

    def get_flattened_obs(self):
        """
        If the obs is of Box type, it will already be flattened.
        If the obs is of Dict type, then concatenate all the
        obs values and flatten them out.
        Returns the concatenated and flattened obs.
        """
        if isinstance(self.observation_space, Box):
            
            if self.create_separate_placeholders_for_each_policy:
                obs = self.env.cuda_data_manager.data_on_device_via_torch(
                    f"{_OBSERVATIONS}_{self.policy}"
                )
            else:
                obs = self.env.cuda_data_manager.data_on_device_via_torch(_OBSERVATIONS)
                # hear pull obs to model
                # obs.shape = torch(env, agent, obs_num)

            flattened_obs = self.reshape_and_flatten_obs(obs)

            
        elif isinstance(self.observation_space, Dict):
            obs_dict = {}
            for key in self.observation_space:
                if self.create_separate_placeholders_for_each_policy:
                    obs = self.env.cuda_data_manager.data_on_device_via_torch(
                        f"{_OBSERVATIONS}_{self.policy}_{key}"
                    )
                else:
                    obs = self.env.cuda_data_manager.data_on_device_via_torch(
                        f"{_OBSERVATIONS}_{key}"
                    )

                if key == _ACTION_MASK:
                    self.action_mask = self.reshape_and_flatten_obs(obs)
                    assert self.action_mask.shape[-1] == sum(self.output_dims)
                else:
                    obs_dict[key] = obs

            flattened_obs_dict = {}
            for key, value in obs_dict.items():
                flattened_obs_dict[key] = self.reshape_and_flatten_obs(value)
            flattened_obs = torch.cat(list(flattened_obs_dict.values()), dim=-1)
        else:
            raise NotImplementedError("Observation space must be of Box or Dict type")
        return flattened_obs
    
    def add_memory(self, input, memory = None):
        
        #ic(input.shape)
        #ic(self.start_game)
        #if not self.start_game:

        if memory is None:  memory = self.memory
        with torch.no_grad():
            output = torch.concat((input, memory), dim=-1)
        return output

        

    def forward(self, obs=None, memory= None, batch_index=None, batch_size=None):
        """
        Forward pass through the model.
        Returns action probabilities and value functions.
        """

        if obs is None:
            assert batch_index < batch_size
            # Read in observation from the placeholders and flatten them
            # before passing through the fully connected layers.
            # This is particularly relevant if the observations space is a Dict.
            obs = self.get_flattened_obs()
            if self.create_separate_placeholders_for_each_policy:
                ip = obs
            else:
                agent_ids_for_policy = self.policy_tag_to_agent_id_map[self.policy]
                ip = obs[:, agent_ids_for_policy]

            # Push the processed (in this case, flattened) obs to the GPU (device).
            # The writing happens to a specific batch index in the processed obs batch.
            # The processed obs batch is required for training.
            
            self.push_processed_obs_to_batch(batch_index, batch_size, ip)
            
            if self.memory_dim > 0:  
                if self.memory is None:  self.memory = torch.zeros([ip.shape[0], ip.shape[1], self.memory_dim], device='cuda:0', requires_grad=False)
                self.push_processed_memory_to_batch(batch_index, batch_size, processed_mem = self.memory)
        else:
            ip = obs
        #memory_ = memory
        # ip just for spot agent
        #if self.memory_dim > 0:  ip = self.add_memory(ip , memory = memory_)

        value, action_probs, memory__ = self.custom_model(input_date = ip, memory = memory, action_mask = self.action_mask, agent_num = len(self.policy_tag_to_agent_id_map[self.policy]))

        self.memory = memory__.detach()
        if memory is not None:  self.memory = None

        #ic(value.shape)
        #ic(self.memory.shape)
        # print("action_mask = ", self.action_mask)
        # print("input = ", len(ip), len(ip[0]), len(ip[0][0]))
        # input shape = [2, 5, 172]
        # print("ouput = v ", len(value), len(value[0]))
        # v shape = [2, 5]
        # print("ouput = p ", len(action_probs), len(action_probs[0]), len(action_probs[0][0]), len(action_probs[0][0][0]))
        # v shape = [3, 2, 5, 21]
        if len(action_probs) > 2:
            data_dict = {
                "input": ip[0][0].tolist(),
                "value": value[0][0].tolist(),
                "action_1": action_probs[0][0][0].tolist(),
                "action_2": action_probs[1][0][0].tolist(),
                "action_3": action_probs[2][0][0].tolist()
            }
        else:
            data_dict = {
                "input": ip[0][0].tolist(),
                "value": value[0][0].tolist(),
                "action_1": action_probs[0][0][0].tolist(),
                "action_2": action_probs[1][0][0].tolist()
            }

        ls_linear = action_probs[0][0][0].tolist()
        index_linear = ls_linear.index(max(ls_linear))
        #print("index_linear = " ,index_linear)
        ls_linear2 = action_probs[1][0][0].tolist()
        index_linear2 = ls_linear2.index(max(ls_linear2))
        if len(action_probs) > 2:
            #print("index_linear2 = " ,index_linear2)
            ls_angular = action_probs[2][0][0].tolist()
            index_angular = ls_angular.index(max(ls_angular))

        if self.training == 'test':
            import os
            file_path = "./custom_model/Spot_3D_NEW/input_output.json"
            with open(file_path, "w") as json_file:
                json.dump(data_dict, json_file)

        elif self.training == 'eval':
            dict_ = {}
            for input_data in range(len(data_dict["input"])):
                dict_[str(input_data)] = data_dict["input"][input_data]
            dict_['action_idx_x'] = index_linear
            dict_['action_idx_y'] = index_linear2
            dict_['action_idx_turn'] = index_angular
            if data_dict["input"][12] == 1:
                self.action_x_dataset.append([self.action_idx, index_linear])
                self.action_y_dataset.append([self.action_idx, index_linear2])
                self.action_turn_dataset.append([self.action_idx, index_angular])
                self.speed.append([
                    data_dict["input"][0],
                    data_dict["input"][1],
                    index_angular
                ])
                self.acc_distribute.append([
                    index_linear,
                    index_linear2,
                    index_angular
                ])
                self.action_idx += 1
                wandb.log(dict_)
        return action_probs, value, memory__

    def push_processed_obs_to_batch(self, batch_index, batch_size, processed_obs):
        name = f"{_PROCESSED_OBSERVATIONS}_batch_{self.policy}"
        if not self.env.cuda_data_manager.is_data_on_device_via_torch(name):
            processed_obs_batch = np.zeros((batch_size,) + processed_obs.shape)
            processed_obs_feed = DataFeed()
            processed_obs_feed.add_data(name=name, data=processed_obs_batch)
            self.env.cuda_data_manager.push_data_to_device(
                processed_obs_feed, torch_accessible=True
            )
        self.env.cuda_data_manager.data_on_device_via_torch(name=name)[
            batch_index
        ] = processed_obs

    def push_processed_memory_to_batch(self, batch_index, batch_size, processed_mem):
        name = f"memory_batch_{self.policy}"
        if not self.env.cuda_data_manager.is_data_on_device_via_torch(name):
            processed_mem_batch = np.zeros((batch_size,) + processed_mem.shape)
            processed_mem_feed = DataFeed()
            processed_mem_feed.add_data(name=name, data=processed_mem_batch)
            self.env.cuda_data_manager.push_data_to_device(
                processed_mem_feed, torch_accessible=True
            )
        self.env.cuda_data_manager.data_on_device_via_torch(name=name)[
            batch_index
        ] = processed_mem