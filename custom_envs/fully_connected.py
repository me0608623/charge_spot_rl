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
from icecream import ic
import json

try:
    import wandb
    from warp_drive.utils.constants import Constants
    from warp_drive.utils.data_feed import DataFeed

    _OBSERVATIONS = Constants.OBSERVATIONS
    _PROCESSED_OBSERVATIONS = Constants.PROCESSED_OBSERVATIONS
    _ACTION_MASK = Constants.ACTION_MASK

    _LARGE_NEG_NUM = -1e20
except:
    print("now is implement")


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
        policy,
        policy_tag_to_agent_id_map,
        env = None,
        fc_dims = [],
        create_separate_placeholders_for_each_policy=False,
        obs_dim_corresponding_to_num_agents="first",
        policy_network = [],
        critic_network = [],
        training = 'train',
        model_type = None,
        memory_dim = 10,
        n_layers = 1,
        observation_space = None,
        action_space = None,
        flattened_obs_size = None,
        episode_length = 180,
        rnn_layer_type = 'RNN'
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
        self.memory_dim = memory_dim
        self.n_layers = n_layers
        self.action_idx = 0
        self.start_game = False
        self.implement = False
        self.seq_train_rnn = False
        self.rnn_layer_type = rnn_layer_type
        self.rnn_seq_len = 5
        self.count_rnn_seq = 1
        rnn_types = ['RNN', 'LSTM', 'GRU']

        assert self.rnn_layer_type in rnn_types
        assert isinstance(fc_dims, list)
        num_fc_layers = len(fc_dims)
        self.policy = policy
        self.policy_tag_to_agent_id_map = policy_tag_to_agent_id_map
        self.create_separate_placeholders_for_each_policy = (
            create_separate_placeholders_for_each_policy
        )
        assert obs_dim_corresponding_to_num_agents in ["first", "last"]
        self.obs_dim_corresponding_to_num_agents = obs_dim_corresponding_to_num_agents

        sample_agent_id = self.policy_tag_to_agent_id_map[self.policy][0]
        # Flatten obs space
        if observation_space is not None: self.observation_space = None
        else: 
            self.observation_space = self.env.env.observation_space[sample_agent_id]
            flattened_obs_size = self.get_flattened_obs_size()
        if action_space is None:
            if isinstance(self.env.env.action_space[sample_agent_id], Discrete):
                action_space = [self.env.env.action_space[sample_agent_id].n]
            elif isinstance(self.env.env.action_space[sample_agent_id], MultiDiscrete):
                action_space = self.env.env.action_space[sample_agent_id].nvec
            else:
                raise NotImplementedError
        else:
            action_space = np.array(action_space)

        input_dims = [flattened_obs_size] + fc_dims[:-1]
        output_dims = fc_dims

        self.fc = nn.ModuleDict()
        for fc_layer in range(num_fc_layers):
            self.fc[str(fc_layer)] = nn.Sequential(
                nn.Linear(input_dims[fc_layer], output_dims[fc_layer]),
                nn.ReLU(),
            )
        rnn_fc_out = fc_dims[-1]
        self.rl_input_dim = 256
        if self.n_layers > 0: 
            self.rl_input = nn.Linear(self.memory_dim + flattened_obs_size, self.rl_input_dim)
            if self.rnn_layer_type == 'LSTM':  self.rnn = nn.LSTM(rnn_fc_out, self.memory_dim, self.n_layers)
            elif self.rnn_layer_type == 'GRU':  self.rnn = nn.GRU(rnn_fc_out, self.memory_dim, self.n_layers)
            else:  self.rnn = nn.RNN(rnn_fc_out, self.memory_dim, num_layers=self.n_layers)
        else:  self.rl_input = nn.Linear(rnn_fc_out + flattened_obs_size, self.rl_input_dim)
        self.memory_max_index = episode_length  // self.rnn_seq_len
        self.memory_banch = [None] * self.memory_max_index
        # policy network (list of heads)
        assert isinstance(policy_network, list)
        num_policy_layers = len(policy_network)
        policy_input_dims = [self.rl_input_dim] + policy_network[:-1]
        policy_output_dims = policy_network
        self.policy_net = nn.ModuleDict()
        for fc_layer in range(num_policy_layers):
            self.policy_net[str(fc_layer)] = nn.Sequential(
                nn.Linear(policy_input_dims[fc_layer], policy_output_dims[fc_layer]),
                nn.ReLU(),
            )

        if num_policy_layers == 0:  self.policy_net = None
        policy_heads = [None for _ in range(len(action_space))]
        self.output_dims = []  # Network output dimension(s)
        for idx, act_space in enumerate(action_space):
            self.output_dims += [act_space]
            if num_policy_layers == 0: policy_heads[idx] = nn.Linear(self.rl_input_dim, act_space)
            else:  policy_heads[idx] = nn.Linear(policy_network[-1], act_space)
        self.policy_head = nn.ModuleList(policy_heads)

        # value-function network head
        assert isinstance(critic_network, list)
        num_critic_layers = len(critic_network)
        critic_input_dims = [self.rl_input_dim] + critic_network[:-1]
        critic_output_dims = critic_network

        self.critic = nn.ModuleDict()
        for fc_layer in range(num_critic_layers):
            self.critic[str(fc_layer)] = nn.Sequential(
                nn.Linear(critic_input_dims[fc_layer], critic_output_dims[fc_layer]),
                nn.ReLU(),
            )
        if num_critic_layers == 0: 
            self.critic = None
            self.vf_head = nn.Linear(self.rl_input_dim, 1)
        else:
            self.vf_head = nn.Linear(critic_network[-1], 1)
        # used for action masking
        self.action_mask = None
        if self.observation_space is not None:
            hidden_dim = 10
            mix_output_dim = 256
            model_config = {
                'policy': policy,
                'policy_tag_to_agent_id_map': policy_tag_to_agent_id_map,
                'fc_dims': fc_dims,
                'create_separate_placeholders_for_each_policy': create_separate_placeholders_for_each_policy,
                'obs_dim_corresponding_to_num_agents':obs_dim_corresponding_to_num_agents,
                'policy_network': policy_network,
                'critic_network': critic_network,
                'training':training,
                'model_type': model_type,
                'memory_dim': memory_dim,
                'n_layers': n_layers,
                'observation_space': "Not required when executing",
                'action_space': action_space.tolist(),
                'flattened_obs_size': flattened_obs_size
            }
    
            file_path = "custom_model/Spot_3D_NEW/model_config.json"
            with open(file_path, "w") as json_file:
                json.dump(model_config, json_file, indent=4)
        """
        self.custom_model = Mix_Process_Model(
            action_space, 
            flattened_obs_size, 
            mix_output_dim, 
            hidden_dim, 
            self.model_type,
            self.memory_dim
        )
        """
        self.action_mask = None

        self.memory_dim = memory_dim
        self.memory = None
        #print("self.custom_model = ",self.custom_model)

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

    def rnn_process(self, input_data):
        """
        nomal input = (banch, banch, in_dim)
            or (seq, banch, banch, in_dim)
        rnn input = (seq, banch,  in_dim)
        """
        #ic(input.shape, input.size(-1), input.dim())
        
        #ic(input.shape, origin_batch)
        if input_data.dim() == 3:  
            output = input_data.reshape(1, -1, input_data.size(-1))
            origin_batch = (input_data.size(0), input_data.size(1), -1)
        elif input_data.dim() == 4: 
            origin_batch = (input_data.size(0), input_data.size(1), input_data.size(2), input_data.size(3))
            seq_date = input_data.permute(1, 2, 0, 3).contiguous()
            seq_date = input_data.reshape(-1, self.rnn_seq_len, origin_batch[3]) 
            output = seq_date.permute(1, 0, 2).contiguous()
        elif input_data.dim() == 1:
            self.implement = True
            output = input_data.unsqueeze(0).unsqueeze(0)
            origin_batch = (-1,)
        else:
            ic("rnn input dim error", input_data.dim())
        return output, origin_batch

    def rnn_recover(self, rout, origin_batch):
        
        if len(origin_batch) == 4:
            rnn_out_dim = rout.size(-1)
            output_size = origin_batch[:-1] + (rnn_out_dim,)
            rout_permute = rout.permute(1, 0, 2).contiguous()
            seq_date = rout_permute.reshape(origin_batch[1], origin_batch[2], -1, rnn_out_dim)
            output = seq_date.permute(2, 0, 1, 3).contiguous()

        else:  output = torch.reshape(rout, origin_batch)
        return output

    def forward(self, obs=None, memory=None, batch_index=None, batch_size=None):
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
        else:
            ip = obs

        
        rl_ip = ip.detach().clone()
        for layer in range(len(self.fc)):
            op = self.fc[str(layer)](ip)
            ip = op
        
        memory_t_1 = None
        if self.n_layers > 0:
            #assert self.n_layers == 1  # rnn hidden reshape will wrong
            if self.observation_space is None:  memory_ = memory.unsqueeze(0).unsqueeze(1) 
            elif memory is not None:  memory_ = memory # memory = layers, banch, memory_dim
            elif self.memory is None and self.observation_space is not None:
                self.count_rnn_seq = 1
                batch_size_ = ip.size(-2) *  ip.size(-3)
                if self.rnn_layer_type == 'LSTM':
                    h_ = torch.zeros(self.n_layers, batch_size_, self.memory_dim).to('cuda:0').detach()
                    c_ = torch.zeros(self.n_layers, batch_size_, self.memory_dim).to('cuda:0').detach()
                    memory_ = (h_, c_)
                else:  memory_ = torch.zeros(self.n_layers, batch_size_, self.memory_dim).to('cuda:0').detach()
                self.memory_banch[0] = memory_
            else:
                if self.rnn_layer_type == 'LSTM': memory_ = (self.memory[0].detach(), self.memory[1].detach())
                else: memory_ = self.memory.detach()
                mem_idx = (self.count_rnn_seq) // self.rnn_seq_len # init self.count_rnn_seq is 1
                if self.count_rnn_seq % self.rnn_seq_len == 0 and mem_idx < self.memory_max_index:  
                    self.memory_banch[mem_idx] = memory_ #store the last h in seq
            rnn_in, origin_batch = self.rnn_process(op)
            r_out_, memory_reg = self.rnn(rnn_in, memory_)
            #ic(memory_reg)
            #ic(batch_size, memory_reg.shape)
            r_out = self.rnn_recover(r_out_, origin_batch)

            if  memory is not None:   memory_t_1 = memory_reg
            elif self.rnn_layer_type == 'LSTM':  memory_t_1 = (memory_reg[0].detach(), memory_reg[1].detach())
            else:  memory_t_1 = memory_reg.detach()
            self.memory = memory_t_1
            self.count_rnn_seq += 1
            if memory is not None:  self.memory = None
        else:  r_out = op
        
        rl_in_ = torch.concat((rl_ip, r_out), dim=-1)
        logits = self.rl_input(rl_in_)

        
        # Compute the action probabilities and the value function estimate
        # Apply action mask to the logits as well.
        if self.policy_net == None:
            action_input_data = logits
        else:
            ip = logits
            for layer in range(len(self.policy_net)):
                op = self.policy_net[str(layer)](ip)
                ip = op
            action_input_data = op
        action_masks = [None for _ in range(len(self.output_dims))]
        if self.action_mask is not None:
            start = 0
            for idx, dim in enumerate(self.output_dims):
                action_masks[idx] = self.action_mask[..., start : start + dim]
                start = start + dim
        action_probs = [
            func.softmax(apply_logit_mask(ph(action_input_data), action_masks[idx]), dim=-1)
            for idx, ph in enumerate(self.policy_head)
        ]
        if self.critic == None:
            vals = self.vf_head(logits)[..., 0]
        else:
            ip = logits
            for layer in range(len(self.critic)):
                op = self.critic[str(layer)](ip)
                ip = op
            vals = self.vf_head(op)[..., 0]
        if self.implement and self.n_layers > 0: 
            new_memory =  torch.reshape(memory_reg.detach(), origin_batch)
            return action_probs, vals, new_memory

        
        ls_linear = action_probs[0][0][0].tolist()
        ls_linear2 = action_probs[1][0][0].tolist()
        index_linear = ls_linear.index(max(ls_linear))
        index_linear2 = ls_linear2.index(max(ls_linear2))
        # input_output test
        if self.n_layers > 0:
            if self.rnn_layer_type == 'LSTM':  memory_out = (memory_[0][:,0,:].tolist(), memory_[1][:,0,:].tolist())
            else: memory_out = memory_[:,0,:].tolist()
        else:
            memory_out = None

        data_dict = {
            "input": rl_ip[0][0].tolist(),
            "memory": memory_out,
            "input shape": rl_ip.shape,
            "value": vals[0][0].tolist(),
            "action_1": ls_linear,
            "action_2": ls_linear2
        }

        if len(action_probs) > 2:
            #print("index_linear2 = " ,index_linear2)
            ls_angular = action_probs[2][0][0].tolist()
            index_angular = ls_angular.index(max(ls_angular))
            data_dict['action_3'] = ls_angular

        if self.training == 'test':
            import os
            file_path = "./custom_model/Spot_3D_NEW/input_output.json"
            with open(file_path, "w") as json_file:
                json.dump(data_dict, json_file)

        elif self.training == 'eval':
            dict_ = {}
            for input_data in range(len(data_dict["input"])):
                dict_[str(input_data)] = data_dict["input"][input_data]
            if self.rnn_layer_type == 'LSTM': 
                h_0 = data_dict["memory"][0][-1]
                c_0 = data_dict["memory"][1][-1]
                for mem_data in range(len(h_0)):
                    dict_["h0_"+str(mem_data)] = h_0[mem_data]
                for mem_data in range(len(c_0)):
                    dict_["c0_"+str(mem_data)] = c_0[mem_data]
            else:
                h_0 = data_dict["memory"][-1]
                for mem_data in range(len(h_0)):
                    dict_["h0_"+str(mem_data)] = h_0[mem_data]
                    
            dict_['action_idx_x'] = index_linear
            dict_['action_idx_y'] = index_linear2
            dict_['action_idx_turn'] = index_angular
            if data_dict["input"][16] == 1:
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

        return action_probs, vals, memory_t_1

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

