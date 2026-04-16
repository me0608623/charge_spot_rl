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
import copy

try:
    import wandb
    from warp_drive.utils.constants import Constants
    from warp_drive.utils.data_feed import DataFeed

    _OBSERVATIONS = Constants.OBSERVATIONS
    _PROCESSED_OBSERVATIONS = Constants.PROCESSED_OBSERVATIONS
    _ACTION_MASK = Constants.ACTION_MASK
    _PREPROCESS = "preprocess_real_data"
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
class CustomModuleConnected(nn.Module):
    """
    Fully connected network implementation in Pytorch
    """

    name = "torch_fully_connected"

    def __init__(
        self,
        policy,
        policy_tag_to_agent_id_map,
        env = None,
        policy_network = [64],
        critic_network = [64],
        module_network = ['rl', 20],
        module_connect_dim = 2,
        rnn_layer_type = 'RNN',
        rnn_num_layers = 1,
        rnn_memory_dim = 10,
        rnn_seq_len = 5,
        rnn_drop = None,
        create_separate_placeholders_for_each_policy=False,
        obs_dim_corresponding_to_num_agents="first",
        training = 'train',
        episode_length = 180,
        concat_rnn = False,
        observation_space = None,
        action_space = None,
        network_feture_dim = None,
        flattened_obs_size = None
    ):
        """
            module_network [input, fc, rnn, fc, rl_input, fc] =>loss preprocess
            rl_network [concat(input, rl_input), policy and critc network] => loss rl
        """
        super().__init__()

        self.env = env
        self.policy = policy
        self.policy_tag_to_agent_id_map = policy_tag_to_agent_id_map
        self.create_separate_placeholders_for_each_policy = create_separate_placeholders_for_each_policy
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
        else:  action_space = np.array(action_space)
        if network_feture_dim is None:
            network_feture = self.env.env.network_feture[sample_agent_id]
            network_feture_dim = len(network_feture)
        else: network_feture_dim = network_feture_dim

        self.training_type = training
        self.action_x_dataset = []
        self.action_y_dataset = []
        self.action_turn_dataset = []
        self.speed = []
        self.acc_distribute = []
        self.memory_dim = rnn_memory_dim
        self.n_layers = rnn_num_layers
        self.action_idx = 0
        self.start_game = False
        self.implement = False
        self.seq_train_rnn = False
        self.rnn_layer_type = rnn_layer_type
        self.rnn_seq_len = rnn_seq_len
        self.count_rnn_seq = 1
        self.have_rnn_model = False
        self.action_mask = None
        self.memory = None
        self.concat_rnn = concat_rnn
        
        assert isinstance(module_network, list)
        origin_module_network = module_network
        module_network = [flattened_obs_size] + module_network + [network_feture_dim]
        self.rl_input_idx = module_network.index('rl')
        just_module_network = copy.deepcopy(module_network)
        just_module_network[self.rl_input_idx] = module_connect_dim

        self.memory_max_index = episode_length  // self.rnn_seq_len
        self.memory_banch = [None] * self.memory_max_index
        self.rnn_input_idx = None
        self.preprocess_info_middle = None
        self.rnn = None
        if 'rnn' in module_network:
            rnn_types = ['RNN', 'LSTM', 'GRU']
            self.rnn_input_idx = just_module_network.index('rnn')
            assert self.rnn_layer_type in rnn_types
            assert self.rnn_input_idx < self.rl_input_idx
            if self.n_layers > 0: 
                self.have_rnn_model = True
                rnn_input_dim = just_module_network[self.rnn_input_idx - 1]
                if self.rnn_layer_type == 'LSTM':  self.rnn = nn.LSTM(rnn_input_dim, self.memory_dim, self.n_layers, batch_first=False)
                elif self.rnn_layer_type == 'GRU':  self.rnn = nn.GRU(rnn_input_dim, self.memory_dim, self.n_layers, batch_first=False)
                else:  self.rnn = nn.RNN(rnn_input_dim, self.memory_dim, num_layers=self.n_layers, batch_first=False, nonlinearity = 'relu')
            else:   module_network.remove('rnn')

        if self.have_rnn_model: first_module_idx = self.rnn_input_idx - 1
        else: first_module_idx = self.rl_input_idx + 1
        print("preprcess shold be caculate")
        assert first_module_idx - 1 >= 0
        if first_module_idx - 1 == 0:   
            module_input_dims = [just_module_network[0]]
            module_output_dims = [just_module_network[1]]
        else:  
            module_input_dims = just_module_network[: first_module_idx - 1]
            module_output_dims = just_module_network[1: first_module_idx]
        num_module_layers = len(module_input_dims)
          
        self.preprocess_info_front = nn.ModuleDict()
        for fc_layer in range(num_module_layers):
            self.preprocess_info_front[str(fc_layer)] = nn.Sequential(
                nn.Linear(module_input_dims[fc_layer], module_output_dims[fc_layer]),
                nn.ReLU(),
            )
        

        if self.have_rnn_model:
            if self.concat_rnn:  just_module_network[self.rnn_input_idx] = self.memory_dim + just_module_network[self.rnn_input_idx - 1]
            else:  just_module_network[self.rnn_input_idx] = self.memory_dim
            middle_module_idx = self.rl_input_idx + 1
            if self.rnn_input_idx + 1 == self.rl_input_idx:
                module_input_dims = [just_module_network[self.rnn_input_idx]]
                module_output_dims = [just_module_network[self.rl_input_idx]]
            else:
                module_input_dims = just_module_network[self.rnn_input_idx: self.rl_input_idx]
                module_output_dims = just_module_network[self.rnn_input_idx + 1: self.rl_input_idx + 1]
            num_module_layers = len(module_input_dims)
            self.preprocess_info_middle = nn.ModuleDict()
            for fc_layer in range(num_module_layers):
                self.preprocess_info_middle[str(fc_layer)] = nn.Sequential(
                    nn.Linear(module_input_dims[fc_layer], module_output_dims[fc_layer]),
                    nn.ReLU(),
                )
 
        back_module_input_dims = just_module_network[self.rl_input_idx : -1]
        back_module_output_dims = just_module_network[self.rl_input_idx + 1:]

        num_module_layers = len(back_module_input_dims)
        self.preprocess_info_back = nn.ModuleDict()
        for fc_layer in range(num_module_layers):
            self.preprocess_info_back[str(fc_layer)] = nn.Sequential(
                nn.Linear(back_module_input_dims[fc_layer], back_module_output_dims[fc_layer]),
                nn.ReLU(),
            )

        if rnn_drop is not None: self.preprocess_dropout = nn.Dropout(p = rnn_drop)
        else: self.preprocess_dropout = None
            
        # RL model
        rl_input_dim = module_connect_dim + flattened_obs_size
        #self.rl_input = nn.Linear(module_connect_dim + flattened_obs_size, rl_input_dim)

        
        # policy network (list of heads)
        assert isinstance(policy_network, list)
        num_policy_layers = len(policy_network)
        policy_input_dims = [rl_input_dim] + policy_network[:-1]
        policy_output_dims = policy_network
        self.policy_network = nn.ModuleDict()
        for fc_layer in range(num_policy_layers):
            self.policy_network[str(fc_layer)] = nn.Sequential(
                nn.Linear(policy_input_dims[fc_layer], policy_output_dims[fc_layer]),
                nn.ReLU(),
            )

        if num_policy_layers == 0:  self.policy_network = None
        policy_heads = [None for _ in range(len(action_space))]
        self.output_dims = []  # Network output dimension(s)
        for idx, act_space in enumerate(action_space):
            self.output_dims += [act_space]
            if num_policy_layers == 0: policy_heads[idx] = nn.Linear(rl_input_dim, act_space)
            else:  policy_heads[idx] = nn.Linear(policy_network[-1], act_space)
        self.policy_head = nn.ModuleList(policy_heads)

        # value-function network head
        assert isinstance(critic_network, list)
        num_critic_layers = len(critic_network)
        critic_input_dims = [rl_input_dim] + critic_network[:-1]
        critic_output_dims = critic_network

        self.critic_network = nn.ModuleDict()
        for fc_layer in range(num_critic_layers):
            self.critic_network[str(fc_layer)] = nn.Sequential(
                nn.Linear(critic_input_dims[fc_layer], critic_output_dims[fc_layer]),
                nn.ReLU(),
            )
        if num_critic_layers == 0: 
            self.critic_network = None
            self.vf_head = nn.Linear(rl_input_dim, 1)
        else:
            self.vf_head = nn.Linear(critic_network[-1], 1)

        
        if self.observation_space is not None:
            model_config = {
                'policy': policy,
                'policy_tag_to_agent_id_map': policy_tag_to_agent_id_map,
                'policy_network': policy_network,
                'critic_network': critic_network,
                'module_network': origin_module_network,
                'module_connect_dim': module_connect_dim,
                'rnn_layer_type': rnn_layer_type,
                'rnn_num_layers': rnn_num_layers,
                'rnn_memory_dim': rnn_memory_dim,
                'create_separate_placeholders_for_each_policy': create_separate_placeholders_for_each_policy,
                'obs_dim_corresponding_to_num_agents':obs_dim_corresponding_to_num_agents,
                'training': training,
                'episode_length': episode_length,
                'observation_space': "Not required when executing",
                'action_space': action_space.tolist(),
                'network_feture_dim': network_feture_dim,
                'flattened_obs_size': flattened_obs_size,
                'concat_rnn': concat_rnn
            }
            file_path = "custom_model/Spot_3D_NEW/model_config.json"
            with open(file_path, "w") as json_file:
                json.dump(model_config, json_file, indent=4)

        

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
        #ic(self.create_separate_placeholders_for_each_policy)
        #ic(type(self.observation_space), self.create_separate_placeholders_for_each_policy)
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

    def rnn_recover(self, rout, origin_batch, rin = None):
        
        if len(origin_batch) == 4:
            rnn_out_dim = rout.size(-1)
            output_size = origin_batch[:-1] + (rnn_out_dim,)
            rout_permute = rout.permute(1, 0, 2).contiguous()
            seq_date = rout_permute.reshape(origin_batch[1], origin_batch[2], -1, rnn_out_dim)
            output = seq_date.permute(2, 0, 1, 3).contiguous()
            #output = seq_date.reshape(output_size)
        else:  output = torch.reshape(rout, origin_batch)
        if rin is not None:  rnn_output = torch.concat((output, rin), dim=-1)
        else:  rnn_output = output
        return rnn_output

    def forward(self, obs=None, memory=None, batch_index=None, batch_size=None, training = False):
        """
            Forward pass through the model.
            Returns action probabilities and value functions.
        """
        if obs is None:
            assert not training
            assert batch_index < batch_size
            preprocess_data = self.env.cuda_data_manager.data_on_device_via_torch(f"{_PREPROCESS}")
            obs = self.get_flattened_obs()
            
            if self.create_separate_placeholders_for_each_policy:
                ip = obs
                pre_ip = preprocess_data
            else:
                # hear let "obs" to "obs-policy" 
                agent_ids_for_policy = self.policy_tag_to_agent_id_map[self.policy]
                ip = obs[:, agent_ids_for_policy]
                pre_ip = preprocess_data[:, agent_ids_for_policy]

            data_name = f"{_PROCESSED_OBSERVATIONS}_batch_{self.policy}"
            self.push_processed_data_to_batch(batch_index, batch_size, data_name = data_name, data = ip)

            data_name = f"{_PREPROCESS}_batch_{self.policy}"
            self.push_processed_data_to_batch(batch_index, batch_size, data_name = data_name, data = pre_ip)
        else:
            ip = obs
            #ic(ip[:,1,1,2:4])

        
        rl_ip = ip.detach()
        for layer in range(len(self.preprocess_info_front)):
            op = self.preprocess_info_front[str(layer)](ip)
            ip = op

        memory_t_1 = None
        if self.rnn is not None:
            if self.observation_space is None:  memory_ = memory.unsqueeze(0).unsqueeze(1) 
            elif memory is not None:  memory_ = None  #memory.detach().contiguous() # memory = layers, banch, memory_dim
            elif self.memory is None and self.observation_space is not None:
                self.count_rnn_seq = 1
                batch_size_ = ip.size(-2) *  ip.size(-3)
                if self.rnn_layer_type == 'LSTM':
                    h_ = torch.zeros(self.n_layers, batch_size_, self.memory_dim).to('cuda:0').detach()
                    c_ = torch.zeros(self.n_layers, batch_size_, self.memory_dim).to('cuda:0').detach()
                    memory_ = (h_, c_)
                else:  memory_ = torch.zeros(self.n_layers, batch_size_, self.memory_dim).to('cuda:0').detach()
                self.memory_banch[0] = memory_  # [batch / seq_len, h or (h, c)] = [seq_batch, n_layer, batch, mem_dim]
            else:
                if self.rnn_layer_type == 'LSTM': memory_ = (self.memory[0].detach(), self.memory[1].detach())
                else: memory_ = self.memory.detach()
                mem_idx = (self.count_rnn_seq) // self.rnn_seq_len # init self.count_rnn_seq is 1
                if self.count_rnn_seq % self.rnn_seq_len == 0 and mem_idx < self.memory_max_index:  
                    self.memory_banch[mem_idx] = memory_ #store the first h in seq

            rnn_in, origin_batch = self.rnn_process(op)
            r_out_, memory_reg = self.rnn(rnn_in, memory_)
            if self.concat_rnn: rin_ = op
            else: rin_ = None
            r_out = self.rnn_recover(r_out_, origin_batch, rin = rin_)

            if  memory is not None:   memory_t_1 = memory_reg
            elif self.rnn_layer_type == 'LSTM':  memory_t_1 = (memory_reg[0].detach(), memory_reg[1].detach())
            else:  memory_t_1 = memory_reg.detach()
            self.memory = memory_t_1
            self.count_rnn_seq += 1
            if memory is not None:  self.memory = None
        else:  
            #r_out = op
            rnn_in, origin_batch = self.rnn_process(op)
            if self.concat_rnn: rin_ = op
            else: rin_ = None
            r_out = self.rnn_recover(rnn_in, origin_batch, rin = rin_)
        

        if self.preprocess_info_middle is not None: 
            ip = r_out
            for layer in range(len(self.preprocess_info_middle)):
                op = self.preprocess_info_middle[str(layer)](ip)
                ip = op
            pre_process_data = op
        else: pre_process_data = r_out
        pre_process_output = None
        # output 1
        if training:
            ip = pre_process_data
            for layer in range(len(self.preprocess_info_back)):
                op = self.preprocess_info_back[str(layer)](ip)
                ip = op
            pre_process_output = op

        if self.preprocess_dropout is not None:  pre_process_data = self.preprocess_dropout(pre_process_data)
        
        # rl network
        rl_in_ = torch.concat((rl_ip, pre_process_data), dim=-1)
        #
        #concat_input = self.rl_input(rl_in_.detach())
        concat_input = rl_in_.detach()

        # rl policy
        ip = concat_input
        for layer in range(len(self.policy_network)):
            op = self.policy_network[str(layer)](ip)
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

        # rl crtic
        ip = concat_input
        for layer in range(len(self.critic_network)):
            op = self.critic_network[str(layer)](ip)
            ip = op
        vals = self.vf_head(op)[..., 0]
        if self.implement: 
            new_memory =  torch.reshape(memory_reg.detach(), origin_batch)
            return action_probs, vals, new_memory, pre_process_output

        # input output pair
        ls_linear = action_probs[0][0][0].tolist()
        ls_linear2 = action_probs[1][0][0].tolist()
        index_linear = ls_linear.index(max(ls_linear))
        index_linear2 = ls_linear2.index(max(ls_linear2))

        # input_output test
        if self.rnn is None: memory_out = None
        elif self.rnn_layer_type == 'LSTM':  memory_out = (memory_reg[0][:,0,:].tolist(), memory_reg[1][:,0,:].tolist())
        else: memory_out = memory_reg[:,0,:].tolist()

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

        if self.training_type == 'test':
            import os
            file_path = "./custom_model/Spot_3D_NEW/input_output.json"
            with open(file_path, "w") as json_file:
                json.dump(data_dict, json_file)

        elif self.training_type == 'eval':
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
        return action_probs, vals, memory_t_1, pre_process_output


    def push_processed_data_to_batch(self, batch_index, batch_size, data_name, data):
        name = f"{_PROCESSED_OBSERVATIONS}_batch_{self.policy}"
        if not self.env.cuda_data_manager.is_data_on_device_via_torch(data_name):
            processed_obs_batch = np.zeros((batch_size,) + data.shape)
            processed_obs_feed = DataFeed()
            processed_obs_feed.add_data(name = data_name, data=processed_obs_batch)
            self.env.cuda_data_manager.push_data_to_device(
                processed_obs_feed, torch_accessible=True
            )
        self.env.cuda_data_manager.data_on_device_via_torch(name = data_name)[
            batch_index
        ] = data
    

