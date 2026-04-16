import torch.nn as nn
import torch
import torch.nn.functional as func
from gym.spaces import Box, Dict, Discrete, MultiDiscrete
import numpy as np
try:
    from warp_drive.utils.constants import Constants
    from warp_drive.utils.data_feed import DataFeed
    _OBSERVATIONS = Constants.OBSERVATIONS
    _PROCESSED_OBSERVATIONS = Constants.PROCESSED_OBSERVATIONS
except:
    _OBSERVATIONS = "observations"
    PROCESSED_OBSERVATIONS = "processed_observations"
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


class Critic_Model(nn.Module):
    def __init__(
        self,
        model_type
    ):
        super(Critic_Model, self).__init__()
        self.critic_model = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )


class Actor_Model(nn.Module):
    def __init__(
        self,
        action_space,
        mix_output_dim,
        model_type
    ):
        super(Actor_Model, self).__init__(model_type)
        self.output_dims = []
        policy_heads = [None for _ in range(len(action_space))]
        for idx, act_space in enumerate(action_space):
            self.output_dims += [act_space]
            policy_heads[idx] = nn.Linear(mix_output_dim, act_space)
        self.actor_model = nn.ModuleList(policy_heads)
        self.actor_MLP_model = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, mix_output_dim)
        )


class Mix_Process_Model( Actor_Model, Critic_Model):
    def __init__(
        self,
        action_space,
        input_dim,
        mix_output_dim,
        hidden_dim,
        model_type = None,
    ):
        """
            Recurrent Layers 
                self.xx = nn.RNN(input_dim , hidden_dim)
                self.xx = nn.GRU(input_dim , hidden_dim)
            Dropout Layers 
                self.xx = nn.Dropout(p=0.2)
        """
        super(Mix_Process_Model, self).__init__(action_space, mix_output_dim, model_type)
        #self.rnn_test = nn.RNN(input_dim , hidden_dim)
        self.hidden_dim = hidden_dim
        self.model_type = model_type

        self.linear_test = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
        )
        #self.lstm_test = nn.GRU(mix_output_dim , hidden_dim)
        
        self.observation_len = input_dim
        #self.Encoder_test = nn.TransformerEncoderLayer(d_model= 256, nhead= 16, dim_feedforward= 128, dropout=0)

    def __call__(self, input_date, action_mask, agent_num):  
        """
            Recurrent Layers 
                output_date = self.Layers_process(input_date, agent_num,  self.xx(model_layer), layer_type)
            Dropout Layers 
                output_date = self.xx(input_date)
            input_date dim = [batch, batch, input_date] or [batch, batch, batch, input_date] (time_state, env, agent_id, input_dim)
            value  dim = [batch, batch]
            action_probs[0].shape = [batch, batch, level]
            action_probs[1].shape = [batch, batch, level]
        """
        
        input_dims = input_date.dim()
        #print("input_date.shape = ",input_date.shape, " input_dims = ",input_dims)
        origin_dim_1_size = input_date.shape[0]
        if input_dims == 3 and input_date.shape[2] == self.observation_len: 
            reshape_input_date = input_date.view(-1, self.observation_len)
        elif input_dims == 4 and input_date.shape[3] == self.observation_len: 
            origin_dim_2_size = input_date.shape[1]
            reshape_input_date = input_date.view(-1, self.observation_len)
        elif input_dims > 4: print("input dim error",  input_date.shape)
        else: reshape_input_date = input_date
        #print("reshape_input_date.shape = ",reshape_input_date.shape)
        front_proc = self.linear_test(reshape_input_date)
        #tran_out = self.Layers_process(front_proc, agent_num, self.Encoder_test, 'transformer')
        #mix_output = self.Mix_model(front_proc)
        actor_input = self.actor_MLP_model(front_proc)
        action_masks = [None for _ in range(len(self.output_dims))]
        if action_mask is not None:
            start = 0
            for idx, dim in enumerate(self.output_dims):
                action_masks[idx] = action_mask[..., start : start + dim]
                start = start + dim
        action_probs = [
            func.softmax(apply_logit_mask(ph(actor_input), action_masks[idx]), dim=-1)
            for idx, ph in enumerate(self.actor_model)
        ]
        value = self.critic_model(front_proc)[..., 0]
        
        if input_dims == 3:
            value = value.view(origin_dim_1_size, -1)
            action_probs[0] = action_probs[0].view(origin_dim_1_size, -1, action_probs[0].shape[1])
            action_probs[1] = action_probs[1].view(origin_dim_1_size, -1, action_probs[1].shape[1])
            action_probs[2] = action_probs[2].view(origin_dim_1_size, -1, action_probs[2].shape[1])
        if input_dims == 4:
            value = value.view(origin_dim_1_size, origin_dim_2_size, -1)
            action_probs[0] = action_probs[0].view(origin_dim_1_size, origin_dim_2_size, -1, action_probs[0].shape[1])
            action_probs[1] = action_probs[1].view(origin_dim_1_size, origin_dim_2_size, -1, action_probs[1].shape[1])
            action_probs[2] = action_probs[2].view(origin_dim_1_size, origin_dim_2_size, -1, action_probs[2].shape[1])
        #print("value.shape = ",value.shape)
        #print("action_probs.shape = ",action_probs[0].shape)
        return value, action_probs


    def Layers_process(self, input_date, agent_num, model, layer_type):
        """
            Recurrent Layers cannot train without 3 dimation ,
            so this funtion chage dimation to 3 and recover it.
            3 dimation is([batch env, batch in env(agent num) , input dimation ])
        """
        input_date_size =  input_date.size()

        if input_date_size[2] == agent_num:          
            input_date = torch.reshape(input_date, (input_date_size[0],-1, input_date_size[3]))
        if layer_type == 'rnn':
            output_date, _ = model(input_date, None)
        elif layer_type == 'transformer':
            output_date = model(input_date)
        else:
            print("layer_type need to lower case")
        if input_date_size[2] == agent_num:          
            output_date = torch.reshape(output_date, (input_date_size[0], input_date_size[1],input_date_size[2], -1))
  
        return output_date
