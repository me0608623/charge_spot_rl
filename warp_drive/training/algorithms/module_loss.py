import torch
import torch.nn as nn
from icecream import ic
import os
import json
import copy

class PreProcess_Module_Loss:
    def __init__(self, info_weight=None):
        self.weight = info_weight

    def compute_loss(self, target, input_data):
        weight_len = input_data.size(-1) 
        if self.weight is None:  self.weight = [1] * weight_len
        else:  assert len(self.weight) == weight_len, "Weight length must match input_data's last dimension."

        display_loss = {}
        preprcess_loss = 0.0
        for idx in range(weight_len):
            input_data_flattened = input_data[..., idx].view(-1)  
            target_flattened = target[..., idx].view(-1).detach()
            #if idx == 3: ic(target_flattened[:30])
            loss_ = nn.L1Loss(reduction='none')(input_data_flattened, target_flattened)
            loss_ = torch.clamp(loss_, min=0.01)
            loss_p = torch.log(loss_)  # obs loss
            """
            if idx < 6 : 
                
            else:  loss_p = loss_ * loss_ # goal loss
            """
            loss_ = loss_p * self.weight[idx]  

            display_loss["module_feture_" + str(idx) + "_loss"] = loss_.mean().item()
            preprcess_loss += loss_.mean()
        display_loss["preprcess_loss"] = preprcess_loss

        return preprcess_loss, display_loss

class Buffer:
    def __init__(self, buffer_large, device='cuda', seq_len=1, rnn_layer_type='RNN', buffer_replace_size = 2, buffer_file='buffer_data.json'):
        self.device = device
        self.buffer_large = buffer_large
        self.obs_buffer = None
        self.memory_buffer = None
        self.preprocess_buffer = None
        self.h0_buffer = None
        self.c0_buffer = None
        self.valid_mask = None
        self.current_size = 0
        self.seq_len = seq_len
        self.rnn_layer_type = rnn_layer_type
        self.buffer_file = buffer_file


    def save_to_file(self):
        """ 將 buffer 內容保存為二進制檔案 """
        try:
            buffer_data = {
                'obs_buffer': self.obs_buffer.cpu() if self.obs_buffer is not None else None,
                'preprocess_buffer': self.preprocess_buffer.cpu() if self.preprocess_buffer is not None else None,
                'h0_buffer': self.h0_buffer.cpu() if self.h0_buffer is not None else None,
                'c0_buffer': self.c0_buffer.cpu() if self.c0_buffer is not None else None,
                'valid_mask': self.valid_mask.cpu() if self.valid_mask is not None else None,
                'current_size': self.current_size
            }

            torch.save(buffer_data, self.buffer_file)
        except Exception as e:
            print(f"Failed to save data to {self.buffer_file}: {e}")

    def load_from_file(self):
        """ 從二進制檔案加載 buffer 內容進行初始化 """
        if os.path.exists(self.buffer_file):
            try:
                buffer_data = torch.load(self.buffer_file)
                
                # 直接使用載入的 tensor
                self.obs_buffer = buffer_data['obs_buffer'] if buffer_data['obs_buffer'] is not None else None
                self.preprocess_buffer = buffer_data['preprocess_buffer'] if buffer_data['preprocess_buffer'] is not None else None
                self.h0_buffer = buffer_data['h0_buffer'] if buffer_data['h0_buffer'] is not None else None
                self.c0_buffer = buffer_data['c0_buffer'] if buffer_data['c0_buffer'] is not None else None
                self.valid_mask = buffer_data['valid_mask'] if buffer_data['valid_mask'] is not None else None
                self.current_size = buffer_data['current_size']
                
                print(f"Buffer loaded from {self.buffer_file}")
            except Exception as e:
                print(f"Failed to load data from {self.buffer_file}: {e}")
        else:
            print(f"No existing buffer file found at {self.buffer_file}")

    def initialize(self, obs, memory, preprocess):
        """
            obs = (batch, batch, batch, data_dim)
            preprocess = (batch, batch, batch, preprocess_dim)
            h = (rnn_layers, batch_seq, memory_dim) batch_seq = batch / seq_len
            memory = h or (h, c)
        """
        #ic(obs.size())
        self.load_from_file()
        #ic(obs.size())
        if memory is None:
            self.batch_size = obs.size(0) * obs.size(1) * obs.size(2)
            self.seq_len = 1
            h_0 = None
            c_0 = None
        elif self.rnn_layer_type == 'LSTM':
            self.batch_size = memory[0].size(1)
            self.origin_h0_dim = (memory[0].size(0), memory[0].size(1), memory[0].size(2))
            h_0 = memory[0]
            c_0 = memory[1]
        else:
            self.batch_size = memory.size(1)
            self.origin_h0_dim = (memory.size(0), memory.size(1), memory.size(2))
            h_0 = memory
            c_0 = None
            
        self.buffer_size = self.batch_size * self.buffer_large
        assert self.buffer_large < 400
        self.buffer_replace_size = int(self.buffer_size / 400.0)

        
        self.origin_obs_dim = (obs.size(0), obs.size(1), obs.size(2), obs.size(3))
        self.origin_preprocess_dim = (preprocess.size(0), preprocess.size(1), preprocess.size(2), preprocess.size(3))
        
        # 調整序列資料的形狀
        permute_obs = obs.permute(1, 2, 0, 3).contiguous()
        permute_preprocess = preprocess.permute(1, 2, 0, 3).contiguous()
        #ic(self.origin_obs_dim, permute_obs.size() ,h_0.size())
        batch_seq_obs = permute_obs.reshape(-1, self.seq_len, self.origin_obs_dim[3]).cpu()
        batch_seq_preprocess = permute_preprocess.reshape(-1, self.seq_len, self.origin_preprocess_dim[3]).cpu()
        #ic(batch_seq_obs.size(), h_0.size())
        if memory is not None:  assert batch_seq_obs.size(0) == h_0.size(1)

        # 只在 LSTM 的情況下初始化記憶狀態
        if h_0 is not None:  batch_seq_h0 = h_0.permute(1, 0, 2).contiguous().cpu()
        else:  batch_seq_h0 = None
            
        if c_0 is not None:  batch_seq_c0 = c_0.permute(1, 0, 2).contiguous().cpu()
        else:  batch_seq_c0 = None

        # 將初始資料放入 buffer 的前部
        obs_buffer_array = torch.zeros((self.buffer_size, *batch_seq_obs.shape[1:]))
        if self.obs_buffer is None or self.obs_buffer.size() != obs_buffer_array.size():  
            self.obs_buffer = obs_buffer_array.cpu()
            self.obs_buffer[:self.batch_size] = batch_seq_obs

        # 初始化 buffer，保持在 CPU 上
        preprocess_buffer_array = torch.zeros((self.buffer_size, *batch_seq_preprocess.shape[1:]))
        if self.preprocess_buffer is None or self.preprocess_buffer.size() != preprocess_buffer_array.size():
            self.preprocess_buffer = preprocess_buffer_array.cpu()
            self.preprocess_buffer[:self.batch_size] = batch_seq_preprocess
        
        if memory is not None:
            h0_buffer_array = torch.zeros((self.buffer_size, *batch_seq_h0.shape[1:]))
            if self.h0_buffer is None or self.h0_buffer.size() != h0_buffer_array.size():  
                self.h0_buffer = h0_buffer_array.cpu()
                self.h0_buffer[:self.batch_size] = batch_seq_h0
            if self.rnn_layer_type == 'LSTM':
                c0_buffer_array = torch.zeros((self.buffer_size, *batch_seq_c0.shape[1:]))
                if self.c0_buffer is None or self.c0_buffer.size() != c0_buffer_array.size():
                    self.c0_buffer = c0_buffer_array.cpu()
                    self.c0_buffer[:self.batch_size] = batch_seq_c0
            else:  self.c0_buffer = None
        else:  self.h0_buffer = None
        valid_mask_array = torch.zeros(self.buffer_size, dtype=torch.bool)
        if self.valid_mask is None or self.valid_mask.size != valid_mask_array.size():  
            self.valid_mask = valid_mask_array
            self.valid_mask[:self.batch_size] = True
        if self.current_size == 0:  self.current_size = self.batch_size  # 初始化當前大小
        self.max_batch_size = 256
        self.batch_size = min(self.batch_size, self.max_batch_size)
        
    def update(self, obs, memory, preprocess):
        

        permute_obs = obs.permute(1, 2, 0, 3).contiguous()
        permute_preprocess = preprocess.permute(1, 2, 0, 3).contiguous()
        
        seq_obs = permute_obs.reshape(-1, self.seq_len, self.origin_obs_dim[3]).cpu()
        seq_preprocess = permute_preprocess.reshape(-1, self.seq_len, self.origin_preprocess_dim[3]).cpu()

        # 計算新的資料大小
        new_size = seq_obs.size(0)

        # 根據 current_size 決定需要替換的舊資料數量
        if self.current_size == self.buffer_size:  replace_size = self.buffer_replace_size
        else:  replace_size = 0
        if memory is None:
            h0_seq = None
            c0_seq = None
        elif self.rnn_layer_type == 'LSTM':
            h0_seq = memory[0].permute(1, 0, 2).contiguous()
            c0_seq = memory[1].permute(1, 0, 2).contiguous()
        else:
            h0_seq = memory.permute(1, 0, 2).contiguous()

        if replace_size > 0:
            replace_indices = torch.randint(0, self.buffer_size, (replace_size,))
            data_idx = torch.randint(0, new_size, (replace_size,))

            # 用新的觀測資料替換舊資料
            self.obs_buffer[replace_indices] = seq_obs[data_idx]
            self.preprocess_buffer[replace_indices] = seq_preprocess[data_idx]
            #ic(replace_indices,self.preprocess_buffer[replace_indices,:,3], seq_preprocess[data_idx,:,3])
            
            if self.h0_buffer is not None and memory is not None:
                self.h0_buffer[replace_indices] = h0_seq.cpu()[data_idx]
                if self.rnn_layer_type == 'LSTM':  self.c0_buffer[replace_indices] = c0_seq.cpu()[data_idx]
        else:
            start_index = self.current_size
            end_index = min(start_index + new_size, self.buffer_size)

            self.obs_buffer[start_index:end_index] = seq_obs.cpu()[:end_index - start_index]
            self.preprocess_buffer[start_index:end_index] = seq_preprocess.cpu()[:end_index - start_index]
            self.valid_mask[start_index:end_index] = True
            self.current_size = end_index
            if self.h0_buffer is not None:
                self.h0_buffer[start_index:end_index] = h0_seq.cpu()[:end_index - start_index]
                if self.rnn_layer_type == 'LSTM': self.c0_buffer[start_index:end_index] = c0_seq.cpu()[:end_index - start_index]

    def get_data(self):
        actual_size = min(self.current_size, self.buffer_size)
        valid_indices = torch.where(self.valid_mask[:actual_size] == True)[0]
        random_indices = valid_indices[torch.randperm(len(valid_indices))[:self.batch_size]]
        
        if self.batch_size != self.max_batch_size:
            data_obs_batch = self.origin_obs_dim
            data_preprocess_batch = self.origin_preprocess_dim
            h_data_batch = self.origin_h0_dim
        else:
            assert self.batch_size % 4 == 0
            #data_batch = (2 * self.seq_len, 2, int(self.batch_size * 0.25), -1)
            data_obs_batch = (2 * self.seq_len, 2, int(self.batch_size * 0.25),  self.origin_obs_dim[3])
            data_preprocess_batch = (2 * self.seq_len, 2, int(self.batch_size * 0.25),  self.origin_preprocess_dim[3])
            h_data_batch = (self.origin_h0_dim[0], self.batch_size, self.origin_h0_dim[2])
        #ic(self.origin_obs_dim, data_batch)
        #ic(self.origin_h0_dim, h_data_batch)
        # 獲取有效的觀測資料
        obs_data = self.obs_buffer[random_indices].contiguous()
        preprocess_data = self.preprocess_buffer[random_indices].contiguous()
        obs_data_reshape = obs_data.reshape(-1, data_obs_batch[1], data_obs_batch[2], self.seq_len, data_obs_batch[3])
        preprocess_data_reshape = preprocess_data.reshape(-1, data_preprocess_batch[1], data_preprocess_batch[2], self.seq_len, data_preprocess_batch[3])
        #ic(random_indices, preprocess_data[:5,:,3])

        # 將資料轉換回原始格式
        seq_obs = obs_data_reshape.permute(0, 3, 1, 2, 4).contiguous().reshape(data_obs_batch).to(self.device)
        seq_preprocess = preprocess_data_reshape.permute(0, 3, 1, 2, 4).contiguous().reshape(data_preprocess_batch).to(self.device)

        if self.h0_buffer is not None:
            h0_data = self.h0_buffer[random_indices].permute(1, 0, 2).contiguous().reshape(h_data_batch)
            if self.rnn_layer_type == 'LSTM' and self.c0_buffer is not None:
                c0_data = self.c0_buffer[random_indices].permute(1, 0, 2).contiguous().reshape(h_data_batch) 
                memory = (h0_data.to(self.device), c0_data.to(self.device))
            else:
                memory = h0_data.to(self.device)
        else:  memory = None
        #ic(seq_obs[:, 0,5,15]) # [[time_banch,seq_len], batch, batch, data_dim]
        #ic(seq_obs[:, 0,5,15], seq_preprocess[:, 0,5,6])
        return seq_obs, seq_preprocess, memory

if __name__ == "__main__":
    # 範例使用
    test_PreProcess_Module_Loss = False
    test_Buffer = True
    if test_PreProcess_Module_Loss:
        input_data = torch.rand(5, 3)  # 模擬5x3的輸入數據
        target = torch.rand(5, 3)      # 模擬5x3的目標數據
        weight = [0.5, 1.0, 1.5]       # 權重列表

        module_loss = PreProcess_Module_Loss(weight=weight)  # 初始化類，傳入權重
        loss = module_loss.compute_loss(target, input_data)  # 計算損失
        print(f"Total Loss: {loss}")  # 輸出總損失
    elif test_Buffer:
        #unittest.main()
        rnn_buffer = Buffer(
            buffer_large = 10, 
            device = 'cuda:0',
            seq_len = 5,
            rnn_layer_type = 'RNN',
            buffer_replace_size = 10,
            buffer_file = './custom_model/Spot_3D_NEW/buffer_data.json'
        )
        rnn_buffer.save_to_file()
        
