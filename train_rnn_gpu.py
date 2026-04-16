import torch
import copy
import time
import random
import shutil
from custom_envs.spot_3dmodule.spot_3dmodule import Spot_3d
from warp_drive.env_wrapper import EnvWrapper
from custom_envs.custom_trainer import Trainer
from matplotlib import animation
from custom_envs.spot_3d.generate_rollout_animation_NEW import generate_tag_env_rollout_animation
from IPython.display import HTML
#import yaml
import logging
import wandb 
import os
from datetime import datetime
import json

class Spot_Env():
    def __init__(self):
        self.Training = True
        self.START_PHASE = 10
        self.RL_env_name = "Spot_3D_NEW"
        self.env_dir_ = "./custom_model/Spot_3D_NEW/example/"
        self.results_dir_ = "obs_delay_1s_rnn_obs_diff"
        self.wandb_columns = ["state", "video", "weight", "PreTrain", "PHASE", "Get Goals Probability", "Total Goal", "Hit the obstacle times"]
        self.notes = None #'need to cheak cuda version to push'
        self.rl_env_versioin = 'RL_Env_v21'
        self.wandb_table_history = []
        self.error_msg_list = []
        self.train_num_env = 180  # 300
        self.wandb_phase_output_video_num = 10 # 10
        self.env_train_times = 100  # 101
        self.model_params_save_freq = 100
        self.phase1_spot_batch = 61000
        self.phase1_goal_batch = 25000
        self.num5_spot_batch = 87000
        self.num3_spot_batch = 135000
        self.num2_spot_batch = 200000
        self.phase2_spot_batch = 130000
        self.phase2_goal_batch = 35000
        self.phase3_spot_batch = 140000
        self.phase3_goal_batch = 110000
        self.phase5_spot_batch = 400000
        self.phase5_goal_batch = 120000
        self.phase7_spot_batch = 200000
        self.phase7_goal_batch = 115000
        self.phase8_spot_batch = 380000
        self.phase8_goal_batch = 115000

    def Calculate_PHASE_Parameter(self, PHASE = 1, old_config = None):
        # all phase same parameter        
        self.phase_config = dict(
            Training = self.Training,
            PHASE = PHASE,
            num_wall = 1,
            wall_length = 6,
            wall_thickness = 0.6,
            num_stairs = 1,
            goal_size = 0.24,
            spot_penalty_hit = -30,
            goal_penalty_hit = -10,
            spot_cost_operate = 0,
            stairs_weight_rand = 2,
            stairs_weight_bias = 4,
            floor_width_rand = 3,
            floor_width_bias = 5,
            grid_length_bias = 20.0,
            grid_length_rand = 0.0,
            grid_length_height = 2.0,
            grid_density = 0.5,
            seed_s = int(10000 * random.random()),
            seed_g = int(10000 * random.random()),
            seed = None,
            random_agent = True,
            random_map = True,
            need_reverse_env = True,
            num_obs_see = 10,
            spot_lidar_angle = 10,
            spot_reward_get_goal = 40.0,  # main reward not suitable for change
            spot_cost_speed_y_rate_with_acc_x = 3.0,
            spot_cost_turn_rate_with_acc_x = 0.5,
            spot_action_levels = 19,
            spot_size_bias = 0.28,
            spot_head_distance = 0.58,
            spot_ass_distance = 0.67,
            obs_size_rand = 0.0,
            reward_on_floor = 30.0, # goal main reward not suitable for change
            goals_reward_step = 0.8,
            goals_reward_speed = 0.5,
            obstacle_speed_rate_with_spot = 0.8,
            goal_speed_rate_with_spot = 0.8,
            four_sign_distance = 3,
            spot_state_obs_agent_rate = 0,  # if it need shoud be large like 0.8
            preprocess_info_weight = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )
        self.train_goal_fix_spot_model = False
        spot_entropy_coeff = 0.12
        spot_action2_entropy_coeff = 0.15
        spot_action3_entropy_coeff = 0.15
        goal_entropy_coeff = 0.04
        goal_action3_entropy_coeff = 0.08
        self.train_goal_num_spot = None
        self.train_goal_num_goal = None
        self.spot_times_train_env = True
        self.goal_times_train_env = True
        rl_env_times = 60
        if PHASE == 1:
            self.train_spot_batch_size = self.phase1_spot_batch
            self.train_goal_batch_size = self.phase1_goal_batch
            self.train_goal_num_spot = 1
            self.phase_config["num_spot"] = 6
            self.phase_config["num_goals"] = 20  # 太多不行
            self.phase_config["num_obstacle"] = 3
            self.phase_config["num_stairs"] = 2
            self.phase_config["rl_fps"] = 3
            self.phase_config["goal_size"] = 0.3
            self.phase_config["spot_reward_on_floor"] = 30.0
            self.phase_config["spot_penalty_hit"] = -5.0
            self.phase_config["spot_cost_operate"] = 0.03
            self.phase_config["goals_reward_step"] = 0.8
            self.phase_config["goals_reward_speed"] = 0.5
            spot_entropy_coeff = 0.04
        elif PHASE == 2:
            self.train_spot_batch_size = self.phase2_spot_batch
            self.train_goal_batch_size = self.phase2_goal_batch
            self.train_goal_num_spot = 1
            self.phase_config["num_spot"] = 3  # 降低數量為了讓 spot 增加存活的期望值
            self.phase_config["num_goals"] = 16  # 降低數量為了讓 spot 爬樓梯
            self.phase_config["num_obstacle"] = 3
            self.phase_config["num_stairs"] = 2
            self.phase_config["rl_fps"] = 3
            self.phase_config["goal_size"] = 0.29
            self.phase_config["spot_reward_on_floor"] = 20.0
            self.phase_config["spot_penalty_hit"] = -10.0  # 為了上樓梯降低沒用，動態障礙物避障效果會很差，一下就死了
            self.phase_config["goals_reward_step"] = 0.8
            self.phase_config["goals_reward_speed"] = 0.5
            self.phase_config["obstacle_speed_rate_with_spot"] = 0.85
        elif PHASE == 3:
            self.train_spot_batch_size = self.num3_spot_batch
            self.train_goal_batch_size = self.phase5_goal_batch
            self.train_goal_num_spot = 1
            self.train_goal_num_goal = 5
            rl_env_times = 100
            spot_action3_entropy_coeff = 0.20
            self.spot_times_train_env = False
            self.phase_config["spot_cooperate"] = True
            self.phase_config['Testing_type'] = 4
            self.phase_config["num_spot"] = 3  # 多 spot 也沒用，競爭的姿態控制會太強，樓梯技能減弱
            self.phase_config["num_goals"] = 1
            self.phase_config["num_obstacle"] = 2
            self.phase_config["num_stairs"] = 2
            self.phase_config["rl_fps"] = 3
            self.phase_config["goal_size"] = 0.29
            self.phase_config["spot_reward_on_floor"] = 5.0
            self.phase_config["spot_penalty_hit"] = -12.0
            self.phase_config["goals_reward_step"] = 0.6
            self.phase_config["goals_reward_speed"] = 0.3
            self.phase_config["obs_size_rand"] = 0.1
            self.phase_config["obstacle_speed_rate_with_spot"] = 0.85
            self.phase_config["floor_width_bias"] = 7.0
            self.phase_config["stairs_weight_rand"] = 2.5
            self.phase_config["goal_speed_rate_with_spot"] = 0.7
        elif PHASE == 4:
            self.train_spot_batch_size = self.num3_spot_batch
            self.train_goal_batch_size = self.phase5_goal_batch
            self.train_goal_num_spot = 1
            self.train_goal_num_goal = 5
            spot_action3_entropy_coeff = 0.20
            rl_env_times = 70  # 短高機率 goal 還在平面， 長 spot 策略會保守(前期保守，後期不會)
            self.spot_times_train_env = False
            self.phase_config["spot_cooperate"] = True
            self.phase_config['Testing_type'] = 4
            self.phase_config["num_spot"] = 3
            self.phase_config["num_goals"] = 1  # 1 個 goal 能出現在樓梯上的機率不高，除非增加spot或goal數量
            self.phase_config["num_obstacle"] = 2
            self.phase_config["num_stairs"] = 2.5
            self.phase_config["num_wall"] = 1
            self.phase_config["wall_length"] = 4.5
            self.phase_config["rl_fps"] = 3
            self.phase_config["goal_size"] = 0.29
            self.phase_config["spot_reward_on_floor"] = 2.0
            self.phase_config["spot_penalty_hit"] = -12.0
            self.phase_config["goals_reward_step"] = 0.4
            self.phase_config["goals_reward_speed"] = 0.2
            self.phase_config["obs_size_rand"] = 0.2
            self.phase_config["obstacle_speed_rate_with_spot"] = 0.85
            self.phase_config["floor_width_bias"] = 6.5  #太大根本找不到樓梯口
            self.phase_config["stairs_weight_rand"] = 2.0
            self.phase_config["goal_speed_rate_with_spot"] = 0.7
        elif PHASE == 5:
            # 優化樓梯，策略，不知道是否有比 PHASE 6 好
            self.train_spot_batch_size = self.num2_spot_batch
            self.train_goal_batch_size = self.phase5_goal_batch
            self.train_goal_num_spot = 1
            self.train_goal_num_goal = 5
            rl_env_times = 100
            self.spot_times_train_env = False
            self.phase_config["spot_cooperate"] = True
            self.phase_config['Testing_type'] = 4
            self.phase_config["num_spot"] = 2
            self.phase_config["num_goals"] = 1  # 1 個 goal 能出現在樓梯上的機率不高，除非增加spot或goal數量
            self.phase_config["num_obstacle"] = 2
            self.phase_config["num_stairs"] = 2.9
            self.phase_config["num_wall"] = 1
            self.phase_config["wall_length"] = 5.0
            self.phase_config["rl_fps"] = 3
            self.phase_config["goal_size"] = 0.29
            self.phase_config["spot_reward_on_floor"] = 0.0
            self.phase_config["spot_penalty_hit"] = -12.0
            self.phase_config["goals_reward_step"] = 0.3
            self.phase_config["goals_reward_speed"] = 0.2
            self.phase_config["obs_size_rand"] = 0.3
            self.phase_config["obstacle_speed_rate_with_spot"] = 0.85
            self.phase_config["floor_width_bias"] = 6.00
            self.phase_config["stairs_weight_rand"] = 2.5
            self.phase_config["stairs_weight_bias"] = 3
            self.phase_config["goal_speed_rate_with_spot"] = 0.7
        elif PHASE == 6:
            # 目前看不出優點
            self.train_spot_batch_size = self.num3_spot_batch
            self.train_goal_batch_size = self.phase7_goal_batch
            self.train_goal_num_spot = 1
            self.train_goal_num_goal = 5
            rl_env_times = 200
            self.spot_times_train_env = False
            self.phase_config["spot_cooperate"] = True
            self.phase_config['Testing_type'] = 4
            self.phase_config["num_spot"] = 3  # spot 2 avoid obs effent is not well
            self.phase_config["num_goals"] = 1
            self.phase_config["num_obstacle"] = 10
            self.phase_config["num_stairs"] = 2.7
            self.phase_config["num_wall"] = 1
            self.phase_config["rl_fps"] = 3
            self.phase_config["goal_size"] = 0.29
            self.phase_config["spot_reward_on_floor"] = 0.0
            self.phase_config["spot_penalty_hit"] = -15.0 
            self.phase_config["goals_reward_step"] = 0.3
            self.phase_config["goals_reward_speed"] = 0.2
            self.phase_config["obs_size_rand"] = 0.3
            self.phase_config["obstacle_speed_rate_with_spot"] = 0.85
            self.phase_config["floor_width_bias"] = 6.0
            self.phase_config["stairs_weight_rand"] = 2.5
            self.phase_config["stairs_weight_bias"] = 3
            self.phase_config["goal_speed_rate_with_spot"] = 0.7
        elif PHASE == 7:
            # 優化動態避障（保守策略）
            self.train_spot_batch_size = self.num2_spot_batch
            self.train_goal_batch_size = self.phase8_goal_batch
            self.train_goal_num_spot = 1
            self.train_goal_num_goal = 5
            rl_env_times = 200
            self.spot_times_train_env = False
            self.phase_config['Testing_type'] = 4
            self.phase_config["num_spot"] = 2
            self.phase_config["num_goals"] = 1
            self.phase_config["num_obstacle"] = 8
            self.phase_config["num_stairs"] = 2.5
            self.phase_config["num_wall"] = 2
            self.phase_config["wall_length"] = 3.5
            self.phase_config["rl_fps"] = 3
            self.phase_config["goal_size"] = 0.29
            self.phase_config["spot_reward_on_floor"] = 0.0
            self.phase_config["spot_penalty_hit"] = -50.0
            self.phase_config["goals_reward_step"] = 0.3
            self.phase_config["goals_reward_speed"] = 0.2
            self.phase_config["obs_size_rand"] = 0.35
            self.phase_config["obstacle_speed_rate_with_spot"] = 1.15
            self.phase_config["floor_width_bias"] = 6.0
            self.phase_config["stairs_weight_rand"] = 2.5
            self.phase_config["stairs_weight_bias"] = 3
            self.phase_config["goal_speed_rate_with_spot"] = 0.7 #不能低
        elif PHASE == 8:
            # 優化動態避障（保守策略）
            self.train_spot_batch_size = self.num2_spot_batch
            self.train_goal_batch_size = self.phase8_goal_batch
            self.train_goal_num_spot = 1
            self.train_goal_num_goal = 5
            rl_env_times = 200
            self.spot_times_train_env = False
            self.phase_config['Testing_type'] = 4
            self.phase_config["num_spot"] = 2
            self.phase_config["num_goals"] = 1
            self.phase_config["num_obstacle"] = 6
            self.phase_config["num_stairs"] = 2.3
            self.phase_config["num_wall"] = 2
            self.phase_config["wall_length"] = 3.5
            self.phase_config["rl_fps"] = 3
            self.phase_config["goal_size"] = 0.29
            self.phase_config["spot_reward_on_floor"] = 0.0
            self.phase_config["spot_penalty_hit"] = -85.0
            self.phase_config["goals_reward_step"] = 0.3
            self.phase_config["goals_reward_speed"] = 0.2
            self.phase_config["obs_size_rand"] = 0.25
            self.phase_config["obstacle_speed_rate_with_spot"] = 1.1
            self.phase_config["floor_width_bias"] = 6.0
            self.phase_config["stairs_weight_rand"] = 2.5
            self.phase_config["stairs_weight_bias"] = 3
            self.phase_config["goal_speed_rate_with_spot"] = 0.7
        else: 
            # 優化動態避障（保守策略）
            self.train_spot_batch_size = self.num2_spot_batch
            self.train_goal_batch_size = self.phase8_goal_batch
            self.train_goal_num_spot = 1
            self.train_goal_num_goal = 5
            rl_env_times = 200
            self.spot_times_train_env = False
            self.phase_config['Testing_type'] = 4
            self.phase_config["num_spot"] = 2
            self.phase_config["num_goals"] = 1
            self.phase_config["num_obstacle"] = 6
            self.phase_config["num_stairs"] = 2.0
            self.phase_config["num_wall"] = 2
            self.phase_config["wall_length"] = 3.5
            self.phase_config["rl_fps"] = 3
            self.phase_config["goal_size"] = 0.29
            self.phase_config["spot_reward_on_floor"] = 0.0
            self.phase_config["spot_penalty_hit"] = -70.0
            self.phase_config["goals_reward_step"] = 0.3
            self.phase_config["goals_reward_speed"] = 0.2
            self.phase_config["obs_size_rand"] = 0.25
            self.phase_config["obstacle_speed_rate_with_spot"] = 0.9
            self.phase_config["floor_width_bias"] = 6.0
            self.phase_config["stairs_weight_rand"] = 2.5
            self.phase_config["stairs_weight_bias"] = 3
            self.phase_config["goal_speed_rate_with_spot"] = 0.7

        # related phase parameter
        self.train_spot_num_spot =  self.phase_config['num_spot']
        self.train_spot_num_goal = self.phase_config["num_goals"]

        if self.train_goal_num_spot is None: self.train_goal_num_spot = self.phase_config["num_spot"]
        if self.train_goal_num_goal is None: self.train_goal_num_goal = self.phase_config["num_goals"]

        self.phase_config["spot_max_speed_x"] = 1
        self.phase_config["spot_min_speed_x"] = -0.9
        self.phase_config["spot_max_acceleration_x"] = self.phase_config["spot_max_speed_x"] / self.phase_config["rl_fps"]
        self.phase_config["spot_min_acceleration_x"] = -1 * self.phase_config["spot_max_speed_x"] / self.phase_config["rl_fps"]
        self.phase_config["spot_max_speed_y"] = 0.88
        self.phase_config["spot_max_acceleration_y"] = self.phase_config["spot_max_speed_y"] / self.phase_config["rl_fps"]
        self.phase_config["spot_max_turn_pi_"] = 4 * self.phase_config["rl_fps"]
        self.phase_config["episode_length"] = int(rl_env_times * self.phase_config["rl_fps"])
        self.phase_config["spot_cost_operate"] = self.phase_config["spot_cost_operate"] / self.phase_config["rl_fps"]
        self.phase_config["goals_reward_step"] = self.phase_config["goals_reward_step"] / self.phase_config["rl_fps"]
        self.phase_config["goals_reward_speed"] = self.phase_config["goals_reward_speed"] / self.phase_config["rl_fps"]
        
        one_sec_spot_gamma = 0.94
        one_sec_goal_gamma = 0.94
        spot_gamma = 1 - ((1 - one_sec_spot_gamma) / self.phase_config["rl_fps"])
        goal_gamma = 1 - ((1 - one_sec_goal_gamma) / self.phase_config["rl_fps"])

        self.wandb_config = copy.deepcopy(self.phase_config)
        self.wandb_config['spot_algorithm'] = "A2C"
        self.wandb_config['goal_algorithm'] = "A2C"
        self.wandb_config['spot_gamma'] = spot_gamma
        self.wandb_config['goal_gamma'] = goal_gamma
        self.wandb_config['train_spot_batch_size'] = self.train_spot_batch_size
        self.wandb_config['train_goal_batch_size'] = self.train_goal_batch_size
        self.wandb_config['goal_model'] = [512]  # layer big is useful but need to test other thing
        self.wandb_config['goal_policy_network'] = [256, 256]
        self.wandb_config['goal_critic_network'] = [256, 256]
        self.wandb_config['spot_model'] = [256, 32]
        self.wandb_config['spot_policy_network'] = [256, 256, 256, 512]
        self.wandb_config['spot_critic_network'] = [256, 256, 256, 512, 512]
        self.wandb_config['spot_module_network'] = [64, 'rnn', 32, 'rl']
        self.wandb_config['spot_module_connect_dim'] = 12 # can less than len(self.phase_config["preprocess_info_weight"])
        self.wandb_config['memory_dim'] = 30
        self.wandb_config['rnn_layer'] = 1
        self.wandb_config['rnn_layer_type'] = 'RNN'
        self.wandb_config['spot_clip_grad_norm'] = True
        self.wandb_config['spot_max_grad_norm'] = 1.0
        self.wandb_config['spot_rnn_seq_len'] = 15
        self.wandb_config['spot_rnn_drop'] = None
        self.wandb_config['spot_random_agent'] = True
        if PHASE == 1 or PHASE == 2:  self.wandb_config['spot_random_map'] = False
        else:  self.wandb_config['spot_random_map'] = True
        self.wandb_config['reset_model_mentum'] = True
        self.wandb_config['goal_random_agent'] = True
        self.wandb_config['goal_random_map'] = True
        self.wandb_config['spot_random_all_time'] = 301
        self.wandb_config['goal_random_all_time'] = 301
        self.wandb_config['goal_lr'] = 0.002  # big suitable for phase1，but after
        self.wandb_config['spot_lr'] = 0.0002
        self.wandb_config['spot_vf_loss_coeff'] = 0.025
        if PHASE > 5:  self.wandb_config['spot_preprocess_model_lr'] = 0.0005
        else:  self.wandb_config['spot_preprocess_model_lr'] = 0
        self.wandb_config['spot_rnn_model_lr'] = 0.0005

        self.wandb_config['spot_entropy_coeff'] = spot_entropy_coeff * 2.5
        self.wandb_config['spot_action2_entropy_coeff'] = spot_action2_entropy_coeff * 2.5
        self.wandb_config['spot_action3_entropy_coeff'] = spot_action3_entropy_coeff * 2.5
        self.wandb_config['goal_entropy_coeff'] = goal_entropy_coeff
        self.wandb_config['goal_action3_entropy_coeff'] = goal_action3_entropy_coeff
        self.wandb_config['rl_env_versioin'] = self.rl_env_versioin
        self.wandb_config['train_goal_rate'] = 3
        self.wandb_config['spot_times_train_env'] = self.spot_times_train_env
        self.wandb_config['goal_times_train_env'] = self.goal_times_train_env

        #self.wandb_config['spot_model'] = 'normal'  #'BN'
        self.wandb_config['spot_model_version'] = 'v1'  #'BN'
        if old_config is not None:
            for config in self.phase_config:
                self.phase_config[config] = old_config[config]
            for config in old_config:
                self.wandb_config[config] = old_config[config]

    def wandb_init(self):
        currentDateAndTime = datetime.now()
        wandb_name = self.results_dir_ + "_PHASE_" + str(self.PHASE)
        tag_phase = "PHASE_" + str(self.PHASE)

        tag = ['Training', 'RL', tag_phase, self.rl_env_versioin]
        self.wandb_project = wandb.init(
            project = self.RL_env_name,
            name = wandb_name,
            config = self.wandb_config,
            tags = tag,
            notes = self.notes
            #mode = "offline"
        )
        self.wandb_table = wandb.Table(columns = self.wandb_columns)
                
    def cuda_init(self):    
        pytorch_cuda_init_success = torch.cuda.FloatTensor(8)
        logging.getLogger().setLevel(logging.ERROR)
        torch.cuda.empty_cache()
        torch.cuda.init()
        time.sleep(10)

    def env_init(self):
        self.train_goal = True
        self.train_spot = True
        if self.train_env_num % self.wandb_config['train_goal_rate'] == 1: 
            self.train_spot = False
            self.train_batch_size = self.train_goal_batch_size
        else: 
            self.train_goal = False
            self.train_batch_size = self.train_spot_batch_size
        if self.train_type == "test": test_compensat = 1
        else: test_compensat = 0

        self.goal_load_weight_train_env = int((self.train_env_num + self.wandb_config['train_goal_rate'] - 2 + test_compensat) / self.wandb_config['train_goal_rate'])
        self.spot_load_weight_train_env = self.train_env_num - self.goal_load_weight_train_env - 1 + test_compensat
        self.num_envs = int(self.train_batch_size / self.phase_config["episode_length"])
        self.spot_load_weight_name = f"{int(self.train_spot_batch_size * self.spot_load_weight_train_env * self.env_train_times)}.state_dict"
        self.goal_load_weight_name = f"{int(self.train_goal_batch_size * self.goal_load_weight_train_env * self.env_train_times)}.state_dict"
        if self.train_goal_fix_spot_model and self.train_goal and self.train_env_num > 1 : self.spot_load_weight_name = "1.state_dict"
        if self.train_type == "train": 
            new_spot_train_env = self.spot_load_weight_train_env
            new_goal_train_env = self.goal_load_weight_train_env
            if self.train_spot: 
                new_spot_train_env = self.spot_load_weight_train_env + 1
                self.new_spot_load_weight_name = f"{int(self.train_spot_batch_size * new_spot_train_env * self.env_train_times)}.state_dict"
                self.new_goal_load_weight_name = f"{int(self.train_goal_batch_size * new_goal_train_env * self.env_train_times)}.state_dict"
            if self.train_goal:
                new_goal_train_env = self.goal_load_weight_train_env + 1
                self.new_spot_load_weight_name = f"{int(self.train_spot_batch_size * new_spot_train_env * self.env_train_times)}.state_dict"
                self.new_goal_load_weight_name = f"{int(self.train_goal_batch_size * new_goal_train_env * self.env_train_times)}.state_dict"
        else:
            self.new_spot_load_weight_name = f"{int(self.train_spot_batch_size * self.spot_load_weight_train_env * self.env_train_times)}.state_dict"
            self.new_goal_load_weight_name = f"{int(self.train_goal_batch_size * self.goal_load_weight_train_env * self.env_train_times)}.state_dict"
    
    def rl_env_all_config(self):
        env_config = dict(
            name="spot_3d",
            env=self.phase_config,
            trainer = dict(
                num_envs = self.num_envs,
                train_batch_size = self.train_batch_size,
                num_episodes = self.env_train_times * self.train_batch_size / self.phase_config["episode_length"],
            ),

            policy=dict(
                obstacle = dict(
                    to_train = self.train_goal,
                    algorithm = "A2C",
                    gamma = 0.98,
                    lr = 0.005,
                    entropy_coeff = 0.1,  # 0.2 is to high
                    action2_entropy_coeff = None,
                    action3_entropy_coeff = None,
                    model = dict(
                        type = "fully_connected", fc_dims=[256, 256], model_ckpt_filepath = ""
                    ),
                ),
                goal = dict(
                    to_train = self.train_goal,
                    algorithm =  self.wandb_config['goal_algorithm'],
                    gamma = self.wandb_config['goal_gamma'],
                    lr = self.wandb_config['goal_lr'],
                    entropy_coeff = self.wandb_config['goal_entropy_coeff'],
                    action2_entropy_coeff = None,
                    action3_entropy_coeff = self.wandb_config['goal_action3_entropy_coeff'],
                    model = dict(
                        type = "fully_connected", 
                        fc_dims = self.wandb_config['goal_model'],
                        policy_network = self.wandb_config['goal_policy_network'],
                        critic_network = self.wandb_config['goal_critic_network'],
                        model_ckpt_filepath=""
                    ),
                ),
                spot = dict(
                    to_train = self.train_spot,
                    algorithm = self.wandb_config['spot_algorithm'],
                    gamma = self.wandb_config['spot_gamma'],
                    vf_loss_coeff = self.wandb_config['spot_vf_loss_coeff'],   # Critic loos default = 0.01
                    entropy_coeff = self.wandb_config['spot_entropy_coeff'],  # max(0.0001, 0.01 * pow(0.97,int(phase_ori_/60))) ,   # mean_entropy loos default = 0.01
                    action2_entropy_coeff = self.wandb_config['spot_action2_entropy_coeff'],
                    action3_entropy_coeff = self.wandb_config['spot_action3_entropy_coeff'],
                    lr = self.wandb_config['spot_lr'],
                    preprocess_model_lr = self.wandb_config['spot_preprocess_model_lr'],
                    rnn_model_lr = self.wandb_config['spot_rnn_model_lr'],
                    preprocess_info_weight = self.wandb_config['preprocess_info_weight'],
                    clip_grad_norm = self.wandb_config['spot_clip_grad_norm'],
                    max_grad_norm = self.wandb_config['spot_max_grad_norm'],
                    model = dict(
                        type = "module_model",
                        rl_policy_network = self.wandb_config['spot_policy_network'],
                        rl_critic_network = self.wandb_config['spot_critic_network'],
                        module_network = self.wandb_config['spot_module_network'],
                        module_connect_dim = self.wandb_config['spot_module_connect_dim'],
                        rnn_layer_type = self.wandb_config['rnn_layer_type'],
                        rnn_layer = self.wandb_config['rnn_layer'],
                        rnn_memory_dim = self.wandb_config['memory_dim'],
                        rnn_seq_len = self.wandb_config['spot_rnn_seq_len'],
                        spot_rnn_drop = self.wandb_config['spot_rnn_drop'],
                        concat_rnn = True,
                        model_ckpt_filepath="",
                    ),
                ),
            ),
            saving=dict(
                metrics_log_freq = 10,  # how often to print the metrics
                model_params_save_freq = self.model_params_save_freq,  # how often to save the model parameters
                basedir = "./custom_model",  # base folder used for saving
                name = self.RL_env_name,  # experiment name
                tag = "example",  # experiment tag
                at_last_store_freq = 29 * self.model_params_save_freq,
                current_step = self.train_env_num * self.model_params_save_freq,
            ),
        )
        file_path = "custom_model/Spot_3D_NEW/rl_config.json"
        with open(file_path, "w") as json_file:
            json.dump(self.wandb_config, json_file)
        return env_config
    
    def creat_RL_ENV(self):
        run_config = self.rl_env_all_config()
        self.env_wrapper = EnvWrapper(
            Spot_3d(**run_config["env"]),
            num_envs = run_config["trainer"]["num_envs"],
            use_cuda = True,
        )
        policy_tag_to_agent_id_map = {
            "spot": list(self.env_wrapper.env.spots),
            "obstacle": list(self.env_wrapper.env.obstacles),
            "goal": list(self.env_wrapper.env.goals),
        }
        self.trainer = Trainer(
            env_wrapper = self.env_wrapper,
            config = run_config,
            policy_tag_to_agent_id_map = policy_tag_to_agent_id_map,
            USE_WANDB = True,
            results_dir = self.results_dir_,
            training = self.train_type
        )

    def search_files_with_keywords(self, directory, keywords):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if all(keyword in file for keyword in keywords):
                    matching_files = os.path.join(root, file)
        return matching_files
    
    def load_weight(self, wandb_path=None):
        if self.train_env_num == 1 and self.PHASE > 1 or wandb_path is not None:
            # from wandb
            if self.download_weight_artifact_name is not None and self.PHASE == self.START_PHASE or wandb_path is not None:
                self.wandb_old_file = self.wandb_project.use_artifact(self.download_weight_artifact_name)
            else: 
                Artifact_name = "PHASE_" + str(self.PHASE - 1)
                self.wandb_old_file = self.wandb_project.use_artifact(f'{Artifact_name}:latest')
            artifact_dir = self.wandb_old_file.download()
            spot_path = self.search_files_with_keywords(artifact_dir, keywords = ['spot', '.state_dict'])
            goal_path = self.search_files_with_keywords(artifact_dir, keywords = ['goal', '.state_dict'])
            obs_path = self.search_files_with_keywords(artifact_dir, keywords = ['obs', '.state_dict'])
            
            spot_reg_file = os.path.join(self.env_dir_, "spot_reg.state_dict")
            shutil.copy2(spot_path, spot_reg_file)
            goal_reg_file = os.path.join(self.env_dir_, "goal_reg.state_dict")
            shutil.copy2(goal_path, goal_reg_file)
            obs_reg_file = os.path.join(self.env_dir_, "obs_reg.state_dict")
            shutil.copy2(obs_path, obs_reg_file)
            os.rename(spot_reg_file, self.env_dir_+ self.results_dir_ + "/spot_" + self.spot_load_weight_name)
            shutil.copy2(spot_path, spot_reg_file)
            os.rename(spot_reg_file, self.env_dir_+ self.results_dir_ + "/spot_1.state_dict")
            os.rename(goal_reg_file, self.env_dir_+ self.results_dir_ + "/goal_" + self.goal_load_weight_name)
            os.rename(obs_reg_file, self.env_dir_+ self.results_dir_ + "/obstacle_" + self.goal_load_weight_name)
            shutil.rmtree(artifact_dir)
            
        if self.train_env_num > 1 or self.PHASE > 1 or wandb_path is not None:
            self.spot_file_path = self.env_dir_+ self.results_dir_ + "/spot_" + self.spot_load_weight_name
            self.goal_file_path = self.env_dir_+ self.results_dir_ + "/goal_" + self.goal_load_weight_name
            self.obs_file_path = self.env_dir_ + self.results_dir_ + "/obstacle_" + self.goal_load_weight_name
            self.new_spot_file_path = self.env_dir_+ self.results_dir_ + "/spot_" + self.new_spot_load_weight_name
            self.new_goal_file_path = self.env_dir_+ self.results_dir_ + "/goal_" + self.new_goal_load_weight_name
            self.new_obs_file_path = self.env_dir_ + self.results_dir_ + "/obstacle_" + self.new_goal_load_weight_name
            self.trainer.load_model_checkpoint(
                {
                    "spot": self.spot_file_path,
                    "goal": self.goal_file_path,
                    "obstacle": self.obs_file_path,
                }
            )

    def video_test(self, test_num_time =None):
        if test_num_time == None: test_num_time = self.train_env_num
        anim = generate_tag_env_rollout_animation(
            self.trainer,
            fps = 10,
            use_wandb = True,
            PHASE_ = self.PHASE,
            state = test_num_time,
            phase_env_num = self.train_num_env,
            wandb_table = self.wandb_table,
            pre_train = False,
            wandb_name = self.results_dir_ + "_PHASE_" + str(self.PHASE),
            wandb_table_history = self.wandb_table_history,
            num_goals = self.phase_config["num_goals"],
        )

    def remove_agents_old_weight(self):
        if self.train_env_num > 1:
            if self.train_goal:
                self.trainer.remove_old_weight({"goal": self.env_dir_+ self.results_dir_ + "/goal_" + self.goal_load_weight_name})
                self.trainer.remove_old_weight({"obstacle": self.env_dir_+ self.results_dir_ + "/obstacle_" + self.goal_load_weight_name})
            if self.train_spot:
                self.trainer.remove_old_weight({"spot": self.env_dir_+ self.results_dir_ + "/spot_" + self.spot_load_weight_name})
        
    def clear_reg(self):
        self.trainer.graceful_close()
        torch.cuda.synchronize()
        del self.trainer
        del self.env_wrapper

    def log_all_data_to_wandb(self):
        if self.PHASE > 1:
            origin_Artifact_name = "PHASE_" + str(self.PHASE - 1)
            self.phase_weight = self.wandb_project.use_artifact(f'{origin_Artifact_name}:latest')
        wandb_model = wandb.Artifact("PHASE_" + str(self.PHASE), type="model", metadata=self.phase_config)
        model_path = "./custom_envs/module_connected.py"
        wandb_model.add_file(model_path, name='spot_model_code.py')
        wandb_model.add_file(self.new_spot_file_path, name="spot_phase_" + str(self.PHASE)+"_final_0.state_dict")
        wandb_model.add_file(self.new_goal_file_path, name="goal_phase_" + str(self.PHASE)+"_final_0.state_dict")
        wandb_model.add_file(self.new_obs_file_path, name="obs_phase_" + str(self.PHASE)+"_final_0.state_dict")
        python_code_path = "custom_envs/spot_3dmodule/spot_3dmodule.py"
        cuda_code_path = "custom_envs/spot_3dmodule/spot_3dmodule_step.cu"
        wandb_model.add_file(python_code_path, name="python_code.py")
        wandb_model.add_file(cuda_code_path, name="cuda_code.cu")
        rl_config_path = "custom_model/Spot_3D_NEW/rl_config.json"
        wandb_model.add_file(rl_config_path, name="rl_config.json")
        model_config_path = "custom_model/Spot_3D_NEW/model_config.json"
        wandb_model.add_file(model_config_path, name="model_config.json")
        csv_file_path = "./custom_model/Spot_3D_NEW/input_output.json"
        wandb_model.add_file(csv_file_path, name='input_output.json')
        self.wandb_project.log_artifact(wandb_model) 
        phase_table = wandb.Table(
            columns = self.wandb_columns, 
            data = self.wandb_table_history
        )
        wandb.log({"Phase " + str(self.PHASE) + " Table": phase_table})
        self.wandb_table_history = []

    def error_condition(self, error_msg):
        self.error_msg_list.append(
            [
                self.PHASE_num_env,
                str(error_msg)
            ]
        ) 
        self.START_PHASE_ = self.PHASE
        error_msg_columns = ["error time", "error"]
        error_table = wandb.Table(
                columns = error_msg_columns, 
                data = self.error_msg_list
            )
        wandb.log({"error msg": error_table})

        try:
            del self.trainer
        except:
            print("not del trainer")    
        try:
            del self.env_wrapper
        except:
            print("not del env_wrapper")  
        if len(self.error_msg_list) > 30: break_train = True
        else: break_train = False
        wandb.finish()
        return break_train
        
    def test(self, artifact_version):
        self.Training = False
        self.train_type = "eval"
        Start_Testing_type = 1
        End_test_type = 2
        test_num_time = 5
        download_weight_artifact_name = 'jqu314159/Spot_3D_NEW/' + artifact_version
        self.download_weight_artifact_name = download_weight_artifact_name
        for Testing_type_ in range(5):
            Testing_type = Start_Testing_type + Testing_type_
            wandb_name = 'Evaluation test ' + str(Testing_type) + " " + artifact_version
            tag = ['RL', 'Evaluation', str(download_weight_artifact_name), 'implement']
            self.wandb_project = wandb.init(
                project = self.RL_env_name,
                name = wandb_name,
                tags = tag,
            )
            self.wandb_table = wandb.Table(columns = self.wandb_columns)
            wandb_path = self.wandb_project.use_artifact(download_weight_artifact_name).download()
            rl_config_path = self.search_files_with_keywords(wandb_path, keywords = ['rl_config'])
            old_config = {}
            with open(rl_config_path, "r") as json_file:
                old_config = json.load(json_file)
            self.Calculate_PHASE_Parameter(old_config = old_config)
            self.phase_config['Training'] = False
            self.phase_config['random_agent'] = True
            self.phase_config['random_map'] = True
            self.phase_config['Testing_type'] = Testing_type
            self.train_env_num = 1
            #self.train_type = "train"
            self.PHASE = self.phase_config['PHASE']
            for time in range(test_num_time):
                self.phase_config['seed'] = int(10000 * random.random())
                self.cuda_init()
                self.env_init()
                self.creat_RL_ENV()
                self.load_weight(wandb_path)
                self.video_test(time + 1)
                self.clear_reg()
            phase_table = wandb.Table(
                columns = self.wandb_columns, 
                data = self.wandb_table_history
            )
            wandb.log({'Evaluation test ' + str(Testing_type) + " table": phase_table})
            wandb.finish()
            if Testing_type == End_test_type: break
            self.wandb_table_history = []

    def del_preprocess_buffer(self):
        ckpt_filepath = './custom_model/Spot_3D_NEW/buffer_data.json'
        if os.path.exists(ckpt_filepath):
            os.remove(ckpt_filepath)

    def train(self, download_weight_artifact_name = None):
        self.Training = True
        self.download_weight_artifact_name = download_weight_artifact_name
        self.del_preprocess_buffer()
        for PHASE in range(99):
            #try:
                if PHASE < self.START_PHASE: continue
                self.PHASE_num_env = 0
                self.PHASE = PHASE
                self.Calculate_PHASE_Parameter(PHASE = PHASE)
                phase_total_env = self.train_num_env + self.wandb_phase_output_video_num
                self.wandb_output_video_env = int(phase_total_env / self.wandb_phase_output_video_num)
                self.train_env_num = 0
                self.wandb_init()
                for PHASE_num_env in range(phase_total_env):
                    self.PHASE_num_env = PHASE_num_env
                    if PHASE_num_env % self.wandb_output_video_env == self.wandb_output_video_env - 1: self.train_type = "test"
                    #if PHASE_num_env % self.wandb_output_video_env == 1: self.train_type = "test"
                    else: 
                        self.train_type = "train"
                        self.train_env_num += 1
                    self.cuda_init()
                    self.env_init()
                    if PHASE == 1 and PHASE_num_env > 90: 
                        self.phase_config["spot_cost_operate"] = 0.0000001
                        #self.phase_config["goals_reward_speed"] = 0.0000001
                    if PHASE_num_env % self.wandb_config['spot_random_all_time']  == self.wandb_config['spot_random_all_time'] - 1: self.phase_config['seed_s'] = int(10000 * random.random())
                    if PHASE_num_env % self.wandb_config['goal_random_all_time']  == self.wandb_config['goal_random_all_time'] - 1: self.phase_config['seed_g'] = int(10000 * random.random())
                    if self.train_spot:
                        self.phase_config['num_spot'] = self.train_spot_num_spot 
                        self.phase_config['num_goals'] = self.train_spot_num_goal 
                        self.phase_config['seed'] = self.phase_config['seed_s']
                        self.phase_config['random_agent'] = self.wandb_config['spot_random_agent'] 
                        self.phase_config['random_map'] = self.wandb_config['spot_random_map']
                        self.phase_config['Training'] = self.wandb_config['spot_times_train_env']
                    elif self.train_goal:
                        self.phase_config['num_spot'] = self.train_goal_num_spot
                        self.phase_config['num_goals'] = self.train_goal_num_goal
                        self.phase_config['seed'] = self.phase_config['seed_g']
                        self.phase_config['random_agent'] = self.wandb_config['goal_random_agent']
                        self.phase_config['random_map'] = self.wandb_config['goal_random_map']
                        self.phase_config['Training'] = self.wandb_config['goal_times_train_env']
                    else:
                        self.phase_config['num_spot'] = self.train_spot_num_spot
                        self.phase_config['num_goals'] = self.train_spot_num_goal 
                        self.phase_config['seed'] = int(10000 * random.random())
                        self.phase_config['random_agent'] = True
                        self.phase_config['random_map'] = True
                        self.phase_config['Training'] = self.wandb_config['spot_times_train_env']

                    self.creat_RL_ENV()
                    self.load_weight()
                    if self.train_spot and PHASE_num_env <= 2: self.video_test()
                    if self.train_type == "test": self.video_test()
                    else: 
                        self.trainer.train(reset_model_mentum = self.wandb_config['reset_model_mentum'])
                        self.remove_agents_old_weight()
                    self.clear_reg()
                self.log_all_data_to_wandb()
                self.del_preprocess_buffer()
                wandb.finish()
                time.sleep(100)
            #except Exception as error_msg:
            #    break_train = self.error_condition(error_msg)
            #    if break_train: break

if __name__ == "__main__":
    spot_env = Spot_Env()
    spot_env.train(download_weight_artifact_name = 'PHASE_9:v12')
    #spot_env.test(artifact_version = 'PHASE_1:v93')