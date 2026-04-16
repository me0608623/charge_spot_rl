import torch
import copy
import time
import random


from warp_drive.env_wrapper import EnvWrapper

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
        self.START_PHASE = 1
        self.RL_env_name = "Spot_3D_NEW"
        self.env_dir_ = "./custom_model/Spot_3D_NEW/example/"
        self.results_dir_ = "cheak_four_distance_1_5"
        self.wandb_columns = ["state", "video", "weight", "PreTrain", "PHASE", "Get Goals Probability", "Total Goal", "Hit the obstacle times"]
        self.notes = None # 'phase2-3 num_goal 改成 20，實在沒有必要降低 goal 的數量'
        self.rl_env_versioin = 'RL_Env_v11'
        self.wandb_table_history = []
        self.error_msg_list = []
        self.train_num_env = 300  # 300
        self.wandb_phase_output_video_num = 10 # 10
        self.env_train_times = 100  # 101
        self.model_params_save_freq = 100
        self.phase1_spot_batch = 70000
        self.phase1_goal_batch = 20000
        self.phase3_spot_batch = 100000
        self.phase3_goal_batch = 20000
        self.phase5_spot_batch = 140000
        self.phase5_goal_batch = 40000
        self.action_x_data = []
        self.action_y_data = []
        self.action_turn_data = []
        self.speed_distribute = []
        self.acc_distribute = []
        self.run_self_goal_model()

    def Calculate_PHASE_Parameter(self, PHASE = 1, old_config = None):
        # all phase same parameter        
        self.phase_config = dict(
            Training = None,
            PHASE = None,
            num_wall = None,
            wall_length = None,
            wall_thickness = None,
            num_stairs = None,
            goal_size = None,
            spot_penalty_hit = None,
            goal_penalty_hit = None,
            spot_cost_operate = None,
            stairs_weight_rand = None,
            stairs_weight_bias = None,
            floor_width_rand = None,
            floor_width_bias = None,
            grid_length_bias = None,
            grid_length_rand = None,
            grid_length_height = None,
            grid_density = None,
            seed_s = None,
            seed_g = None,
            seed = None,
            random_agent = None,
            random_map = None,
            num_obs_see = None,
            spot_lidar_angle = None,
            spot_reward_get_goal = None,
            spot_cost_speed_y_rate_with_acc_x = None,
            spot_cost_turn_rate_with_acc_x = None,
            spot_action_levels = None,
            spot_size_bias = None,
            obs_size_rand =None,
            goals_reward_step =None,
            goals_reward_speed = None,
            obstacle_speed_rate_with_spot = None,
            goal_speed_rate_with_spot = None,
            four_sign_distance = None,
            num_spot = None,
            num_goals = None,
            num_obstacle = None,
            rl_fps = None,
            reward_on_floor = None,
            spot_reward_on_floor = None,
            spot_max_speed_x = None,
            spot_min_speed_x = None,
            spot_max_acceleration_x = None,
            spot_min_acceleration_x = None,
            spot_max_speed_y = None,
            spot_max_acceleration_y = None,
            spot_max_turn_pi_ = None,
            episode_length = None
        )
        self.wandb_config = {}
        if old_config is not None:
            for config in self.phase_config:
                self.phase_config[config] = old_config[config]
            for config in old_config:
                self.wandb_config[config] = old_config[config]
                
    def cuda_init(self):    
        pytorch_cuda_init_success = torch.cuda.FloatTensor(8)
        logging.getLogger().setLevel(logging.ERROR)
        torch.cuda.empty_cache()
        torch.cuda.init()

    def env_init(self):
        self.train_goal = True
        self.train_spot = True
        self.train_spot_batch_size = self.phase1_spot_batch
        self.train_goal_batch_size = self.phase1_goal_batch
        if self.train_env_num % self.wandb_config['train_goal_rate'] == 1: 
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
        
        if self.train_type == "train": 
            new_spot_train_env = self.spot_load_weight_train_env
            new_goal_train_env = self.goal_load_weight_train_env
            if self.train_spot: new_spot_train_env = self.spot_load_weight_train_env + 1
            if self.train_goal: new_goal_train_env = self.goal_load_weight_train_env + 1
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
                    entropy_coeff = 0.1,
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
                    #vf_loss_coeff = 1,   # Critic loos default = 0.01
                    entropy_coeff = self.wandb_config['spot_entropy_coeff'],  # max(0.0001, 0.01 * pow(0.97,int(phase_ori_/60))) ,   # mean_entropy loos default = 0.01
                    action2_entropy_coeff = self.wandb_config['spot_action2_entropy_coeff'],
                    action3_entropy_coeff = self.wandb_config['spot_action3_entropy_coeff'],
                    lr = self.wandb_config['spot_lr'],
                    model = dict(
                        type = "custom_model",
                        model_type = self.wandb_config['spot_model'] ,
                        model_ckpt_filepath=""
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

    def rl_env_all_config_fc(self):
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
                    entropy_coeff = 0.1,
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
                    #vf_loss_coeff = 1,   # Critic loos default = 0.01
                    entropy_coeff = self.wandb_config['spot_entropy_coeff'],  # max(0.0001, 0.01 * pow(0.97,int(phase_ori_/60))) ,   # mean_entropy loos default = 0.01
                    action2_entropy_coeff = self.wandb_config['spot_action2_entropy_coeff'],
                    action3_entropy_coeff = self.wandb_config['spot_action3_entropy_coeff'],
                    lr = self.wandb_config['spot_lr'],
                    model = dict(
                        type = "fully_connected",
                        fc_dims = self.wandb_config['spot_model'],
                        policy_network = self.wandb_config['spot_policy_network'],
                        critic_network = self.wandb_config['spot_critic_network'],
                        model_ckpt_filepath=""
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
        from custom_envs.spot_3d.spot_3d import Spot_3d
        from custom_envs.custom_trainer import Trainer
        if self.model_connect == 'fc': run_config = self.rl_env_all_config_fc()
        else: run_config = self.rl_env_all_config()
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

    def search_files_with_keywords(self, directory, keywords:list):
        matching_files = None
        for root, dirs, files in os.walk(directory):
            for file in files:
                if all(keyword in file for keyword in keywords):
                    matching_files = os.path.join(root, file)
        return matching_files
        
        
    
    def load_weight(self, wandb_path=None):
        import shutil
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
            #goal_path = self.env_dir_+ "compete/goal_0.state_dict"
            #obs_path = self.env_dir_+ "compete/obstacle_0.state_dict"
            
            spot_reg_file = os.path.join(self.env_dir_, "spot_reg.state_dict")
            shutil.copy2(spot_path, spot_reg_file)
            goal_reg_file = os.path.join(self.env_dir_, "goal_reg.state_dict")
            shutil.copy2(goal_path, goal_reg_file)
            obs_reg_file = os.path.join(self.env_dir_, "obs_reg.state_dict")
            shutil.copy2(obs_path, obs_reg_file)
            os.rename(spot_reg_file, self.env_dir_+ self.results_dir_ + "/spot_" + self.spot_load_weight_name)
            os.rename(goal_reg_file, self.env_dir_+ self.results_dir_ + "/goal_" + self.goal_load_weight_name)
            os.rename(obs_reg_file, self.env_dir_+ self.results_dir_ + "/obstacle_" + self.goal_load_weight_name)
            shutil.rmtree(artifact_dir)
        if self.train_env_num > 1 or self.PHASE > 1 or wandb_path is not None:
            self.spot_file_path = self.env_dir_+ self.results_dir_ + "/spot_" + self.spot_load_weight_name
            if self.artifact_version in self.self_goal_version:
                self.goal_file_path = self.env_dir_+ self.results_dir_+ "/goal_" + self.goal_load_weight_name
                self.obs_file_path = self.env_dir_ + self.results_dir_ + "/obstacle_" + self.goal_load_weight_name
            else:
                self.goal_file_path = self.env_dir_+ "independ/goal_" + self.goal_load_weight_name
                self.obs_file_path = self.env_dir_ + "independ/obstacle_" + self.goal_load_weight_name

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
            wandb_name = self.artifact_version,
            wandb_table_history = self.wandb_table_history,
            num_goals = self.phase_config["num_goals"],
        )
        
    def clear_reg(self):
        if self.model_connect != 'fc': 
            action, speed, acc_dis = self.trainer.callback_custom_model_action(policy_name = "spot")
            
            self.action_x_data.extend(action['x'])
            self.action_y_data.extend(action['y'])
            self.action_turn_data.extend(action['turn'])
            self.speed_distribute.extend(speed)
            
            self.acc_distribute.extend(acc_dis)
        self.trainer.graceful_close()
        torch.cuda.synchronize()
        del self.trainer
        del self.env_wrapper

    def display_spot_custom_model_action(self):
        table_1 = wandb.Table(data=self.action_x_data, columns=["step", "action_x_idx"])
        table_2 = wandb.Table(data=self.action_y_data, columns=["step", "action_y_idx"])
        table_3 = wandb.Table(data=self.action_turn_data, columns=["step", "action_turn_idx"])
        table_4 = wandb.Table(data=self.speed_distribute, columns=["speed_x", "speed_y", "turn_idx"])
        table_5 = wandb.Table(data=self.acc_distribute, columns=["acc_x_idx", "acc_y_idx", "turn_idx"])
        scatter_4 = wandb.plot.scatter(
            table_4,
            x='speed_x',
            y='speed_y',
            title='speed distributed'
        )
        scatter_5 = wandb.plot.scatter(
            table_5,
            x='acc_x_idx',
            y='acc_y_idx',
            title='acc distributed'
        )
        scatter_6 = wandb.plot.scatter(
            table_5,
            x='acc_y_idx',
            y='turn_idx',
            title='speed y turn distributed'
        )
        scatter_7 = wandb.plot.scatter(
            table_5,
            x='acc_x_idx',
            y='turn_idx',
            title='acc turn distributed'
        )

        # Log custom tables, which will show up in customizable charts in the UI
        wandb.log({
            'action x idx': table_1, 
            'action y idx': table_2, 
            'action turn idx': table_3,
            'scatter_4': scatter_4,
            'scatter_5': scatter_5,
            'scatter_6': scatter_6,
            'scatter_7': scatter_7,
        })
        
    def replace_old_file(self, wandb_path):
        import shutil
        env_dir = "spot_3d"
        reg_dir = "old_env"
        """
        now_python_path =  "./custom_envs/" + env_dir + "/spot_3d.py"
        reg_python_path = "./custom_envs/" + reg_dir + "/spot_3d.py"
        python_code_path = self.search_files_with_keywords(wandb_path, keywords = ['python_code'])
        shutil.copy2(now_python_path, reg_python_path)
        shutil.copy2(python_code_path, now_python_path)
        """

        now_model_path =  "./custom_envs/models.py"
        reg_model_path = "./custom_envs/" + reg_dir + "/models.py"
        model_code_path = self.search_files_with_keywords(wandb_path, keywords = ['model_code'])
        shutil.copy2(now_model_path, reg_model_path)
        shutil.copy2(model_code_path, now_model_path)

    def recover_old_file(self):
        import shutil
        env_dir = "spot_3d"
        reg_dir = "old_env"
        """
        now_python_path =  "./custom_envs/" + env_dir + "/spot_3d.py"
        reg_python_path = "./custom_envs/" + reg_dir + "/spot_3d.py"
        shutil.copy2(reg_python_path, now_python_path)
        """

        now_model_path =  "./custom_envs/models.py"
        reg_model_path = "./custom_envs/" + reg_dir + "/models.py"
        shutil.copy2(reg_model_path, now_model_path)
        
    def test(self, artifact_version, Start_Testing_type = 1, End_test_type = 2 ,test_num_time = 5, model_connect = None):
        self.Training = False
        self.train_type = "eval"
        self.model_connect = model_connect
        self.artifact_version = artifact_version
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
            self.replace_old_file(wandb_path)
            self.Calculate_PHASE_Parameter(old_config = old_config)
            self.phase_config['Training'] = False
            self.phase_config['random_agent'] = False
            self.phase_config['random_map'] = False
            self.phase_config['Testing_type'] = Testing_type
            self.phase_config['episode_length'] = 300
            self.phase_config['num_spot'] = 5
            self.phase_config['num_goals'] = 15
            self.phase_config['num_obstacle'] = 7
            self.phase_config['num_stairs'] = 2
            self.phase_config['num_wall'] = 1
            self.phase_config["goal_speed_rate_with_spot"] = 0.6
            self.phase_config["obstacle_speed_rate_with_spot"] = 0.9
            self.phase_config["goal_size"] = 0.29
            self.phase_config["obs_size_rand"] = 0.1
            self.phase_config["stairs_weight_rand"] = 2.5
            self.phase_config["floor_width_bias"] = 5.5
            
            self.train_env_num = 1
            #self.train_type = "train"
            self.PHASE = self.phase_config['PHASE']
            avoid_idx = 0
            avoid_seed = [5, 8, 21, 23, 24, 38]  #9, 25, 28, 33
            for time in range(test_num_time):
                if 2 + time + avoid_idx in avoid_seed:
                    avoid_idx += 1
                    if 2 + time + avoid_idx in avoid_seed:
                        avoid_idx += 1
                        if 2 + time + avoid_idx in avoid_seed:
                            avoid_idx += 1
                self.phase_config['seed'] = 2 + time + avoid_idx
                print("test_num_time = ",time + 1)
                self.cuda_init()
                self.env_init()
                self.creat_RL_ENV()
                self.load_weight(wandb_path)
                self.video_test(self.phase_config['seed'])
                self.clear_reg()
            phase_table = wandb.Table(
                columns = self.wandb_columns, 
                data = self.wandb_table_history
            )
            self.recover_old_file()
            if self.model_connect != 'fc':  self.display_spot_custom_model_action()
            wandb.log({'Evaluation test ' + str(Testing_type) + " table": phase_table})
            wandb.finish()
            if Testing_type == End_test_type: break
            self.wandb_table_history = []

    def run_self_goal_model(self):
        self.self_goal_version = [
            'PHASE_6:v18',
            'PHASE_6:v17',
            'PHASE_5:v37',
            'PHASE_5:v40',
            'PHASE_4:v57',
            'PHASE_4:v54',
            'PHASE_3:v113',
            'PHASE_3:v115',
            'PHASE_3:v112',
            'PHASE_3:v123',
            'PHASE_3:v141',
            'PHASE_3:v142',
            'PHASE_3:v103',
            'PHASE_3:v149',
            'PHASE_3:v150',
            'PHASE_3:v154',
            'PHASE_3:v155',
            'PHASE_3:v156',
            'PHASE_2:v147',
            'PHASE_2:v146',
            'PHASE_2:v143',
            'PHASE_2:v155',
            'PHASE_2:v174',
            'PHASE_2:v173',
            'PHASE_2:v136',
            'PHASE_2:v181',
            'PHASE_2:v182',
            'PHASE_2:v186',
            'PHASE_2:v187',
            'PHASE_1:v190',
            'PHASE_1:v202',
            'PHASE_1:v221',
            'PHASE_1:v220',
            'PHASE_1:v193',
            'PHASE_1:v194',
            'PHASE_1:v121',
            'PHASE_1:v228',
            'PHASE_1:v229',
            'PHASE_1:v233',
            'PHASE_1:v234',
        ]

if __name__ == "__main__":
    """

    #phase 3 136 error 2_174
    """ #this all need in self_goal_version
    """
    old_finsh{
        '6':[['18'], ['17', 'fc']],
        '5':[['40'], ['37', 'fc']],
        '4':[['57'],['49'], ['54', 'fc']],
        '3': [

            ['98'], ['100'], ['101'], ['142'], ['99'] ,['146'],['144'], ['134'],   
            ['104'], ['112'], ['145'], ['131'], ['149'], 
            ['105', 'fc'],  ['109', 'fc'],  ['110', 'fc'],  
            ['141', 'fc'], ['115', 'fc'], ['113', 'fc'], ['111', 'fc'],  ['120', 'fc']
        ],
        '2': [
            ['129'], ['177'], ['143'], ['167'], ['171'], ['145'], ['176'], ['178'], ['131'], ['149'],['174'], 
            ['133'], ['134'], ['137'], ['155'], ['163'], ['165'], ['160'], ['179'],
            ['158'], ['164'], ['128'], ['148'], ['161'], ['172'],
            ['138', 'fc'], ['139', 'fc'], ['140', 'fc'], ['142', 'fc'], ['144', 'fc'],
            ['141', 'fc'],  ['152', 'fc'] , ['173', 'fc'], ['147', 'fc'], ['146', 'fc'], 
        ],
        '1': [
            ['211'], ['218'], ['225'], ['209'], ['175'], ['178'], ['195'], ['196'], ['208'], ['221'], 
            ['184'], ['176'], ['179'], ['181'], ['224'], ['190'], ['202'], ['210'], ['214'],
            ['205'], ['192'], ['207'], ['226'], ['212'], ['219'],
            ['185', 'fc'],['194', 'fc'], ['220', 'fc'],  ['193', 'fc'],
            ['188', 'fc'], ['186', 'fc'], ['187', 'fc'], ['189', 'fc'], ['191', 'fc'], ['199', 'fc']
        ],
    }
    """
    run_version = {
        '3':[
            ['154'], ['155'], ['156'],
        ],
        '2':[
            ['187'],

        ],
    }

    for phase in run_version:
        for version in run_version[phase]:

            spot_env = Spot_Env()
            print('phase',phase,' version', version)
            artifact_version = 'PHASE_' + phase + ':v' + version[0]
            if  len(version) > 1: model_connect = version[1]
            else: model_connect = None
            print("now run version ", artifact_version)
            spot_env.test(
                artifact_version = artifact_version,
                Start_Testing_type = 2,
                End_test_type = 2 ,
                test_num_time = 30,
                model_connect = model_connect
            )
            del spot_env
            time.sleep(15)

