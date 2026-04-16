# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
The Trainer, PerfStats and Metrics classes
"""
from icecream import ic
import json
import logging
import os
import random
import time
import pycuda.autoinit
import numpy as np
import torch
import yaml
import wandb

from gym.spaces import Discrete, MultiDiscrete
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from warp_drive.managers.function_manager import CUDASampler
from warp_drive.training.algorithms.a2c_entropy import A2CE
from warp_drive.training.algorithms.a2c_ken import A2CK
from warp_drive.training.algorithms.a2c import A2C
from warp_drive.training.algorithms.ppo import PPO
from warp_drive.training.algorithms.module_loss import PreProcess_Module_Loss
from warp_drive.training.algorithms.module_loss import Buffer
from warp_drive.training.models.fully_connected import FullyConnected
from custom_envs.fully_connected import CustomModelConnected


from warp_drive.training.utils.data_loader import create_and_push_data_placeholders
from warp_drive.training.utils.param_scheduler import ParamScheduler

from warp_drive.utils.common import get_project_root
from warp_drive.utils.constants import Constants


_ROOT_DIR = get_project_root()

_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS
_DONE_FLAGS = Constants.DONE_FLAGS
_PROCESSED_OBSERVATIONS = Constants.PROCESSED_OBSERVATIONS
_COMBINED = "combined"
_PREPROCESS = "preprocess_real_data"
_EPSILON = 1e-10  # small number to prevent indeterminate divisions


def all_equal(iterable):
    """
    Check all elements of an iterable (e.g., list) are identical
    """
    return len(set(iterable)) <= 1


def recursive_merge_config_dicts(config, default_config):
    """
    Merge the configuration dictionary with the default configuration
    dictionary to fill in any missing configuration keys.
    """
    assert isinstance(config, dict)
    assert isinstance(default_config, dict)

    for k, v in default_config.items():
        if k not in config:
            config[k] = v
        else:
            if isinstance(v, dict):
                recursive_merge_config_dicts(config[k], v)
    return config


def verbose_print(message, device_id=None):
    if device_id is None:
        device_id = 0
    print(f"[Device {device_id}]: {message} ")


class Trainer:
    """
    The trainer object. Contains modules train(), save_model_checkpoint() and
    fetch_episode_global_states()
    """

    def __init__(
        self,
        env_wrapper = None,
        config = None,
        policy_tag_to_agent_id_map = None,
        create_separate_placeholders_for_each_policy = False,
        obs_dim_corresponding_to_num_agents = "first",
        num_devices=1,
        device_id=0,
        results_dir = None,
        verbose = True,
        USE_WANDB = False,
        training = 'train'
    ):
        """
        Args:
            env_wrapper: the wrapped environment object.
            config: the experiment run configuration.
            policy_tag_to_agent_id_map:
                a dictionary mapping policy tag to agent ids.
            create_separate_placeholders_for_each_policy:
                a flag indicating whether there exist separate observations,
                actions and rewards placeholders, for each policy,
                as designed in the step function. The placeholders will be
                used in the step() function and during training.
                When there's only a single policy, this flag will be False.
                It can also be True when there are multiple policies, yet
                all the agents have the same obs and action space shapes,
                so we can share the same placeholder.
                Defaults to "False".
            obs_dim_corresponding_to_num_agents:
                indicative of which dimension in the observation corresponds
                to the number of agents, as designed in the step function.
                It may be "first" or "last". In other words,
                observations may be shaped (num_agents, *feature_dim) or
                (*feature_dim, num_agents). This is required in order for
                WarpDrive to process the observations correctly. This is only
                relevant when a single obs key corresponds to multiple agents.
                Defaults to "first".
            num_devices: number of GPU devices used for (distributed) training.
                Defaults to 1.
            device_id: device ID. This is set in the context of multi-GPU training.
            results_dir: (optional) name of the directory to save results into.
            verbose:
                if False, training metrics are not printed to the screen.
                Defaults to True.
        """
        assert env_wrapper is not None
        assert env_wrapper.use_cuda
        assert config is not None
        assert isinstance(create_separate_placeholders_for_each_policy, bool)
        assert obs_dim_corresponding_to_num_agents in ["first", "last"]
        self.obs_dim_corresponding_to_num_agents = obs_dim_corresponding_to_num_agents
        self.wandb = USE_WANDB
        self.training = training

        self.cuda_envs = env_wrapper


        # Load in the default configuration
        default_config_path = os.path.join(
            _ROOT_DIR, "warp_drive", "training", "run_configs", "default_configs.yaml"
        )
        with open(default_config_path, "r", encoding="utf8") as fp:
            default_config = yaml.safe_load(fp)

        self.config = config
        # Fill in any missing configuration parameters using the default values
        # Trainer-related configurations
        self.config["trainer"] = recursive_merge_config_dicts(
            self.config["trainer"], default_config["trainer"]
        )
        # Policy-related configurations
        for key in config["policy"]:
            self.config["policy"][key] = recursive_merge_config_dicts(
                self.config["policy"][key], default_config["policy"]
            )
        # Saving-related configurations
        self.config["saving"] = recursive_merge_config_dicts(
            self.config["saving"], default_config["saving"]
        )
        

        if results_dir is None:
            # Use the current time as the name for the results directory.
            results_dir = f"{time.time():10.0f}"

        # Directory to save model checkpoints and metrics
        self.save_dir = os.path.join(
            self._get_config(["saving", "basedir"]),
            self._get_config(["saving", "name"]),
            self._get_config(["saving", "tag"]),
            results_dir,
        )
            

        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        # Save the run configuration
        config_filename = os.path.join(self.save_dir, "run_config.json")
        with open(config_filename, "a+", encoding="utf8") as fp:
            json.dump(self.config, fp)
            fp.write("\n")

        # Flag to determine whether to print training metrics
        self.verbose = verbose

        # Number of GPU devices in the train
        self.num_devices = num_devices
        self.device_id = device_id

        # Policies
        self.policy_tag_to_agent_id_map = policy_tag_to_agent_id_map
        self.policies = list(self._get_config(["policy"]).keys())
        self.policies_to_train = [
            policy
            for policy in self.policies
            if self._get_config(["policy", policy, "to_train"])
        ]

        # Flag indicating whether there needs to be separate placeholders / arrays
        # for observation, actions and rewards, for each policy
        self.create_separate_placeholders_for_each_policy = (
            create_separate_placeholders_for_each_policy
        )
        # Note: separate placeholders are needed only when there are
        # multiple policies
        if self.create_separate_placeholders_for_each_policy:
            assert len(self.policies) > 1

        # Number of iterations algebra
        self.num_episodes = self._get_config(["trainer", "num_episodes"])
        assert self.num_episodes > 0
        self.training_batch_size = self._get_config(["trainer", "train_batch_size"])
        self.num_envs = self._get_config(["trainer", "num_envs"])      
        
        self.training_batch_size_per_env = self.training_batch_size // self.num_envs
        assert self.training_batch_size_per_env > 0

        # Push all the data and tensor arrays to the GPU
        # upon resetting environments for the very first time.
        self.cuda_envs.reset_all_envs()
        #data_reward = DataFeed()
        #data_reward.add_data(name="diffent_reward", 
        #                     data=np.array([1.0, 0.5, 0.7]),
        #                     save_copy_and_apply_at_reset = True)
        #self.cuda_envs.cuda_data_manager.push_data_to_device(data_reward, torch_accessible=True)
        # Create and push data placeholders to the device
        create_and_push_data_placeholders(
            self.cuda_envs,
            self.policy_tag_to_agent_id_map,
            self.create_separate_placeholders_for_each_policy,
            self.obs_dim_corresponding_to_num_agents,
            self.training_batch_size_per_env,
        )

        self.cuda_sample_controller = CUDASampler(self.cuda_envs.cuda_function_manager)

        # Register action placeholders
        if self.create_separate_placeholders_for_each_policy:
            for policy in self.policies:
                sample_agent_id = self.policy_tag_to_agent_id_map[policy][0]
                action_space = self.cuda_envs.env.action_space[sample_agent_id]

                self._register_actions(action_space, suffix=f"_{policy}")
        else:
            sample_policy = self.policies[0]
            sample_agent_id = self.policy_tag_to_agent_id_map[sample_policy][0]
            action_space = self.cuda_envs.env.action_space[sample_agent_id]

            self._register_actions(action_space)

        # Seeding (device_id is included for distributed training)
        seed = (
            self.config["trainer"].get("seed", np.int32(time.time())) + self.device_id
        )
        self.cuda_sample_controller.init_random(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Define models, optimizers, and learning rate schedules
        self.models = {}
        self.optimizers = {}
        self.optimizers_module = {}
        self.lr_schedules = {}
        self.pre_process_lr_schedules = {}
        self.rnn_lr_schedules = {}

        # For logging episodic reward
        self.num_completed_episodes = {}
        self.episodic_reward_sum = {}
        self.reward_running_sum = {}

        # Indicates the current timestep of the policy model
        self.current_timestep = {}

        self.total_steps = self.cuda_envs.episode_length * self.num_episodes
        self.num_iters = int(self.total_steps // self.training_batch_size)
        if self.num_iters == 0:
            raise ValueError(
                "Not enough steps to even perform a single training iteration!. "
                "Please increase the number of episodes or reduce the training "
                "batch size."
            )

        for policy in self.policies:
            self.current_timestep[policy] = 0
            self._initialize_policy_model(policy)

        # Load the model parameters (if model checkpoints are specified)
        # Note: Loading the model checkpoint may also update the current timestep!
        self.load_model_checkpoint()

        for policy in self.policies:
            # Push the models to the GPU
            # If distributed train, sync model using DDP
            if self.num_devices > 1:
                self.models[policy] = DDP(
                    self.models[policy], device_ids=[self.device_id]
                )

            # Initialize the (ADAM) optimizer
            lr_config = self._get_config(["policy", policy, "lr"])
            preprocess_model_lr = lr_config
            if "preprocess_model_lr" in self._get_config(["policy", policy]): preprocess_model_lr = self._get_config(["policy", policy, "preprocess_model_lr"])
            rnn_lr = preprocess_model_lr
            if "rnn_model_lr" in self._get_config(["policy", policy]): rnn_lr = self._get_config(["policy", policy, "rnn_model_lr"])
            self.lr_schedules[policy] = ParamScheduler(lr_config)
            initial_lr = self.lr_schedules[policy].get_param_value(
                timestep=self.current_timestep[policy]
            )
            self.pre_process_lr_schedules[policy] = ParamScheduler(preprocess_model_lr)
            initial_pre_process_lr = self.pre_process_lr_schedules[policy].get_param_value(
                timestep=self.current_timestep[policy]
            )
            self.rnn_lr_schedules[policy] = ParamScheduler(rnn_lr)
            initial_rnn_lr = self.rnn_lr_schedules[policy].get_param_value(
                timestep=self.current_timestep[policy]
            )

            self.preprocess_params = [param for name, param in self.models[policy].named_parameters() if 'preprocess_' in name]
            self.rnn_params = [param for name, param in self.models[policy].named_parameters() if name.startswith('rnn')]
            self.rl_params = [param for name, param in self.models[policy].named_parameters() if not name.startswith('rnn') and 'preprocess_' not in name]
            self.optimizers[policy] = torch.optim.Adam(
                [
                    {'params': self.rl_params, 'lr': initial_lr},
                    {'params': self.preprocess_params, 'lr': 0},         
                    {'params': self.rnn_params, 'lr': 0}
                ]
            )
            self.optimizers_module[policy] = torch.optim.Adam(
                [
                    {'params': self.rl_params, 'lr': 0},
                    {'params': self.preprocess_params, 'lr': initial_pre_process_lr},         
                    {'params': self.rnn_params, 'lr': initial_rnn_lr}
                ]
            )

            # Initialize episodic rewards and push to the GPU
            num_agents_for_policy = len(self.policy_tag_to_agent_id_map[policy])
            self.num_completed_episodes[policy] = 0
            self.episodic_reward_sum[policy] = (
                torch.tensor(0).type(torch.float32)
            ).to('cuda:0')
            self.reward_running_sum[policy] = torch.zeros(
                (self.num_envs, num_agents_for_policy)
            ).to('cuda:0')
        # Initialize the trainers
        self.trainers = {}
        self.trainers_module = {}
        self.clip_grad_norm = {}
        self.max_grad_norm = {}
        for policy in self.policies_to_train:
            self._initialize_policy_algorithm(policy)

        # Performance Stats
        self.perf_stats = PerfStats()

        # Metrics
        self.metrics = Metrics()

    def _initialize_policy_algorithm(self, policy):
        algorithm = self._get_config(["policy", policy, "algorithm"])
        assert algorithm in ["A2C", "PPO"]
        entropy_coeff = self._get_config(["policy", policy, "entropy_coeff"])
        vf_loss_coeff = self._get_config(["policy", policy, "vf_loss_coeff"])

        action2_entropy_coeff = self._get_config(["policy", policy, "action2_entropy_coeff"])
        action3_entropy_coeff = self._get_config(["policy", policy, "action3_entropy_coeff"])
        self.clip_grad_norm[policy] = self._get_config(
            ["policy", policy, "clip_grad_norm"]
        )
        if self.clip_grad_norm[policy]:
            self.max_grad_norm[policy] = self._get_config(
                ["policy", policy, "max_grad_norm"]
            )
        normalize_advantage = self._get_config(
            ["policy", policy, "normalize_advantage"]
        )
        normalize_return = self._get_config(["policy", policy, "normalize_return"])
        gamma = self._get_config(["policy", policy, "gamma"])
        if "preprocess_info_weight" in self._get_config(["policy", policy]): info_weight = self._get_config(["policy", policy, "preprocess_info_weight"])
        else: info_weight = None
        if algorithm == "A2C":
            # Advantage Actor-Critic
            if policy == 'spot':
                print("policy = ",policy)
                self.trainers[policy] = A2CK(
                    discount_factor_gamma = gamma,
                    normalize_advantage = normalize_advantage,
                    normalize_return = normalize_return,
                    vf_loss_coeff = vf_loss_coeff,
                    entropy_coeff = entropy_coeff,
                    action2_entropy_coeff = action2_entropy_coeff,
                    action3_entropy_coeff = action3_entropy_coeff,
                )
                self.trainers_module[policy] = PreProcess_Module_Loss(
                    info_weight = info_weight
                )
            else:
                self.trainers[policy] = A2C(
                    discount_factor_gamma = gamma,
                    normalize_advantage = normalize_advantage,
                    normalize_return = normalize_return,
                    vf_loss_coeff = vf_loss_coeff,
                    entropy_coeff = entropy_coeff,
                    action2_entropy_coeff = action2_entropy_coeff,
                    action3_entropy_coeff = action3_entropy_coeff,
                )

                #print("mmm",self.trainers[policy].loss_coefficient_entropy)
            logging.info(f"Initializing the A2C trainer for policy {policy}")
        elif algorithm == "PPO":
            # Proximal Policy Optimization
            clip_param = self._get_config(["policy", policy, "clip_param"])
            self.trainers[policy] = PPO(
                discount_factor_gamma=gamma,
                clip_param=clip_param,
                normalize_advantage=normalize_advantage,
                normalize_return=normalize_return,
                vf_loss_coeff=vf_loss_coeff,
                entropy_coeff=entropy_coeff,
            )
            logging.info(f"Initializing the PPO trainer for policy {policy}")
        else:
            raise NotImplementedError

    def _initialize_policy_model(self, policy):
        policy_model_config = self._get_config(["policy", policy, "model"])
        if policy_model_config["type"] == "fully_connected":
            if 'policy_network' in policy_model_config: policy_network_ = policy_model_config["policy_network"]
            else: policy_network_ = []
            if 'critic_network' in policy_model_config: critic_network_ = policy_model_config["critic_network"]
            else: critic_network_ = []
            model = FullyConnected(
                self.cuda_envs,
                policy_model_config["fc_dims"],
                policy,
                self.policy_tag_to_agent_id_map,
                self.create_separate_placeholders_for_each_policy,
                self.obs_dim_corresponding_to_num_agents,
                policy_network = policy_network_,
                critic_network = critic_network_,
            ).to('cuda:0')
        elif policy_model_config["type"] == "custom_model":
            if 'rl_policy_network' in policy_model_config: policy_network_ = policy_model_config["rl_policy_network"]
            else: policy_network_ = []
            if 'rl_critic_network' in policy_model_config: critic_network_ = policy_model_config["rl_critic_network"]
            else: critic_network_ = []
            if "rnn_memory_dim"  in policy_model_config: memory_dim_ = policy_model_config["rnn_memory_dim"]
            else: memory_dim_ = 0
            if "rnn_layer" in policy_model_config: n_layers = policy_model_config["rnn_layer"]
            else: n_layers = 1
            if "rnn_layer_type" in policy_model_config: rnn_layer_type_ = policy_model_config["rnn_layer_type"]
            else: rnn_layer_type_ = 'RNN'

            model = CustomModelConnected(
                env = self.cuda_envs,
                policy = policy,
                policy_tag_to_agent_id_map = self.policy_tag_to_agent_id_map,
                fc_dims = [128, 32],
                create_separate_placeholders_for_each_policy = self.create_separate_placeholders_for_each_policy,
                obs_dim_corresponding_to_num_agents = self.obs_dim_corresponding_to_num_agents,
                training = self.training,
                model_type = policy_model_config["rnn_layer_type"],
                memory_dim = memory_dim_,
                policy_network = policy_network_,
                critic_network = critic_network_,
                n_layers = n_layers,
                episode_length = self.cuda_envs.episode_length,
                rnn_layer_type = rnn_layer_type_
            ).to('cuda:0')
        elif policy_model_config["type"] == "module_model" or policy_model_config["type"] == "test_model":
            if policy_model_config["type"] == "test_model":  from custom_envs.test_model import CustomModuleConnected
            else:  from custom_envs.module_connected import CustomModuleConnected
            policy_network_ = policy_model_config["rl_policy_network"]
            critic_network_ = policy_model_config["rl_critic_network"]
            if "module_network" in policy_model_config:
                print('"rl" need to be inside module_network，and "rl" mean rl input place')
                assert 'rl' in policy_model_config["module_network"]
                module_network_ = policy_model_config["module_network"]
            else: module_network_ = ['rl', 20]
            if "module_connect_dim" in policy_model_config:  module_connect_dim_ = policy_model_config["module_connect_dim"]
            else: module_connect_dim_ = 2
            if "rnn_layer_type" in policy_model_config: rnn_layer_type_ = policy_model_config["rnn_layer_type"]
            else: rnn_layer_type_ = 'RNN'
            if "rnn_layer" in policy_model_config: rnn_num_layers_ = policy_model_config["rnn_layer"]
            else: rnn_num_layers_ = 1
            if "rnn_memory_dim"  in policy_model_config: rnn_memory_dim_ = policy_model_config["rnn_memory_dim"]
            else: rnn_memory_dim_ = 0
            if "rnn_seq_len"  in policy_model_config: rnn_seq_len_ = policy_model_config["rnn_seq_len"]
            else: rnn_seq_len_ = 10
            if "spot_rnn_drop"  in policy_model_config: spot_rnn_drop_ = policy_model_config["spot_rnn_drop"]
            else: spot_rnn_drop_ = None
            if "preprocess_buffer_replace_size" in policy_model_config: buffer_replace_size_ = policy_model_config["preprocess_buffer_replace_size"]
            else: buffer_replace_size_ = 1
            if "concat_rnn" in policy_model_config: concat_rnn_ = policy_model_config["concat_rnn"]
            else: concat_rnn_ = False

            model = CustomModuleConnected(
                policy = policy,
                policy_tag_to_agent_id_map = self.policy_tag_to_agent_id_map,
                env = self.cuda_envs,
                policy_network = policy_network_,
                critic_network = critic_network_,
                module_network = module_network_,
                module_connect_dim = module_connect_dim_,
                rnn_layer_type = rnn_layer_type_,
                rnn_num_layers = rnn_num_layers_,
                rnn_memory_dim = rnn_memory_dim_,
                rnn_seq_len = rnn_seq_len_,
                rnn_drop = spot_rnn_drop_,
                create_separate_placeholders_for_each_policy = self.create_separate_placeholders_for_each_policy,
                obs_dim_corresponding_to_num_agents = self.obs_dim_corresponding_to_num_agents,
                training = self.training,
                episode_length = self.cuda_envs.episode_length,
                concat_rnn = concat_rnn_
            ).to('cuda:0')

            self.rnn_buffer = Buffer(
                buffer_large = 10, 
                device = 'cuda:0',
                seq_len = rnn_seq_len_,
                rnn_layer_type = rnn_layer_type_,
                buffer_replace_size = buffer_replace_size_,
                buffer_file = './custom_model/Spot_3D_NEW/buffer_data.json'
            )
        else:
            raise NotImplementedError
        self.models[policy] = model.to('cuda:0')

    def callback_custom_model_action(self, policy_name):
        action = {}
        action['x'] = self.models[policy_name].action_x_dataset
        action['y'] = self.models[policy_name].action_y_dataset
        action['turn'] = self.models[policy_name].action_turn_dataset
        speed = self.models[policy_name].speed
        acc_dis = self.models[policy_name].acc_distribute
        return action, speed, acc_dis

    def _get_config(self, args):
        assert isinstance(args, (tuple, list))
        config = self.config
        for arg in args:
            try:
                config = config[arg]
            except ValueError:
                logging.error("Missing configuration '{arg}'!")
        return config

    def train(self, reset_model_mentum = False):
        """
        Perform training.
        """
        # Ensure env is reset before the start of training, and done flags are False
        self.cuda_envs.reset_all_envs()
        self.reset_model_mentum = reset_model_mentum
        for iteration in range(self.num_iters):
            start_time = time.time()

            # Generate a batched rollout for every CUDA environment.
            self._generate_rollout_batch()

            # Train / update model parameters.
            metrics = self._update_model_params(iteration)

            self.perf_stats.iters = iteration + 1
            self.perf_stats.steps = self.perf_stats.iters * self.training_batch_size
            end_time = time.time()
            self.perf_stats.total_time += end_time - start_time

            # Log the training metrics
            self._log_metrics(metrics)

            # Save torch model
            self.save_model_checkpoint(iteration)
            

    def _generate_rollout_batch(self):
        """
        Generate an environment rollout batch.
        """
        # Code timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        X_copy_at_host = self.cuda_envs.cuda_data_manager.pull_data_from_device(name="display_reward")
        
        for batch_index in range(self.training_batch_size_per_env):

            # Evaluate policies to compute action probabilities
            start_event.record()
            
            probabilities = self._evaluate_policies(batch_index=batch_index)
            end_event.record()
            torch.cuda.synchronize()
            self.perf_stats.policy_eval_time += (
                start_event.elapsed_time(end_event) / 1000
            )

            # Sample actions using the computed probabilities
            # and push to the batch of actions
            start_event.record()
            self._sample_actions(probabilities, batch_index=batch_index)
            end_event.record()
            torch.cuda.synchronize()
            self.perf_stats.action_sample_time += (
                start_event.elapsed_time(end_event) / 1000
            )

            # Step through all the environments
            start_event.record()
            self.cuda_envs.step_all_envs()

            # Bookkeeping rewards and done flags
            _, done_flags = self._bookkeep_rewards_and_done_flags(batch_index)

            X_copy_at_host = self.cuda_envs.cuda_data_manager.pull_data_from_device(name="display_reward")
            

            # Reset all the environments that are in done state.
            if done_flags.any():
                self.cuda_envs.reset_only_done_envs()

            end_event.record()
            torch.cuda.synchronize()
            self.perf_stats.env_step_time += start_event.elapsed_time(end_event) / 1000
        self.host_display_reward_data = np.mean(X_copy_at_host, axis=0)
        del X_copy_at_host

    def _evaluate_policies(self, batch_index=0):
        """
        Perform the policy evaluation (forward pass through the models)
        and compute action probabilities
        """
        assert isinstance(batch_index, int)
        
        probabilities = {}
        for policy in self.policies:
            if self._get_config(["policy", policy, "model", "type"]) == "custom_model":
                probabilities[policy], _, m = self.models[policy](
                    batch_index=batch_index, 
                    batch_size=self.training_batch_size_per_env,
                )
            elif self._get_config(["policy", policy, "model", "type"]) == "module_model" or self._get_config(["policy", policy, "model", "type"]) == "test_model":
                probabilities[policy], _, memory, _ = self.models[policy](
                    batch_index=batch_index, 
                    batch_size=self.training_batch_size_per_env,
                )
            else:
                probabilities[policy], _ = self.models[policy](
                    batch_index=batch_index, batch_size=self.training_batch_size_per_env
                )
            

        # Combine probabilities across policies if there are multiple policies,
        # yet they share the same action placeholders.
        # The action sampler will then need to run just once on each action type.
        if (
            len(self.policies) > 1
            and not self.create_separate_placeholders_for_each_policy
        ):
            # Assert that all the probabilities are of the same length
            # in other words the number of action types for each policy
            # is the same.
            num_action_types = {}
            for policy in self.policies:
                num_action_types[policy] = len(probabilities[policy])
            assert all_equal(list(num_action_types.values()))

            # Initialize combined_probabilities.
            first_policy = list(probabilities.keys())[0]
            num_action_types = num_action_types[first_policy]

            first_action_idx = 0
            num_envs = len(probabilities[first_policy][first_action_idx])
            num_agents = self.cuda_envs.env.num_agents

            combined_probabilities = [None for _ in range(num_action_types)]
            for action_type in range(num_action_types):
                # action_dim = probabilities[first_policy][action_type].shape[-1]
                action_dim = len(probabilities[first_policy][first_action_idx][0][0])
                combined_probabilities[action_type] = torch.zeros(
                    (num_envs, num_agents, action_dim)
                ).to('cuda:0')
            
            # Combine the probabilities across policies
            for action_idx in range(num_action_types):
                for policy, prob_values in probabilities.items():

                    agent_to_id_mapping = self.policy_tag_to_agent_id_map[policy]
                    data = torch.tensor(prob_values[action_idx])
                    combined_probabilities[action_idx][
                        :, agent_to_id_mapping
                    ] = data.to('cuda:0')

            probabilities = {_COMBINED: combined_probabilities}

        return probabilities

    def _register_actions(self, action_space, suffix=""):
        if isinstance(action_space, Discrete):
            # Single action
            action_dim = [action_space.n]
        elif isinstance(action_space, MultiDiscrete):
            # Multiple actions
            action_dim = action_space.nvec
        else:
            raise NotImplementedError(
                "Only 'Discrete' or 'MultiDiscrete' type action spaces are supported!"
            )
        if len(action_dim) == 1:
            self.cuda_sample_controller.register_actions(
                self.cuda_envs.cuda_data_manager,
                action_name=_ACTIONS + suffix,
                num_actions=action_dim[0],
            )
        else:
            for action_idx, _ in enumerate(action_dim):
                self.cuda_sample_controller.register_actions(
                    self.cuda_envs.cuda_data_manager,
                    action_name=f"{_ACTIONS}_{action_idx}" + suffix,
                    num_actions=action_dim[action_idx],
                )

    def _sample_actions(self, probabilities, batch_index=0):
        """
        Sample action probabilities (and push the sampled actions to the device).
        """
        assert isinstance(batch_index, int)
        if self.create_separate_placeholders_for_each_policy:
            for policy in self.policies:
                suffix = f"_{policy}"
                self._sample_actions_helper(
                    probabilities[policy], batch_index, suffix=suffix
                )
        else:
            assert len(probabilities) == 1
            policy = list(probabilities.keys())[0]
            self._sample_actions_helper(probabilities[policy], batch_index)

    def _sample_actions_helper(self, probabilities, batch_index, suffix=""):

        num_action_types = len(probabilities)

        if num_action_types == 1:
            action_name = _ACTIONS + suffix
            self.cuda_sample_controller.sample(
                self.cuda_envs.cuda_data_manager, probabilities[0], action_name
            )
            # Push actions to the batch of actions
            actions = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                action_name
            )
            self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                name=f"{_ACTIONS}_batch" + suffix
            )[batch_index, :, :, num_action_types - 1] = actions

        else:
            for action_idx, probs in enumerate(probabilities):
                action_name = f"{_ACTIONS}_{action_idx}" + suffix
                self.cuda_sample_controller.sample(
                    self.cuda_envs.cuda_data_manager, probs, action_name
                )
                # Push (indexed) actions to 'actions' and 'actions_batch'
                actions = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    action_name
                )
                self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    name=_ACTIONS + suffix
                )[:, :, action_idx] = actions
                self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    name=f"{_ACTIONS}_batch" + suffix
                )[batch_index, :, :, action_idx] = actions

    def _bookkeep_rewards_and_done_flags(self, batch_index):
        """
        Push rewards and done flags to the corresponding batched versions.
        Also, update the episodic reward
        """
        assert isinstance(batch_index, int)
        # Push done flags to done_flags_batch
        done_flags = (
            self.cuda_envs.cuda_data_manager.data_on_device_via_torch("_done_") > 0
        )
        self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
            name=f"{_DONE_FLAGS}_batch"
        )[batch_index] = done_flags


        done_env_ids = done_flags.nonzero()

        # Push rewards to rewards_batch and update the episodic rewards
        if self.create_separate_placeholders_for_each_policy:
            for policy in self.policies:
                rewards = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    f"{_REWARDS}_{policy}"
                )
                self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    name=f"{_REWARDS}_batch_{policy}"
                )[batch_index] = rewards

                # Update the episodic rewards
                self._update_episodic_rewards(rewards, done_env_ids, policy)

        else:
            rewards = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                _REWARDS
            )
            self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                name=f"{_REWARDS}_batch"
            )[batch_index] = rewards

            # Update the episodic rewards
            # (sum of individual step rewards over an episode)
            for policy in self.policies:
                self._update_episodic_rewards(
                    rewards[:, self.policy_tag_to_agent_id_map[policy]],
                    done_env_ids,
                    policy,
                )

        return rewards, done_flags

    def _update_episodic_rewards(self, rewards, done_env_ids, policy):

        self.reward_running_sum[policy] += rewards.to('cuda:0')

        num_completed_episodes = len(done_env_ids)
        if num_completed_episodes > 0:
            # Update the episodic rewards
            self.episodic_reward_sum[policy] += torch.sum(
                self.reward_running_sum[policy][done_env_ids]
            ).to('cuda:0')
            self.num_completed_episodes[policy] += num_completed_episodes
            # Reset the reward running sum
            self.reward_running_sum[policy][done_env_ids] = 0


    def _update_model_params(self, iteration):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        # Flag for logging (which also happens after the last iteration)
        logging_flag = (
            iteration % self.config["saving"]["metrics_log_freq"] == 0
            or iteration == self.num_iters - 1
        )

        metrics_dict = {}
        loss_display = {}

        # Fetch the actions and rewards batches for all agents
        if not self.create_separate_placeholders_for_each_policy:
            all_actions_batch = (
                self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    f"{_ACTIONS}_batch"
                )
            )
            all_rewards_batch = (
                self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    f"{_REWARDS}_batch"
                )
            )
        done_flags_batch = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
            f"{_DONE_FLAGS}_batch"
        )

        # On the device, observations_batch, actions_batch,
        # rewards_batch are all shaped
        # (batch_size, num_envs, num_agents, *feature_dim).
        # done_flags_batch is shaped (batch_size, num_envs)
        # Perform training sequentially for each policy
        reg = {}
        for policy in self.policies_to_train:
            if self.create_separate_placeholders_for_each_policy:
                actions_batch = (
                    self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                        f"{_ACTIONS}_batch_{policy}"
                    )
                )
                rewards_batch = (
                    self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                        f"{_REWARDS}_batch_{policy}"
                    )
                )
            else:
                # Filter the actions and rewards only for the agents
                # corresponding to a particular policy
                agent_ids_for_policy = self.policy_tag_to_agent_id_map[policy]
                actions_batch = all_actions_batch[:, :, agent_ids_for_policy, :]
                rewards_batch = all_rewards_batch[:, :, agent_ids_for_policy]

            # Fetch the (processed) observations batch to pass through the model
            processed_obs_batch = (
                self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    f"{_PROCESSED_OBSERVATIONS}_batch_{policy}"
                )
            )

            self.models[policy].train() 

            # Policy evaluation for the entire batch
            if self._get_config(["policy", policy, "model", "type"]) == "custom_model":
                n_layers = self.models[policy].n_layers
                memory_dim = self.models[policy].memory_dim
                if n_layers <= 0:
                    processed_memory_batch = None
                elif self.models[policy].rnn_layer_type == 'LSTM':
                    h0_banch = []
                    c0_banch = []
                    for memory in self.models[policy].memory_banch:
                        h0_banch.append(memory[0])
                        c0_banch.append(memory[1])
                    stack_h = torch.stack(h0_banch)
                    stack_h = stack_h.permute(1, 0, 2, 3).contiguous()
                    processed_h0_batch = stack_h.reshape(n_layers, -1, memory_dim)
                    stack_c = torch.stack(c0_banch)
                    stack_c = stack_h.permute(1, 0, 2, 3).contiguous()
                    processed_c0_batch = stack_c.reshape(n_layers, -1, memory_dim)
                    processed_memory_batch = (processed_h0_batch, processed_c0_batch)
                else:
                    stack_memory = torch.stack(self.models[policy].memory_banch) # time_banch, layers, banch, data_dim
                    stack_memory = stack_memory.permute(1, 0, 2, 3).contiguous()
                    processed_memory_batch = stack_memory.reshape(n_layers, -1, memory_dim)
                probabilities_batch, value_functions_batch, _ = self.models[policy](
                    obs = processed_obs_batch,
                    memory = processed_memory_batch
                )
            elif self._get_config(["policy", policy, "model", "type"]) == "module_model" or self._get_config(["policy", policy, "model", "type"])== "test_model":
                n_layers = self.models[policy].n_layers
                memory_dim = self.models[policy].memory_dim
                if n_layers <= 0:  processed_memory_batch = None
                elif self.models[policy].rnn_layer_type == 'LSTM':
                    h0_banch = []
                    c0_banch = []
                    for memory in self.models[policy].memory_banch:
                        h0_banch.append(memory[0])
                        c0_banch.append(memory[1])
                    stack_h = torch.stack(h0_banch)
                    stack_h = stack_h.permute(1, 0, 2, 3).contiguous()
                    processed_h0_batch = stack_h.reshape(n_layers, -1, memory_dim)
                    stack_c = torch.stack(c0_banch)
                    stack_c = stack_h.permute(1, 0, 2, 3).contiguous()
                    processed_c0_batch = stack_c.reshape(n_layers, -1, memory_dim)
                    processed_memory_batch = (processed_h0_batch, processed_c0_batch)
                else:
                    stack_memory = torch.stack(self.models[policy].memory_banch) # time_banch, layers, banch, data_dim
                    stack_memory = stack_memory.permute(1, 0, 2, 3).contiguous()
                    processed_memory_batch = stack_memory.reshape(n_layers, -1, memory_dim)
                  
                data_name = f"{_PREPROCESS}_batch_{policy}"
                preprocess_real_batch = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(data_name)
                #ic(processed_obs_batch.size(),processed_memory_batch.size(), preprocess_real_batch.size())
                if self.rnn_buffer.obs_buffer is None: 
                    self.rnn_buffer.initialize(obs = processed_obs_batch , memory = processed_memory_batch, preprocess = preprocess_real_batch)
                else:  self.rnn_buffer.update(obs = processed_obs_batch , memory = processed_memory_batch, preprocess = preprocess_real_batch)
                if iteration == self.num_iters - 1: self.rnn_buffer.save_to_file()
                
                obs_, preprocess_, memory_ = self.rnn_buffer.get_data()
                #ic(processed_memory_batch.shape, memory_.shape)
                _, _, _, pre_process_output = self.models[policy](
                    obs = obs_,
                    memory = memory_,
                    training = True,
                )

                # pre process module update
                loss, display_loss = self.trainers_module[policy].compute_loss(target = preprocess_, input_data = pre_process_output)
                
                self.optimizers_module[policy].zero_grad()
                if self.reset_model_mentum:
                    for param in self.optimizers_module[policy].state.values():
                        if 'exp_avg' in param:  param['exp_avg'].zero_() 
                        if 'exp_avg_sq' in param:  param['exp_avg_sq'].zero_() 
                for feture in display_loss:
                    loss_display[policy + "_" + feture] = display_loss[feture]
                loss.backward()
                
                if self.clip_grad_norm[policy]:
                    nn.utils.clip_grad_norm_(
                        self.models[policy].parameters(), self.max_grad_norm[policy]
                    )
                self.optimizers_module[policy].step()

                probabilities_batch, value_functions_batch, _, _ = self.models[policy](
                    obs = processed_obs_batch,
                    memory = processed_memory_batch,
                    training = True,
                )
     
            else:
                probabilities_batch, value_functions_batch = self.models[policy](
                    obs = processed_obs_batch
                )
            

            # Loss and metrics computation
            loss, metrics = self.trainers[policy].compute_loss_and_metrics(
                self.current_timestep[policy],
                actions_batch,
                rewards_batch,
                done_flags_batch,
                probabilities_batch,
                value_functions_batch,
                perform_logging=logging_flag,
            )
            loss_display[policy + " loss"] = loss
            # Compute the gradient norm
            
            grad_norm = 0.0
            for param in list(
                filter(lambda p: p.grad is not None, self.models[policy].parameters())
            ):
                grad_norm += param.grad.data.norm(2).item()

            # Update the timestep and learning rate based on the schedule
            # self.optimizers "lr" setting is have set before
            self.current_timestep[policy] += self.training_batch_size
            lr = self.lr_schedules[policy].get_param_value(
                self.current_timestep[policy]
            )
            #for param_group in self.optimizers[policy].param_groups:
            #    if 'params' in param_group and param_group['params'] == self.rl_params:
            #        param_group['lr'] = lr
   
            # Loss backpropagation and optimization step
            self.optimizers[policy].zero_grad()
            if self.reset_model_mentum:
                for param in self.optimizers[policy].state.values():
                    if 'exp_avg' in param:  # 'exp_avg' 是一階動量
                        param['exp_avg'].zero_()  # 清除一階動量
                    if 'exp_avg_sq' in param:  # 'exp_avg_sq' 是二階動量
                        param['exp_avg_sq'].zero_()  # 清除二階動量
                self.reset_model_mentum = False
            
            loss.backward()
            if self.clip_grad_norm[policy]:
                nn.utils.clip_grad_norm_(
                    self.models[policy].parameters(), self.max_grad_norm[policy]
                )

            self.optimizers[policy].step()

            # Logging
            if logging_flag:
                metrics_dict[policy] = metrics
                # Update the metrics dictionary
                metrics_dict[policy].update(
                    {
                        "Current timestep": self.current_timestep[policy],
                        "Gradient norm": grad_norm,
                        "Learning rate": lr,
                        "train grid length": float(self.cuda_envs.env.grid_length),
                        "Mean episodic reward": self.episodic_reward_sum[policy].item()
                        / (self.num_completed_episodes[policy] + _EPSILON),
                    }
                )

                if self.config["policy"][policy]["to_train"] == True:
                    reg[policy + " reward"] =  self.episodic_reward_sum[policy].item() / (self.num_completed_episodes[policy] + _EPSILON)
                    if self.perf_stats.iters < 5:
                        reg[policy + " long time reward"] =  self.episodic_reward_sum[policy].item() / (self.num_completed_episodes[policy] + _EPSILON)
                # Reset sum and counter
                self.episodic_reward_sum[policy] = (
                    torch.tensor(0).type(torch.float32)
                ).to('cuda:0')
                self.num_completed_episodes[policy] = 0

        
        for key in self.config["policy"]:
            if self.config["policy"][key]["to_train"] == True and self.config["name"] == "spot_3d":
                for feture in loss_display:
                    #ic(feture)
                    reg[feture] = loss_display[feture]
                #reg[ key + " loss"] = loss_display[key]
                reg[ key + " loss coefficient entropy"] = self.trainers[key].loss_coefficient_entropy
                goal_dynamic_obstacle_reward = 0
                goal_obstacle_reward = 0
                goal_spot_reward = 0
                goal_step_reward = 0
                goal_move_reward = 0
                goal_floor_reward = 0
                spot_dynamic_obstacle_reward = 0
                spot_obstacle_reward = 0
                spot_goal_reward = 0 
                spot_dynamic_reward = 0
                spot_floor_reward = 0
                spot_hit_probability = 0
                goal_remain_probability = 0
                spot_born_died = 0
                for all_data in range(len(self.host_display_reward_data)):
                    if all_data % 14 == 0:
                        goal_dynamic_obstacle_reward += self.host_display_reward_data[all_data]
                    elif all_data % 14 == 1:
                        goal_obstacle_reward += self.host_display_reward_data[all_data]
                    elif all_data % 14 == 2:
                        goal_spot_reward += self.host_display_reward_data[all_data]
                    elif all_data % 14 == 3:
                        goal_step_reward += self.host_display_reward_data[all_data]
                    elif all_data % 14 == 4:
                        goal_move_reward += self.host_display_reward_data[all_data]
                    elif all_data % 14 == 5:
                        goal_floor_reward += self.host_display_reward_data[all_data]
                    elif all_data % 14 == 6:
                        spot_dynamic_obstacle_reward += self.host_display_reward_data[all_data]
                    elif all_data % 14 == 7:
                        spot_obstacle_reward += self.host_display_reward_data[all_data]
                    elif all_data % 14 == 8:
                        spot_goal_reward += self.host_display_reward_data[all_data]
                    elif all_data % 14 == 9:
                        spot_dynamic_reward += self.host_display_reward_data[all_data]
                    elif all_data % 14 == 10:
                        spot_floor_reward += self.host_display_reward_data[all_data]
                    elif all_data % 14 == 11:
                        spot_hit_probability = self.host_display_reward_data[all_data]
                    elif all_data % 14 == 12:
                        goal_remain_probability = self.host_display_reward_data[all_data]
                    elif all_data % 14 == 13:
                        spot_born_died += self.host_display_reward_data[all_data]
                reg["goal dynamic obstacle reward"] = goal_dynamic_obstacle_reward / self.config["env"]["num_goals"]
                reg["goal static obstacle reward"] = goal_obstacle_reward / self.config["env"]["num_goals"]
                reg["goal spot reward"] = goal_spot_reward / self.config["env"]["num_goals"]
                reg["goal step reward"] = goal_step_reward / self.config["env"]["num_goals"]
                reg["goal move reward"] = goal_move_reward / self.config["env"]["num_goals"]
                reg["goal floor reward"] = goal_floor_reward / self.config["env"]["num_goals"]
                reg["spot dynamic obstacle reward"] = spot_dynamic_obstacle_reward / self.config["env"]["num_spot"]
                reg["spot static obstacle reward"] = spot_obstacle_reward / self.config["env"]["num_spot"]
                reg["spot goal reward"] = spot_goal_reward / self.config["env"]["num_spot"]
                reg["spot dynamic reward"] = spot_dynamic_reward / self.config["env"]["num_spot"]
                reg["spot floor reward"] = spot_floor_reward / self.config["env"]["num_spot"]
                reg["spot hit probability"] = spot_hit_probability / self.config["env"]["num_spot"]
                reg["goal remain probability"] = 1 - goal_remain_probability / self.config["env"]["num_goals"]
                reg["spot dies at birth probability"] = spot_born_died / self.config["env"]["num_spot"]
                if key == 'spot':
                    reg["spot time goal remain probability"] = 1 - goal_remain_probability / self.config["env"]["num_goals"]
                    reg["spot time survive expected value"] = (self.config["env"]["num_goals"] * (1 - reg["goal remain probability"]) * self.config["env"]["spot_reward_get_goal"] + (reg["spot hit probability"] - reg["spot dies at birth probability"] + 0.0001) * self.config["env"]["spot_penalty_hit"]) / self.config["env"]["num_spot"]
                    reg["spot time action expected value"] = (1 - reg["spot dies at birth probability"]) * self.config["env"]["episode_length"] * self.config["env"]["spot_cost_operate"]
                elif key == 'goal':
                    reg["goal time goal remain probability"] = 1 - goal_remain_probability / self.config["env"]["num_goals"]
                    reg["goal time survive expected value"] = reg["goal move reward"] + reg["goal step reward"] + reg["goal dynamic obstacle reward"] + reg["goal spot reward"]
                    reg["goal time move expected value"] = reg["goal move reward"] + reg["goal static obstacle reward"] + reg["goal floor reward"] * 0.1 + reg["goal spot reward"] * 0.7
                    reg["goal time stair expected value(Estimate)"] = reg["goal floor reward"] + reg["goal static obstacle reward"] * 0.6 + reg["goal spot reward"] * 0.5
            elif self.config["policy"][key]["to_train"] == True and self.config["name"] == "new_car":
                reg[ key + " loss"] = loss_display[key]
                reg[ key + " loss coefficient entropy"] = self.trainers[key].loss_coefficient_entropy
                goal_dynamic_obstacle_reward = 0
                goal_obstacle_reward = 0
                goal_car_reward = 0
                goal_step_reward = 0
                goal_move_reward = 0
                goal_floor_reward = 0
                car_dynamic_obstacle_reward = 0
                car_obstacle_reward = 0
                car_goal_reward = 0 
                car_dynamic_reward = 0
                car_floor_reward = 0
                car_hit_probability = 0
                goal_remain_probability = 0
                car_born_died = 0
                for all_data in range(len(self.host_display_reward_data)):
                    if all_data % 14 == 0:
                        goal_dynamic_obstacle_reward += self.host_display_reward_data[all_data]
                    elif all_data % 14 == 1:
                        goal_obstacle_reward += self.host_display_reward_data[all_data]
                    elif all_data % 14 == 2:
                        goal_car_reward += self.host_display_reward_data[all_data]
                    elif all_data % 14 == 3:
                        goal_step_reward += self.host_display_reward_data[all_data]
                    elif all_data % 14 == 4:
                        goal_move_reward += self.host_display_reward_data[all_data]
                    elif all_data % 14 == 5:
                        goal_floor_reward += self.host_display_reward_data[all_data]
                    elif all_data % 14 == 6:
                        car_dynamic_obstacle_reward += self.host_display_reward_data[all_data]
                    elif all_data % 14 == 7:
                        car_obstacle_reward += self.host_display_reward_data[all_data]
                    elif all_data % 14 == 8:
                        car_goal_reward += self.host_display_reward_data[all_data]
                    elif all_data % 14 == 9:
                        car_dynamic_reward += self.host_display_reward_data[all_data]
                    elif all_data % 14 == 10:
                        car_floor_reward += self.host_display_reward_data[all_data]
                    elif all_data % 14 == 11:
                        car_hit_probability = self.host_display_reward_data[all_data]
                    elif all_data % 14 == 12:
                        goal_remain_probability = self.host_display_reward_data[all_data]
                    elif all_data % 14 == 13:
                        car_born_died += self.host_display_reward_data[all_data]
                reg["goal dynamic obstacle reward"] = goal_dynamic_obstacle_reward / self.config["env"]["num_goals"]
                reg["goal static obstacle reward"] = goal_obstacle_reward / self.config["env"]["num_goals"]
                reg["goal car reward"] = goal_car_reward / self.config["env"]["num_goals"]
                reg["goal step reward"] = goal_step_reward / self.config["env"]["num_goals"]
                reg["goal move reward"] = goal_move_reward / self.config["env"]["num_goals"]
                reg["goal floor reward"] = goal_floor_reward / self.config["env"]["num_goals"]
                reg["car dynamic obstacle reward"] = car_dynamic_obstacle_reward / self.config["env"]["num_car"]
                reg["car static obstacle reward"] = car_obstacle_reward / self.config["env"]["num_car"]
                reg["car goal reward"] = car_goal_reward / self.config["env"]["num_car"]
                reg["car dynamic reward"] = car_dynamic_reward / self.config["env"]["num_car"]
                reg["car floor reward"] = car_floor_reward / self.config["env"]["num_car"]
                reg["car hit probability"] = car_hit_probability / self.config["env"]["num_car"]
                reg["goal remain probability"] = 1 - goal_remain_probability / self.config["env"]["num_goals"]
                reg["car dies at birth probability"] = car_born_died / self.config["env"]["num_car"]
                if key == 'car':
                    reg["car time goal remain probability"] = 1 - goal_remain_probability / self.config["env"]["num_goals"]
                elif key == 'goal':
                    reg["goal time goal remain probability"] = 1 - goal_remain_probability / self.config["env"]["num_goals"]
                    
        if self.wandb:
            wandb.log(reg)

        end_event.record()
        torch.cuda.synchronize()
        self.perf_stats.training_time += start_event.elapsed_time(end_event) / 1000
        return metrics_dict

    def _log_metrics(self, metrics):
        # Log the metrics if it is not empty
        if len(metrics) > 0:
            perf_stats = self.perf_stats.get_perf_stats()

            if self.verbose:
                print("\n")
                print("=" * 40)
                print(f"Device: {self.device_id}")
                print(
                    f"{'Iterations Completed':40}: "
                    f"{self.perf_stats.iters} / {self.num_iters}"
                )
                self.perf_stats.pretty_print(perf_stats)
                self.metrics.pretty_print(metrics)
                print("=" * 40, "\n")

            # Log metrics and performance stats
            logs = {"Iterations Completed": self.perf_stats.iters}
            logs.update(metrics)
            logs.update({"Perf. Stats": perf_stats})

            if self.num_devices > 1:
                fn = f"results_device_{self.device_id}.json"
            else:
                fn = "results.json"
            results_filename = os.path.join(self.save_dir, fn)
            if self.verbose:
                verbose_print(
                    f"Saving the results to the file '{results_filename}'",
                    self.device_id,
                )
            with open(results_filename, "a+", encoding="utf8") as fp:
                json.dump(logs, fp)
                fp.write("\n")
            self._heartbeat_check_printout(metrics)

    def _heartbeat_check_printout(self, metrics, check=False):
        if check is False:
            return

        if self.num_devices > 1:
            heartbeat_print = (
                "Iterations Completed: "
                + f"{self.perf_stats.iters} / {self.num_iters}: \n"
            )
            for policy in self.policies:
                
                heartbeat_print += (
                    f"policy '{policy}' - Mean episodic reward: "
                    f"{metrics[policy]['Mean episodic reward']} \n"
                )
                
                
            verbose_print(heartbeat_print, self.device_id)

    def load_model_checkpoint(self, ckpts_dict=None):
        """
        Load the model parameters if a checkpoint path is specified.
        """

        if ckpts_dict is None:
            logging.info(
                "Loading trainer model checkpoints from the run configuration."
            )
            for policy in self.policies:
                ckpt_filepath = self.config["policy"][policy]["model"][
                    "model_ckpt_filepath"
                ]
                self._load_model_checkpoint_helper(policy, ckpt_filepath)
        else:
            assert isinstance(ckpts_dict, dict)
            if self.verbose:
                verbose_print(
                    "Loading the provided trainer model checkpoints.", self.device_id
                )
            for policy, ckpt_filepath in ckpts_dict.items():
                assert policy in self.policies
                print("ckpt_filepath = ",ckpt_filepath)
                self._load_model_checkpoint_helper(policy, ckpt_filepath)
                

    def remove_old_weight(self ,ckpts_dict=None):
        if ckpts_dict is not None:
            for policy, ckpt_filepath in ckpts_dict.items():
                if self.config["saving"]["at_last_store_freq"] > self.config["saving"]["model_params_save_freq"] and self.wandb == True:
                    if self.config["saving"]["current_step"] % self.config["saving"]["at_last_store_freq"] != 0:
                        print("remove ",ckpt_filepath)
                        os.remove(ckpt_filepath)

    def _load_model_checkpoint_helper(self, policy, ckpt_filepath):
        if ckpt_filepath != "":
            assert os.path.isfile(ckpt_filepath), "Invalid model checkpoint path!"
            if self.verbose:
                verbose_print(
                    f"Loading the '{policy}' torch model "
                    f"from the previously saved checkpoint: '{ckpt_filepath}'",
                    self.device_id,
                )
            self.models[policy].load_state_dict(torch.load(ckpt_filepath))

            # Update the current timestep using the saved checkpoint filename
            timestep = int(ckpt_filepath.split(".state_dict")[0].split("_")[-1])
            if self.verbose:
                verbose_print(
                    f"Updating the timestep for the '{policy}' model to {timestep}.",
                    self.device_id,
                )
            self.current_timestep[policy] = timestep

    def save_model_checkpoint(self, iteration=0):
        """
        Save the model parameters
        """
        # If multiple devices, save the synced-up model only for device id 0
        if self.device_id == 0:
            # Save model checkpoints if specified (and also for the last iteration)
            if (
                (iteration % self.config["saving"]["model_params_save_freq"] == 0
                or iteration == self.num_iters - 1) and iteration > 0
            ):
                for policy, model in self.models.items():
                    filepath = os.path.join(
                        self.save_dir,
                        f"{policy}_{self.current_timestep[policy]}.state_dict",
                    )
                    if self.verbose:
                        verbose_print(
                            f"Saving the '{policy}' torch model "
                            f"to the file: '{filepath}'.",
                            self.device_id,
                        )

                    torch.save(model.state_dict(), filepath)


    def graceful_close(self):
        # Delete the sample controller to clear
        # the random seeds defined in the CUDA memory heap.
        # Warning: Not closing gracefully could lead to a memory leak.
        del self.cuda_sample_controller
        del self.models
        if self.verbose:
            verbose_print("Trainer exits gracefully", self.device_id)

    def fetch_episode_states(
        self,
        list_of_states=None,  # list of states (data array names) to fetch
        env_id=0,  # environment id to fetch the states from
    ):
        """
        Step through env and fetch the desired states (data arrays on the GPU)
        for an entire episode. The trained models will be used for evaluation.
        """
        assert 0 <= env_id < self.num_envs
        assert list_of_states is not None
        assert isinstance(list_of_states, list)
        assert len(list_of_states) > 0

        logging.info(f"Fetching the episode states: {list_of_states} from the GPU.")
        # Ensure env is reset before the start of training, and done flags are False
        self.cuda_envs.reset_all_envs()
        env = self.cuda_envs.env

        episode_states = {}

        for state in list_of_states:
            assert self.cuda_envs.cuda_data_manager.is_data_on_device(
                state
            ), f"{state} is not a valid array name on the GPU!"
            # Note: Discard the first dimension, which is the env dimension
            array_shape = self.cuda_envs.cuda_data_manager.get_shape(state)[1:]

            # Initialize the episode states
            episode_states[state] = np.nan * np.stack(
                [np.ones(array_shape) for _ in range(env.episode_length + 1)]
            )

        for policy in self.policies:
            self.models[policy].eval()

        for timestep in range(env.episode_length):
            # Update the episode states
            for state in list_of_states:
                episode_states[state][
                    timestep
                ] = self.cuda_envs.cuda_data_manager.pull_data_from_device(state)[
                    env_id
                ]
            # Evaluate policies to compute action probabilities
            probabilities = self._evaluate_policies()

            # Sample actions
            self._sample_actions(probabilities)

            # Step through all the environments
            self.cuda_envs.step_all_envs()

            # Fetch the states when episode is complete
            if env.cuda_data_manager.pull_data_from_device("_done_")[env_id]:
                for state in list_of_states:
                    episode_states[state][
                        timestep + 1
                    ] = self.cuda_envs.cuda_data_manager.pull_data_from_device(state)[
                        env_id
                    ]
                break


        return episode_states
    
    def fetch_states_init(
        self,
        list_of_states = None,  # list of states (data array names) to fetch
        env_id=0,  # environment id to fetch the states from
    ):
        """
        Step through env and fetch the desired states (data arrays on the GPU)
        for an entire episode. The trained models will be used for evaluation.
        origin code is from coustom_trainer fetch_episode_states.
        because of the fetch_episode_states sample full episode action,
        this function can test other agent by human move result.
        """
        assert 0 <= env_id < self.num_envs
        assert list_of_states is not None
        assert isinstance(list_of_states, list)
        assert len(list_of_states) > 0

        logging.info(f"Fetching the episode states: {list_of_states} from the GPU.")
        # Ensure env is reset before the start of training, and done flags are False
        self.cuda_envs.reset_all_envs()


        episode_states = {}

        for state in list_of_states:
            assert self.cuda_envs.cuda_data_manager.is_data_on_device(
                state
            ), f"{state} is not a valid array name on the GPU!"
            # Note: Discard the first dimension, which is the env dimension
            array_shape = self.cuda_envs.cuda_data_manager.get_shape(state)[1:]

            # Initialize the episode states
            episode_states[state] = np.nan * np.stack(
                [np.ones(array_shape) for _ in range(1000)]
            )
        return episode_states

    def fetch_states_continue(
        self,
        list_of_states=None,  # list of states (data array names) to fetch
        env_id = 0,  # environment id to fetch the states from
        timestep = 0,
        episode_states = {}
    ):
        for state in list_of_states:
            episode_states[state][
                timestep
            ] = self.cuda_envs.cuda_data_manager.pull_data_from_device(state)[
                env_id
            ]
        # Evaluate policies to compute action probabilities
        probabilities = self._evaluate_policies()
        env = self.cuda_envs.env
        # Sample actions
        self._sample_actions(probabilities)

        # Step through all the environments
        self.cuda_envs.step_all_envs()
        if timestep + 1 > 1000 - 1:
            timestep = 0
        # Fetch the states when episode is complete
        if env.cuda_data_manager.pull_data_from_device("_done_")[env_id]:
            for state in list_of_states:
                episode_states[state][
                    timestep + 1
                ] = self.cuda_envs.cuda_data_manager.pull_data_from_device(state)[
                    env_id
                ]

        return episode_states


class PerfStats:
    """
    Performance stats that will be included in rollout metrics.
    """

    def __init__(self):
        self.iters = 0
        self.steps = 0
        self.policy_eval_time = 0.0
        self.action_sample_time = 0.0
        self.env_step_time = 0.0
        self.training_time = 0.0
        self.total_time = 0.0

    def get_perf_stats(self):
        return {
            "Mean policy eval time per iter (ms)": self.policy_eval_time
            * 1000
            / self.iters,
            "Mean action sample time per iter (ms)": self.action_sample_time
            * 1000
            / self.iters,
            "Mean env. step time per iter (ms)": self.env_step_time * 1000 / self.iters,
            "Mean training time per iter (ms)": self.training_time * 1000 / self.iters,
            "Mean total time per iter (ms)": self.total_time * 1000 / self.iters,
            "Mean steps per sec (policy eval)": self.steps / self.policy_eval_time,
            "Mean steps per sec (action sample)": self.steps / self.action_sample_time,
            "Mean steps per sec (env. step)": self.steps / self.env_step_time,
            "Mean steps per sec (training time)": self.steps / self.training_time,
            "Mean steps per sec (total)": self.steps / self.total_time,
        }

    @staticmethod
    def pretty_print(stats):
        print("=" * 40)
        print("Speed performance stats")
        print("=" * 40)
        for k, v in stats.items():
            print(f"{k:40}: {v:10.2f}")


class Metrics:
    """
    Metrics class to log and print the key metrics
    """

    def __init__(self):
        pass

    def pretty_print(self, metrics):
        assert metrics is not None
        assert isinstance(metrics, dict)

        for policy in metrics:
            print("=" * 40)
            print(f"Metrics for policy '{policy}'")
            print("=" * 40)
            for k, v in metrics[policy].items():
                print(f"{k:40}: {v:10.5f}")
