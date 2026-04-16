# `train_rnn` / `train_car` 每次 commit 改了什麼

產出時間: 2026-04-16  
資料來源: 本地 git patch，自動抽取 relevant commit 的作者、檔案、增減行數、關鍵參數/函式變化。

說明:
- 這份不是只看 commit message，而是直接看 patch。
- 每一筆都列出作者、日期、改動檔案、`+/-` 行數、與 patch 中最能代表變更意圖的片段。
- 因為原始 commit message 大多失真，以下內容以實際 diff 為準。

## RNN 線
### `2024-09-26 f545d5d soc123`
- 改名/搬移 `train_stable_rnn.py` (similarity 098%)；關鍵片段: `class Spot_Env():` ; `def __init__(self):` ; `self.Training = True` ; `self.START_PHASE = 1` ; `self.RL_env_name = "Spot_3D_NEW"`

### `2024-09-28 4c98b06 gpu`
- 修改 `train_stable_rnn.py` `+2/-2`；關鍵片段: `self.results_dir_ = "RNN_noDetach_h0_grad_norm"` ; `self.results_dir_ = "RNN_num_layer2"`

### `2024-09-28 38119f8 soc123`
- 修改 `train_stable_rnn.py` `+1/-1`；關鍵片段: `self.results_dir_ = "RNN_noDetach_h0_grad_norm"` ; `self.results_dir_ = "RNN_detach_init_memory_at_collect"`

### `2024-09-28 357937d soc123`
- 修改 `train_stable_rnn.py` `+5/-1`；關鍵片段: `self.results_dir_ = "RNN_num_layer2"`

### `2024-09-30 53a7af3 soc123`
- 修改 `train_stable_rnn.py` `+1/-5`；關鍵片段: `self.results_dir_ = "RNN_detach_init_memory_at_collect"` ; `self.results_dir_ = "RNN_num_layer2"` ; `self.results_dir_ = "RNN_seq5_rnn_loss_01"`

### `2024-09-30 35ec222 gpu`
- 修改 `train_stable_rnn.py` `+1/-5`；關鍵片段: `self.results_dir_ = "RNN_detach_init_memory_at_collect"` ; `self.results_dir_ = "RNN_num_layer2"` ; `self.results_dir_ = "RNN_merge_seq5"`

### `2024-09-30 7ba848b gpu`
- 修改 `train_stable_rnn.py` `+4/-0`；關鍵片段: `self.results_dir_ = "RNN_seq5_rnn_loss_01"`

### `2024-10-03 33fe687 soc123`
- 修改 `train_stable_rnn.py` `+3/-3`；關鍵片段: `self.results_dir_ = "RNN_seq5_rnn_loss_01"` ; `self.results_dir_ = "rnn_layer_short"`

### `2024-10-03 2e3ae93 gpu`
- 修改 `train_stable_rnn.py` `+1/-5`；關鍵片段: `self.results_dir_ = "RNN_merge_seq5"` ; `self.results_dir_ = "RNN_seq5_rnn_loss_01"` ; `self.results_dir_ = "RNN_all_rnn"`

### `2024-10-03 c470e97 gpu`
- 修改 `train_stable_rnn.py` `+6/-2`；關鍵片段: `self.results_dir_ = "rnn_layer_short"`

### `2024-10-03 6cc9de1 gpu`
- 新增 `train_rnn_module.py` `+734/-0`；關鍵片段: `class Spot_Env():` ; `def __init__(self):` ; `self.Training = True` ; `self.START_PHASE = 1` ; `self.RL_env_name = "Spot_3D_NEW"`
- 新增 `train_rnn_module_compair.py` `+734/-0`；關鍵片段: `class Spot_Env():` ; `def __init__(self):` ; `self.Training = True` ; `self.START_PHASE = 1` ; `self.RL_env_name = "Spot_3D_NEW"`

### `2024-10-09 9253c46 soc123`
- 修改 `train_rnn_module.py` `+23/-14`；關鍵片段: `self.results_dir_ = "rnn_module"` ; `self.results_dir_ = "rnn_module_obs_all_static"` ; `self.notes = 'dont git push'` ; `self.notes = None#'dont git push.'` ; `self.train_num_env = 180  # 300`
- 修改 `train_stable_rnn.py` `+0/-4`；關鍵片段: `self.results_dir_ = "RNN_all_rnn"`

### `2024-10-09 d415642 gpu`
- 修改 `train_rnn_module_compair.py` `+23/-14`；關鍵片段: `self.START_PHASE = 2` ; `self.START_PHASE = 1` ; `self.results_dir_ = "no_rnn_module_compair"` ; `self.results_dir_ = "rnn_module_compair_nomodule"` ; `self.notes = None# 'rnn all param detach, just input dynamic info give state'`
- 新增 `train_rnn_module_compair_no_sucess.py` `+734/-0`；關鍵片段: `class Spot_Env():` ; `def __init__(self):` ; `self.Training = True` ; `self.START_PHASE = 2` ; `self.RL_env_name = "Spot_3D_NEW"`

### `2024-10-09 29c44e4 gpu`
- 修改 `train_rnn_module.py` `+3/-3`；關鍵片段: `model_path = "./custom_envs/fully_connected.py"` ; `model_path = "./custom_envs/module_connected.py"` ; `python_code_path = "custom_envs/spot_3dIn/spot_3dIn.py"` ; `cuda_code_path = "custom_envs/spot_3dIn/spot_3dIn_step.cu"` ; `python_code_path = "custom_envs/spot_3dmodule/spot_3dmodule.py"`
- 修改 `train_rnn_module_compair.py` `+6/-5`；關鍵片段: `self.notes = None#'dont git push.'` ; `self.notes = 'reset_model_mentum, 目標復刻both_state_have_dyob的數據，train goal spot訓練數據不一樣沒關係'` ; `model_path = "./custom_envs/fully_connected.py"` ; `model_path = "./custom_envs/module_connected.py"` ; `python_code_path = "custom_envs/spot_3dIn/spot_3dIn.py"`

### `2024-10-09 b693a21 soc123`
- 修改 `train_rnn_module.py` `+4/-3`；關鍵片段: `self.notes = None#'dont git push.'` ; `self.notes = 'reset_model_mentum'` ; `self.phase2_spot_batch = 100000` ; `self.phase2_spot_batch = 130000`
- 修改 `train_rnn_module_compair.py` `+1/-1`；關鍵片段: `self.phase2_spot_batch = 100000` ; `self.phase2_spot_batch = 130000`

### `2024-10-11 bb9e640 soc123`
- 修改 `train_rnn_module.py` `+4/-4`；關鍵片段: `self.results_dir_ = "rnn_module_obs_all_static"` ; `self.results_dir_ = "rnn_module_weight_change"` ; `preprocess_info_weight = [0.5, 0.5, 0.2, 0.2, 1, 1]` ; `preprocess_info_weight = [0.1, 0.1, 0.03, 0.03, 1, 1]`

### `2024-10-11 d0d7e61 soc123`
- 修改 `train_rnn_module.py` `+1/-1`；關鍵片段: `self.results_dir_ = "rnn_module_weight_change"` ; `self.results_dir_ = "rnn_module_obs_state_change"`

### `2024-10-11 23921dd gpu`
- 修改 `train_rnn_module.py` `+3/-1`；關鍵片段: `self.results_dir_ = "rnn_module_obs_state_change"` ; `self.results_dir_ = "rnn_compair_dropout01"` ; `spot_rnn_drop = self.wandb_config['spot_rnn_drop'],`

### `2024-10-11 d55a24a soc123`
- 修改 `train_rnn_module.py` `+1/-3`；關鍵片段: `self.results_dir_ = "rnn_compair_dropout01"` ; `self.results_dir_ = "rnn_module_obs_state_change"` ; `spot_rnn_drop = self.wandb_config['spot_rnn_drop'],`

### `2024-10-11 4937b69 gpu`
- 修改 `train_rnn_module.py` `+6/-7`；關鍵片段: `self.train_goal_fix_spot_model = False`

### `2024-10-11 a58b086 gpu`
- 修改 `train_rnn_module.py` `+1/-3`

### `2024-10-11 f44ecec gpu`
- 修改 `train_rnn_module.py` `+3/-1`；關鍵片段: `self.results_dir_ = "rnn_module_obs_state_change"` ; `self.results_dir_ = "rnn_compair_dropout01"` ; `spot_rnn_drop = self.wandb_config['spot_rnn_drop'],`

### `2024-10-12 a6d1eae gpu`
- 修改 `train_rnn_module.py` `+3/-3`；關鍵片段: `self.results_dir_ = "rnn_compair_dropout01"` ; `self.results_dir_ = "rnn_loss_log"` ; `preprocess_info_weight = [0.1, 0.1, 0.03, 0.03, 1, 1]` ; `preprocess_info_weight = [1, 1, 0.7, 0.7, 0.8, 0.8]`

### `2024-10-14 34f7af4 gpu`
- 修改 `train_rnn_module.py` `+2/-2`；關鍵片段: `self.START_PHASE = 1` ; `self.START_PHASE = 3` ; `self.notes = 'reset_model_mentum'` ; `self.notes = None  #'reset_model_mentum'`

### `2024-10-15 9c162a4 gpu`
- 修改 `train_rnn_module.py` `+8/-7`；關鍵片段: `self.START_PHASE = 3` ; `self.START_PHASE = 1` ; `self.results_dir_ = "rnn_loss_log"` ; `self.results_dir_ = "rnn_model_rl_input_del_policy_loss_04"` ; `vf_loss_coeff = self.wandb_config['spot_vf_loss_coeff'],   # Critic loos default = 0.01`
- 新增 `train_rnn_module_stable.py` `+746/-0`；關鍵片段: `class Spot_Env():` ; `def __init__(self):` ; `self.Training = True` ; `self.START_PHASE = 1` ; `self.RL_env_name = "Spot_3D_NEW"`

### `2024-10-18 e1de4f2 gpu`
- 修改 `train_rnn_module.py` `+3/-3`；關鍵片段: `self.results_dir_ = "rnn_model_rl_input_del_policy_loss_04"` ; `self.results_dir_ = "rnn_model_policy_loss04_stair_state_diff"` ; `self.train_num_env = 300  # 300` ; `self.train_num_env = 180  # 300`
- 修改 `train_rnn_module_stable.py` `+1/-1`

### `2024-10-21 adff7eb gpu`
- 修改 `train_rnn_module.py` `+15/-9`；關鍵片段: `self.results_dir_ = "rnn_model_policy_loss04_stair_state_diff"` ; `self.results_dir_ = "module_test_mlp"` ; `self.notes = None  #'reset_model_mentum'` ; `self.rl_env_versioin = 'RL_Env_v20'` ; `self.notes = None  #'module_loss_diffent'`

### `2024-10-21 329405c gpu`
- 修改 `train_rnn_module.py` `+5/-6`；關鍵片段: `self.results_dir_ = "module_test_mlp"` ; `self.results_dir_ = "rnn_huge_input"`

### `2024-10-28 f8ca040 gpu`
- 修改 `train_rnn_module.py` `+7/-5`；關鍵片段: `self.results_dir_ = "rnn_huge_input"` ; `self.results_dir_ = "module_rnn_batch_256"` ; `self.notes = None  #'module_loss_diffent'` ; `self.notes = None  #'concat_relu_state_goal_dt_xy'` ; `preprocess_info_weight = [1.0, 1.0, 0.7, 0.7, 10.0, 10.0, 0.1]`

### `2024-10-28 cf9a6e5 gpu`
- 修改 `train_rnn_module.py` `+3/-3`；關鍵片段: `self.START_PHASE = 1` ; `self.START_PHASE = 5` ; `self.phase7_spot_batch = 190000` ; `self.phase7_goal_batch = 120000` ; `self.phase7_spot_batch = 200000`

### `2024-10-30 2f20dfa gpu`
- 修改 `train_rnn_module.py` `+6/-5`；關鍵片段: `self.START_PHASE = 5` ; `self.START_PHASE = 6` ; `self.phase5_spot_batch = 350000` ; `self.phase5_spot_batch = 400000` ; `ckpt_filepath = './custom_model/Spot_3D_NEW/buffer_data.json'`

### `2024-10-30 9cf5ecf gpu`
- 修改 `train_rnn_module.py` `+7/-5`；關鍵片段: `def del_preprocess_buffer(self):` ; `ckpt_filepath = './custom_model/Spot_3D_NEW/buffer_data.json'`

### `2024-10-30 cbc1539 soc123`
- 修改 `train_rnn_module_compair.py` `+10/-11`；關鍵片段: `self.START_PHASE = 7` ; `self.START_PHASE = 1` ; `self.results_dir_ = "module_test_rnn_train_no_h0"` ; `self.results_dir_ = "lstm"` ; `self.notes = None  #'module_loss_diffent'`
- 修改 `train_rnn_module_stable.py` `+18/-11`；關鍵片段: `self.results_dir_ = "rnn_module_stable"` ; `self.results_dir_ = "module_stable"` ; `self.notes = None  #'reset_model_mentum'` ; `self.rl_env_versioin = 'RL_Env_v20'` ; `self.notes = None  #'concat_relu_state_goal_dt_xy'`

### `2024-11-02 8f83db6 soc123`
- 修改 `train_rnn_module.py` `+5/-5`；關鍵片段: `self.START_PHASE = 6` ; `self.START_PHASE = 4` ; `self.results_dir_ = "module_rnn_batch_256"` ; `self.results_dir_ = "batch_256_phase45_change_obs"` ; `preprocess_info_weight = [1.0, 1.0, 0.7, 0.7, 30.0, 30.0, 0]`
- 修改 `train_rnn_module_compair.py` `+2/-2`；關鍵片段: `self.results_dir_ = "just_dynanic_obs"` ; `self.results_dir_ = "dynanic_obs_add_distance_loss"` ; `preprocess_info_weight = [1.0, 1.0, 0.7, 0.7, 0.0, 0.0, 0]` ; `preprocess_info_weight = [1.0, 1.0, 1.0, 0.7, 0.7, 0.7, 0]`

### `2024-11-08 aa20643 soc123`
- 修改 `train_rnn_module.py` `+13/-13`；關鍵片段: `self.results_dir_ = "batch_256_phase45_change_obs"` ; `self.results_dir_ = "phase45_new_pra_num_spot2"` ; `self.train_spot_batch_size = self.phase5_spot_batch` ; `self.train_spot_batch_size = self.phase7_spot_batch` ; `rl_env_times = 50`

### `2024-11-10 e9db8e5 soc123`
- 修改 `train_rnn_module.py` `+44/-40`

### `2024-11-14 f608823 soc123`
- 修改 `train_rnn_module.py` `+21/-16`；關鍵片段: `self.START_PHASE = 1` ; `self.START_PHASE = 3` ; `self.results_dir_ = "stair_side_30_1m"` ; `self.results_dir_ = "phase12_env_reverse"` ; `self.notes = None #'num_spot_high_env_len_low_num_goal_low'`

### `2024-11-14 36f5b87 soc123`
- 修改 `train_rnn_module.py` `+4/-4`；關鍵片段: `self.START_PHASE = 3` ; `self.START_PHASE = 4` ; `self.results_dir_ = "phase12_env_reverse"` ; `self.results_dir_ = "phase4_long_env"` ; `self.notes = 'phase12 random map=false'`

### `2024-11-16 4a3c82d soc123`
- 修改 `train_rnn_module.py` `+7/-5`；關鍵片段: `self.START_PHASE = 4` ; `self.START_PHASE = 5` ; `self.results_dir_ = "phase4_long_env"` ; `self.results_dir_ = "spot_coperate_at_all"` ; `self.notes = 'spot coperate_ 0.25 0.35 0.5 is not important'`

### `2024-11-19 cd75c20 soc123`
- 修改 `train_rnn_module.py` `+48/-19`；關鍵片段: `self.START_PHASE = 5` ; `self.START_PHASE = 7` ; `self.results_dir_ = "spot_coperate_at_all"` ; `self.results_dir_ = "phase6_stair_and_avoid"` ; `self.notes = 'origin is stair diff'`

### `2024-11-20 98a5a15 soc123`
- 修改 `train_rnn_module.py` `+4/-4`；關鍵片段: `self.START_PHASE = 7` ; `self.START_PHASE = 3` ; `self.results_dir_ = "phase6_stair_and_avoid"` ; `self.results_dir_ = "add_two_grid_wall_grad_change"` ; `self.notes = None #'origin is stair diff'`

### `2024-11-22 52a0dca soc123`
- 修改 `train_rnn_module.py` `+2/-2`；關鍵片段: `self.START_PHASE = 3` ; `self.START_PHASE = 4`

### `2024-11-22 ef6b2ac soc123`
- 修改 `train_rnn_module.py` `+1/-1`；關鍵片段: `self.notes = 'dont push'` ; `self.notes = None #''`

### `2024-11-23 ded932f soc123`
- 修改 `train_rnn_module.py` `+14/-13`；關鍵片段: `self.START_PHASE = 4` ; `self.START_PHASE = 6` ; `self.num2_spot_batch = 200000` ; `self.train_spot_batch_size = self.num3_spot_batch` ; `self.train_spot_batch_size = self.num2_spot_batch`

### `2024-12-04 85311e4 soc123`
- 修改 `train_rnn_module.py` `+8/-7`；關鍵片段: `self.START_PHASE = 6` ; `self.START_PHASE = 4` ; `self.results_dir_ = "add_two_grid_wall_grad_change"` ; `self.results_dir_ = "add_noise_grid_wall"` ; `self.notes = None #''`

### `2024-12-04 9359eb8 gpu`
- 修改 `train_rnn_module.py` `+6/-5`；關鍵片段: `self.results_dir_ = "add_two_grid_wall_grad_change"` ; `self.results_dir_ = "final_model_rnn_init_model_change"` ; `self.train_spot_batch_size = self.num2_spot_batch` ; `self.train_spot_batch_size = self.num3_spot_batch`
- 修改 `train_rnn_module_compair.py` `+6/-5`；關鍵片段: `self.results_dir_ = "add_two_grid_wall_grad_change"` ; `self.results_dir_ = "two_gradw_2stair_wall"` ; `self.train_spot_batch_size = self.num2_spot_batch` ; `self.train_spot_batch_size = self.num3_spot_batch`

### `2024-12-04 f666b53 gpu`
- 修改 `train_rnn_module.py` `+7/-3`；關鍵片段: `self.results_dir_ = "add_noise_grid_wall"`

### `2024-12-05 3f369c7 gpu`
- 修改 `train_rnn_module.py` `+2/-5`；關鍵片段: `self.results_dir_ = "final_model_rnn_init_model_change"`
- 修改 `train_rnn_module_compair.py` `+8/-7`；關鍵片段: `self.START_PHASE = 6` ; `self.START_PHASE = 4` ; `self.results_dir_ = "two_gradw_2stair_wall"` ; `self.results_dir_ = "stair_skill_should_less_45_angle"` ; `self.notes = None #''`

### `2024-12-09 740c40e soc123`
- 修改 `train_rnn_module.py` `+5/-5`；關鍵片段: `self.START_PHASE = 4` ; `self.START_PHASE = 7` ; `self.results_dir_ = "add_noise_grid_wall"` ; `self.results_dir_ = "stair_grid_starlink"`

### `2024-12-17 84d1306 gpu`
- 改名/搬移 `train_rnn_module_star.py` (similarity 095%)；關鍵片段: `class Spot_Env():` ; `def __init__(self):` ; `self.Training = True` ; `self.START_PHASE = 7` ; `self.RL_env_name = "Spot_3D_NEW"`

### `2024-12-18 c3f3778 soc123`
- 修改 `train_rnn_module.py` `+34/-7`；關鍵片段: `self.START_PHASE = 7` ; `self.START_PHASE = 5` ; `self.results_dir_ = "stair_grid_starlink"` ; `self.results_dir_ = "stair_grid_net"` ; `self.notes = 'need to cheak cuda version to push'`

### `2024-12-23 9a915ca soc123`
- 修改 `train_rnn_module.py` `+40/-14`；關鍵片段: `self.START_PHASE = 5` ; `self.START_PHASE = 10` ; `self.train_spot_batch_size = self.num2_spot_batch` ; `self.train_goal_batch_size = self.phase8_goal_batch` ; `self.train_goal_num_spot = 1`

### `2024-12-23 9dca858 gpu`
- 修改 `train_rnn_module.py` `+29/-2`；關鍵片段: `self.START_PHASE = 5` ; `self.START_PHASE = 9` ; `self.train_spot_batch_size = self.num2_spot_batch` ; `self.train_goal_batch_size = self.phase8_goal_batch` ; `self.train_goal_num_spot = 1`
- 修改 `train_rnn_module_star.py` `+31/-3`；關鍵片段: `self.results_dir_ = "starlink_1_5"` ; `self.results_dir_ = "stair_avoid_first"` ; `self.notes = None#'need to cheak cuda version to push'` ; `self.notes = None #'need to cheak cuda version to push'` ; `self.train_spot_batch_size = self.num2_spot_batch`

### `2024-12-23 4b8730a gpu`
- 修改 `train_rnn_module.py` `+22/-11`；關鍵片段: `self.START_PHASE = 10`

### `2024-12-24 1518d62 soc123`
- 新增 `train_rnn_car.py` `+800/-0`；關鍵片段: `class Spot_Env():` ; `def __init__(self):` ; `self.Training = True` ; `self.START_PHASE = 6` ; `self.RL_env_name = "Spot_3D_NEW"`

### `2024-12-24 1086b26 soc123`
- 新增 `train_rnn_car.py` `+800/-0`；關鍵片段: `class Spot_Env():` ; `def __init__(self):` ; `self.Training = True` ; `self.RL_env_name = "Spot_3D_NEW"` ; `self.env_dir_ = "./custom_model/Spot_3D_NEW/example/"`
- 新增 `train_rnn_car_banch_high.py` `+800/-0`

### `2024-12-24 437c16c gpu`
- 修改 `train_rnn_module.py` `+0/-12`；關鍵片段: `self.START_PHASE = 9`
- 修改 `train_rnn_module_star.py` `+41/-43`；關鍵片段: `self.START_PHASE = 7` ; `self.START_PHASE = 10` ; `self.results_dir_ = "stair_avoid_first"` ; `self.results_dir_ = "stair_grid_net_hit_add"` ; `self.train_spot_batch_size = self.num2_spot_batch`

### `2025-02-25 af7afb6 soc123`
- 修改 `train_rnn_car.py` `+4/-4`；關鍵片段: `self.START_PHASE = 5` ; `self.START_PHASE = 7`
- 修改 `train_rnn_car_banch_high.py` `+6/-6`；關鍵片段: `self.START_PHASE = 5` ; `self.START_PHASE = 7`
- 修改 `train_rnn_module.py` `+3/-31`；關鍵片段: `self.results_dir_ = "stair_grid_net"` ; `self.results_dir_ = "final_spot_obs_delay_1s"` ; `self.train_spot_batch_size = self.num2_spot_batch` ; `self.train_goal_batch_size = self.phase8_goal_batch` ; `self.train_goal_num_spot = 1`

### `2025-03-01 9a9a94d soc123`
- 修改 `train_rnn_module.py` `+4/-4`；關鍵片段: `self.results_dir_ = "final_spot_obs_delay_1s"` ; `self.results_dir_ = "final_spot_obs_delay_0.7s"`

### `2025-03-02 5f096a0 soc123`
- 修改 `train_rnn_module.py` `+8/-7`；關鍵片段: `self.results_dir_ = "final_spot_obs_delay_0.7s"` ; `self.results_dir_ = "obs_delay_1s_rnn_obs_diff"` ; `preprocess_info_weight = [1.0, 1.0, 1.0, 0.7, 0.7, 0.7, 0]` ; `preprocess_info_weight = [0.9, 0.9, 0.8, 0.8, 1.0, 1.0, 1.0]` ; `self.download_weight_artifact_name = None`

### `2025-03-02 0656c7a soc123`
- 修改 `train_rnn_module.py` `+29/-2`；關鍵片段: `self.START_PHASE = 10` ; `self.START_PHASE = 11` ; `self.train_spot_batch_size = self.num2_spot_batch` ; `self.train_goal_batch_size = self.phase8_goal_batch` ; `self.train_goal_num_spot = 1`

### `2025-03-02 ed86a05 soc123`
- 修改 `train_rnn_module.py` `+1/-1`

## Car 線
### `2023-01-31 1e9c24f soc123`
- 新增 `custom_envs/car_continuous/car_continuous.py` `+828/-0`；關鍵片段: `_OBSERVATIONS = Constants.OBSERVATIONS` ; `_ACTIONS = Constants.ACTIONS` ; `_REWARDS = Constants.REWARDS` ; `_LOC_X = "loc_x"` ; `_LOC_Y = "loc_y"`
- 新增 `custom_envs/car_continuous/car_continuous_step.cu` `+392/-0`
- 新增 `custom_envs/final_car/final_car.py` `+985/-0`；關鍵片段: `_OBSERVATIONS = Constants.OBSERVATIONS` ; `_ACTIONS = Constants.ACTIONS` ; `_REWARDS = Constants.REWARDS` ; `_LOC_X = "loc_x"` ; `_LOC_Y = "loc_y"`
- 新增 `custom_envs/final_car/final_car_step.cu` `+591/-0`；關鍵片段: `neighbor_ids_sorted_by_distance_arr_idx =` ; `distance_arr_idx = kThisAgentArrayIdxOffset + idx;` ; `i_index = kThisAgentArrayIdxOffset + i;` ; `j_index = kThisAgentArrayIdxOffset + j;` ; `check_car_can_look_goal = true;`
- 新增 `custom_envs/train_double_car/Double_Car_step.cu` `+580/-0`；關鍵片段: `min_dis = sqrt(` ; `neighbor_ids_sorted_by_distance_arr_idx =` ; `distance_arr_idx = kThisAgentArrayIdxOffset + idx;` ; `i_index = kThisAgentArrayIdxOffset + i;` ; `j_index = kThisAgentArrayIdxOffset + j;`
- 新增 `custom_envs/train_double_car/double_car.py` `+1062/-0`；關鍵片段: `_OBSERVATIONS = Constants.OBSERVATIONS` ; `_ACTIONS = Constants.ACTIONS` ; `_REWARDS = Constants.REWARDS` ; `_LOC_X = "loc_x"` ; `_LOC_Y = "loc_y"`
- 新增 `train_double_car.py` `+131/-0`；關鍵片段: `pytorch_cuda_init_success = torch.cuda.FloatTensor(8)` ; `Training = True` ; `run_config = dict(` ; `name="train_double_car",` ; `env=dict(`

### `2023-02-01 06ca442 ken`
- 修改 `custom_envs/final_car/final_car.py` `+3/-3`；關鍵片段: `agent_size = self.np_random.random(self.num_agents) * 0.6 + 0.2` ; `agent_size = self.np_random.rand(self.num_agents) * 0.6 + 0.2` ; `starting_location_x = self.grid_length * self.np_random.random(` ; `starting_location_x = self.grid_length * self.np_random.rand(` ; `starting_location_y = self.grid_length * self.np_random.random(`
- 修改 `custom_envs/final_car/final_car_step.cu` `+0/-4`

### `2023-02-01 a645221 soc123`
- 修改 `custom_envs/final_car/final_car.py` `+3/-3`；關鍵片段: `agent_size = self.np_random.rand(self.num_agents) * 0.6 + 0.2` ; `agent_size = self.np_random.random(self.num_agents) * 0.6 + 0.2` ; `starting_location_x = self.grid_length * self.np_random.rand(` ; `starting_location_x = self.grid_length * self.np_random.random(` ; `starting_location_y = self.grid_length * self.np_random.rand(`

### `2023-04-27 e326094 linsea94`
- 修改 `custom_envs/final_car/final_car.py` `+13/-10`；關鍵片段: `agent_size = self.np_random.random(self.num_agents) * 0.6 + 0.2` ; `agent_size = (self.np_random.rand(self.num_agents) * 0.6 + 0.8)` ; `starting_location_x = self.grid_length * self.np_random.random(` ; `starting_location_x = self.grid_length * self.np_random.rand(` ; `starting_location_y = self.grid_length * self.np_random.random(`
- 修改 `custom_envs/final_car/final_car_step.cu` `+10/-3`

### `2023-08-08 21e8cac jqu314159`
- 新增 `custom_envs/car/car.py` `+1404/-0`；關鍵片段: `_OBSERVATIONS = Constants.OBSERVATIONS` ; `_ACTIONS = Constants.ACTIONS` ; `_REWARDS = Constants.REWARDS` ; `_LOC_X = "loc_x"` ; `_LOC_Y = "loc_y"`
- 新增 `custom_envs/car/car_step.cu` `+1064/-0`；關鍵片段: `point_dx = 0.5 * (grid_map_z[big_dx_id] - grid_map_z[small_dx_id]);` ; `point_dy = 0.5 * (grid_map_z[big_dy_id] - grid_map_z[small_dy_id]);` ; `neighbor_ids_sorted_by_distance_arr_idx =` ; `distance_arr_idx = kThisAgentArrayIdxOffset + idx;` ; `i_index = kThisAgentArrayIdxOffset + i;`

### `2023-10-23 6470757 jqu314159@gmail.com`
- 改名/搬移 `custom_envs/new_car/new_car.py` (similarity 083%)；關鍵片段: `_OBSERVATIONS = Constants.OBSERVATIONS` ; `_ACTIONS = Constants.ACTIONS` ; `_REWARDS = Constants.REWARDS` ; `_LOC_X = "loc_x"` ; `_LOC_Y = "loc_y"`
- 改名/搬移 `custom_envs/new_car/new_car_step.cu` (similarity 096%)；關鍵片段: `point_dx = 0.5 * (grid_map_z[big_dx_id] - grid_map_z[small_dx_id]);` ; `point_dy = 0.5 * (grid_map_z[big_dy_id] - grid_map_z[small_dy_id]);` ; `neighbor_ids_sorted_by_distance_arr_idx =` ; `distance_arr_idx = kThisAgentArrayIdxOffset + idx;` ; `i_index = kThisAgentArrayIdxOffset + i;`

### `2023-10-25 a6a6a07 gpu`
- 修改 `custom_envs/new_car/new_car.py` `+14/-100`；關鍵片段: `_SP_Y = "speed_y"` ; `_ACC_Y = "acceleration_y"` ; `car_max_speed_y = 0.3,` ; `car_max_acceleration_y = 0.05,` ; `car_cost_speed_y_rate_with_acc_x  = 0.4,`
- 修改 `custom_envs/new_car/new_car_step.cu` `+51/-131`；關鍵片段: `agent_types = 1;` ; `agent_speed_y = speed_y_arr[kOtherAgentArrayIdx] / (kMaxSpeed_y + kEpsilon);` ; `kRewardOnFloor_ = - 1 * kRewardcarFloorDirectionBase;`

### `2023-11-13 e5018e2 gpu`
- 修改 `custom_envs/new_car/new_car.py` `+11/-22`；關鍵片段: `floor_width = floor_width_bias * np.ones(num_stairs) + floor_width_rand * random.random()` ; `floor_width = floor_width_bias + floor_width_rand * random.random()` ; `def seed(self, seed=None):` ; `hit_goal = False` ; `hit_goal = True`
- 修改 `custom_envs/new_car/new_car_step.cu` `+6/-1`；關鍵片段: `goal_hit = true;`

### `2023-11-21 d978a19 jqu314159@gmail.com`
- 修改 `custom_envs/new_car/new_car.py` `+12/-0`；關鍵片段: `def seed(self, seed=None):`

### `2024-04-16 ffece4b soc123`
- 修改 `custom_envs/train_double_car/Double_Car_step.cu` `+3/-2`
- 修改 `custom_envs/train_double_car/double_car.py` `+16/-4`；關鍵片段: `starting_location_x = self.grid_length * self.np_random.rand(` ; `starting_location_x = self.grid_length * self.np_random.random(` ; `starting_location_y = self.grid_length * self.np_random.rand(` ; `starting_location_y = self.grid_length * self.np_random.random(` ; `display_reward = np.zeros(14 * self.num_agents, dtype=self.float_dtype)`
- 修改 `train_double_car.py` `+11/-8`；關鍵片段: `action2_entropy_coeff = None,` ; `action3_entropy_coeff = None,` ; `model_type = 'normal',` ; `type="fully_connected", fc_dims=[256, 256, 256], model_ckpt_filepath=""` ; `type="fully_connected",`

### `2024-04-17 6cac546 soc123`
- 修改 `train_double_car.py` `+7/-2`；關鍵片段: `run = wandb.init(` ; `project = 'test',` ; `name = "train_double_car",` ; `train_batch_size=1000,  # total batch size used for training environments` ; `train_batch_size=10000,  # total batch size used for training environments`

### `2024-04-17 db0c350 soc123`
- 修改 `custom_envs/train_double_car/Double_Car_step.cu` `+12/-8`
- 修改 `custom_envs/train_double_car/double_car.py` `+1/-1`；關鍵片段: `tag_reward_for_tagger = 2.0,` ; `tag_reward_for_tagger = 5.0,`
- 修改 `train_double_car.py` `+522/-132`；關鍵片段: `pytorch_cuda_init_success = torch.cuda.FloatTensor(8)` ; `Training = True` ; `run = wandb.init(` ; `project = 'test',` ; `name = "train_double_car",`
