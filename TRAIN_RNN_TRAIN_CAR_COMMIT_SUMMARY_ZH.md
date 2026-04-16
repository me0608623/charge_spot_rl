# `train_rnn` / `train_car` 人話摘要版

產出時間: 2026-04-16  
用途: 把逐 commit 的 patch 改寫成研究日誌式中文，重點放在「作者當時在試什麼」。

## 先看總體方向

- `train_rnn` 前半段主要在試 RNN 記憶形式、序列長度、dropout、module 前處理權重。
- 中段開始把焦點移到 phase 設計、樓梯地圖、動態避障、長短 episode 與 batch 配置。
- 後段再從主線分出 `train_rnn_car.py`，把 module + RNN 的經驗搬到 car/spot 預訓練場景。
- `train_car` 比較像多條環境分支並行演化: `final_car`、`car`、`new_car`、`double_car` 各自代表不同車體/地圖/訓練方向。

## RNN 線逐筆摘要

### `2024-09-26 f545d5d soc123`
- 把舊的 `train_stable_rnn_loss_change.py` 正式改名成 `train_stable_rnn.py`，等於把 RNN 穩定版訓練腳本立成一條獨立主線。

### `2024-09-28 4c98b06 gpu`
- 把 stable RNN 實驗切到 `RNN_num_layer2`，看起來是在試兩層 RNN 結構。

### `2024-09-28 38119f8 soc123`
- 把 stable RNN 的實驗焦點改成 `detach_init_memory_at_collect`，也就是想測試收資料時是否要把初始記憶斷開。

### `2024-09-28 357937d soc123`
- 合併兩條 stable RNN 實驗分支，主要是在處理 `results_dir` 與 env 設定衝突，讓前面的兩個實驗方向收回同一條線。

### `2024-09-30 53a7af3 soc123`
- 把 stable RNN 切到 `seq5 + rnn_loss` 這組設定，代表開始強調固定序列長度與額外 RNN loss。

### `2024-09-30 35ec222 gpu`
- 把 stable RNN 切到 `RNN_merge_seq5`，看起來是在試把多段序列資訊合併後再訓練。

### `2024-09-30 7ba848b gpu`
- 再做一次 stable RNN 合併，收斂 `merge_seq5` 與 `seq5_rnn_loss` 兩條分支。

### `2024-10-03 33fe687 soc123`
- 把 stable RNN 切到 `rnn_layer_short`，方向是縮短/簡化 RNN 層配置。

### `2024-10-03 2e3ae93 gpu`
- 把 stable RNN 改成 `RNN_all_rnn`，意思是更大範圍地讓觀測或處理流程都走 RNN。

### `2024-10-03 c470e97 gpu`
- 整理並合併前面 stable RNN 的不同嘗試，把分支結果重新匯回主線。

### `2024-10-03 6cc9de1 gpu`
- 從 stable RNN 複製出 `train_rnn_module.py` 與 `train_rnn_module_compair.py`，正式展開「module + RNN」這條新訓練線。

### `2024-10-09 9253c46 soc123`
- 把 module 版實驗切成 `obs_all_static` 路線，同時把訓練輪數拉高、phase1 batch 加大，表示開始做更重的主訓練。

### `2024-10-09 d415642 gpu`
- 建立 `no module` 對照線，另外把 `train_rnn_module_compair_no_sucess.py` 留存下來，等於把失敗/對照版本也固化成檔案。

### `2024-10-09 29c44e4 gpu`
- 把訓練腳本真正從 `fully_connected + spot_3dIn` 轉到 `module_connected + spot_3dmodule`，這是 module 線最關鍵的架構切換。

### `2024-10-09 b693a21 soc123`
- 把 phase2 的 spot batch 從 100000 拉到 130000，並標記要重設 momentum，表示想讓第二階段訓練更重、更穩。

### `2024-10-11 bb9e640 soc123`
- 大幅重調 `preprocess_info_weight`，把前段權重壓低，切成 `rnn_module_weight_change` 實驗。

### `2024-10-11 d0d7e61 soc123`
- 把實驗焦點從權重調整切到 `obs_state_change`，代表開始比較 observation/state 組成的影響。

### `2024-10-11 23921dd gpu`
- 加入 `dropout=0.1` 對照，想測試 RNN module 線在 dropout 下的表現。

### `2024-10-11 d55a24a soc123`
- 把上一筆的 dropout 嘗試又切回 `obs_state_change`，等於回退或重新對照。

### `2024-10-11 4937b69 gpu`
- 把 `train_goal_fix_spot_model` 調成 `False`，表示 goal 訓練時不再把 spot 模型完全鎖死。

### `2024-10-11 a58b086 gpu`
- 合併 dropout 與 obs_state_change 兩條 module 分支，主要是把衝突整理掉。

### `2024-10-11 f44ecec gpu`
- 再把 module 線切回 `dropout01` 路線，等於重開一次 dropout 測試。

### `2024-10-12 a6d1eae gpu`
- 把實驗改成 `rnn_loss_log`，並重新調整 preprocess 權重，開始更明確觀察 RNN loss 的反應。

### `2024-10-14 34f7af4 gpu`
- 把起始 phase 從 1 改到 3，表示這時候訓練已經比較偏續訓/跳過前期暖身。

### `2024-10-15 9c162a4 gpu`
- 新增 `train_rnn_module_stable.py`，同時在主 module 線切到 `policy_loss_04` 版本，開始把 stable 分支獨立出來。

### `2024-10-18 e1de4f2 gpu`
- 把 module 主線切到 `stair_state_diff`，並把訓練輪數從 300 降回 180，顯然在換一組更專注樓梯/狀態差異的實驗。

### `2024-10-21 adff7eb gpu`
- 把 module 主線切到 `module_test_mlp`，同時重調 RL env 版本和 preprocess 權重，像是在測 module 前處理改成更接近 MLP 的效果。

### `2024-10-21 329405c gpu`
- 把實驗改成 `rnn_huge_input`，開始測更大的輸入組合。

### `2024-10-28 f8ca040 gpu`
- 切到 `module_rnn_batch_256`，把 batch 與 preprocess 權重再調一次，並打開 `concat_rnn`，方向是讓 RNN 吃更完整拼接輸入。

### `2024-10-28 cf9a6e5 gpu`
- 把起始 phase 改到 5，並調整 phase7 的 batch，代表訓練焦點往更後面的高難 phase 移動。

### `2024-10-30 2f20dfa gpu`
- 把起始 phase 往後推到 6，並把 phase5 的 spot batch 從 350000 拉到 400000。

### `2024-10-30 9cf5ecf gpu`
- 新增 `del_preprocess_buffer()`，開始在訓練前明確刪除 buffer 檔，避免舊序列資料污染新訓練。

### `2024-10-30 cbc1539 soc123`
- stable 線切成 `module_stable`，compair 線切成 `lstm`，等於把 stable 與 LSTM 對照兩條路一起明確命名。

### `2024-11-02 8f83db6 soc123`
- 主 module 線改成 `phase45_change_obs`，重新調 observation 權重；compair 線則改成 `add_distance_loss`，開始把距離 loss 納進對照。

### `2024-11-08 aa20643 soc123`
- 重新分配 phase4/5 的 batch 與 episode 長度，並用 `phase45_new_pra_num_spot2` 命名，表示這時候主要在刷 phase4/5 參數組合。

### `2024-11-10 e9db8e5 soc123`
- 大合併一次 phase 參數分支，實際上重寫了多個 phase 的地圖、牆、樓梯、reward 與 batch 設計，這是一筆很重的訓練日誌型 commit。

### `2024-11-14 f608823 soc123`
- 把主線切到 `phase12_env_reverse`，開始明確測試 reverse env / 反向場景，以及更偏樓梯地形的配置。

### `2024-11-14 36f5b87 soc123`
- 把主線切到 `phase4_long_env`，核心是把環境跑更久，觀察長 episode 對 spot 策略的影響。

### `2024-11-16 4a3c82d soc123`
- 把主線切到 `spot_coperate_at_all`，看起來是在試更強的 spot 協作設定。

### `2024-11-19 cd75c20 soc123`
- 把主線切到 `phase6_stair_and_avoid`，並新增 phase8 batch 參數，焦點明顯轉向「樓梯 + 避障」雙任務。

### `2024-11-20 98a5a15 soc123`
- 把主線切到 `add_two_grid_wall_grad_change`，表示開始強測牆、格子與地形梯度對策略的影響。

### `2024-11-22 52a0dca soc123`
- 只把起始 phase 往後調，屬於續訓節奏調整。

### `2024-11-22 ef6b2ac soc123`
- 把註解裡的 `dont push` 清掉，內容變動不大，像是整理一次腳本狀態。

### `2024-11-23 ded932f soc123`
- 把起始 phase 改到 6，並改用 `num2/num3/phase8` 這組 batch 邏輯重排訓練節奏。

### `2024-12-04 85311e4 soc123`
- 把主線切到 `add_noise_grid_wall`，還特別註記 CUDA 版本要確認，表示這組實驗和 CUDA/環境耦合比較高。

### `2024-12-04 9359eb8 gpu`
- 主 module 線切到 `final_model_rnn_init_model_change`，compair 線切到 `two_gradw_2stair_wall`，開始做接近最終模型的初始化與地圖組合測試。

### `2024-12-04 f666b53 gpu`
- 把 `final_model_rnn_init_model_change` 與 `add_noise_grid_wall` 兩條線合併，主線最後保留 `add_noise_grid_wall` 這個方向。

### `2024-12-05 3f369c7 gpu`
- 主 module 線延續前面的最終模型方向；compair 線改成 `stair_skill_should_less_45_angle`，明顯在試樓梯角度限制對技能學習的影響。

### `2024-12-09 740c40e soc123`
- 把主線切到 `stair_grid_starlink`，開始進入星鏈/網格型樓梯場景。

### `2024-12-17 84d1306 gpu`
- 把 `train_rnn_module_compair.py` 改名成 `train_rnn_module_star.py`，代表 star 這條樓梯/網格分支正式獨立。

### `2024-12-18 c3f3778 soc123`
- 主線切到 `stair_grid_net`，調整批次與 phase 佈局，繼續沿著樓梯網格場景推進。

### `2024-12-23 9a915ca soc123`
- 把主線直接推到 phase10，加入更偏保守動態避障的設定，包含一個 spot 對多 goal 的長時程訓練。

### `2024-12-23 9dca858 gpu`
- 主線改成 phase9 版本；star 線則切到 `stair_avoid_first`，表示先學避障再處理樓梯。

### `2024-12-23 4b8730a gpu`
- 合併 phase9 與 phase10 兩條後期分支，把「保守動態避障」這套 phase 結構整理回一條主線。

### `2024-12-24 1518d62 soc123`
- 從 `train_rnn_module.py` 複製出 `train_rnn_car.py`，正式把 RNN module 線分叉到 car 專用訓練腳本，初始實驗名是 `car_new_gamma`。

### `2024-12-24 1086b26 soc123`
- 把 `train_rnn_car.py` 和 `train_rnn_car_banch_high.py` 一起併進主線，最終採用 `START_PHASE=5` 與 `car_spot_pretrain` 這組 car 設定。

### `2024-12-24 437c16c gpu`
- star 線切到 `stair_grid_net_hit_add`，而主 module 線則刪去一部分後期 phase 內容，像是在做主線與 star 線的職責切分。

### `2025-02-25 af7afb6 soc123`
- 把 `train_rnn_car.py` 與 `train_rnn_car_banch_high.py` 的起始 phase 往後推到 7，同時主 module 線改成 `final_spot_obs_delay_1s`。

### `2025-03-01 9a9a94d soc123`
- 把最終觀測延遲實驗從 `1.0s` 改成 `0.7s`。

### `2025-03-02 5f096a0 soc123`
- 把主線切到 `obs_delay_1s_rnn_obs_diff`，重調 preprocess 權重，並取消自動下載舊 artifact，變成更獨立的一次訓練。

### `2025-03-02 0656c7a soc123`
- 把起始 phase 往後推到 11，表示訓練已進入非常後段的續訓階段。

### `2025-03-02 ed86a05 soc123`
- 把主程式入口從 `train(download_weight_artifact_name=...)` 改成單純 `train()`，不再強制載入 `PHASE_9:v12` 權重。

## Car 線逐筆摘要

### `2023-01-31 1e9c24f soc123`
- 一次加入 `car_continuous`、`final_car`、`train_double_car` 等整套 car 環境與訓練腳本，等於 car 線的起點。

### `2023-02-01 06ca442 ken`
- 微調 `final_car` 的隨機初始化寫法，主要是 numpy random API 與相關 CUDA 配套的小修。

### `2023-02-01 a645221 soc123`
- 把前一筆對 random API 的改法又切回另一種寫法，屬於初始化行為的來回修正。

### `2023-04-27 e326094 linsea94`
- 調整 `final_car` 的 agent 尺寸範圍與 observation 內容，讓車子觀測到鄰近車型別/尺寸資訊。

### `2023-08-08 21e8cac jqu314159`
- 新增 `custom_envs/car/*`，建立更完整的 car 環境，包含 2D 速度、方向與地形梯度處理。

### `2023-10-23 6470757 jqu314159@gmail.com`
- 把 `spot_pfsp` 這條線改名搬成 `new_car`，同時對應產生新的 car 訓練腳本，代表 car 線的一次正式轉型。

### `2023-10-25 a6a6a07 gpu`
- 在 `new_car` 裡加入 y 軸速度與加速度，連 reward/cost 也一起改，代表車的動態模型更完整了。

### `2023-11-13 e5018e2 gpu`
- 修正 `new_car` 的樓梯寬度生成、seed 與 hit_goal 邏輯，讓環境可重現性和碰 goal 判斷更穩。

### `2023-11-21 d978a19 jqu314159@gmail.com`
- 補強 `new_car` 的 seed 相關邏輯，延續前一筆把環境做得更可重現。

### `2024-04-16 ffece4b soc123`
- `train_double_car` 增加 `display_reward` 顯示資料，並調整 trainer/model 參數，開始讓雙車訓練更可觀察。

### `2024-04-17 6cac546 soc123`
- 在 `train_double_car.py` 裡加上 wandb 初始化，並把 train batch size 從 1000 拉到 10000。

### `2024-04-17 db0c350 soc123`
- 大改 `train_double_car.py`，把訓練腳本擴成完整 wandb/custom trainer 版本，同時把 tagger reward 從 2.0 提到 5.0。
