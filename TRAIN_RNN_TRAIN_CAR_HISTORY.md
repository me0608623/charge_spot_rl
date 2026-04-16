# `train_rnn` / `train_car` 歷史紀錄

產出時間: 2026-04-16  
資料來源: 本地 clone `/home/aa/new_warp_drive`，remote=`git@github.com:jqu314159/new_warp_drive.git`

## 1. 先講結論

- `train_rnn` 主線不是單檔演化，而是一條分支鏈:
  - `train_stable_rnn.py` 在 `f545d5d`（2024-09-26）由 `train_stable_rnn_loss_change.py` 改名而來。
  - `train_rnn_module.py` 在 `6cc9de1`（2024-10-03）由 `train_stable_rnn.py` 複製而來，copy similarity 99%。
  - `train_rnn_module_stable.py` 在 `9c162a4`（2024-10-15）由 `train_rnn_module.py` 複製而來，copy similarity 98%。
  - `train_rnn_module_star.py` 在 `84d1306`（2024-12-17）由 `train_rnn_module_compair.py` 改名而來，rename similarity 95%。
  - `train_rnn_car.py` 在 `1518d62`（2024-12-24）由 `train_rnn_module.py` 複製而來，copy similarity 91%。
- `train_car` 不是只有一條線，而是至少四個分支族群:
  - `train_double_car.py` + `custom_envs/train_double_car/*`
  - `custom_envs/final_car/*`
  - `custom_envs/car/*`
  - `custom_envs/new_car/*`
- `custom_envs/new_car/*` 不是原生新檔，而是在 `6470757`（2023-10-23）由 `custom_envs/spot_pfsp/*` 改名轉出。
- 目前工作樹不是乾淨的:
  - `train_rnn_car.py` 有未提交修改。
  - 未追蹤檔案有 `train_rnn_car_used_files_manifest.txt` 與 `論文終.pdf`。

## 2. Push 歷史能不能完整還原

不能完整還原。

我能精準還原的是 commit 時間線與 copy/rename 血統，但「所有 push 批次」無法從這份 clone 完整重建，原因如下:

- `git reflog --all` 只看到這台機器在 `2026-04-10 17:01:16 +0800` 的 clone 紀錄。
- repo 目前只有 `main` 與 `origin/main`，沒有保留下來的本地推送 reflog。
- 這個 repo 很多 commit 的 message 都誤把編輯器模板整段提交進去，裡面的 `ahead of origin/main by N commits` / `up to date` 文字只能當時點線索，不能當正式 push log。

所以這份文件把:

- `commit` 歷史做成精確紀錄
- `push` 只註明「無法從目前 clone 完整還原」

## 3. 重要檔案摘要

### RNN 族群

| 路徑 | direct commits | 首個 direct commit | 最後 direct commit | 血統說明 |
|---|---:|---|---|---|
| `train_stable_rnn.py` | 11 | `f545d5d` 2024-09-26 | `9253c46` 2024-10-09 | 由 `train_stable_rnn_loss_change.py` 改名而來 |
| `train_rnn_module.py` | 44 | `6cc9de1` 2024-10-03 | `ed86a05` 2025-03-02 | 由 `train_stable_rnn.py` 複製而來 |
| `train_rnn_module_stable.py` | 3 | `9c162a4` 2024-10-15 | `cbc1539` 2024-10-30 | 由 `train_rnn_module.py` 複製而來 |
| `train_rnn_module_star.py` | 3 | `84d1306` 2024-12-17 | `437c16c` 2024-12-24 | 由 `train_rnn_module_compair.py` 改名而來 |
| `train_rnn_module_compair_no_sucess.py` | 1 | `d415642` 2024-10-09 | `d415642` 2024-10-09 | 一次性分支檔案 |
| `train_rnn_car.py` | 3 | `1518d62` 2024-12-24 | `af7afb6` 2025-02-25 | 由 `train_rnn_module.py` 複製而來 |
| `train_rnn_car_banch_high.py` | 2 | `1086b26` 2024-12-24 | `af7afb6` 2025-02-25 | 與 `train_rnn_car.py` 同批出現在主線 |

補充:

- 若用 `git log --follow` 去追祖先鏈，`train_rnn_module.py` 可往前追到 2024-03-01 附近的舊訓練腳本族系，`train_rnn_car.py` 也會繼承這整條祖先鏈。
- 但下面第 4 節紀錄的是「這些檔案名本身在主線上的 direct commit」，避免把祖先腳本和後來分叉腳本混成一坨。

### Car 族群

| 路徑 | direct commits | 首個 direct commit | 最後 direct commit | 血統說明 |
|---|---:|---|---|---|
| `train_double_car.py` | 4 | `1e9c24f` 2023-01-31 | `db0c350` 2024-04-17 | 初始即存在 |
| `custom_envs/train_double_car/double_car.py` | 3 | `1e9c24f` 2023-01-31 | `db0c350` 2024-04-17 | 初始即存在 |
| `custom_envs/train_double_car/Double_Car_step.cu` | 3 | `1e9c24f` 2023-01-31 | `db0c350` 2024-04-17 | 初始即存在 |
| `custom_envs/final_car/final_car.py` | 4 | `1e9c24f` 2023-01-31 | `e326094` 2023-04-27 | 初始即存在 |
| `custom_envs/final_car/final_car_step.cu` | 3 | `1e9c24f` 2023-01-31 | `e326094` 2023-04-27 | 初始即存在 |
| `custom_envs/car_continuous/car_continuous.py` | 1 | `1e9c24f` 2023-01-31 | `1e9c24f` 2023-01-31 | 初始即存在 |
| `custom_envs/car_continuous/car_continuous_step.cu` | 1 | `1e9c24f` 2023-01-31 | `1e9c24f` 2023-01-31 | 初始即存在 |
| `custom_envs/car/car.py` | 1 | `21e8cac` 2023-08-08 | `21e8cac` 2023-08-08 | 舊 car 分支，前面祖先在更舊 env |
| `custom_envs/car/car_step.cu` | 1 | `21e8cac` 2023-08-08 | `21e8cac` 2023-08-08 | 舊 car 分支，前面祖先在更舊 env |
| `custom_envs/new_car/new_car.py` | 4 | `6470757` 2023-10-23 | `d978a19` 2023-11-21 | 由 `custom_envs/spot_pfsp/spot_pfsp.py` 改名而來 |
| `custom_envs/new_car/new_car_step.cu` | 3 | `6470757` 2023-10-23 | `e5018e2` 2023-11-13 | 由 `custom_envs/spot_pfsp/spot_pfsp_step.cu` 改名而來 |

## 4. RNN 族群 direct commit 時間線

只列真正改到下列檔案的 commit:

- `train_stable_rnn.py`
- `train_rnn_module.py`
- `train_rnn_module_stable.py`
- `train_rnn_module_star.py`
- `train_rnn_module_compair_no_sucess.py`
- `train_rnn_car.py`
- `train_rnn_car_banch_high.py`

```text
2024-09-26 f545d5d soc123  train_stable_rnn.py
2024-09-28 4c98b06 gpu     train_stable_rnn.py
2024-09-28 38119f8 soc123  train_stable_rnn.py
2024-09-28 357937d soc123  train_stable_rnn.py
2024-09-30 53a7af3 soc123  train_stable_rnn.py
2024-09-30 35ec222 gpu     train_stable_rnn.py
2024-09-30 7ba848b gpu     train_stable_rnn.py
2024-10-03 33fe687 soc123  train_stable_rnn.py
2024-10-03 2e3ae93 gpu     train_stable_rnn.py
2024-10-03 c470e97 gpu     train_stable_rnn.py
2024-10-03 6cc9de1 gpu     train_rnn_module.py
2024-10-09 9253c46 soc123  train_rnn_module.py, train_stable_rnn.py
2024-10-09 d415642 gpu     train_rnn_module_compair_no_sucess.py
2024-10-09 29c44e4 gpu     train_rnn_module.py
2024-10-09 b693a21 soc123  train_rnn_module.py
2024-10-11 bb9e640 soc123  train_rnn_module.py
2024-10-11 d0d7e61 soc123  train_rnn_module.py
2024-10-11 23921dd gpu     train_rnn_module.py
2024-10-11 d55a24a soc123  train_rnn_module.py
2024-10-11 4937b69 gpu     train_rnn_module.py
2024-10-11 a58b086 gpu     train_rnn_module.py
2024-10-11 f44ecec gpu     train_rnn_module.py
2024-10-12 a6d1eae gpu     train_rnn_module.py
2024-10-14 34f7af4 gpu     train_rnn_module.py
2024-10-15 9c162a4 gpu     train_rnn_module.py, train_rnn_module_stable.py
2024-10-18 e1de4f2 gpu     train_rnn_module.py, train_rnn_module_stable.py
2024-10-21 adff7eb gpu     train_rnn_module.py
2024-10-21 329405c gpu     train_rnn_module.py
2024-10-28 f8ca040 gpu     train_rnn_module.py
2024-10-28 cf9a6e5 gpu     train_rnn_module.py
2024-10-30 2f20dfa gpu     train_rnn_module.py
2024-10-30 9cf5ecf gpu     train_rnn_module.py
2024-10-30 cbc1539 soc123  train_rnn_module_stable.py
2024-11-02 8f83db6 soc123  train_rnn_module.py
2024-11-08 aa20643 soc123  train_rnn_module.py
2024-11-10 e9db8e5 soc123  train_rnn_module.py
2024-11-14 f608823 soc123  train_rnn_module.py
2024-11-14 36f5b87 soc123  train_rnn_module.py
2024-11-16 4a3c82d soc123  train_rnn_module.py
2024-11-19 cd75c20 soc123  train_rnn_module.py
2024-11-20 98a5a15 soc123  train_rnn_module.py
2024-11-22 52a0dca soc123  train_rnn_module.py
2024-11-22 ef6b2ac soc123  train_rnn_module.py
2024-11-23 ded932f soc123  train_rnn_module.py
2024-12-04 85311e4 soc123  train_rnn_module.py
2024-12-04 9359eb8 gpu     train_rnn_module.py
2024-12-04 f666b53 gpu     train_rnn_module.py
2024-12-05 3f369c7 gpu     train_rnn_module.py
2024-12-09 740c40e soc123  train_rnn_module.py
2024-12-17 84d1306 gpu     train_rnn_module_star.py
2024-12-18 c3f3778 soc123  train_rnn_module.py
2024-12-23 9a915ca soc123  train_rnn_module.py
2024-12-23 9dca858 gpu     train_rnn_module.py, train_rnn_module_star.py
2024-12-23 4b8730a gpu     train_rnn_module.py
2024-12-24 1518d62 soc123  train_rnn_car.py
2024-12-24 1086b26 soc123  train_rnn_car.py, train_rnn_car_banch_high.py
2024-12-24 437c16c gpu     train_rnn_module.py, train_rnn_module_star.py
2025-02-25 af7afb6 soc123  train_rnn_car.py, train_rnn_car_banch_high.py, train_rnn_module.py
2025-03-01 9a9a94d soc123  train_rnn_module.py
2025-03-02 5f096a0 soc123  train_rnn_module.py
2025-03-02 0656c7a soc123  train_rnn_module.py
2025-03-02 ed86a05 soc123  train_rnn_module.py
```

## 5. Car 族群 direct commit 時間線

只列真正改到下列檔案的 commit:

- `train_double_car.py`
- `custom_envs/train_double_car/*`
- `custom_envs/final_car/*`
- `custom_envs/car/*`
- `custom_envs/new_car/*`
- `custom_envs/car_continuous/*`

```text
2023-01-31 1e9c24f soc123             custom_envs/car_continuous/car_continuous.py, custom_envs/car_continuous/car_continuous_step.cu, custom_envs/final_car/final_car.py, custom_envs/final_car/final_car_step.cu, custom_envs/train_double_car/Double_Car_step.cu, custom_envs/train_double_car/double_car.py, train_double_car.py
2023-02-01 06ca442 ken                custom_envs/final_car/final_car.py, custom_envs/final_car/final_car_step.cu
2023-02-01 a645221 soc123             custom_envs/final_car/final_car.py
2023-04-27 e326094 linsea94           custom_envs/final_car/final_car.py, custom_envs/final_car/final_car_step.cu
2023-08-08 21e8cac jqu314159          custom_envs/car/car.py, custom_envs/car/car_step.cu
2023-10-23 6470757 jqu314159@gmail.com custom_envs/new_car/new_car.py, custom_envs/new_car/new_car_step.cu
2023-10-25 a6a6a07 gpu                custom_envs/new_car/new_car.py, custom_envs/new_car/new_car_step.cu
2023-11-13 e5018e2 gpu                custom_envs/new_car/new_car.py, custom_envs/new_car/new_car_step.cu
2023-11-21 d978a19 jqu314159@gmail.com custom_envs/new_car/new_car.py
2024-04-16 ffece4b soc123             custom_envs/train_double_car/Double_Car_step.cu, custom_envs/train_double_car/double_car.py, train_double_car.py
2024-04-17 6cac546 soc123             train_double_car.py
2024-04-17 db0c350 soc123             custom_envs/train_double_car/Double_Car_step.cu, custom_envs/train_double_car/double_car.py, train_double_car.py
```

## 6. 幾個關鍵血統節點

### `train_rnn_car.py`

- `1518d62`:
  - `git show --find-copies-harder 1518d62`
  - 顯示 `train_rnn_module.py => train_rnn_car.py`，copy similarity 91%
- 直接歷史只有 3 個 commit:
  - `1518d62`
  - `1086b26`
  - `af7afb6`
- 若用 `--follow` 往前追，祖先會接回 `train_rnn_module.py` 的整條歷史。

### `train_rnn_module.py`

- `6cc9de1`:
  - `git show --find-copies-harder 6cc9de1`
  - 顯示 `train_stable_rnn.py => train_rnn_module.py`，copy similarity 99%
- 這是整條 `train_rnn` 主線裡 commit 最密集的主檔。

### `train_rnn_module_stable.py`

- `9c162a4`:
  - `git show --find-copies-harder 9c162a4`
  - 顯示 `train_rnn_module.py => train_rnn_module_stable.py`，copy similarity 98%

### `train_rnn_module_star.py`

- `84d1306`:
  - `git show --find-renames 84d1306`
  - 顯示 `train_rnn_module_compair.py => train_rnn_module_star.py`，rename similarity 95%

### `custom_envs/new_car/*`

- `6470757`:
  - `git show --find-renames 6470757`
  - 顯示:
    - `custom_envs/spot_pfsp/spot_pfsp.py => custom_envs/new_car/new_car.py`
    - `custom_envs/spot_pfsp/spot_pfsp_step.cu => custom_envs/new_car/new_car_step.cu`

## 7. 目前工作樹狀態

這些不是 commit 歷史的一部分，但你現在本地有:

- `train_rnn_car.py` 已修改:
  - `git diff --stat -- train_rnn_car.py`
  - 結果是 `137 insertions(+), 12 deletions(-)`
- 未追蹤:
  - `train_rnn_car_used_files_manifest.txt`
  - `論文終.pdf`

## 8. 我實際使用的查詢方式

```bash
git remote -v
git status --short --branch
git reflog --date=iso --all

git log --oneline -- train_rnn_car.py
git log --oneline -- train_rnn_module.py
git log --oneline -- train_stable_rnn.py

git show --stat --summary --find-copies-harder 1518d62
git show --stat --summary --find-copies-harder 6cc9de1
git show --stat --summary --find-copies-harder 9c162a4
git show --stat --summary --find-renames 84d1306
git show --stat --summary --find-renames 6470757
```

如果你下一步要的是「把每個 commit 的實際程式 diff 再拆成逐版功能變化摘要」，可以直接在這份文件的基礎上繼續擴成第二份 `diff-by-diff` 版。
