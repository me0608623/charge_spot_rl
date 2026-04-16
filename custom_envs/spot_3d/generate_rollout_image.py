import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import art3d
from matplotlib.cm import Greys
from matplotlib.colors import Normalize, to_rgba
import math
import wandb
import os


def generate_tag_env_rollout_image(
    trainer,  # 訓練器物件，包含環境和智能體信息
    spot_color='gold',  # Spot 的顏色
    spot_direction_color='orange',  # Spot 方向指示器的顏色
    obstacle_color='black',  # 障礙物的顏色
    goal_color='lightgreen',  # 目標的顏色
    goal_not_in_game_color="#666666",  # 目標不在遊戲中的顏色
    Training=True,  # 是否為訓練模式
    use_wandb=False,  # 是否使用 wandb 記錄
    state=0,  # 當前狀態
    phase_env_num=300,  # 階段環境數量
    wandb_table=None,  # wandb 表格
    PHASE_=1,  # 階段
    pre_train=None,  # 是否為預訓練
    wandb_name=None,  # wandb 名稱
    wandb_table_history=None,  # wandb 表格歷史記錄
    num_goals=15,  # 目標數量
    time_idx=0
):
    assert trainer is not None
    episode_states = trainer.fetch_episode_states(  # 獲取 episode 的狀態信息
        ["loc_x", "loc_y", "direction", "still_in_the_game",
         "num_hit_obstacle", "num_get_goal"]
    )
    assert isinstance(episode_states, dict)
    env = trainer.cuda_envs.env  # 獲取環境物件
    fig_width = 10  # 圖表寬度
    fig_height = 10  # 圖表高度

    # 找到達到目標或碰到障礙物的索引
    end_index = 300
    goal_deid = -1
    for i in range(env.episode_length):
        if episode_states["num_get_goal"][i] == 1 or episode_states["num_hit_obstacle"][i] == 1:
            end_index = i
            if episode_states["num_get_goal"][i] == 1:
                goal_deid = 1
            if episode_states["num_hit_obstacle"][i] == 1:
                goal_deid = 0
            break

    # 創建存放圖片的資料夾
    if not os.path.exists(f"spot_image"):
        os.makedirs(f"spot_image")

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))  # 創建 matplotlib 圖表
    ax = fig.add_subplot(projection="3d")  # 設置為 3D 圖表

    # 隱藏坐標軸刻度
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    # 隱藏坐標軸
    ax.set_axis_off()

    z_values = env.map_z  # 獲取 z 軸的值
    # 標準化 z 值以進行顏色映射
    norm = Normalize(vmin=-1 * env.grid_length_height *
                     2, vmax=env.grid_length_height * 1.5)
    colors = Greys(1 - norm(z_values))  # 使用灰度顏色映射

    # 繪製表面並應用顏色
    ax.plot_surface(env.map_x, env.map_y, env.map_z,
                    facecolors=colors, rstride=1, cstride=1, alpha=0.2)
    rate =  2 * env.grid_length_height / env.grid_length
    ax.set_box_aspect([1, 1, rate])

    # 設置相機视角
    ax.elev = 40
    ax.azim = -55
    ax.dist = 10

    # 繪製初始數據
    point = [None for _ in range(
        env.num_agents)]  # 初始化智能體的繪圖點
    grid_map_height = env.map_z  # 地圖高度信息
    grid_density_ = env.grid_density  # 網格密度
    max_j = env.map_max_x  # x 軸最大值
    s_ag_x = 0
    g_ag_x = 0
    g_ag_y = 0
    s_ag_y = 0
    g_ag_z = 0
    s_ag_z = 0

    for idx in range(env.num_agents):  # 遍歷每個智能體
        # 計算 Marker 大小
        Markersize = (env.agent_size[idx] *
                      env.agent_size[idx] + 0.02) * 5 * 20

        if idx in env.obstacles:  # 如果是障礙物
            # Initialize loc_x, loc_y, loc_z before the loop
            loc_x = episode_states["loc_x"][0, idx]
            loc_y = episode_states["loc_y"][0, idx]
            loc_z = grid_map_height[min(max_j - 1, int(loc_y / grid_density_)),
                                    min(max_j - 1, int(loc_x / grid_density_))]

            for t in range(1, end_index + 1, 3):
                loc_x = episode_states["loc_x"][t, idx]
                loc_y = episode_states["loc_y"][t, idx]
                loc_z = grid_map_height[min(max_j - 1, int(loc_y / grid_density_)),
                                        min(max_j - 1, int(loc_x / grid_density_))]

                # 繪製障礙物的歷史軌跡，顏色較淺, alpha 隨時間遞增, linestyle='None' 移除連線
                ax.plot3D(
                    [episode_states["loc_x"][t, idx]],  # Plot only the current point
                    [episode_states["loc_y"][t, idx]],
                    [grid_map_height[min(max_j - 1, int(episode_states["loc_y"][t, idx] / grid_density_)),
                                    min(max_j - 1, int(episode_states["loc_x"][t, idx] / grid_density_))]],
                    # 更加明顯的 alpha 差異
                    color=to_rgba(obstacle_color,
                                  alpha=0.35 * t / end_index + 0.1),
                    marker="o",
                    markersize=Markersize,
                    zorder=1,  # 設置障礙物歷史軌跡zorder為1，确保其在最下面
                    linestyle='None'  # 移除連線
                )

            # 繪製障礙物的當前位置，顏色較深
            ax.plot3D(
                [loc_x],
                [loc_y],
                [loc_z],
                color=obstacle_color,
                marker="o",
                markersize=Markersize,
                zorder=2  # 設置障礙物當前位置zorder為2
            )
            point[idx] = ax.plot3D(
                [loc_x],
                [loc_y],
                [loc_z],
                color=obstacle_color,
                marker="o",
                markersize=Markersize,
                zorder=2  # 設置障礙物當前位置zorder為2
            )[0]

        elif idx in env.spots:  # 如果是 Spot
            for t in range(1, end_index + 1, 3):
                loc_x = episode_states["loc_x"][t, idx]
                loc_y = episode_states["loc_y"][t, idx]
                dir_x = math.cos(episode_states["direction"][t, idx])
                dir_y = math.sin(episode_states["direction"][t, idx])
                loc_z = grid_map_height[min(max_j - 1, int(loc_y / grid_density_)),
                                        min(max_j - 1, int(loc_x / grid_density_))]

                # 繪製 Spot 的歷史軌跡，顏色較淺，alpha 隨時間遞增
                ax.plot3D(
                    [loc_x],
                    [loc_y],
                    [loc_z],
                    # 調整 alpha 值，拉開差距
                    color=to_rgba(spot_color,
                                  alpha=0.35 * t / end_index + 0.2),
                    marker="o",
                    markersize=Markersize,
                    zorder=3  # 設置Spot歷史軌跡zorder為3
                )

                # 繪製 Spot 的 head，顏色較淺，alpha 隨時間遞增
                ax.plot3D(
                    [loc_x + dir_x * env.spot_head_distance],
                    [loc_y + dir_y * env.spot_head_distance],
                    [loc_z],
                    # 調整 alpha 值，拉開差距
                    color=to_rgba(spot_direction_color,
                                  alpha=0.35 * t / end_index + 0.2),
                    marker="o",
                    markersize=Markersize * 0.7,
                    zorder=4  # 設置Spot頭部歷史軌跡zorder為4
                )

                # 繪製 Spot 的 ass，顏色較淺，alpha 隨時間遞增
                ax.plot3D(
                    [loc_x - dir_x * env.spot_ass_distance],
                    [loc_y - dir_y * env.spot_ass_distance],
                    [loc_z],
                    # 調整 alpha 值，拉開差距
                    color=to_rgba(spot_color,
                                  alpha=0.35 * t / end_index + 0.2),
                    marker="o",
                    markersize=Markersize,
                    zorder=4  # 設置Spot尾部歷史軌跡zorder為4
                )

            #  繪製 Spot 的當前位置，顏色較深 (loop 外面只畫一次)
            loc_x = episode_states["loc_x"][end_index, idx]
            loc_y = episode_states["loc_y"][end_index, idx]
            dir_x = math.cos(episode_states["direction"][end_index, idx])
            dir_y = math.sin(episode_states["direction"][end_index, idx])
            loc_z = grid_map_height[min(max_j - 1, int(loc_y / grid_density_)),
                                    min(max_j - 1, int(loc_x / grid_density_))]

            s_ag_x = episode_states["loc_x"][0, idx]
            s_ag_y = episode_states["loc_y"][0, idx]
            s_ag_z = grid_map_height[min(max_j - 1, int(s_ag_y / grid_density_)),
                                    min(max_j - 1, int(s_ag_x / grid_density_))]

            ax.plot3D(
                [loc_x],  # 取最後一個時間點的位置
                [loc_y],
                [loc_z],
                color=spot_color,
                marker="o",
                markersize=Markersize,
                zorder=5  # 設置Spot當前位置zorder為5
            )

            # 繪製 Spot 的 head (最後一個時間點)
            ax.plot3D(
                [loc_x + dir_x * env.spot_head_distance],
                [loc_y + dir_y * env.spot_head_distance],
                [loc_z],
                color=spot_direction_color,
                marker="o",
                markersize=Markersize * 0.7,
                zorder=6  # 設置Spot頭部當前位置zorder為6
            )

            # 繪製 Spot 的 ass (最後一個時間點)
            ax.plot3D(
                [loc_x - dir_x * env.spot_ass_distance],
                [loc_y - dir_y * env.spot_ass_distance],
                [loc_z],
                color=spot_color,
                marker="o",
                markersize=Markersize,
                zorder=6  # 設置Spot尾部當前位置zorder為6
            )

        else:  # 如果是目標
            # 繪製目標的歷史軌跡
            for t in range(1, end_index + 1, 3):
                loc_x = episode_states["loc_x"][t, idx]
                loc_y = episode_states["loc_y"][t, idx]
                loc_z = grid_map_height[min(max_j - 1, int(loc_y / grid_density_)),
                                        min(max_j - 1, int(loc_x / grid_density_))]
                ax.plot3D(
                    [loc_x],
                    [loc_y],
                    [loc_z],
                    color=to_rgba(goal_color,
                                  alpha=0.35 * t / end_index + 0.1),
                    marker="o",
                    markersize=Markersize,
                    zorder=4,  # 设置goal历史轨迹zorder为4，Spot历史轨迹之上
                    linestyle='None'  # 移除连线
                )
            # 繪製目標的當前位置
            point[idx] = ax.plot3D(
                episode_states["loc_x"][end_index - 1, idx],
                episode_states["loc_y"][end_index - 1, idx],
                grid_map_height[min(max_j - 1, int(episode_states["loc_y"]
                                                [end_index - 1, idx] / grid_density_)),
                                min(max_j - 1, int(episode_states["loc_x"]
                                                [end_index - 1, idx] / grid_density_))],
                color=goal_color,
                marker="o",
                markersize=Markersize,
                zorder=5  # 设置goal的zorder为5，最终Spot之下
            )[0]

            g_ag_x = episode_states["loc_x"][end_index - 1, idx]
            g_ag_y = episode_states["loc_y"][end_index - 1, idx]
            g_ag_z = grid_map_height[min(max_j - 1, int(g_ag_y / grid_density_)),
                                    min(max_j - 1, int(g_ag_x / grid_density_))]

    # Place the goal_dis calculation here, outside the loop
    # print("goal_dis", goal_dis, " ",s_ag_x,g_ag_x,(abs(s_ag_x - g_ag_x)**2), s_ag_y, g_ag_y)
    goal_dis = ((abs(s_ag_x - g_ag_x)**2) + (abs(s_ag_y - g_ag_y)**2) +
                (abs(s_ag_z - g_ag_z)**2)) ** 0.5
    if env.Training:
        text = "Training"
    else:
        text = "Testing"

    def _get_label(timestep, n_goals_alive, n_hit_obstacle):
        line1 = "Spot 3D " + text + "\n"
        line2 = "Time Step:".ljust(14) + f"{timestep:2.0f}\n"
        still_in = n_goals_alive
        pct_goals_alive = f"{still_in:2}"
        line3 = "Get Goals:".ljust(14) + pct_goals_alive
        line4 = "\nhit the obstacle times: ".ljust(14) + \
            f"{n_hit_obstacle:2}"
        line5 = "\ngrid_length: ".ljust(14) + f"{env.grid_length:2}"
        return line1 + line2 + line3 + line4 + line5

    # 在圖表上添加文字信息
    label = ax.text(
        0,
        0,
        2.5,
        _get_label(end_index, int(episode_states["num_get_goal"][end_index]),
                   int(episode_states["num_hit_obstacle"][end_index])).lower(),
    )

    label.set_fontsize(14)
    label.set_fontweight("normal")
    label.set_color("#666666")

    save_path = "custom_model/image/" + wandb_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(
        f"{save_path}/seed_{state}_surcess_{goal_deid}_distance_{str(goal_dis)}_timestep_{end_index}_phase_{PHASE_}_testid_{time_idx}_.png")
    plt.close()  # 关闭图表

    del env  # 删除环境对象释放内存
    return wandb_table_history