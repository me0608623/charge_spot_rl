"""
Helper file for generating an environment rollout
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import art3d
import math
import wandb


def generate_tag_env_rollout_animation(
    trainer,
    fps=10,
    spot_color='yellow',
    spot_direction_color='deepskyblue',
    obstacle_color='black',
    goal_color='fuchsia',
    goal_not_in_game_color="#666666",
    Training = True,
    use_wandb = False,
    state = 0,
    phase_env_num = 300,
    wandb_table = None,
    PHASE_ = 1,
    pre_train = None,
    wandb_name = None,
    wandb_table_history = None,
    num_goals = 15,
):
    assert trainer is not None
    episode_states = trainer.fetch_episode_states(
        ["loc_x", "loc_y", "direction", "still_in_the_game", "num_hit_obstacle","num_get_goal"]
    )
    assert isinstance(episode_states, dict)
    env = trainer.cuda_envs.env
    fig_width = 10
    fig_height = 10
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))  # , constrained_layout=True
    ax = fig.add_subplot(projection="3d")
    
    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    # Hide axes
    ax.set_axis_off()
    ax.plot_wireframe(env.yy, env.xx, env.grid_map_z, color='gainsboro')

    # Set camera
    ax.elev = 40
    ax.azim = -55
    ax.dist = 10


    # Plot init data
    lines = [None for _ in range(env.num_agents)]
    lines2 = [None for _ in range(env.num_spot)]
    init_count_goal= env.num_goals
    init_hit_obstacle = env.num_hit_obstacle
    grid_map_height = env.grid_map_z
    grid_density_ = env.grid_density

    spot_list = []
    num_spot_reg = 0

    for idx in range(env.num_agents):
        Markersize = (env.agent_size[idx] * env.agent_size[idx] + 0.02) * 5 * 20

        if idx in env.spots:
            loc_x = episode_states["loc_x"][:1, idx]
            loc_y = episode_states["loc_y"][:1, idx] 
            loc_z = grid_map_height[int(loc_x / grid_density_), int(loc_y / grid_density_)]

            lines[idx] = ax.plot3D(
                loc_x,
                loc_y,
                loc_z,
                color=spot_color,
                marker="o",
                markersize = Markersize,
            )[0]
            spot_list.append(idx)
            lines2[num_spot_reg] = ax.plot3D(
                loc_x,
                loc_y,
                loc_z,
                color=spot_direction_color,
                marker="o",
                markersize = int(Markersize * 0.5),
            )[0]
            lines2[num_spot_reg].set_zorder(10)
            num_spot_reg += 1
            

        elif idx in env.obstacles:
            lines[idx] = ax.plot3D(
                episode_states["loc_x"][:1, idx],
                episode_states["loc_y"][:1, idx],
                grid_map_height[int(episode_states["loc_x"][:1, idx] / grid_density_), int(episode_states["loc_y"][:1, idx] / grid_density_)],
                color = obstacle_color,
                marker = "o",
                markersize = Markersize,
            )[0]
        else:
            lines[idx] = ax.plot3D(
                episode_states["loc_x"][:1, idx],
                episode_states["loc_y"][:1, idx],
                grid_map_height[int(episode_states["loc_x"][:1, idx] / grid_density_), int(episode_states["loc_y"][:1, idx] / grid_density_)],
                color = goal_color,
                marker = "o",
                markersize = Markersize,
            )[0]
        lines[idx].set_zorder(9)


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
        line4 = "\nhit the obstacle times: ".ljust(14) + f"{n_hit_obstacle:2}"
        line5 = "\ngrid_length: ".ljust(14) + f"{env.grid_length:2}"
        return line1 + line2 + line3 + line4 + line5


    label = ax.text(
        0,
        0,
        2.5,
        _get_label(0, init_count_goal, init_hit_obstacle).lower(),
    )

    label.set_fontsize(14)
    label.set_fontweight("normal")
    label.set_color("#666666")

    def animate(i):

        for idx, line in enumerate(lines):

            still_in_game = episode_states["still_in_the_game"][i, idx]

            line.set_data_3d(
                (episode_states["loc_x"][i : i + 1, idx]),
                (episode_states["loc_y"][i : i + 1, idx]),
                ([grid_map_height[int(episode_states["loc_x"][i : i + 1, idx] / grid_density_), int(episode_states["loc_y"][i : i + 1, idx] / grid_density_)]]),
            )

            if still_in_game:
                pass
            else:
                line.set_color(goal_not_in_game_color)
                line.set_marker("")

        for idx, line in enumerate(lines2):
            spot_idx = spot_list[idx]
            dir_x = math.cos(episode_states["direction"][i : i + 1, spot_idx]) * 0.2
            dir_y = math.sin(episode_states["direction"][i : i + 1, spot_idx]) * 0.2
            line.set_data_3d(
                (episode_states["loc_x"][i : i + 1, spot_idx] + dir_x),
                (episode_states["loc_y"][i : i + 1, spot_idx] + dir_y),
                ([grid_map_height[int(episode_states["loc_x"][i : i + 1, spot_idx] / grid_density_), int(episode_states["loc_y"][i : i + 1, spot_idx] / grid_density_)]]),
            )
            still_in_game = episode_states["still_in_the_game"][i, spot_idx]

            if still_in_game:
                pass
            else:
                line.set_color(goal_not_in_game_color)
                line.set_marker("")
        
        label.set_text(_get_label(i, int(episode_states["num_get_goal"][i]), int(episode_states["num_hit_obstacle"][i])).lower())
    ani = animation.FuncAnimation(
        fig, animate, np.arange(0, env.episode_length + 1), interval=1000.0 / fps
    )
    
    if use_wandb:
        writer = animation.FFMpegWriter(fps=10)
        mp4_name = "./custom_model/video/spot_state_" + str(state) + ".mp4"
        ani.save(mp4_name, writer)
        del writer
        Pre_train = False
        if pre_train is not None:
            if pre_train.split('_')[1] != "final":
                Pre_train = True
        if state % phase_env_num == 0:
            wandb_state = PHASE_ * phase_env_num
        else:
            wandb_state = (PHASE_ - 1) * phase_env_num + state % phase_env_num
            
        wandb_table.add_data(
            wandb_state,
            wandb.Video(mp4_name, fps = 10),
            wandb_name,
            Pre_train,
            PHASE_,
            int(episode_states["num_get_goal"][-1]),
            num_goals,
            int(episode_states["num_hit_obstacle"][-1])
        )
        if Training: wandb.log({"Last training situation": wandb_table})  
        wandb_table_history.append(
            [
                wandb_state,
                wandb.Video(mp4_name, fps = 10),
                wandb_name,
                Pre_train,
                PHASE_,
                int(episode_states["num_get_goal"][-1]),
                num_goals,
            int(episode_states["num_hit_obstacle"][-1])
            ]
        )

    del env   
    plt.close()
    return ani, wandb_table_history
