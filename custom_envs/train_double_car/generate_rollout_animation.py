"""
Helper file for generating an environment rollout
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import art3d
import wandb


def generate_tag_env_rollout_animation(
    trainer,
    fps=10,
    spot_color='gold',
    spot_direction_color='orange',
    obstacle_color='black',
    goal_color='lightgreen',
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
    car1_color = spot_color
    car2_color = 'red'
    wall_color = obstacle_color
    not_in_game_color = goal_not_in_game_color
    assert trainer is not None

    episode_states = trainer.fetch_episode_states(
        ["loc_x", "loc_y", "still_in_the_game"]
    )
    assert isinstance(episode_states, dict)
    env = trainer.cuda_envs.env
    fig_width = 10
    fig_height = 10
    fig, ax = plt.subplots(
        1, 1, figsize=(fig_width, fig_height)
    )  # , constrained_layout=True
    ax.remove()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Bounds
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(-0.01, 0.01)

    # Surface
    corner_points = [(0, 0), (0, 1), (1, 1), (1, 0)]

    poly = Polygon(corner_points, color=(0.1, 0.2, 0.5, 0.15))
    ax.add_patch(poly)
    art3d.pathpatch_2d_to_3d(poly, z=0, zdir="z")

    # "Hide" side panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Hide axes
    ax.set_axis_off()

    # Set camera
    ax.elev = 40
    ax.azim = -55
    ax.dist = 10

    # Try to reduce whitespace
    fig.subplots_adjust(left=0, right=1, bottom=-0.2, top=1)

    # Plot init data
    lines = [None for _ in range(env.num_agents)]
    init_count_goal = env.num_taggers + env.testing_num_goals
    

    for idx in range(env.num_agents):
        if env.phase == 1:
            Markersize = 10
        else:
            Markersize = (env.agent_size[idx] * env.agent_size[idx] + 0.02) * 4 * 7

        if idx in env.taggers:
            if env.agent_type[idx] == 0:
                lines[idx] = ax.plot3D(
                    episode_states["loc_x"][:1, idx] / env.grid_length,
                    episode_states["loc_y"][:1, idx] / env.grid_length,
                    0,
                    color = car1_color,
                    marker="o",
                    markersize = Markersize,
                )[0]
            else:
                lines[idx] = ax.plot3D(
                    episode_states["loc_x"][:1, idx] / env.grid_length,
                    episode_states["loc_y"][:1, idx] / env.grid_length,
                    0,
                    color = car2_color,
                    marker="o",
                    markersize = Markersize,
                )[0]
        elif idx in env.walls:
            lines[idx] = ax.plot3D(
                episode_states["loc_x"][:1, idx] / env.grid_length,
                episode_states["loc_y"][:1, idx] / env.grid_length,
                0,
                color = wall_color,
                marker="o",
                markersize = Markersize,
            )[0]
        else:
            lines[idx] = ax.plot3D(
                episode_states["loc_x"][:1, idx] / env.grid_length,
                episode_states["loc_y"][:1, idx] / env.grid_length,
                0,
                color = goal_color,
                marker="o",
                markersize=7,
            )[0]

    if env.Training:
        text = "Training"
    else:
        text = "Testing"

    def _get_label(timestep, n_goals_alive, init_n_goals):

        line1 = "Double Car " + text + "\n"
        line2 = "Time Step:".ljust(14) + f"{timestep:4.0f}\n"
        still_in = init_n_goals - n_goals_alive
        frac_goals_alive = still_in / init_n_goals
        pct_goals_alive = f"{still_in:4} ({frac_goals_alive * 100:.0f}%)"
        line3 = "Get Goals:".ljust(14) + pct_goals_alive
        return line1 + line2 + line3


    label = ax.text(
        0,
        0,
        0.02,
        _get_label(0, init_count_goal, init_count_goal).lower(),
    )

    label.set_fontsize(14)
    label.set_fontweight("normal")
    label.set_color("#666666")

    def animate(i):
        # i = time step
        count_goal = env.num_taggers + env.testing_num_goals
        for idx, line in enumerate(lines):
            line.set_data_3d(
                episode_states["loc_x"][i : i + 1, idx] / env.grid_length,
                episode_states["loc_y"][i : i + 1, idx] / env.grid_length,
                np.zeros(1),
            )

            still_in_game = episode_states["still_in_the_game"][i, idx]

            if still_in_game:
                pass
            else:
                line.set_color(not_in_game_color)
                line.set_marker("")
        for x in range(env.num_agents):
            if episode_states["still_in_the_game"][i][x] == 0:
                count_goal -=1
        label.set_text(_get_label(i, count_goal, init_count_goal).lower())

    ani = animation.FuncAnimation(
        fig, animate, np.arange(0, env.episode_length + 1), interval=1000.0 / fps
    )
    if use_wandb:
        writer = animation.FFMpegWriter(fps=10)
        mp4_name = "./custom_model/video/spot_state_" + str(state) + ".mp4"
        ani.save(mp4_name, writer)
        del writer
        Pre_train = False

        if state % phase_env_num == 0:
            wandb_state = PHASE_ * phase_env_num + 1
        else:
            wandb_state = (PHASE_ - 1) * phase_env_num + state % phase_env_num + 1
        wandb_table.add_data(
            wandb_state,
            wandb.Video(mp4_name, fps = 10),
            wandb_name,
            Pre_train,
            PHASE_,
            "no data",
            num_goals,
            "no data"
        )
        wandb.log({"Last training situation": wandb_table})  
        wandb_table_history.append(
            [
                wandb_state,
                wandb.Video(mp4_name, fps = 10),
                wandb_name,
                Pre_train,
                PHASE_,
                "no data",
                num_goals,
                "no data"
            ]
        )

    del env
    plt.close()

    return ani
