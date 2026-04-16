"""
Helper file for generating an environment rollout
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Polygon
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import art3d
import wandb
from wandb.keras import WandbCallback
import csv
import os
import json




def generate_tag_env_rollout_animation(
    trainer,
    fps=10,
    tagger_color="#C843C3",
    wall_color='black',
    goal_color="#245EB6",
    goal_not_in_game_color="#666666",
):
    assert trainer is not None

    episode_states = trainer.fetch_episode_states(
        ["loc_x", "loc_y", "still_in_the_game", "num_hit_wall","num_get_goal" ]
    )
    assert isinstance(episode_states, dict)
    env = trainer.cuda_envs.env
    fig_width = 10
    fig_height = 10
    fig, ax = plt.subplots(
        1, 1, figsize=(fig_width, fig_height)
    )  # , constrained_layout=True

    fig_2d, ax_2d = plt.subplots(
        1, 1, figsize=(fig_width, fig_height)
    )  



    ax.remove()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Bounds
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(-0.01, 0.01)

    ax_2d.set_xlim(-0.05, 1.05)
    ax_2d.set_ylim(-0.05, 1.05)
    ax_2d.set_alpha(.7)
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
    lines_2d = [None for _ in range(env.num_agents)]
    init_count_goal= env.num_goals
    init_hit_wall = env.num_hit_wall
    

    for idx in range(env.num_agents):

        if env.phase ==1:
            Markersize = 10
        else:
            Markersize = (env.agent_size[idx] * env.agent_size[idx] + 0.02) * 4 * 7
        if idx in env.taggers:
            lines[idx] = ax.plot3D(
                episode_states["loc_x"][:1, idx] / env.grid_length,
                episode_states["loc_y"][:1, idx] / env.grid_length,
                0,
                color=tagger_color,
                marker="o",
                markersize = Markersize,
            )[0]
            # 2D graph
            lines_2d[idx] = ax_2d.plot(
                episode_states["loc_x"][:1, idx] / env.grid_length,
                episode_states["loc_y"][:1, idx] / env.grid_length,
                color=tagger_color,
                marker="o",
                markersize = Markersize,
                alpha = 0.0,
            )[0]
        elif idx in env.walls:
            lines[idx] = ax.plot3D(
                episode_states["loc_x"][:1, idx] / env.grid_length,
                episode_states["loc_y"][:1, idx] / env.grid_length,
                [0],
                color=wall_color,
                marker="o",
                markersize = Markersize,
            )[0]
            # 2D graph
            lines_2d[idx] = ax_2d.plot(
                episode_states["loc_x"][:1, idx] / env.grid_length,
                episode_states["loc_y"][:1, idx] / env.grid_length,
                color=wall_color,
                marker="o",
                markersize = Markersize,
                alpha = 0.4,
            )[0]
        else:
            lines[idx] = ax.plot3D(
                episode_states["loc_x"][:1, idx] / env.grid_length,
                episode_states["loc_y"][:1, idx] / env.grid_length,
                [0],
                color=goal_color,
                marker="o",
                markersize=7,
            )[0]
            # 2D graph
            lines_2d[idx] = ax_2d.plot(
                episode_states["loc_x"][:1, idx] / env.grid_length,
                episode_states["loc_y"][:1, idx] / env.grid_length,
                color='w',
                marker="o",
                markersize = 10,
                alpha = 0.0,
                markerfacecolor=goal_color,
            )[0]

    if env.Training:
        text = "Training"
    else:
        text = "Testing"

    def _get_label(timestep, n_goals_alive, n_hit_wall):
        line1 = "final Car " + text + "\n"
        line2 = "Time Step:".ljust(14) + f"{timestep:4.0f}\n"
        still_in = n_goals_alive
        pct_goals_alive = f"{still_in:4}" 
        line3 = "Get Goals:".ljust(14) + pct_goals_alive
        line4 = "\nhit the wall times: ".ljust(14) + f"{n_hit_wall:4}"
        return line1 + line2 + line3 + line4


    label = ax.text(
        0,
        0,
        0.02,
        _get_label(0, init_count_goal, init_hit_wall).lower(),
    )

    label.set_fontsize(14)
    label.set_fontweight("normal")
    label.set_color("#666666")

    def animate(i):
        # i = time step
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
                line.set_color(goal_not_in_game_color)
                line.set_marker("")
        label.set_text(_get_label(i, int(episode_states["num_get_goal"][i]), int(episode_states["num_hit_wall"][i])).lower())

    ani = animation.FuncAnimation(
        fig, animate, np.arange(0, env.episode_length + 1), interval=1000.0 / fps
    )

    # 2D graph generate
    for j in range(0, env.episode_length + 1):
        for idx, line in enumerate(lines_2d):
            line.set_data(
                episode_states["loc_x"][ 400: 600: 4 , idx] / env.grid_length,
                episode_states["loc_y"][ 400: 600: 4, idx] / env.grid_length,
            )

            still_in_game = episode_states["still_in_the_game"][j, idx]

            if still_in_game:
                pass
            else:
                line.set_color(goal_not_in_game_color)
                line.set_marker("")
            
    tagger_legend = mlines.Line2D([0], [0], marker='o', color='w', label='Tagger',markerfacecolor=tagger_color, markersize=10)
    wall_legend = mlines.Line2D([0], [0], marker='o', color='w', label='Wall',markerfacecolor=wall_color, markersize=10)
    goal_legend = mlines.Line2D([0], [0], marker='o', color='w', label='Goal',markerfacecolor=goal_color, markersize=10)

    # ax_2d.legend(handles=[tagger_legend, wall_legend, goal_legend], loc='upper right')
    # ax_2d.set_title("Pedestrian model trajectory")
    ax_2d.set_title("Random obstacle trajectory")
    ax_2d.set_xlabel("x")
    ax_2d.set_ylabel("y")
    ax_2d.grid()
    # fig_2d.savefig('./custom_model/final_result/png/test.png')
    
    plt.close()

    wandb.log({
        "collision": int(episode_states["num_hit_wall"][env.episode_length]),
        "Get goal": int(episode_states["num_get_goal"][env.episode_length])
    })

    return ani
