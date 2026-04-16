"""
Helper file for generating an environment rollout
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import art3d
import wandb
from wandb.keras import WandbCallback


def generate_rollout_wandb(
    trainer,
    wandb_dir,
    wandb_file,
    fps=10,
    tagger_color="#C843C3",
    wall_color='black',
    goal_color="#245EB6",
    goal_not_in_game_color="#666666",
):
    assert trainer is not None
    assert wandb_dir is not None
    assert wandb_file is not None
    online_wandb = wandb.init(project = wandb_dir)
    online_wandb.name = wandb_file
    reg = {}
    episode_states = trainer.fetch_episode_states(
        ["loc_x", "loc_y", "still_in_the_game", "num_hit_wall","num_get_goal" ]
    )
    assert isinstance(episode_states, dict)
    env = trainer.cuda_envs.env

    # Plot init data
    lines = [None for _ in range(env.num_agents)]
    init_count_goal= env.num_goals
    init_hit_wall = env.num_hit_wall
    

    for idx in range(env.num_agents):

        if idx in env.taggers:
            lines[idx] = ax.plot3D(
                episode_states["loc_x"][:1, idx] / env.grid_length,
                episode_states["loc_y"][:1, idx] / env.grid_length,
                0,
                color=tagger_color,
                marker="o",
                markersize = Markersize,
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
        else:
            lines[idx] = ax.plot3D(
                episode_states["loc_x"][:1, idx] / env.grid_length,
                episode_states["loc_y"][:1, idx] / env.grid_length,
                [0],
                color=goal_color,
                marker="o",
                markersize=7,
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
        #label.set_text(_get_label(i, int(episode_states["num_get_goal"][i]), int(episode_states["num_hit_wall"][i])).lower())
    ani = animation.FuncAnimation(
        fig, animate, np.arange(0, env.episode_length + 1), interval=1000.0 / fps
    )
    plt.close()

    return ani
