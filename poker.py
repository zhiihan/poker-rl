import argparse
import os
from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter
import time

import torch
import random
import numpy as np
import supersuit as ss
import gymnasium as gym
from pettingzoo.classic import texas_holdem_no_limit_v6
from pettingzoo.utils import save_observation
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import time
from agents import QLearningAgent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gym-name",
        type=str,
        help="the envrionment to train the agent",
        default="texas_holdem_no_limit_v6",
    )
    parser.add_argument(
        "--rl-agent", type=str, help="the type of algorithm to use", default="ppo"
    )
    parser.add_argument(
        "--learning-rate", type=float, help="the learning rate", default=10**-4
    )
    parser.add_argument("--seed", type=int, help="the seed", default=42)
    parser.add_argument(
        "--timesteps", type=int, help="the total timesteps for the agent", default=25000
    )
    parser.add_argument(
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--cuda", default=True, type=lambda x: bool(strtobool(x)), nargs="?", const=True
    )
    parser.add_argument(
        "--num_envs",
        default=16,
        type=int,
        nargs="?",
        const=True,
        help="the number of environments to run in parallel",
    )
    parser.add_argument(
        "--track",
        default=False,
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        help="use wandb to track the agents training progress for cloud implementations.",
    )
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    run_name = f"{args.gym_name}__{args.rl_agent}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        from dotenv import load_dotenv

        load_dotenv()

        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            sync_tensorboard=True,
            config=vars(args),
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "param|value|\n|-|-|\n %s"
        % ("\n".join([f"{key}|{value}" for key, value in vars(args).items()])),
    )
    for i in range(100):
        writer.add_scalar("test_loss", i * 2, global_step=i)

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = texas_holdem_no_limit_v6.env(num_players=2, render_mode="ansi")
    env.reset(seed=42)

    my_agents = [QLearningAgent(), QLearningAgent()]

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            mask = observation["action_mask"]

            if agent == "player_0":
                action_array = my_agents[0].get_action_value(observation, mask)
                print(f"Player 0 actions = {action_array}")
            elif agent == "player_1":
                action_array = my_agents[1].get_action_value(observation, mask)
                print(f"Player 1 actions = {action_array}")

            with torch.no_grad():
                # print(observation, reward, termination, truncation, info)
                action_array = torch.Tensor(mask) * action_array
                action = int(torch.argmax(action_array))

        env.step(action)
    env.close()
    writer.close()
