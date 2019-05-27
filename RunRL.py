import argparse
import sys

from RL_implementations.policy_gradients import runPolicyGradients
from RL_implementations.DDQN import DDQN
from RL_implementations.DDQN_NEW import DDQN1
from RL_implementations.benchmark import Dijkstras

# Broken:
# from q_learning import Q
# from deep_q import main


class RunPolicyGradients:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Run Policy Gradients Deep RL on any SNDlib Network Architecture"
        )
        # parser.add_argument("-H", "--Help", help="Prints custom help text", required=False, default="")
        parser.add_argument(
            "-n",
            "--network",
            help="Select a network, should be the filename without the .xml extension, defaults to germany50",
            required=False,
            default="germany50",
        )
        parser.add_argument(
            "-r",
            "--render",
            help="whether to visualize the network during training (will considerably slow down training), to render simply put -r as a command line argument, no need to type True after",
            required=False,
            action="store_true",
            default=False,
        )

        # mode is currently broken/irrelevant, will always render as human currently:
        parser.add_argument(
            "-m",
            "--mode",
            help="'human' or 'rgb_array', the render mode to use if render==True",
            required=False,
            default="human",
        )

        parser.add_argument(
            "-l",
            "--logging",
            help="Logging level to use for current run, recommended and default level: 'INFO'",
            required=False,
            default="INFO",
        )

        parser.add_argument(
            "-s",
            "--seed",
            help="Seed to use for selecting the random start and end nodes, keeping this a constant integer will ensure the start and end nodes chosen are the same. Set to None if you want them to change each step. Default value 1",
            required=False,
            default=0,
        )
        parser.add_argument(
            "-e",
            "--episodes",
            help="Int, Number of episodes to run for, defaults to 10,000",
            required=False,
            type=int,
            default=10000,
        )

        parser.add_argument(
            "-a",
            "--algorithm",
            help="Reinforcement learning algorithm to use, e.g. 'policy_gradients', 'ddqn',...",
            required=False,
            default="ddqn",
        )

        parser.add_argument(
            "-env",
            "--env",
            help="OpenAI Gym environment to use, currently a choice between BasicPacketRoutingEnv-v0 and PathFindingNetworkEnv-v1. Defaults to PathFindingNetworkEnv-v1",
            required=False,
            default="PathFindingNetworkEnv-v1"
        )
        
        args = parser.parse_args()

        if args.render:
            print(args.render)

        if args.algorithm == "policy_gradients":
            runPolicyGradients(
                env=args.env,
                network=args.network,
                render=args.render,
                mode=args.mode,
                log_level=args.logging,
                seed=args.seed,
                num_episodes=args.episodes,
            )

        elif args.algorithm == "ddqn":
            DDQN(
                env=args.env,
                network=args.network,
                render=args.render,
                mode=args.mode,
                log_level=args.logging,
                seed=args.seed,
                num_episodes=args.episodes,
            )


        elif args.algorithm == "ddqn_new":
            DDQN1(
                env=args.env,
                network=args.network,
                render=args.render,
                mode=args.mode,
                log_level=args.logging,
                seed=args.seed,
                num_episodes=args.episodes,
            )

        elif args.algorithm == "dijkstras":
            Dijkstras(
                env=args.env,
                network=args.network,
                render=args.render,
                mode=args.mode,
                log_level=args.logging,
                seed=args.seed
            )

if __name__ == "__main__":
    policyGradients = RunPolicyGradients()
