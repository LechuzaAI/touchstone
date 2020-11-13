import argparse

import gym

from touchstone.modules import PPOLightning


def main(params) -> None:
    agent =PPOLightning.load_from_checkpoint('tb_logs/default/version_0/checkpoints/epoch=99999.ckpt')

    print(agent.hparams)
    agent.eval()
    agent.env.close()

    env = gym.make(params.env)

    state = env.reset()
    done = False

    total_reward = 0

    while not done:
        value, action_dist = agent(state)
        action = action_dist.mean
        state, reward, done, _ = env.step(action.detach().numpy())
        total_reward += reward
        env.render()

    print(total_reward)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
parser.add_argument("--betas", type=tuple, default=(0.9, 0.999), help="betas for ADAM")
parser.add_argument("--env", type=str, default="Pendulum-v0", help="gym environment tag")
parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
parser.add_argument("--sync_rate", type=int, default=10,
                    help="how many frames do we update the target network")
parser.add_argument("--replay_size", type=int, default=1000,
                    help="capacity of the replay buffer")
parser.add_argument("--warm_start_size", type=int, default=1000,
                    help="how many samples do we use to fill our buffer at the start of training")
parser.add_argument("--eps_last_frame", type=int, default=1000,
                    help="what frame should epsilon stop decaying")
parser.add_argument("--eps_start", type=float, default=1.0, help="starting value of epsilon")
parser.add_argument("--eps_end", type=float, default=0.01, help="final value of epsilon")
parser.add_argument("--episode_length", type=int, default=200, help="max length of an episode")
parser.add_argument("--max_episode_reward", type=int, default=200,
                    help="max episode reward in the environment")
parser.add_argument("--warm_start_steps", type=int, default=1000,
                    help="max episode reward in the environment")
parser.add_argument("--time_steps", type=int, default=512,
                    help="time steps for collecting data")
parser.add_argument("--num_actors", type=int, default=1,
                    help="number of actors to run in parallel")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--iterations", type=int, default=10000)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--clip_param", type=float, default=0.1)
parser.add_argument("--action_std", type=float, default=1.25)

args, _ = parser.parse_known_args()

if __name__ == '__main__':
    main(args)
