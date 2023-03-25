import load_script
import gymnasium as gym
import simulate

env = gym.make("CartPole-v1", render_mode="human")
simulate.simulate(env, load_script.actor_critic, episodes=1)