import numpy as np
from tqdm.auto import tqdm

def simulate(env, estimator, episodes = 1000):
    observation, info = env.reset()

    data = np.zeros(episodes)
    
    print("TQDM")
    for episode in tqdm(range(episodes), "Episodes"):
        episode_length = 0
        end = False

        print("while")
        while not end:
            action = estimator.choose_action(observation)

            _observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                reward = 0
            episode_length += reward

            estimator.update(observation, action, _observation, reward)
            observation = _observation

            end = terminated or truncated

        data[episode] = episode_length
        observation, info = env.reset()

    return data