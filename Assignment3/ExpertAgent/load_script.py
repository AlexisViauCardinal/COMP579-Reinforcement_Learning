import numpy as np
import gymnasium as gym
import random
import scipy

# DISCRETIZATION
BINS = 11
ANGLE_SOFT_LIMIT = np.radians(12)

class Discretizer:
    def __init__(self):
        self.position = np.linspace(-2.4, 2.4, BINS)
        self.velocity = np.linspace(-3, 3, BINS)
        self.angle = np.linspace(-ANGLE_SOFT_LIMIT, ANGLE_SOFT_LIMIT, BINS)
        self.angular_velocity = np.linspace(-3.5, 3.5, BINS)

    def discretize(self, state):
        _position, _velocity, _angle, _angular_velocity = state

        position = np.digitize(_position, self.position)
        velocity = np.digitize(_velocity, self.velocity)
        angle = np.digitize(_angle, self.angle)
        angular_velocity = np.digitize(_angular_velocity, self.angular_velocity)

        return tuple([position-1, velocity-1, angle-1, angular_velocity-1])
    

# ESTIMATOR
MIN = -0.01
MAX =  0.01

class TabularEstimator:
    def __init__(self, discretizer):
        self.table = np.random.uniform(low=MIN, high=MAX, size=[BINS-1]*4)
        self.discretizer = discretizer

    def evaluate(self, state):
        discrete = self.discretizer.discretize(state)
        if -1 == min(discrete) or BINS-1 == max(discrete):
            return 0
        return self.table[discrete]

    def update(self, state, add):
        discrete = self.discretizer.discretize(state)
        if -1 == min(discrete) or BINS-1 == max(discrete):
            return
        self.table[discrete] += add

# Q-LEARNING
class QLearning:
    def __init__(self, action_space, discretizer, estimator, alpha, gamma, alpha_decay = lambda x: x, epsilon = 0.1, epsilon_decay = lambda x : x):
        self.action_space = action_space
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.estimators = {}
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        for action in action_space:
            self.estimators[action] = estimator(discretizer)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            self.epsilon = self.epsilon_decay(self.epsilon)
            return random.choice(self.action_space)
        return self.best_action(state)

    def evaluate(self, state, action):
        return self.estimators[action].evaluate(state)

    def state_value(self, state):
        return self.estimators[self.best_action(state)].evaluate(state)

    def best_action(self, state):
        return max(self.estimators, key=lambda x: self.estimators[x].evaluate(state))

    def update(self, state, action, next_state, reward):
        current_eval = self.evaluate(state, action)
        next_state_eval = self.state_value(next_state)
        self.estimators[action].update(state, self.alpha * (reward + self.gamma*next_state_eval - current_eval))
        self.alpha = self.alpha_decay(self.alpha)


# ACTOR-CRITIC
class Actor:
    def __init__(self, action_space, discretizer, estimator):
        self.approximators = {}
        self.discretizer = discretizer
        for action in action_space:
            self.approximators[action] = estimator(discretizer)

    def evaluate(self, state):
        approximations = {name: approximator.evaluate(state) for name, approximator in self.approximators.items()}
        return dict(zip(approximations.keys(), scipy.special.softmax(np.array(list(approximations.values())))))

    def update(self, state, action, alpha, I, delta):
        eval = self.evaluate(state)
        self.approximators[action].update(state, alpha * I * delta * (1 - eval[action]))
        for _action in self.approximators:
            if _action == action:
                continue
            self.approximators[_action].update(state, - alpha * I * delta * (1 - eval[action]))

class Critic:
    def __init__(self, discretizer, estimator):
        self.estimator = estimator(discretizer)

    def evaluate(self, state):
        return self.estimator.evaluate(state)

    def update(self, state, alpha, delta):
        self.estimator.update(state, alpha * delta)

def sample_from_dict(d):
    a = []
    p = []
    for action, prob in d.items():
        a += [action]
        p += [prob]
    return np.random.choice(a, p=p)

class ActorCritic:
    def __init__(self, action_space, actor, critic, gamma, alpha_theta, alpha_w, epsilon, alpha_theta_decay = lambda x : x, alpha_w_decay = lambda x : x, epsilon_decay = lambda x : x):
        self.action_space = action_space
        self.actor = actor
        self.critic = critic
        self.I = 1
        self.gamma = gamma
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.epsilon = epsilon
        self.alpha_theta_decay = alpha_theta_decay
        self.alpha_w_decay = alpha_w_decay
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state):
        if random.random() < self.epsilon:
            self.epsilon = self.epsilon_decay(self.epsilon)
            return random.choice(self.action_space)
        return sample_from_dict(self.actor.evaluate(state))

    def update(self, state, action, next_state, reward):
        delta = reward + self.gamma * self.critic.evaluate(next_state) - self.critic.evaluate(state)

        self.actor.update(state, action, self.alpha_theta, self.I, delta)
        self.critic.update(state, self.alpha_w, delta)

        self.I *= self.gamma
        self.alpha_theta = self.alpha_theta_decay(self.alpha_theta)
        self.alpha_w = self.alpha_w_decay(self.alpha_w)


# LOAD AGENTS
env = gym.make("CartPole-v1")

action_space = np.arange(start=env.action_space.start, stop = env.action_space.start + env.action_space.n)
discretizer = Discretizer()

## Q-LEARNING
q_learning = QLearning(action_space, discretizer, TabularEstimator, 0, 0.9, epsilon = 0)

## ACTOR
actor = Actor(action_space, discretizer, TabularEstimator)

## CRITIC
critic = Critic(discretizer, TabularEstimator)

## ACTOR-CRITIC
actor_critic = ActorCritic(action_space, actor, critic, 0.9999, 0, 0, 0)

## LOADING
with open("q_learning_0.pkl", 'rb') as f:
    q_learning.estimators[0].table = np.load(f)

with open("q_learning_1.pkl", 'rb') as f:
    q_learning.estimators[1].table = np.load(f)


with open("actor_0.pkl", 'rb') as f:
    actor.approximators[0].table = np.load(f)

with open("actor_1.pkl", 'rb') as f:
    actor.approximators[1].table = np.load(f)

with open("critic.pkl", 'rb') as f:
    critic.estimator.table = np.load(f)

env.close()