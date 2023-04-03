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

# Q-LEARNING
class QLearning:
    def __init__(self, action_space, discretizer, estimator):
        self.action_space = action_space
        self.estimators = {}
        for action in action_space:
            self.estimators[action] = estimator(discretizer)

    def choose_action(self, state):
        return self.best_action(state)

    def evaluate(self, state, action):
        return self.estimators[action].evaluate(state)

    def state_value(self, state):
        return self.estimators[self.best_action(state)].evaluate(state)

    def best_action(self, state):
        return max(self.estimators, key=lambda x: self.estimators[x].evaluate(state))



# ACTOR-CRITIC
class Actor:
    def __init__(self, action_space, discretizer, estimator):
        self.discretizer = discretizer
        self.approximators = {}
        for action in action_space:
            self.approximators[action] = estimator(discretizer)

    def evaluate(self, state):
        approximations = {name: approximator.evaluate(state) for name, approximator in self.approximators.items()}
        return dict(zip(approximations.keys(), scipy.special.softmax(np.array(list(approximations.values())))))

class Critic:
    def __init__(self, discretizer, estimator):
        self.estimator = estimator(discretizer)

    def evaluate(self, state):
        return self.estimator.evaluate(state)

def sample_from_dict(d):
    a = []
    p = []
    for action, prob in d.items():
        a += [action]
        p += [prob]
    return np.random.choice(a, p=p)

class ActorCritic:
    def __init__(self, action_space, actor, critic):
        self.action_space = action_space
        self.actor = actor
        self.critic = critic

    def choose_action(self, state):
        return sample_from_dict(self.actor.evaluate(state))


# LOAD AGENTS
env = gym.make("CartPole-v1")

action_space = np.arange(start=env.action_space.start, stop = env.action_space.start + env.action_space.n)
discretizer = Discretizer()

## Q-LEARNING
q_learning = QLearning(action_space, discretizer, TabularEstimator)

## ACTOR
actor = Actor(action_space, discretizer, TabularEstimator)

## CRITIC
critic = Critic(discretizer, TabularEstimator)

## ACTOR-CRITIC
actor_critic = ActorCritic(action_space, actor, critic)

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