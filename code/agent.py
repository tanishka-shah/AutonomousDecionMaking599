import random
import numpy as np
from multi_armed_bandits import *

"""
 Base class of an autonomously acting and learning agent.
"""
class Agent:

    def __init__(self, params):
        self.nr_actions = params["nr_actions"]

    """
     Behavioral strategy of the agent. Maps states to actions.
    """
    def policy(self, state):
        pass

    """
     Learning method of the agent. Integrates experience into
     the agent's current knowledge.
    """
    def update(self, state, action, reward, next_state, terminated, truncated):
        pass
        

"""
 Randomly acting agent.
"""
class RandomAgent(Agent):

    def __init__(self, params):
        super(RandomAgent, self).__init__(params)
        
    def policy(self, state):
        return random.choice(range(self.nr_actions))

"""
 Autonomous agent base for learning Q-values.
"""
class TemporalDifferenceLearningAgent(Agent):
    def __init__(self, params):
        self.params = params
        self.gamma = params["gamma"]
        self.nr_actions = params["nr_actions"]
        self.Q_values = {}
        self.alpha = params["alpha"]
        self.epsilon_decay = params["epsilon_decay"]
        self.exploration_decay = params["exploration_decay"]
        self.temperature_decay = params["temperature_decay"]
        self.epsilon = params['epsilon']
        self.exploration_constant = params['exploration_constant']
        self.temperature = params['temperature']
        self.action_counts = {}

    def Q(self, state):
        state = np.array2string(state)
        if state not in self.Q_values:
            self.Q_values[state] = np.zeros(self.nr_actions)
        return self.Q_values[state]

    def policy(self, state, evaluation = True):
        pass
    
    def decay_exploration(self):
        self.epsilon = max(self.epsilon-self.epsilon_decay, self.epsilon_decay)

"""
 Autonomous agent using on-policy SARSA.
"""
class SARSALearner(TemporalDifferenceLearningAgent):
        
    def update(self, state, action, reward, next_state, terminated, truncated):
        self.decay_exploration()
        Q_old = self.Q(state)[action]
        TD_target = reward
        if not terminated:
            next_action = self.policy(next_state)
            Q_next = self.Q(next_state)[next_action]
            TD_target += self.gamma*Q_next
        TD_error = TD_target - Q_old
        self.Q(state)[action] += self.alpha*TD_error
        
"""
 Autonomous agent using off-policy Q-Learning.
"""
class QLearner(TemporalDifferenceLearningAgent):
    '''
     Update function for Q-values and decaying the exploration parameter
    '''
    def update(self, state, action, reward, next_state, terminated, truncated, q_val = None):
        if q_val != None:
            self.Q_values = q_val
        self.decay_exploration()
        Q_old = self.Q(state)[action]
        TD_target = reward
        if not terminated:
            Q_next = max(self.Q(next_state))
            TD_target += self.gamma*Q_next
        TD_error = TD_target - Q_old
        self.Q(state)[action] += self.alpha*TD_error
        return self.Q_values

class QLearner_EpsilonGreedy(QLearner):
    '''
     Forming a policy by choosing an action based on epsilon-greedy bandit
    '''
    def policy(self, state, evaluation = True):
        statev = np.array2string(state)
        if statev not in self.action_counts:
            self.action_counts[statev] = np.zeros(self.nr_actions)
        action_count = self.action_counts[statev]
        Q_values = self.Q(state)
        epsilon1 = self.epsilon
        if evaluation == True:
            epsilon1 = 0
        else:
            epsilon1 = self.epsilon
        action = epsilon_greedy(Q_values, action_count, epsilon=epsilon1)
        self.action_counts[statev][action] += 1
        return action



class QLearner_UCB1(QLearner):
    '''
     Forming a policy by choosing an action based on UCB1 bandit
    '''
    
    def policy(self, state, evaluation = True):
        statev = np.array2string(state)
        if statev not in self.action_counts:
            self.action_counts[statev] = np.zeros(self.nr_actions)
        action_count = self.action_counts[statev]
        Q_values = self.Q(state)
        exp_constant = self.exploration_constant
        if evaluation == True:
            exp_constant = 0
        else:
            exp_constant = self.exploration_constant
        action = UCB1(Q_values, action_count, exploration_constant=exp_constant)
        self.action_counts[statev][action] += 1
        return action
    
    def decay_exploration(self):
        self.exploration_constant = max(self.exploration_constant-self.exploration_decay, self.exploration_decay)

    

class QLearner_Boltzmann(QLearner):
    '''
     Forming a policy by choosing an action based on Boltzmann bandit
    '''
    def policy(self, state, evaluation = True):
        statev = np.array2string(state)
        if statev not in self.action_counts:
            self.action_counts[statev] = np.zeros(self.nr_actions)
        action_count = self.action_counts[statev]
        Q_values = self.Q(state)
        temp = self.temperature
        if evaluation == True:
            temp = 0
        else:
            temp = self.temperature
        action = boltzmann(Q_values, action_count, temperature=temp)
        self.action_counts[statev][action] += 1
        return action
    
    def decay_exploration(self):
        self.temperature = max(self.temperature-self.temperature_decay, self.temperature_decay)

    