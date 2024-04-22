import random
import numpy as np
import math

def random_bandit(Q_values, action_counts):
    return random.choice(range(len(Q_values)))
    
def epsilon_greedy(Q_values, action_counts, epsilon=0.1):
    if np.random.rand() <= epsilon:
        return random.choice(range(len(Q_values)))
    else:
        '''
        To avoid taking same actions again and again, we take into account all actions with high Q-values, and choose the action least frequent. 
        '''
        max_indices = np.where(Q_values == np.max(Q_values))[0]
        min_count_action = np.argmin([action_counts[i] for i in max_indices])
        return max_indices[min_count_action]
        
def boltzmann(Q_values, action_counts, temperature=1.0):
    E = np.exp(Q_values/temperature)
    N_total = sum(action_counts)
    if N_total == 0:
        N_total = 1
    prob = E/sum(E)
    prob = prob * (1 - action_counts/ N_total)
    '''
        To avoid taking same actions again and again, we take into account all actions with high Q-values, and choose the action least frequent. 
    '''
    prob = prob/sum(prob) # Normalizing probabilities
    prob[np.isnan(prob)] = 0 # Replacing NaN probabilities by 0
    '''
        If the sum of probabilities is less than 1, normalize probabilities respectively by adding the offset equally to all probability values
    '''
    if np.sum(prob) < 1:
        val = 1 - np.sum(prob)
        val = val/len(prob) 
        prob += val
    return np.random.choice(range(len(Q_values)), p=prob)
        
def UCB1(Q_values, action_counts, exploration_constant=1):
    UCB1_values = []
    N_total = sum(action_counts)
    for Q, N in zip(Q_values, action_counts):
        if N == 0:
            UCB1_values.append(math.inf)
        else:
            exploration_term = exploration_constant
            exploration_term *= np.sqrt(np.log(N_total)/N)
            UCB1_values.append(Q + exploration_term)
    return np.argmax(UCB1_values)

