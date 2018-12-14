# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 16:21:05 2018

@author: Bruce
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def sample_jester_data(file_name, context_dim = 32, num_actions = 8, num_contexts = 19181,
    shuffle_rows=True, shuffle_cols=False):
    """Samples bandit game from (user, joke) dense subset of Jester dataset.
    Args:
    file_name: Route of file containing the modified Jester dataset.
    context_dim: Context dimension (i.e. vector with some ratings from a user).
    num_actions: Number of actions (number of joke ratings to predict).
    num_contexts: Number of contexts to sample.
    shuffle_rows: If True, rows from original dataset are shuffled.
    shuffle_cols: Whether or not context/action jokes are randomly shuffled.
    Returns:
    dataset: Sampled matrix with rows: (context, rating_1, ..., rating_k).
    opt_vals: Vector of deterministic optimal (reward, action) for each context.
    """
    np.random.seed(0)
    with tf.gfile.Open(file_name, 'rb') as f:
        dataset = np.load(f)
    
    if shuffle_cols:
        dataset = dataset[:, np.random.permutation(dataset.shape[1])]
    if shuffle_rows:
        np.random.shuffle(dataset)
    dataset = dataset[:num_contexts, :]
    assert context_dim + num_actions == dataset.shape[1], 'Wrong data dimensions.'
    
    opt_actions = np.argmax(dataset[:, context_dim:], axis=1)
    opt_rewards = np.array([dataset[i, context_dim + a] for i, a in enumerate(opt_actions)])
    return dataset, opt_rewards, opt_actions


dataset, opt_rewards, opt_actions = sample_jester_data('jester_data_40jokes_19181users.npy', context_dim = 32, num_actions = 8, num_contexts = 19181,
    shuffle_rows=False, shuffle_cols=False)

class LinUCB:
    def __init__(self, alpha, num_feature=32, num_action=8):
        self.alpha = alpha
        self.num_feature = num_feature
        self.num_action = num_action
        self.A_list = [np.identity(self.num_feature) for i in range(self.num_action)]
        self.b_list = [np.matrix(np.zeros(self.num_feature)).T for i in range(self.num_action)]
        self.theta_list = [0 for i in range(self.num_action)]
        self.A_inverse_list = [np.identity(self.num_feature) for i in range(self.num_action)]
        
    def train(self, context, rewards_list):
        context = np.matrix(context)
        p_list = [0 for i in range(self.num_action)]
        for a in range(self.num_action):
            self.theta_list[a] = self.A_inverse_list[a]*self.b_list[a]
            p_list[a] = context*self.theta_list[a] + self.alpha*np.sqrt(context*self.A_inverse_list[a]*context.T)
        potential_a = np.where(np.array(p_list)==np.max(p_list))[0]
        a_opt = np.random.choice(potential_a)
        self.A_list[a_opt] += context.T*context
        self.b_list[a_opt] += rewards_list[a_opt]*context.T
        self.A_inverse_list[a_opt] = np.linalg.solve(self.A_list[a_opt], np.identity(self.num_feature))
        return None
    
    def predict(self, context):
        context = np.matrix(context) # context is a 1*d matrix
        p_list = [0 for i in range(self.num_action)]
        for a in range(self.num_action):
            self.theta_list[a] = self.A_inverse_list[a]*self.b_list[a]
            p_list[a] = context*self.theta_list[a] + self.alpha*np.sqrt(context*self.A_inverse_list[a]*context.T)
        potential_a = np.where(np.array(p_list)==np.max(p_list))[0]
        a_opt = np.random.choice(potential_a)
        return a_opt


#for alpha in np.linspace(0.5, 0.9, 11):       
#    model = LinUCB(alpha)
#    
#    N = 18000
#    for i in range(N):
#        context = dataset[i, :32]
#        rewards = dataset[i, 32:]
#        model.train(context, rewards)
#    #    if i%100==0:
#    #        print(i)
#    
#    regret = 0
#    for i in range(N, dataset.shape[0]):
#        context = dataset[i, :32]
#        rewards = dataset[i, 32:]
#        pred_a = model.predict(context)
#        regret += opt_rewards[i] - rewards[pred_a]
#    #    print(opt_rewards[i] - rewards[pred_a])
#    regret /= (dataset.shape[0]-N)
#    print(alpha, regret)
        
alpha = 0.9
model = LinUCB(alpha)

N = 18000
for i in range(N):
    context = dataset[i, :32]
    rewards = dataset[i, 32:]
    model.train(context, rewards)

regret_list=[]
regret = 0
for i in range(N, dataset.shape[0]):
    context = dataset[i, :32]
    rewards = dataset[i, 32:]
    pred_a = model.predict(context)
    regret += opt_rewards[i] - rewards[pred_a]
    regret_list.append(regret)

print(alpha, regret)
plt.plot(regret_list)
plt.xlabel('Num of test datapoints')
plt.ylabel('Total regrets')
plt.show()
