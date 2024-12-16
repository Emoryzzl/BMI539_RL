# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:51:30 2024
@author: zzha962
"""
### Macos:  CC=gcc-14 CXX=g++-14 python3 -m pip install contextualbandits


import pandas as pd, numpy as np, re
from copy import deepcopy
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import SGDClassifier
from contextualbandits.linreg import LinearRegression
from contextualbandits.online import LinUCB, AdaptiveGreedy, \
        SoftmaxExplorer, ActiveExplorer, EpsilonGreedy

# Step1: Reading the data 
def parse_data(filename):
    with open(filename, "rb") as f:
        infoline = f.readline()
        infoline = re.sub(r"^b'", "", str(infoline))
        n_features = int(re.sub(r"^\d+\s(\d+)\s\d+.*$", r"\1", infoline))
        features, labels = load_svmlight_file(f, n_features=n_features, multilabel=True)
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)
    features = np.array(features.todense())
    features = np.ascontiguousarray(features)
    return features, labels

X, y = parse_data("Bibtex/Bibtex_data.txt")
print(X.shape)
print(y.shape)


# Step2: Streaming models
nchoices = y.shape[1]
base_sgd = SGDClassifier(random_state=123, loss='log_loss', warm_start=False)
base_ols = LinearRegression(lambda_=10., fit_intercept=True, method="sm")

## Metaheuristic using different base algorithms and configurations
linucb = LinUCB(nchoices = nchoices, beta_prior = None, alpha = 0.1,
                ucb_from_empty = False, random_state = 1111)
### Important!!! the default hyperparameters for LinUCB in the reference paper
### are very different from what's used in this example
adaptive_active_greedy = AdaptiveGreedy(deepcopy(base_ols), nchoices = nchoices,
                                        smoothing = None, beta_prior = ((3./nchoices,4.), 2),
                                        active_choice = 'weighted', decay_type = 'percentile',
                                        decay = 0.9997, batch_train = True,
                                        random_state = 2222)
softmax_explorer = SoftmaxExplorer(deepcopy(base_sgd), nchoices = nchoices,
                                   smoothing = (1,2), beta_prior = None, batch_train = True,
                                   refit_buffer = 50, deep_copy_buffer = False, random_state = 3333)
adaptive_greedy_perc = AdaptiveGreedy(deepcopy(base_ols), nchoices = nchoices,
                                      smoothing = (1,2), beta_prior = None,
                                      decay_type = 'percentile', decay = 0.9997, batch_train = True,
                                      random_state = 4444)
#active_explorer = ActiveExplorer(deepcopy(base_sgd), smoothing = None, nchoices = nchoices,
#                                 beta_prior = ((3./nchoices, 4.), 2), batch_train = True, refit_buffer = 50,
#                                 deep_copy_buffer = False, random_state = 5555)
epsilon_greedy_nodecay = EpsilonGreedy(deepcopy(base_ols), nchoices = nchoices,
                                       smoothing = (1,2), beta_prior = None,
                                       decay = None, batch_train = True,
                                       deep_copy_buffer = False, random_state = 6666)

#models = [linucb, adaptive_active_greedy, softmax_explorer, adaptive_greedy_perc,
#          active_explorer, epsilon_greedy_nodecay]
models = [linucb, adaptive_active_greedy, softmax_explorer, adaptive_greedy_perc,epsilon_greedy_nodecay]
# Step3: Running the experiment
# These lists will keep track of the rewards obtained by each policy
#rewards_lucb, rewards_aac, rewards_sft, rewards_agr, \
#rewards_ac, rewards_egr = [list() for i in range(len(models))]
rewards_lucb, rewards_aac, rewards_sft, rewards_agr, rewards_egr = [list() for i in range(len(models))]
# lst_rewards = [rewards_lucb, rewards_aac, rewards_sft,
#                rewards_agr, rewards_ac, rewards_egr]
lst_rewards = [rewards_lucb, rewards_aac, rewards_sft,rewards_agr, rewards_egr]
# batch size - algorithms will be refit after N rounds
batch_size=50

# initial seed - all policies start with the same small random selection of actions/rewards
first_batch = X[:batch_size, :]
np.random.seed(1)
action_chosen = np.random.randint(nchoices, size=batch_size)
rewards_received = y[np.arange(batch_size), action_chosen]

# fitting models for the first time
for model in models:
    model.fit(X=first_batch, a=action_chosen, r=rewards_received)
    
# these lists will keep track of which actions does each policy choose
lst_a_lucb, lst_a_aac, lst_a_sft, lst_a_agr, lst_a_egr = [action_chosen.copy() for i in range(len(models))]

lst_actions = [lst_a_lucb, lst_a_aac, lst_a_sft,
               lst_a_agr, lst_a_egr]

# rounds are simulated from the full dataset
def simulate_rounds_stoch(model, rewards, actions_hist, X_batch, y_batch, rnd_seed):
    np.random.seed(rnd_seed)
    
    ## choosing actions for this batch
    actions_this_batch = model.predict(X_batch).astype('uint8')
    
    # keeping track of the sum of rewards received
    rewards.append(y_batch[np.arange(y_batch.shape[0]), actions_this_batch].sum())
    
    # adding this batch to the history of selected actions
    new_actions_hist = np.append(actions_hist, actions_this_batch)
    
    # rewards obtained now
    rewards_batch = y_batch[np.arange(y_batch.shape[0]), actions_this_batch]
    
    # now refitting the algorithms after observing these new rewards
    np.random.seed(rnd_seed)
    model.partial_fit(X_batch, actions_this_batch, rewards_batch)
    
    return new_actions_hist

Epoch_num = 100
# running all the simulation
for jj in range(Epoch_num):
    for i in range(int(np.floor(X.shape[0] / batch_size))):
        batch_st = (i + 1) * batch_size
        batch_end = (i + 2) * batch_size
        batch_end = np.min([batch_end, X.shape[0]])
        
        X_batch = X[batch_st:batch_end, :]
        y_batch = y[batch_st:batch_end, :]
        
        for model in range(len(models)):
            lst_actions[model] = simulate_rounds_stoch(models[model],
                                                       lst_rewards[model],
                                                       lst_actions[model],
                                                       X_batch, y_batch,
                                                       rnd_seed = batch_st)
        print('{}/{} {}/{} done'.format(i+1,int(np.floor(X.shape[0] / batch_size)), jj+1, Epoch_num))

# Step4: Visualizing results
import matplotlib.pyplot as plt
from pylab import rcParams

def get_mean_reward(reward_lst, batch_size=batch_size):
    mean_rew=list()
    for r in range(len(reward_lst)):
        mean_rew.append(sum(reward_lst[:r+1]) * 1.0 / ((r+1)*batch_size))
    return mean_rew

rcParams['figure.figsize'] = 25, 15
lwd = 5
cmap = plt.get_cmap('tab20')
colors=plt.cm.tab20(np.linspace(0, 1, 20))
rcParams['figure.figsize'] = 25, 15

ax = plt.subplot(111)
plt.plot(get_mean_reward(rewards_lucb), label="LinUCB (OLS)", linewidth=lwd,color=colors[0])
plt.plot(get_mean_reward(rewards_aac), label="Adaptive Active Greedy (OLS)", linewidth=lwd,color=colors[16])
plt.plot(get_mean_reward(rewards_sft), label="Softmax Explorer (SGD)", linewidth=lwd,color=colors[17])
plt.plot(get_mean_reward(rewards_agr), label="Adaptive Greedy (p0=30%, decaying percentile, OLS)", linewidth=lwd,color=colors[12])
plt.plot(get_mean_reward(rewards_egr), label="Epsilon-Greedy (p0=20%, decay=0.9999, OLS)",linewidth=lwd,color=colors[6])
plt.plot(np.repeat(y.mean(axis=0).max(),len(rewards_sft)), label="Overall Best Arm (no context)",linewidth=lwd,color=colors[1],ls='dashed')

# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 1.25])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, ncol=3, prop={'size':10})


plt.tick_params(axis='both', which='major', labelsize=20)
#plt.xticks([i*20 for i in range(8)], [i*1000 for i in range(8)])


plt.xlabel('Rounds (models were updated every 50 rounds)', size=15)
plt.ylabel('Cumulative Mean Reward', size=15)
plt.title('Comparison of Online Contextual Bandit Policies (Streaming-data mode) \n Prostate MRI Dataset (5 categories, 1836 attributes)',size=15)
plt.grid()
plt.show()




