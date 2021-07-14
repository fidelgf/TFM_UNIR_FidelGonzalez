import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from multiprocessing import Process, Manager
import sys

#sys.path.append('..')
sys.path.append('/home/pg1551610/CloudSimPy')

from core.machine import MachineConfig
from playground.Non_DAG.algorithm.random_algorithm import RandomAlgorithm
from playground.Non_DAG.algorithm.tetris import Tetris
from playground.Non_DAG.algorithm.first_fit import FirstFitAlgorithm
from playground.Non_DAG.algorithm.DeepJS.DRL import RLAlgorithm
from playground.Non_DAG.algorithm.DeepJS.agent import Agent
from playground.Non_DAG.algorithm.DeepJS.brain import Brain

from playground.Non_DAG.algorithm.DeepJS.reward_giver import MakespanRewardGiver

from playground.Non_DAG.utils.csv_reader import CSVReader
from playground.Non_DAG.utils.feature_functions import features_extract_func, features_normalize_func
from playground.Non_DAG.utils.tools import multiprocessing_run, average_completion, average_slowdown
from playground.Non_DAG.utils.episode import Episode

os.environ['CUDA_VISIBLE_DEVICES'] = ''

np.random.seed(41)
tf.random.set_seed(41)
#tf.random.set_random_seed(41)
# ************************ Parameters Setting Start ************************
machines_number = 5
jobs_len = 10
n_iter = 50
n_episode = 12
jobs_csv = '../jobs_files/jobs.csv'

brain = Brain(6)
reward_giver = MakespanRewardGiver(-1)
features_extract_func = features_extract_func
features_normalize_func = features_normalize_func

name = '%s-%s-m%d' % (reward_giver.name, brain.name, machines_number)
model_dir = './agents/%s' % name
# ************************ Parameters Setting End ************************

if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

agent = Agent(name, brain, 1, reward_to_go=True, nn_baseline=True, normalize_advantages=True,
              model_save_path='%s/model.ckpt' % model_dir)

machine_configs = [MachineConfig(64, 1, 1) for i in range(machines_number)]
csv_reader = CSVReader(jobs_csv)
jobs_configs = csv_reader.generate(0, jobs_len)

#tic = time.time()
#algorithm = RandomAlgorithm()
#episode = Episode(machine_configs, jobs_configs, algorithm, None)
#episode.run()
#print(episode.env.now, time.time() - tic, average_completion(episode), average_slowdown(episode))

#tic = time.time()
#algorithm = FirstFitAlgorithm()
#episode = Episode(machine_configs, jobs_configs, algorithm, None)
#episode.run()
#print(episode.env.now, time.time() - tic, average_completion(episode), average_slowdown(episode))

#tic = time.time()
#algorithm = Tetris()
#episode = Episode(machine_configs, jobs_configs, algorithm, None)
#episode.run()
#print(episode.env.now, time.time() - tic, average_completion(episode), average_slowdown(episode))

columnas = ['makespans', 'average_makespan', 'time', 'average_completions', 'average_slowdowns', 'reward']
storageData = []
tic2 = time.time()

for itr in range(n_iter):
    tic = time.time()
    print("********** Iteration %i ************" % itr)
    processes = []

    manager = Manager()
    trajectories = manager.list([])
    makespans = manager.list([])
    average_completions = manager.list([])
    average_slowdowns = manager.list([])
    for i in range(n_episode):
        algorithm = RLAlgorithm(agent, reward_giver, features_extract_func=features_extract_func,
                                features_normalize_func=features_normalize_func)
        episode = Episode(machine_configs, jobs_configs, algorithm, None)
        algorithm.reward_giver.attach(episode.simulation)
        p = Process(target=multiprocessing_run,
                    args=(episode, trajectories, makespans, average_completions, average_slowdowns))

        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    agent.log('makespan', np.mean(makespans), agent.global_step)
    agent.log('average_completions', np.mean(average_completions), agent.global_step)
    agent.log('average_slowdowns', np.mean(average_slowdowns), agent.global_step)

    print(makespans)
    #print(np.mean(makespans), intervalo, np.mean(average_completions), np.mean(average_slowdowns))

    all_observations = []
    all_actions = []
    all_rewards = []
    for trajectory in trajectories:
        observations = []
        actions = []
        rewards = []
        for node in trajectory:
            observations.append(node.observation)
            actions.append(node.action)
            rewards.append(node.reward)
            

        all_observations.append(observations)
        all_actions.append(actions)
        all_rewards.append(rewards)
        #print("tamano")
        #print(sum(x is not None for x in observations))
        #print(sum(x is not None for x in actions))
        #print(sum(x is not None for x in rewards))
        #time.sleep(2)

    get_rewards = [np.mean(i) for i in all_rewards]
    print("rewards")
    print(get_rewards)
    
    all_q_s, all_advantages = agent.estimate_return(all_rewards)
    #print("despues de estimate return")
    #print("all observations: {}".format(type(all_observations)))
    #print(np.shape(all_observations))
    #for i in all_observations:
    #    print(np.shape(i))
    #print("all actions: {}".format(type(all_actions)))
    #print(np.shape(all_actions))
    #for i in all_actions:
    #    print(np.shape(i))
    #print("all advantages: {}".format(type(all_advantages)))
    #print(np.shape(all[M .F_advantages))
    #for i in all_advantages:
    #    print(np.shape(i))
    #time.sleep(20)
    agent.update_parameters(all_observations, all_actions, all_advantages)
    toc = time.time()
    intervalo = toc - tic
    storageData.append([makespans, np.mean(makespans), intervalo, np.mean(average_completions), np.mean(average_slowdowns), get_rewards])
    df = pd.DataFrame(storageData, columns = columnas)
    file = "deeplearning{}.csv".format(itr)
    df.to_csv(file)

toc2 = time.time()
print("Tiempo final")
print(toc2 - tic2)
df = pd.DataFrame(storageData, columns = columnas)
df.to_csv('deeplearning.csv')
agent.save()
