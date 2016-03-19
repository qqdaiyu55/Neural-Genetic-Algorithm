import numpy as np
import neural_genetic_algorithm as ng

"""load data set and normalize it"""
dataset = np.loadtxt('rat_spinal_cord_data.txt')
for i in range(len(dataset[0])):
    minv = min(dataset[:, i])
    maxv = max(dataset[:, i])
    dataset[:, i] = (dataset[:, i] - minv) / (maxv - minv)

chromosome_set = []
for i in range(120):
    # structure of best chromesome:
    # ({'weight':value, 'RMSE':value}, individual)
    best_chromosome = ng.ga(i, dataset)
    chromosome_set.append(best_chromosome)
    print(best_chromosome[1])