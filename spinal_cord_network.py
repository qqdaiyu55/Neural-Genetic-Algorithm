from random import randint, random, sample
from operator import add
import numpy as np

def individual(n):
    length = randint(1, n)
    # length = 5
    return sample(range(1, n+1), length)

def population(count, n):
    return [ individual(n) for _ in range(count) ]

def fitness(individual, target, dataset, timepoint):
    """
    Input data is chosen as dataset from initial timepoint to given timepoint.
    Using ANN to compute weights, and the initial weights is within
    the permitted range (-10 to 10).
    Error is computed as the root mean squared error (RMSE).
    """
    individual = [i-1 for i in individual]
    target -= 1

    data_in = dataset[individual, :timepoint].T
    data_out = np.array([[x for x in dataset[target, :timepoint]]]).T
    
    """Parameters setting"""
    w = 20 * np.random.rand(len(individual), 1) - 10
    epoch = 100
    beta = 0.9
    momentum = 0.9
    
    w_delta = 0
    for i in range(epoch):
        y = 1 / (1 + np.exp(-(np.dot(data_in, w))))
        y_delta = (data_out - y) * (y * (1-y))
        w_delta = beta * data_in.T.dot(y_delta) + momentum * w_delta
        w += w_delta

    y = 1 / (1 + np.exp(-(np.dot(data_in, w))))

    from sklearn.metrics import mean_squared_error
    RMSE = mean_squared_error(data_out, y) ** 0.5
    return {'weight':w, 'RMSE':RMSE}

# def grade(pop, target):
#     'Find average fitness for a population.'
#     summed = reduce(add, (fitness(x, target) for x in pop))
#     return summed / (len(pop) * 1.0)

def getKey(item):
    return item[0]['RMSE']

def evolve(pop, target, dataset, timepoint=9, retain=0.8, 
    random_select=0.05, mutate=0.1):
    graded = [ (fitness(x, target, dataset, timepoint), x) for x in pop]
    graded = [ x[1] for x in sorted(graded, key=getKey)]
    retain_length = int(len(graded)*retain)
    parents = graded[:retain_length]
    # randomly add other individuals to
    # promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)

    # mutate some individuals
    for individual in parents:
        if mutate > random():
            # this mutation is not ideal, because it
            # may be same as previous element
            mutate_num = randint(1, len(dataset))
            # in case that the mutated numbers have existed in the list
            if mutate_num not in individual:
                pos_to_mutate = randint(0, len(individual)-1)
                individual[pos_to_mutate] = mutate_num
            
    # crossover parents to create children
    parents_length = len(parents)
    while len(parents) < len(pop):
        male = randint(0, parents_length-1)
        female = randint(0, parents_length-1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = int(len(male) / 2)
            children = male[:half] + female[half:]
            parents.append(children)

    return parents


def ga(target, dataset):
    timepoint = 9
    pop_size = 50
    generations = 1000
    pop = population(pop_size, 112)

    # find the best chromosome
    for _ in range(generations):
        pop = evolve(pop, target, dataset, timepoint)
        print("the %d-th generation" %_)
    
    graded = [ (fitness(x, target, dataset, timepoint), x) for x in pop]
    graded = sorted(graded, key=getKey)
    best = graded[0]

    return best


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
    best_chromosome = ga(i, dataset)
    chromosome_set.append(best_chromosome)
    print(best_chromosome[1])