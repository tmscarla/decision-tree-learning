"""
This module contains all the necessary functions to perform the decision_tree_learner function
and the related tests on the resulting trees.
It uses matplotlib.pyplot to show the curves of errors on both training and test set
as a function of tree complexity, which is defined as the number of internal nodes.
"""
from dataset import DataSet
from copy import deepcopy
from decisiontreelearner import decision_tree_learner
from datetime import datetime
import matplotlib.pyplot as plt
import random


def test_and_plot(file, attrnames, target, attrvalues, m_range):
    """
    It calls create_dataset given data file name, attributes names and values, then calls train_and_test
    to split randomly 10 times 80% training set and 20% test set.
    For m_range times it creates different trees using decision_tree_learner according to the pre-pruning
    parameter m, then it stores the number of internal nodes, errors in the training and test set and computes
    the average of them. Finally it shows the plot of the two errors as a function of the number of internal nodes.
    """
    train_err = []
    test_err = []
    internal_nodes = []
    for i in range(10):
        j = 0
        for m in range(m_range, 0, -1):
            dataset = create_dataset(file, attrnames, target, attrvalues)
            dataset.examples, test = train_and_test(dataset, 80)
            tree, internal_node = decision_tree_learner(dataset, m)
            # if is the first iteration, initialize the list
            if i == 0:
                train_err.append(count_errors(dataset.examples, target, tree))
                test_err.append(count_errors(test, target, tree))
                internal_nodes.append(internal_node)
            # if is the last iteration, compute the average
            elif i == 9:
                train_err[j] = float("%.3f" % (train_err[j] / 10))
                test_err[j] = float("%.3f" % (test_err[j] / 10))
                # internal_nodes[j] = math.floor(internal_nodes[j] / 10)
            else:
                train_err[j] += float("%.3f" % count_errors(dataset.examples, target, tree))
                test_err[j] += float("%.3f" % count_errors(test, target, tree))
                # internal_nodes[j] += internal_node
            j += 1

    # the rest of the function uses the previous lists to create a plot and show it
    plt.plot(internal_nodes, train_err, label="Training set")
    plt.plot(internal_nodes, test_err, label="Test set")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.ylabel('Percentage error')
    plt.xlabel('Internal nodes')
    plt.show()


def count_errors(examples, target, tree):
    """Count the errors of a decision tree given a set of examples."""
    i = 0
    for e in examples:
        if e[target] != tree(e):
            i += 1
    return i / len(examples) * 100


def train_and_test(dataset, perc_train):
    """
    Shuffle the dataset examples using the current time as a seed then
    split it into training and test set according to the percentage
    value of the training set.
    """
    random.seed(datetime.now())
    random.shuffle(dataset.examples)
    end = int((len(dataset.examples) / 100) * perc_train)
    train = dataset.examples[0:end]
    test = dataset.examples[end:]
    return train, test

def set_inputs(attributes, target):
    """Returns a list of attributes without the target"""
    inputs = deepcopy(attributes)
    inputs.pop(inputs.index(target))
    return inputs

def create_dataset(file, attrnames, target, values):
    """Create a dataset from a file, given its attributes names, values and index target"""
    dset = list()
    parts = []
    for line in open(file).readlines():
        parts = line.rstrip()
        parts = line.split(',')
        parts = [p.rstrip() for p in parts]
        dset.append(parts)

    examples = []
    for i in range(len(dset)):
        example = {}
        for j in range(len(dset[0])):
            example[j] = dset[i][j]
        examples.append(example)
    attributes = [k for k in range(len(examples[0]))]
    inputs = set_inputs(attributes, target)

    return DataSet(file, examples, inputs, attributes, target, attrnames, values)
