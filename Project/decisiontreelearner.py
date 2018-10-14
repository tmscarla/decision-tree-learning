import math
import numpy as np
from decisiontree import DecisionTree
from decisionleaf import DecisionLeaf
from decisiontreecontinuos import DecisionTreeContinuos

""" TODO: trova un modo dove posizionarla perbenino
    Questa praticamente è una lista di liste dove ogni lista interna è una lista di valori 
    unici dell'attributo ordinati in ordine crescente
"""
def get_values(file, n_attrs):
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

    values_list = []
    for n in range(n_attrs):
        values_list.append(sorted(list(set([e[n] for e in examples]))))

    return values_list


def decision_tree_learner(dataset, m=0, continuos=True):
    """
    This function contains all the functions that decision_tree_learning algorithm needs
    to compute. The complexity of the building tree is encoded in internal_nodes.
    """
    target = dataset.target
    values = dataset.values
    internal_nodes = 0

    values_list = get_values('iris.txt', 4)

    def decision_tree_learning(examples, attributes, m, parent_examples=()):
        if len(examples) == 0:
            return majority_value(parent_examples)
        elif same_classification(examples):
            return DecisionLeaf(examples[0][target])
        elif len(attributes) == 0:
            return majority_value(examples)
        elif misclass_error(examples) < m:
            return majority_value(examples)
        else:
            A = pick_attribute(attributes, examples)
            tree = DecisionTree(A, dataset.attrnames[A])
            nonlocal internal_nodes
            internal_nodes += 1
            for (val_i, exs_i) in split(A, examples):
                subtree = decision_tree_learning(exs_i, removeall(A, attributes), m, examples)
                tree.add(val_i, subtree)
            return tree

    def decision_tree_learning_continuos(examples, attributes, parent_examples=()):
        if len(examples) == 0:
            return majority_value(parent_examples)
        elif same_classification(examples):
            return DecisionLeaf(examples[0][target])
        elif len(attributes) == 0:
            return majority_value(examples)
        else:
            A, treshold = pick_attribute_continuos(attributes, examples)
            tree = DecisionTreeContinuos(A, treshold, dataset.attrnames[A])
            less_equal, greater_than = split_continuous(A, treshold, examples)

            subtree_le = decision_tree_learning_continuos(less_equal, removeall(A, attributes), examples)
            tree.add(treshold, False, subtree_le)
            subtree_gt = decision_tree_learning_continuos(greater_than, removeall(A, attributes), examples)
            tree.add(treshold, True, subtree_gt)
            return tree

    def majority_value(examples):
        """Selects the most common output value among a setof examples, breaking ties randomly."""
        i = 0
        for v in values[target]:
            if (count(target, v, examples) > i):
                i = count(target, v, examples)
                major = v
        return DecisionLeaf(major)

    def same_classification(examples):
        """Returns true iff all the examples have the same target classification"""
        classification = examples[0][target]
        for e in examples:
            if e[target] != classification:
                return False
        return True

    def entropy_bits(examples):
        """Calculates the entropy bits of the set of examples"""
        e = 0
        for v in values[target]:
            if len(examples) != 0:
                p = count(target, v, examples) / len(examples)
                if (p != 0):
                    e += ((-p) * math.log2(p))
        return e

    def information_gain(A, examples):
        """It estimates the expected reduction in entropy choosing A for splitting the tree"""
        N = float(len(examples))
        remainder = 0
        for (v, examples_i) in split(A, examples):
            remainder += ((len(examples_i) / N) * entropy_bits(examples_i))
        return entropy_bits(examples) - remainder

    def information_gain_continuos(attribute, treshold, examples):
        """It estimates the expected reduction in entropy choosing a treshold for the attribute"""
        N = float(len(examples))
        remainder = 0

        less_or_equal, greater_than = split_continuous(attribute, treshold, examples)

        remainder += ((len(less_or_equal) / N) * entropy_bits(less_or_equal))
        remainder += ((len(greater_than) / N) * entropy_bits(greater_than))

        return entropy_bits(examples) - remainder

    def pick_attribute(attributes, examples):
        """Choose the attribute with the best information gain"""
        max = 0
        for a in attributes:
            if information_gain(a, examples) >= max:
                max = information_gain(a, examples)
                best = a
        return best

    def pick_attribute_continuos(attributes, examples):
        max = 0
        treshold = 0
        for a in attributes:
            t = choose_split(a, examples)
            if information_gain_continuos(a, treshold, examples) >= max:
                max = information_gain_continuos(a, treshold, examples)
                best = a
                treshold = t
        return best, float(treshold)

    def split(attribute, examples):
        """
        Return a list of tuples (value, examples) pairs for each value of attribute.
        It reduces the amount of examples by giving a value to an attribute.
        """
        return [(v, [e for e in examples if e[attribute] == v])
                for v in values[attribute]]

    def split_continuous(attribute, treshold, examples):
        less_or_equal = []
        greater_than = []

        for e in examples:
            if float(e[attribute]) <= treshold:
                less_or_equal.append(e)
            else:
                greater_than.append(e)

        return less_or_equal, greater_than

    def choose_split(attribute, examples):
        """
        Choose the best treshold to split the values of the attribute.
        :param attribute: the attribute considered
        :param examples: the list of remained examples
        :return: treshold: the best value to binary split the examples
        """
        max = 0
        treshold = 0
        for v in values_list[attribute]:
            if information_gain_continuos(attribute, float(v), examples) > max:
                max = information_gain_continuos(attribute, float(v), examples)
                treshold = float(v)

        return treshold

    def count(attribute, value, examples):
        """Count the number of examples that have attribute = value."""
        i = 0
        for e in examples:
            if e[attribute] == value:
                i += 1
        return i

    def misclass_error(examples):
        """
        Given a set of examples, counts the majority target value and returns the number
        of examples which have a different target value.
        """
        val_max = majority_value(examples)
        num_val_max = count(target, val_max, examples)
        return len(examples) - num_val_max

    def removeall(item, seq):
        """Return a copy of seq (or string) with all occurences of item removed."""
        if isinstance(seq, str):
            return seq.replace(item, '')
        else:
            return [x for x in seq if x != item]

    if continuos:
        return decision_tree_learning_continuos(dataset.examples, dataset.inputs, m), internal_nodes
    else:
        return decision_tree_learning(dataset.examples, dataset.inputs, m), internal_nodes
