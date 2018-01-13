import math
from decisiontree import DecisionTree
from decisionleaf import DecisionLeaf


def decision_tree_learner(dataset, m=0):
    """
    This function contains all the functions that decision_tree_learning algorithm needs
    to compute. The complexity of the building tree is encoded in internal_nodes.
    """
    target = dataset.target
    values = dataset.values
    internal_nodes = 0

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

    def pick_attribute(attributes, examples):
        """Choose the attribute with the best information gain"""
        max = 0
        for a in attributes:
            if information_gain(a, examples) >= max:
                max = information_gain(a, examples)
                best = a
        return best

    def split(attribute, examples):
        """
        Return a list of tuples (value, examples) pairs for each value of attribute.
        It reduces the amount of examples by giving a value to an attribute.
        """
        return [(v, [e for e in examples if e[attribute] == v])
                for v in values[attribute]]

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

    return decision_tree_learning(dataset.examples, dataset.inputs, m), internal_nodes
