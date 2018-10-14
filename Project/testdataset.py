"""
This module contains three different dataset taken by the UCI Machine Learning Repository. The list of examples are
contained in the respective .txt file. Attributes names and values have been manually inserted in this module.

Uncomment any test_and_plot function to run it for the proper dataset, otherwise uncomment the two lines below
to print a non-pruned tree with the display() function.

The m_range for every dataset was chosen according to the major possible error. This gives the chance to the
decision learning algorithm to return a tree with just a single node.
The nursery data set has m_range = 1000 instead of 8640 for computational reasons.
"""
from test import test_and_plot
from test import create_dataset
from decisiontreelearner import decision_tree_learner


# DataSet 1: CAR EVALUATION_____________________________________________________________________________________________

car_names = {0: 'buying', 1: 'maint', 2: 'doors', 3: 'persons', 4: 'lug_boot', 5: 'safety', 6: 'evaluation'}
car_values = {0: ['vhigh', 'high', 'med', 'low'],
              1: ['vhigh', 'high', 'med', 'low'],
              2: ['2', '3', '4', '5more'],
              3: ['2', '4', 'more'],
              4: ['small', 'med', 'big'],
              5: ['low', 'med', 'high'],
              6: ['unacc', 'acc', 'good', 'vgood']}

# test_and_plot('car.txt', car_names, 6, car_values, 520)

# car_tree, car_nodes = decision_tree_learner(create_dataset('car.txt', car_names, 6, car_values))
# car_tree.display()

# DataSet 2: BALANCE-SCALE______________________________________________________________________________________________

bs_names = {0: 'balanced', 1: 'left-weight', 2: 'left-distance', 3: 'right-weight', 4: 'right-distance'}
bs_values = {0: ['L', 'B', 'R'],
             1: ['1', '2', '3', '4', '5'],
             2: ['1', '2', '3', '4', '5'],
             3: ['1', '2', '3', '4', '5'],
             4: ['1', '2', '3', '4', '5'],
             5: ['1', '2', '3', '4', '5']}

# test_and_plot('balance-scale.txt', bs_names, 0, bs_values, 340)

# bs_tree, bs_nodes = decision_tree_learner(create_dataset('balance-scale.txt', bs_names, 0, bs_values))
# bs_tree.display()

# DataSet 3: NURSERY____________________________________________________________________________________________________

n_names = {0: 'parents', 1: 'has_nurs', 2: 'form', 3: 'children', 4: 'housing',
           5: 'finance', 6: 'social', 7: 'health', 8: 'evaluation'}
n_values = {0: ['usual', 'pretentious', 'great_pret'],
            1: ['proper', 'less_proper', 'improper', 'critical', 'very_crit'],
            2: ['complete', 'completed', 'incomplete', 'foster'],
            3: ['1', '2', '3', 'more'],
            4: ['convenient', 'less_conv', 'critical'],
            5: ['convenient', 'inconv'],
            6: ['nonprob', 'slightly_prob', 'problematic'],
            7: ['recommended', 'priority', 'not_recom'],
            8: ['not-recom', 'recommend', 'priority', 'very_recom', 'spec_prior']}

# test_and_plot('nursery.txt', n_names, 8, n_values, 1000)

#n_tree, n_nodes = decision_tree_learner(create_dataset('nursery.txt', n_names, 8, n_values))
#n_tree.display()

# DataSet 4: IRIS____________________________________________________________________________________________________

print("\n")

iris_names = {0: 'sepal-len', 1: 'sepal-width', 2: 'petal-len', 3: 'petal-width', 4: 'class'}
iris_values = {4: ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']}

iris_dataset = create_dataset('iris.txt', iris_names, 4, iris_values)

iris_tree, n_nodes = decision_tree_learner(iris_dataset, m=0, continuos=True)
iris_tree.display()

print("\n\n")

print(len(iris_dataset.examples))

sum = 0
for e in iris_dataset.examples:
    if e[4] == iris_tree(e):

        #print("EXAMPLE:", e)
        #print("CLASS:", iris_tree(e))
        sum += 1
    print("EXAMPLE:", e)
    print("CLASS:", iris_tree(e))
print(sum)


