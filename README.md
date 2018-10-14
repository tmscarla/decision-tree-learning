# Decision Tree Learning

## Intro
This is a project that I developed for the Artificial Intelligence course during my Bachelor in Computer Engineering at the University of Florence. You can find further details in the file *Relazione.pdf*, although in italian.

## Overview

Decision tree learning is the construction of a decision tree from class-labeled training tuples. A decision tree is a flowchart-like structure in which each internal node represents a "test" on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label. The paths from root to leaf represent classification rules.

## How it works
The algorithm uses a "divide-et-impera" strategy: it selects the most informative attribute among a list of available ones. The test performed in a node divides the problem into two sub-problems solved in a recursive way. The final tree is built in a bottom-up procedure and leaves are generated when one of the following conditions occur:

* All the samples have the same label
* No samples are left
* No attributes are left
* Pre-pruning strategy

## Entropy and information gain
How can we capture the "importance" of an attribute? We can use the notion of entropy:


Information gain is the expected reduction of entropy choosing a particular attribute for testing. Clearly, at each step, the chosen attribute will be the one with the highest information gain.


## Datasets

I used three discrete datasets from the UCI Machine Learning Repository to test the algorithm, here you can see the results:
