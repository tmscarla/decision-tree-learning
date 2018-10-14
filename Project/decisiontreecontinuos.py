class DecisionTreeContinuos:
    """"It holds an attribute and a set of branches, one for each value of the attribute."""

    def __init__(self, attribute, treshold, attrname=None, branches=None):
        self.attribute = attribute
        self.treshold = treshold
        self.attrname = attrname or attribute
        self.branches = branches or {}

    def __call__(self,example):
        "Given an example, classify it using the attribute and the branches."
        attr_value = example[self.attribute]
        if float(attr_value) > self.treshold:
            return self.branches[(self.treshold, True)](example)
        else:
            return self.branches[(self.treshold, False)](example)

    def add(self,value, is_greater, subtree):
        "Add a branch"
        self.branches[(value, is_greater)] = subtree

    def display(self, indent=0):
        name = self.attrname
        print('TEST', name)
        for (val, subtree) in self.branches.items():
            if val[1] is True:
                print(' ' * 4 * indent, name, '>', val[0], '==>', end=' ')
            else:
                print(' ' * 4 * indent, name, '<=', val[0], '==>', end=' ')
            subtree.display(indent + 1)

    def __repr__(self):
        return ('DecisionTree(%r, %r, %r)'
                % (self.attribute, self.attrname, self.branches))
