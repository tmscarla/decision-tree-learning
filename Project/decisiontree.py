class DecisionTree:
    """"It holds an attribute and a set of branches, one for each value of the attribute."""

    def __init__(self, attribute, attrname=None, branches=None):
        self.attribute = attribute
        self.attrname = attrname or attribute
        self.branches = branches or {}

    def __call__(self,example):
        "Given an example, classify it using the attribute and the branches."
        attr_value = example[self.attribute]
        return self.branches[attr_value](example)

    def add(self,value,subtree):
        "Add a branch"
        self.branches[value] = subtree

    def display(self, indent=0):
        name = self.attrname
        print('TEST', name)
        for (val, subtree) in self.branches.items():
            print(' ' * 4 * indent, name, '=', val, '==>', end=' ')
            subtree.display(indent + 1)

    def __repr__(self):
        return ('DecisionTree(%r, %r, %r)'
                % (self.attribute, self.attrname, self.branches))
