class DataSet(object):
    """
    This class defines dataset for a decision tree learning problem. It has the following fields:

       d.examples       A list of examples. Each element is a dictionary that maps
                        {attribute : value, ...} for all attributes in d.attributes.
       d.name           A name which should identify dataset's content
       d.target         The attribute that the decision_learning_tree algorithm
                        will try to predict.
       d.attributes     A list of integers to index the attributes.
       d.attrnames      A dictionary that maps {attribute : attrname, ...} for all attributes
                        in d.attributes.
       d.values         A dictionary of all possible values of every attribute in
                        d.attributes: {attribute : [v1, v2, v3], ...}
    """

    def __init__(self, name='', examples=None,  inputs=None, attributes=None, target=None,
                 attrnames=None, values=None):
        """Initializes DataSet's fields"""
        self.name = name
        self.examples = examples
        self.target = target
        self.values = values
        self.inputs = inputs

        # Attrs are the indices of examples, unless otherwise stated.
        if attributes is None and self.examples is not None:
            attributes = list(range(len(self.examples[0])))
        self.attributes = attributes
        # Initialize attrnames from string or by default
        if isinstance(attrnames, str):
            self.attrnames = attrnames.split()
        else:
            self.attrnames = attrnames or attributes

    def addExample(self,example):
        """
        Add an example to the list of examples.
        First, it checks its validity.
        """
        self.checkExample(example)
        self.examples.append(example)

    def checkExample(self,example):
        """
        Check if every value of the example is valid,
        otherwise raise a ValueError exception
        """
        if self.values:
            for a in self.attributes:
                if example[a] not in self.values[a]:
                    raise ValueError('Unvalid value %s for %s in %s',
                                     example[a], self.attributesNames[a], example)

    def removeExample(self,index):
        if index > len(self.examples):
            raise IndexError('Unvalid index!')
