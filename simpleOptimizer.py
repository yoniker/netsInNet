class SimpleOptimizer:
        def __init__(self,parameters,learning_rate):
                self.parameters=parameters
                self.learning_rate=learning_rate
        def step():
                for p in parameters:
                        p=p-learning_rate*p.grad