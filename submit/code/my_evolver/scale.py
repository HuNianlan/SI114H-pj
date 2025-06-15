class Optimizer:
    def __init__(self, model, lr=0.001):
        self.model = model
        self.lr = lr

    def step(self):
        for param in self.model.parameters():
            param.data -= self.lr * param.grad.data
