import torch
import uuid

class NeuralNetwork:
    def __init__(self, model=None, optimizer_func=None, learning_rate=None, loss_function=None, metrics_function=None, device=None, parent=None, generation=0, parent_ids=None):
        if parent:
            self.model = parent.model
            self.optimizer = parent.optimizer
            self.criterion = parent.criterion
            self.metric = parent.metric
            self.device = parent.device
        else:
            self.model = model
            self.optimizer = optimizer_func(self.model.parameters(), lr=learning_rate)
            self.criterion = loss_function()
            self.metric = metrics_function
            self.device = torch.device(device)

        self.model.to(device)
        self.fitness = 0
        self.accuracy = 0
        self.id = uuid.uuid4()
        self.generation = generation
        self.parent_ids = parent_ids or []

    def train(self, data_loader, epochs):
        self.model.train()
        for _ in range(epochs):
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

    def evaluate(self, data_loader):
        self.model.eval()
        test_loss = 0.0
        accuracy = None
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                if(self.metric):
                    self.metric(outputs, labels)
            if(self.metric):
                accuracy = self.metric.compute()
                self.metric.reset()
        return test_loss / len(data_loader.dataset), accuracy
    
    def cross_validate(self, train_loader, test_loader, epochs):
        self.train(train_loader, epochs)
        self.fitness, self.accuracy = self.evaluate(test_loader)