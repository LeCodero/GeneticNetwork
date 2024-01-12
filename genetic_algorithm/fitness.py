def evaluate_fitness(model, train_loader, test_loader, epochs):
    model.train(train_loader, epochs)
    return model.evaluate(test_loader)

