from torch.utils.data import dataset
import torch
import numpy as np
from torch import nn


class TempData(dataset.Dataset):
    def __init__ (self, start_val=0, num_elements=1000, add_noise=True):
        self.start_val = start_val
        self.num_elements = num_elements
        self.add_noise = add_noise
        self.data = []

        # create celcius temperature data
        celcius = np.arange(start_val, start_val + num_elements, dtype=np.float32)

        # convert to fahrenheit
        fahrenheit = celcius * 9.0 / 5.0 + 32.0

        if add_noise:
            fahrenheit += np.random.normal(0, 0.5, num_elements)

        # data, labels
        self.data = list(zip(fahrenheit, celcius))       

    def __getitem__ (self, index):
        fahrenheit, celcius = self.data[index]
        return torch.tensor([fahrenheit]), torch.tensor([celcius])

    def __len__(self):
        return self.num_elements


class Model(nn.Module):
    def __init__ (self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc1(x)


def evaluate_model(model, test_data, loss_fn=None):
    """ Evaluate the model 
    
    Args:
        model: (torch.nn.Module) the neural network
        data: (torch.utils.data.Dataset) the evaluation data
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
    """
    model.eval()
    predictions = torch.FloatTensor(0)
    gt = torch.FloatTensor(0)

    with torch.no_grad():
        for batch, (inputs, labels) in enumerate(test_data):       
            # forward
            outputs = model(inputs)
            gt = torch.cat((gt, labels), 0)
            predictions = torch.cat((predictions, outputs), 0)
        
        # compute the loss
        mean_error = loss_fn(gt, predictions)
        # print('\nTest set MSE: {:.4f}'.format(mean_error))

        # print the predictions and ground truth
        rand_int = np.random.randint(len(predictions)) # pick a random sample
        print('Predictions:ground truth', predictions.numpy()[rand_int], ':', gt.numpy()[rand_int])


def train_model(model, train_data, cv_data, optimizer, loss_fn, scheduler, num_epochs=1, log_interval=10):
    """ Train the model 
    
    Args:
        model: (torch.nn.Module) the neural network
        data: (torch.utils.data.Dataset) the training data
        optimizer: (torch.optim.Optimizer) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        num_epochs: (int) number of epochs to train for
        log_interval: (int) number of batches to wait before logging training status
        scheduler: (torch.optim.lr_scheduler) the scheduler for the optimizer
    """

    # keep track of training
    train_losses = []
    train_counter = []

    # training loop
    for epoch in range(num_epochs):
        model.train()

        for batch, (inputs, labels) in enumerate(train_data):            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # backward + optimize
            loss.backward()
            optimizer.step()

        # scheduler step
        scheduler.step()

        # keep track of training
        if epoch % log_interval == 0:    
            # print progress
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, loss.item()))

            # keep track of training
            train_losses.append(loss.item())
            train_counter.append(batch)

            # evaluate the model
            evaluate_model(model, cv_data, loss_fn)



if __name__ == '__main__':
    # dataset settings
    startval = -100
    num_elements = 1000

    # training settings
    batch_size = 8
    learning_rate = 0.01
    num_epochs = 100

    # create two datasets, one with noise for training, one without for testing 
    data_train = TempData(start_val=startval, num_elements=num_elements, add_noise=True)
    data_test = TempData(start_val=startval+num_elements+50, num_elements=200, add_noise=False)

    # create two dataloaders, one for training, one for testing
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False)

    # create a model
    model = Model()

    # create adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # create a mean squared error loss function
    criterion = nn.MSELoss()

    # create a learning rate scheduler using MultiStepLR
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * num_elements, 0.75 * num_elements], gamma=0.1)

    # train the model
    print('Training model...')
    train_model(model=model, train_data=train_loader, cv_data=test_loader, 
                optimizer=optimizer, loss_fn=criterion, scheduler=scheduler,
                num_epochs=num_epochs, log_interval=1)

    # test a single temperature sample
    evaluate_model(model=model, test_data=test_loader, loss_fn=criterion)

    # print model parameters
    print('\nModel parameters:')
    for name, param in model.named_parameters():
        print(name, param.data)