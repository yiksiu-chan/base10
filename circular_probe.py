from tqdm import tqdm
import torch
import numpy as np
import logging

class CircularProbe(torch.nn.Module):
    def __init__(self, embedding_size, basis, bias):
        super(CircularProbe, self).__init__()
        self.weights = torch.nn.Linear(embedding_size, 22, bias=bias)
        self.basis = basis

    def forward_digit(self, x):
        original_shape = x.shape

        projected =  self.weights(x)

        projected = projected.view(-1, 2)

        # get the angle of the vector
        angles = torch.atan2(projected[...,1], projected[...,0])

        # signed to unsigned
        angles = torch.where(angles < 0, angles + 2*np.pi, angles)
        
        # get the digit
        digits = (angles) * self.basis / (2*np.pi)

        # back to 2d
        res = digits.view(original_shape[:-1] + (11,))            

        assert res.shape == original_shape[:-1] + (11,)
        
        return res
    
    def forward(self, x):
        original_shape = x.shape

        projected =  self.weights(x)

        projected = projected.view(original_shape[:-1] + (11, 2))

        return projected
        

def train_circular_probe(params, mt, num_to_hidden):
    '''return average accuracy on test set (getting all digits correct) as well as the probe itself'''
    logging.debug(f"Number of layers in model: {mt.num_layers}")
    logging.debug(f'{mt.model.__class__=}')

    embedding_size = num_to_hidden[0][0].shape[-1]
    print(embedding_size)
    # assert embedding_size > 1000
    
    # initalize the circular probe
    circular_probe = CircularProbe(embedding_size, params['basis'], params['bias'])
    circular_probe = circular_probe.to(mt.device)
    X = []
    Y = []

    X_test = []
    Y_test = []

    if params['layers'] == 'all':
        layers = list(range(0,mt.num_layers))
    else:
        layers = params['layers']


    def get_digit(num, index, basis = 10):
        return (num // basis**(-index-1)) % basis

    assert get_digit(1234,-1) == 4
    assert get_digit(1234,-1, basis=1235) == 1234
    assert get_digit(1234,-1, basis=2) == 0
    assert get_digit(1234,-2) == 3
    assert get_digit(1234,-7) == 0

    def get_digits(num, basis = 10, num_len = 11):
        return [get_digit(num, -num_len+i, basis) for i in range(num_len)]
    
    def get_digits_as_vectors(num, basis = 10, num_len = 11):
        digits = get_digits(num, basis, num_len)
        return [[np.cos(2*np.pi*d/basis), np.sin(2*np.pi*d/basis)] for d in digits]
    

    assert get_digits(1234) == [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4]
    assert get_digits(1024, basis=2) == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


    for i in tqdm(range(params['numbers']), delay=120):
        hidden_states = num_to_hidden[i]

        assert params['positions'] == 1, "Only one position supported for now"

        for layer in layers:
            x = hidden_states[layer][0][-1] # last token
            if i in params['exclude']:
                X_test.append(x)
                digits = get_digits(i, basis=params['basis'])
                Y_test.append(torch.tensor(digits))
            else:
                X.append(x)
                digits = get_digits_as_vectors(i, basis=params['basis'])
                Y.append(torch.tensor(digits))

    # train the circular probe
    print(f"{X[0].shape=}")
    X = torch.stack(X)
    
    Y = torch.stack(Y)
    Y = Y.to(mt.device)
    Y = Y.float()

    assert X.shape[0] == Y.shape[0], f"{X.shape=}, {Y.shape=}"

    if params['shuffle']:
        perm = torch.randperm(X.shape[0])
        X = X[perm]
        Y = Y[perm]

    optimizer = torch.optim.Adam(circular_probe.parameters(), lr=params['lr'])
    loss_fn = torch.nn.MSELoss(reduction='mean')

    for epoch in tqdm(range(params['epochs']), delay=120):
        
        for i in range(0, len(X), params['batch_size']):
            X_batch = X[i:i+params['batch_size']]
            Y_batch = Y[i:i+params['batch_size']]

            optimizer.zero_grad()
            y_pred = circular_probe(X_batch)

            assert y_pred.shape == Y_batch.shape, f"{y_pred.shape=}, {Y_batch.shape=}"
            
            loss = loss_fn(y_pred, Y_batch)
            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
            logging.info(f"Epoch {epoch}, Loss: {loss.item()}")

    logging.debug("Finished training circular probe")


    logging.debug("Evaluate on test set")
    X_test = torch.stack(X_test)
    Y_test = torch.stack(Y_test)
    Y_test = Y_test.to(mt.device)
    Y_test = Y_test.float()
    assert X_test.shape[0] == Y_test.shape[0]

    y_pred = circular_probe.forward_digit(X_test)
    loss = loss_fn(y_pred, Y_test)
    logging.debug(f"Test loss: {loss.item()}")

    # now prediction accuracy
    correct = 0
    for i in range(len(Y_test)):
        if torch.all((torch.round(y_pred[i]) % params['basis']) == Y_test[i]):
            correct += 1

    acc = correct/len(Y_test)
    logging.debug(f"Accuracy: {acc}")
    
    return acc, circular_probe
