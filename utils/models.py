import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from IPython.display import clear_output as clc

# Which one is correct (xD)?
# from utils.processdata import mse, mre, num2p
from .processdata import mse, mre, num2p

class SHRED(torch.nn.Module):

    def __init__(self, input_size, output_size, hidden_size = 64, hidden_layers = 2, decoder_sizes = [350, 400], dropout = 0.0):
        '''
        SHRED model accepts:
            -   input size (e.g., number of sensors), 
            -   output size (dimension of low or high-dimensional state,
            -   hidden_size and number of LSTM layers,
            -   l1 and l2 represents the size of the decoder network.
        '''
            
        super(SHRED,self).__init__()

        self.lstm = torch.nn.LSTM(input_size = input_size,
                                  hidden_size = hidden_size,
                                  num_layers = hidden_layers,
                                  batch_first=True)
        
        self.decoder = torch.nn.ModuleList()
        decoder_sizes.insert(0, hidden_size)
        decoder_sizes.append(output_size)

        for i in range(len(decoder_sizes)-1):
            self.decoder.append(torch.nn.Linear(decoder_sizes[i], decoder_sizes[i+1]))
            if i != len(decoder_sizes)-2:
                self.decoder.append(torch.nn.Dropout(dropout))
                self.decoder.append(torch.nn.ReLU())

        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        '''
        Input: input data
        Output: SHRED evaluation at the input data
        Definition of the forward function for the SHRED network.
        '''
        
        h_0 = torch.zeros((self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float)
        c_0 = torch.zeros((self.hidden_layers, x.size(0), self.hidden_size), dtype=torch.float)
        if next(self.parameters()).is_cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        _, (output, _) = self.lstm(x, (h_0, c_0))
        output = output[-1].view(-1, self.hidden_size)

        for layer in self.decoder:
            output = layer(output)

        return output

    def freeze(self):

        self.eval()
        
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):

        self.train()
        
        for param in self.parameters():
            param.requires_grad = True

def fit(model, train_dataset, valid_dataset, 
        batch_size = 64, epochs = 4000, optim = torch.optim.Adam, lr = 1e-3, verbose = False, patience = 5):
    '''
    Function for training SHRED and SDN models.
    Accepts the model (`torch.nn.Module`), training dataset and validation dataset.
    The loss function is `torch.nn.MSELoss()`, the default batch size is 64, the default number of epochs is 4000, the default learning rate is 1e-3.
    Default optimizer is `torch.optim.Adam`.
    '''

    train_loader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size)
    optimizer = optim(model.parameters(), lr = lr)

    train_error_list = []
    valid_error_list = []
    patience_counter = 0
    best_params = model.state_dict()

    for epoch in range(1, epochs + 1):
        
        for k, data in enumerate(train_loader):
            model.train()
            def closure():
                outputs = model(data[0])
                optimizer.zero_grad()
                loss = mse(outputs, data[1])
                loss.backward()
                return loss
            optimizer.step(closure)

        model.eval()
        with torch.no_grad():
            train_error = mre(train_dataset.Y, model(train_dataset.X))
            valid_error = mre(valid_dataset.Y, model(valid_dataset.X))
            train_error_list.append(train_error)
            valid_error_list.append(valid_error)

        if verbose == True:
            # print("\t\tTrain \tValid")
            # print("Epoch "+ str(epoch) + ":\t" + num2p(train_error_list[-1]) + "\t" + num2p(valid_error_list[-1]))
            # clc(wait = True)
            print("Epoch "+ str(epoch) + ": Train = " + num2p(train_error_list[-1]) + " and Valid = " + num2p(valid_error_list[-1]), end="\r")

        if valid_error == torch.min(torch.tensor(valid_error_list)):
            patience_counter = 0
            best_params = deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter == patience:
            if verbose == True:
                print("Epoch "+ str(epoch) + ": Train = " + num2p(train_error_list[-1]) + " and Valid = " + num2p(valid_error_list[-1]))
            model.load_state_dict(best_params)
    
            return torch.tensor(train_error_list).detach().cpu().numpy(), torch.tensor(valid_error_list).cpu()
    
    if verbose == True:
        print("Epoch "+ str(epoch) + ": Train = " + num2p(train_error_list[-1]) + " and Valid = " + num2p(valid_error_list[-1]))
            
    model.load_state_dict(best_params)
    return torch.tensor(train_error_list).detach().cpu().numpy(), torch.tensor(valid_error_list).detach().cpu().numpy()

def forecast(forecaster, steps, test_dataset):
    '''
    Forecast a test time series in time
    Inputs: forecaster model, number of forecasting steps and test time series of dimension (forecaster input size, forcaster output size)
    '''   

    input = test_dataset.clone()

    forecast = torch.zeros(steps, test_dataset.shape[1])
    for i in range(steps):
        forecast[i] = forecaster(input)
        input[:-1] = input[1:]
        input[-1] = forecast[i]

    return forecast














