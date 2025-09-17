import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from IPython.display import clear_output as clc
from .processdata import mse, mre, num2p
from torch.optim.lr_scheduler import StepLR
import numpy as np

class SHRED(torch.nn.Module):

    def __init__(self, input_size, output_size, hidden_size = 64, hidden_layers = 2, decoder_sizes = [350, 400], dropout = 0.0, activation = torch.nn.ReLU()):
        '''
        SHRED model definition
        
        
        Inputs
        	input size (e.g. number of sensors)
        	output size (e.g. full-order variable dimension)
        	size of LSTM hidden layers (default to 64)
        	number of LSTM hidden layers (default to 2)
        	list of decoder layers sizes (default to [350, 400])
        	dropout parameter (default to 0)
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
                self.decoder.append(activation)

        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        
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

def fit(model, train_dataset, valid_dataset, batch_size = 64, epochs = 4000, optim = torch.optim.Adam, lr = 1e-3, loss_fun = mse, loss_output = mre, formatter = num2p, verbose = False, patience = 5, step_size=200):
    '''
    Neural networks training
    
    Inputs
    	model (`torch.nn.Module`)
    	training dataset (`torch.Tensor`)
    	validation dataset (`torch.Tensor`)
    	batch size (default to 64)
    	number of epochs (default to 4000)
    	optimizer (default to `torch.optim.Adam`)
    	learning rate (default to 0.001)
        loss function (defalut to Mean Squared Error)
        loss value to print and return (default to Mean Relative Error)
        loss formatter for printing (default to percentage format)
    	verbose parameter (default to False) 
    	patience parameter (default to 5)
    '''

    train_loader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size)
    optimizer = optim(model.parameters(), lr = lr)

    scheduler = StepLR(optimizer, step_size, gamma=0.1)

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
                loss = loss_fun(outputs, data[1])
                loss.backward()
                return loss
            optimizer.step(closure)
            

        model.eval()
        scheduler.step()

        with torch.no_grad():
            train_error = loss_output(train_dataset.Y, model(train_dataset.X))
            valid_error = loss_output(valid_dataset.Y, model(valid_dataset.X))
            train_error_list.append(train_error)
            valid_error_list.append(valid_error)
        
        if verbose == True:
            print("Epoch "+ str(epoch) + ": Training loss = " + formatter(train_error_list[-1]) + " \t Validation loss = " + formatter(valid_error_list[-1]) + " "*10 + " \t learning rate = " + str(scheduler.get_last_lr()[0]),  end = "\r")

        if valid_error == torch.min(torch.tensor(valid_error_list)):
            patience_counter = 0
            best_params = deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter == patience:
            model.load_state_dict(best_params)
            train_error = loss_output(train_dataset.Y, model(train_dataset.X))
            valid_error = loss_output(valid_dataset.Y, model(valid_dataset.X))
            
            if verbose == True:
                print("Training done: Training loss = " + formatter(train_error) + " \t Validation loss = " + formatter(valid_error) + " \t learning rate = " + str(scheduler.get_last_lr()[0]))
         
            return torch.tensor(train_error_list).detach().cpu().numpy(), torch.tensor(valid_error_list).detach().cpu().numpy()
    
    model.load_state_dict(best_params)
    train_error = loss_output(train_dataset.Y, model(train_dataset.X))
    valid_error = loss_output(valid_dataset.Y, model(valid_dataset.X))
    
    if verbose == True:
    	print("Training done: Training loss = " + formatter(train_error) + " \t Validation loss = " + formatter(valid_error))
    
    return torch.tensor(train_error_list).detach().cpu().numpy(), torch.tensor(valid_error_list).detach().cpu().numpy()
 
def forecast(forecaster, input_data, steps, nsensors):
    '''
    Forecast time series in time
    Inputs
    	forecaster model (`torch.nn.Module`)
        starting time series of dimension (ntrajectories, lag, nsensors+nparams)
    	number of forecasting steps
        number of sensors
    Outputs
        forecast of the time series in time
    '''   

    forecast = []
    for i in range(steps):
        forecast.append(forecaster(input_data))
        temp = input_data.clone()
        input_data[:,:-1] = temp[:,1:]
        input_data[:,-1, :nsensors] = forecast[i]

    return torch.stack(forecast, 1)




# DEFINE lstm deepOnet MODEL

class lstm_deepONet(torch.nn.Module):

    def __init__(self, input_size , output_size, hidden_size = 64, hidden_layers = {'branch':2, 'trunk':2}, dropout = 0.1, basis_functions = 10, activation = torch.nn.ReLU()):
        '''
        Deeponet arcitecture for time series data. 
        In branch net is a lstm network
        Trunk netwrok is a simple feedforward network 
        Inputs
            input size  for branch and trunk                    (`dict`)
        	output size (e.g. full-order variable dimension)    (`int`)
        	size hidden layers                          (`int`)
        	number of hidden layers                        (`int`)
        	list of decoder layers sizes                        (`list[int]`)
        	dropout parameter                                   (`float`)
            activation                                          ('function')
        '''

        super(lstm_deepONet,self).__init__()

        # create lstm branch net for handling sequential data 
        self.branch_lstm = torch.nn.LSTM(input_size = input_size['branch'],
                                  hidden_size = hidden_size,
                                  num_layers = hidden_layers['branch'],
                                  batch_first=True)
        # encode lstm network
        self.branch_encoder = torch.nn.Linear(hidden_size, basis_functions * output_size)
        
        # layers in trunk network 
        layers = []
            
        layers.append(torch.nn.Linear(input_size['trunk'], hidden_size))
        layers.append(activation)
        
        for _ in range(hidden_layers['trunk'] - 1):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            
            layers.append(activation)
        
        layers.append(torch.nn.Linear(hidden_size, basis_functions * output_size) )
        
        self.trunk = torch.nn.Sequential(*layers) #simple FNN 
        
        # store variables for later use 
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.basis_functions = basis_functions
        
    def forward(self, x):
        trunk_input = x[0]
        branch_input = x[1] 
        batch_size = branch_input.size(0)
        device = branch_input.device

        h_0 = torch.zeros(self.hidden_layers['branch'], batch_size, self.hidden_size, device=device) 
        c_0 = torch.zeros(self.hidden_layers['branch'], batch_size, self.hidden_size, device=device) 

        _, (output_branch, _) = self.branch_lstm(branch_input, (h_0, c_0))
        output_branch = self.branch_encoder(output_branch[-1])  # last layer's hidden state
        output_branch = output_branch.view(-1, self.basis_functions, self.output_size) #reshape for innerproduct

        trunk_input = trunk_input.view(batch_size, -1) 
        output_trunk = self.trunk(trunk_input)
        output_trunk = output_trunk.view(-1, self.basis_functions, self.output_size) #reshape for innerproduct

        out = torch.sum(output_trunk * output_branch, dim=1)
        return out


    def freeze(self):

        self.eval()

        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):

        self.train()

        for param in self.parameters():
            param.requires_grad = True



mse = lambda datatrue, datapred: (datatrue - datapred).pow(2).sum(axis = -1).mean()  # Mean Squared Error



def fit_deepONet(model, train_dataset, valid_dataset=None, batch_size=64, epochs=50,
        optim=torch.optim.Adam, lr=1e-3, loss_fun=mre, loss_output=mre,
         verbose=False, patience=5, step_size = 50, device='cpu', formatter = num2p):

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    optimizer = optim(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)
    
    train_error_list = []
    valid_error_list = []
    patience_counter = 0
    best_params = deepcopy(model.state_dict())

    model.to(device)

    for epoch in range(1, epochs + 1):

        model.train()
        for batch in train_loader:
            # unpack branch, trunk, and target
            (trunk_input, branch_input), y = batch  
            trunk_input = trunk_input.to(device)
            branch_input = branch_input.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(trunk_input, branch_input)
            # print(np.shape(outputs), np.shape(y))
            loss = loss_fun(y, outputs)
            loss.backward()
            optimizer.step()

        # evaluation
        model.eval()
        scheduler.step()

        with torch.no_grad():
            if valid_dataset is not None:
                train_error = loss_output(train_dataset.Y.to(device), model(train_dataset.trunk_data, train_dataset.branch_data))
                valid_error = loss_output(valid_dataset.Y.to(device), model(valid_dataset.trunk_data, valid_dataset.branch_data))
            else:
                train_error = loss_output(train_dataset.Y.to(device), model(train_dataset.trunk_data, train_dataset.branch_data))
                valid_error = train_error

            train_error_list.append(train_error.item())
            valid_error_list.append(valid_error.item())

        # verbose
        if verbose:
            print(f"Epoch {epoch}: Train Loss = {formatter(train_error)}, Valid Loss = {formatter(valid_error)}, learning rate = {scheduler.get_last_lr()[0]}")

        # early stopping
        if valid_error == min(valid_error_list):
            patience_counter = 0
            best_params = deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            model.load_state_dict(best_params)
            if verbose:
                print("Early stopping triggered")
            break

    model.load_state_dict(best_params)
    return train_error_list, valid_error_list










