import torch
import torch.nn as nn

class EchoStateNetwork(nn.Module):
    def __init__(self, device, reservoir_size=35, random_seed=None, input_weights=None, spectral_radius=.99, input_scaling=1, n_transient=0):
        super(EchoStateNetwork, self).__init__()
        if random_seed is not None:
            torch.manual_seed(random_seed)

        self.input_scaling = input_scaling
        self.n_transient = n_transient
        self.spectral_radius = spectral_radius
        self.reservoir_size = reservoir_size
        self.device = device
        
        # Gen W_in
        if input_weights is not None:
            self.W_in = input_weights.to(dtype=torch.float64)
        else: 
            self.W_in = (torch.rand((reservoir_size, 1), dtype=torch.float64) * 2.0 - 1.0).to(self.device)

    def get_states(self, X):
        # Initialize the list to store state tensors
        states_list = [torch.zeros(self.reservoir_size, dtype=torch.float64, device=self.device)]

        # Go through samples (steps) and compute states for each of them
        for t in range(1, X.size(0)):
            new_state = torch.tanh(self.W_in @ X[t] + self.W_res @ states_list[-1])
            states_list.append(new_state)

        # Convert the list of states into a tensor
        states = torch.stack(states_list)

        return states
        
    def forward(self, W_res, X):
        # Scale X according to input_scaling
        X = X * self.input_scaling  
        
        self.reservoir_size = W_res.shape[0]
        spectral_radius = torch.max(torch.abs(torch.linalg.eigvals(W_res))).item()
        self.W_res = W_res.to(dtype=torch.float64) / spectral_radius
        
        states = self.get_states(X)
        
        combined_states = torch.cat((states, X), dim=1)
        
        return combined_states
    
    def train_output_weights(self, states, targets):
        if targets.dim() == 1:
            targets = targets.view(-1, 1)
            
        self.n_outputs = targets.shape[1]
        # Solve for W_out using pseudoinverse
        self.W_out = (torch.linalg.pinv(states) @ targets).T
        
    def predict(self, X):
        # Scale X according to input_scaling
        X = X * self.input_scaling
        # X = X.unsqueeze(1)
        
        states = self.get_states(X)
        
        # Initialize predictions tensor
        y_pred = torch.zeros((X.size(0), self.n_outputs), dtype=torch.float64, device=self.device)

        # Compute predictions based on the states and input
        for t in range(1, X.size(0)):
            combined_states = torch.cat((states[t], X[t]), dim=0)
            y_pred[t, :] = self.W_out @ combined_states

        return y_pred
