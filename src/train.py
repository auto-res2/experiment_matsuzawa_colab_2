import torch
import torch.nn as nn
import torch.optim as optim

class AEFTMController(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(AEFTMController, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, error_metrics):
        x = self.fc1(error_metrics)
        x = self.relu(x)
        timestep_adjustment = torch.sigmoid(self.fc2(x))
        return timestep_adjustment

class AEFTMModule(nn.Module):
    def __init__(self, adaptive=True, realign=True, input_dim=4):
        super(AEFTMModule, self).__init__()
        self.adaptive = adaptive
        self.realign = realign
        self.controller = AEFTMController(input_dim=input_dim)

    def forward(self, error_metrics, current_state):
        from preprocess import should_realign, recalc_guidance, cached_guidance
        
        if self.adaptive:
            timestep_adj = self.controller(error_metrics).mean()
        else:
            timestep_adj = torch.tensor(1.0)
        
        if self.realign and should_realign(error_metrics):
            new_cond_guidance = recalc_guidance(current_state)
        else:
            new_cond_guidance = cached_guidance(current_state)

        return timestep_adj, new_cond_guidance

def train_aeftm_controller(controller, data_loader, epochs=10, lr=0.001):
    """
    Train the AEFTM controller with dummy training loop
    """
    optimizer = optim.Adam(controller.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (error_metrics, target_dt) in enumerate(data_loader):
            optimizer.zero_grad()
            predicted_dt = controller(error_metrics)
            loss = criterion(predicted_dt, target_dt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {total_loss/len(data_loader):.4f}')
    
    return controller
