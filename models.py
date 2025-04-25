import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=1456, num_classes=2):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        logits = self.mlp(x)
        return logits
    
class CNN(nn.Module):
    def __init__(self, input_size, input_channels=1, num_classes=2):
        super(CNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First
            nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Second
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Third
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Fourth
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # Fifth
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv_layers(x)
        x = torch.squeeze(self.final_pool(x))
        logits = self.fc(x)

        # x = self.conv_layers(x)
        # x = x.swapaxes(1, 2)
        # logits = self.fc(x)
        # logits = logits.mean(dim=1)

        return logits
