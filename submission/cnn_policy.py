import torch.nn as nn
import torch.nn.functional as F

# Defines the CNN policy network architecture for the PPO agent. 
# The network consists of three convolutional layers followed by two fully connected layers:
class HexCNNPolicy(nn.Module):
    def __init__(self, board_size):
        super().__init__()

        self.board_size = board_size
        self.num_actions = board_size * board_size # Each cell is a potential action

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1) # Input channels: 3 (current player, opponent, empty), Output channels: 64
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # Input channels: 64, Output channels: 64
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.fc_policy = nn.Linear(64 * board_size * board_size, self.num_actions)
        self.fc_value = nn.Linear(64 * board_size * board_size, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # Apply ReLU activation after the first convolutional layer
        x = F.relu(self.conv2(x)) #
        x = F.relu(self.conv3(x))

        x = x.reshape(x.size(0), -1) # Flatten the feature maps

        policy_logits = self.fc_policy(x) # Output raw logits for the policy head
        value = self.fc_value(x).squeeze(-1) 

        return policy_logits, value
    

