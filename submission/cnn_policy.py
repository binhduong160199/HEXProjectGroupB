import torch.nn as nn
import torch.nn.functional as F


class HexCNNPolicy(nn.Module):
    def __init__(self, board_size=None):
        super().__init__()

        self.board_size = board_size

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.policy_head = nn.Conv2d(64, 1, kernel_size=1)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        policy_logits = self.policy_head(x).flatten(start_dim=1)

        pooled_features = F.adaptive_avg_pool2d(x, output_size=1).flatten(start_dim=1)
        value = self.value_head(pooled_features).squeeze(-1)

        return policy_logits, value

