import torch
import torch.nn as nn

class MINE(nn.Module):
    """
    Mutual Information Neural Estimator (MINE).
    This network is trained to estimate the mutual information between two input tensors.
    """
    def __init__(self):
        super(MINE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(3, affine=False),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.InstanceNorm2d(3, affine=False),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(256 * 4 * 4, 1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor.
        Returns:
            torch.Tensor: A scalar value representing the MI estimate.
        """
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x = torch.add(x1, x2)
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x