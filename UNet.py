import torch
import torch.nn as nn
from collections import deque

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) 
        )

    def forward(self, X):
        return self.conv(X)

class UNET(nn.Module):
    """
    Attributes:
      input_channels = num channels of input, int
      output_channels = num channels of output, int
      features = list of features on each layer, list
    Usage:
      model = UNET(input_channels, output_channels, features).to(device)
    """
    def __init__(self, in_channels = 3, out_channels = 1, features = [64,128,256,512]):
        super(UNET, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2,2)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, 2, 2)
            )
            self.ups.append(
                DoubleConv(feature * 2, feature)
            )
    
    def forward(self, x):
        skip_connections =  deque()

        for down in self.downs:
            x = down(x)
            skip_connections.appendleft(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for idx, up in enumerate(self.ups):
            if not (idx%2):
                x = up(x)
                skip_connection = skip_connections.popleft()

                diffY = skip_connection.size()[2] - x.size()[2]
                diffX = skip_connection.size()[3] - x.size()[3]
                skip_cropped = skip_connection[:, :, diffY // 2 : (diffY // 2 + x.size()[2]), diffX // 2 : (diffX // 2 + x.size()[3])]

                concat_skip = torch.cat((skip_cropped, x), 1)
            
            x = up(concat_skip)

        return self.final_conv(x)