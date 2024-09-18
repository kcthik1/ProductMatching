import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, cnn_output_channels, rnn_hidden_size, rnn_num_layers, num_classes):
        super(CRNN, self).__init__()
        
        # CNN part
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d((2,2), (2,1), (0,1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d((2,2), (2,1), (0,1)),
            nn.Conv2d(512, cnn_output_channels, kernel_size=2, padding=0), nn.BatchNorm2d(cnn_output_channels), nn.ReLU()
        )
        
        # RNN part
        self.rnn = nn.GRU(cnn_output_channels, rnn_hidden_size, num_layers=rnn_num_layers, bidirectional=True, batch_first=True)
        
        # Prediction part
        self.predictor = nn.Linear(rnn_hidden_size * 2, num_classes)

    def forward(self, x):
        # x: (batch, 1, height, width)
        conv = self.cnn(x)
        
        # (batch, cnn_output_channels, height/16, width/4) -> (batch, width/4, cnn_output_channels)
        batch, c, h, w = conv.size()
        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)
        
        # RNN
        rnn, _ = self.rnn(conv)
        
        # Prediction
        output = self.predictor(rnn)
        
        # (batch, time_step, num_classes)
        return output.permute(1, 0, 2)
