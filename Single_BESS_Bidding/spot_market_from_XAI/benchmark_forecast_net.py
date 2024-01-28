import torch.nn as nn 


class LSTMNet(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        layer_num=3,
        hidden_dim=512
    ) -> None:
        super().__init__()

        self.lstm_layer = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_num, batch_first=True)
        self.linear = nn.Linear(hidden_dim,output_dim)
    
    def forward(self,x):
        lstm_out,(h_n,c_n) = self.lstm_layer(x)
        out = self.linear(h_n[-1,:,:])
        return out