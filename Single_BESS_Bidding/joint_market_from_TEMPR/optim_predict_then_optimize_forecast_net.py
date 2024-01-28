import torch.nn as nn 


class LSTMNet(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=100
    ) -> None:
        super(LSTMNet,self).__init__() 
        self.lstm_layer = nn.LSTM(input_dim,hidden_dim,batch_first=True)
        # self.lstm_layer = nn.LSTM(hidden_dim,hidden_dim)
        self.linear = nn.Linear(hidden_dim,output_dim)
    
    def forward(self,x):
        # if len(x.shape) == 2: x = torch.reshape(x,shape=(1,))
        # x = torch.reshape
        lstm_out,(h_n,c_n) = self.lstm_layer(x)
        out = self.linear(h_n)
        return out


class TransformerNet(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        feature_dim,
        nhead,
        num_layers,
        init_w=3e-3
    ) -> None:
        super(TransformerNet,self).__init__()

        self.input_embed =  nn.Linear(input_dim,feature_dim)
        self.enc = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=feature_dim,nhead=nhead,batch_first=True),
            num_layers=num_layers,
            norm=nn.LayerNorm(feature_dim)
        )
        self.avg_pool1d = nn.AdaptiveAvgPool1d(output_size=1)
        self.output_layer = nn.Linear(feature_dim,output_dim)

        self.output_layer.weight.data.uniform_(-init_w,init_w)
        self.output_layer.bias.data.uniform_(-init_w,init_w)
    

    def forward(self,x):
        x = self.input_embed(x) # b*s*f 
        x = self.enc(x) # b*s*f
        x = x.permute(0,2,1) # b*f*s
        x = self.avg_pool1d(x).squeeze(-1) # b*f
        x = self.output_layer(x) # b * output_dim

        return x 