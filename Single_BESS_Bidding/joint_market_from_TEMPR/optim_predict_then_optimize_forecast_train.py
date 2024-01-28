import os 
import pandas as pd 
import numpy as np 
import time 
import argparse


import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader


from optim_predict_then_optimize_forecast_net import LSTMNet,TransformerNet
from optim_predict_then_optimize_utils import create_seq,PriceForecastDataset



class Forecaster:
    def __init__(
        self,
        lstm_or_trans,
        model,
        mode,
        state_name,
        market_year
    ) -> None:
        
        self.lstm_or_trans = lstm_or_trans
        self.model = model 

        self.mode = mode 
        self.state_name = state_name
        self.market_year = market_year

        model_save_folder = 'predict_then_optimize/{}_models/{}{}'.format(lstm_or_trans,state_name,market_year)
        os.makedirs(model_save_folder,exist_ok=True)
        model_save_name = 'price_model_params_{}'.format(mode)
        model_save_path = os.path.join(model_save_folder,model_save_name)
        train_loss_save_name = 'price_train_loss_{}.txt'.format(mode)
        train_loss_save_path = os.path.join(model_save_folder,train_loss_save_name)
        eval_loss_save_name = 'price_eval_loss_{}.txt'.format(mode)
        eval_loss_save_path = os.path.join(model_save_folder,eval_loss_save_name)
        eval_actual_pred_y_save_name = 'price_eval_actual_pred_y_{}.csv'.format(mode)
        eval_actual_pred_y_save_path = os.path.join(model_save_folder,eval_actual_pred_y_save_name)
        self.model_save_path = model_save_path
        self.train_loss_save_path = train_loss_save_path
        self.eval_loss_save_path = eval_loss_save_path
        self.eval_actual_pred_y_save_path = eval_actual_pred_y_save_path




    def load_data(
        self,
        input_len,
        device,
        batch_size,
        data_folder='NEM_annual_data',
        train_test_split_ratio=0.8
    ):
        # load price data 
        data_file_name = '{}{}.csv'.format(self.state_name,self.market_year)
        data_path = os.path.join(data_folder,data_file_name)
        data_df = pd.read_csv(data_path,index_col=[0])


        last_two_month_len = int((60/5)*24*31) + int((60/5)*24*30)
        # get the first ten month of data 
        data_df = data_df.iloc[:-last_two_month_len,:].reset_index(drop=True)

        if self.mode == 'ES': price_cols = ['RRP']
        elif self.mode == 'Contingency_FCAS': price_cols = ['RAISE6SECRRP','LOWER6SECRRP','RAISE60SECRRP','LOWER60SECRRP','RAISE5MINRRP','LOWER5MINRRP']
        else: price_cols = ['RRP','RAISE6SECRRP','LOWER6SECRRP','RAISE60SECRRP','LOWER60SECRRP','RAISE5MINRRP','LOWER5MINRRP']

        # ------ prepare forecast model data -----------
        price_df = data_df[price_cols]
        data_len = price_df.shape[0]
        train_price_df = price_df.iloc[:int(train_test_split_ratio*data_len),:]
        test_price_df = price_df.iloc[int(train_test_split_ratio*data_len):,:].reset_index(drop=True)
        train_price_arr = train_price_df.values
        test_price_arr = test_price_df.values 

        # create sequential data 
        train_x,train_y = create_seq(data=train_price_arr,seq_len=input_len)
        test_x,test_y = create_seq(data=test_price_arr,seq_len=input_len)
        # cast them to tensor 
        train_x_te = torch.FloatTensor(train_x).to(device)
        train_y_te = torch.FloatTensor(train_y).to(device)
        test_x_te = torch.FloatTensor(test_x).to(device)
        test_y_te = torch.FloatTensor(test_y).to(device)
        # create Dataset and DataLoader
        train_dataset = PriceForecastDataset(x=train_x_te,y=train_y_te)
        test_dataset = PriceForecastDataset(x=test_x_te,y=test_y_te)
        train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
        test_dataloader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=False)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = train_dataloader
        self.test_dataloader= test_dataloader


    def train(
        self,
        epoch_num,
        lr
    ):
        loss_func = nn.MSELoss() 
        optimizer = torch.optim.Adam(self.model.parameters(),lr=lr)


        # train 
        self.model.train()
        train_loss_lst = []
        for epoch_idx in range(1,epoch_num+1):
            train_loss = 0
            for idx,(batch_x,batch_y) in enumerate(self.train_dataloader):
                pred_y = self.model(batch_x)
                loss = loss_func(pred_y.flatten(),batch_y.flatten())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.detach().cpu().item()
            
            train_loss /= (idx+1)
            train_loss_lst.append(train_loss)
            print('Model:{},Epoch:{},Train Loss:{:.2f}'.format(self.lstm_or_trans,epoch_idx,train_loss))

        # save model params and train losses 
        torch.save(self.model.state_dict(),self.model_save_path)
        np.savetxt(self.train_loss_save_path,np.array(train_loss_lst))


    def eval(self):
        loss_func = nn.MSELoss() 
        
        self.model.eval() 
        eval_loss = 0
        actual_y_lst = []
        pred_y_lst = []
        for idx, (x,y) in enumerate(self.test_dataloader):
            pred_y = self.model(x)
            loss = loss_func(pred_y.flatten(),y.flatten())
            eval_loss += loss.detach().cpu().item()

            pred_y_lst.append(list(pred_y.detach().cpu().numpy().flatten()))
            actual_y_lst.append(list(y.detach().cpu().numpy().flatten()))

        # save eval res 
        res_dict = {'pred_y':pred_y_lst,'actual_y':actual_y_lst}
        res_df = pd.DataFrame(res_dict)
        res_df.to_csv(self.eval_actual_pred_y_save_path)

        # save test loss 
        print('Model:{},Test Loss:{:.2f}'.format(self.lstm_or_trans,eval_loss))
        np.savetxt(self.eval_loss_save_path,[eval_loss])




    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_save_path)) 
        self.model.eval()



    def inference(
        self,
        pred_len,
        his_x,
        device,
        input_len
    ):  
        pred_lst = []
        for idx in range(pred_len):
            if idx == 0: 
                input_x = his_x
            elif idx <= his_x.shape[0]-1:
                input_x = np.concatenate([his_x[idx:his_x.shape[0],:],np.array(pred_lst)],axis=0)
            else:
                input_x = pred_lst[idx-len(his_x):idx-len(his_x)+input_len]
            pred_y = self.model(torch.FloatTensor(input_x).to(device).unsqueeze(0)).detach().cpu().numpy().flatten()
            pred_lst.append(list(pred_y))

        return pred_lst 




if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--lstm_or_trans',default='trans',type=str,help='choose model: lstm or transformer')
    parser.add_argument('--mode',default='Contingency_FCAS',type=str,help='')
    parser.add_argument('--market_year',default=2016,type=int,help='read data begin')
    parser.add_argument('--state_name',default='VIC',type=str,help='state name')
    parser.add_argument('--device',default='cuda:0',type=str,help='training device')
    parser.add_argument('--input_len',default=31,type=int,help='input series length')
    parser.add_argument('--batch_size',default=128,type=int,help='batch size ')
    parser.add_argument('--epoch_num',default=200,type=int,help='training epoch numnber')
    parser.add_argument('--lr',default=1e-3,type=float,help='learning rate')
    parser.add_argument('--trans_feature_dim',default=64,type=int,help='transformer parameter -- embedding dimension')
    parser.add_argument('--trans_nhead',default=8,type=int,help='transformer parameter -- nhead')
    parser.add_argument('--trans_num_layers',default=2,type=int,help='transformer params -- num_layers')
    args = parser.parse_args()

    lstm_or_trans = args.lstm_or_trans
    mode = args.mode
    market_year = args.market_year
    state_name = args.state_name
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    input_len = args.input_len 
    batch_size = args.batch_size 
    epoch_num = args.epoch_num
    lr = args.lr
    trans_feature_dim = args.trans_feature_dim
    trans_nhead = args.trans_nhead
    trans_num_layers = args.trans_num_layers
    
    # build nn
    if mode=='ES': price_dim = 1
    elif mode=='Contingency_FCAS': price_dim = 6 
    else: price_dim = 7
    if lstm_or_trans == 'lstm':
        forecast_net = LSTMNet(
            input_dim=price_dim,
            output_dim=price_dim
        ).to(device)
    else:
        forecast_net = TransformerNet(
           input_dim=price_dim,
           output_dim=price_dim,
           feature_dim=trans_feature_dim,
           nhead=trans_nhead,
           num_layers=trans_num_layers
        ).to(device)

    # build trainer
    forecaster = Forecaster(
        lstm_or_trans=lstm_or_trans,
        model=forecast_net,
        mode=mode,
        state_name=state_name,
        market_year=market_year,
    )


    # load train&test data 
    forecaster.load_data(
        input_len=input_len,
        device=device,
        batch_size=batch_size,
    )

    # train
    forecaster.train(
        epoch_num=epoch_num,
        lr=lr
    )


    # eval 
    forecaster.eval()


    end_time = time.time()


    collpse_time = end_time-start_time

    runtime_save_folder = 'predict_then_optimize/{}_models/{}{}'.format(lstm_or_trans,state_name,market_year)
    runtime_save_name = 'price_forecast_run_time_{}.txt'.format(mode)
    runtime_save_path = os.path.join(runtime_save_folder,runtime_save_name)
    np.savetxt(runtime_save_path,[collpse_time])

    