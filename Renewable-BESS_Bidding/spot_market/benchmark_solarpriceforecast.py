import pandas as pd 
import os 
import numpy as np 
import argparse
import time 

import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader

from benchmark_utils import split_train_test
from benchmark_utils import SolarPriceForecastDataset
from benchmark_utils import LSTMNet

class SolarPriceForecaster():
    def __init__(
        self,
        solar_or_price,
        model
    ) -> None:
        super().__init__()

        self.solar_or_price = solar_or_price
        
        self.model = model 
        
        self.model_save_folder = 'benchmark_forecast_model/price' if solar_or_price=='price' else 'benchmark_forecast_model/solar'
        os.makedirs(self.model_save_folder,exist_ok=True)
        self.model_file_name = 'price_forecast' if solar_or_price=='price' else 'solar_forecast' 
        self.model_save_path = os.path.join(self.model_save_folder,self.model_file_name)

        res_save_folder = 'benchmark_forecast_res/price' if solar_or_price=='price' else 'benchmark_forecast_res/solar' 
        os.makedirs(res_save_folder,exist_ok=True)
        train_res_file_name = 'price_forecast_train.txt' if solar_or_price=='price' else 'solar_forecast_train.txt'
        self.train_res_save_path = os.path.join(res_save_folder,train_res_file_name)
        eval_res_file_name = 'price_forecast_eval.csv' if solar_or_price=='price' else 'solar_forecast_eval.csv'
        self.eval_res_save_path = os.path.join(res_save_folder,eval_res_file_name)



    # load train and eval data 
    def load_data(
        self,
        market_year,
        state_name,
        solar_farm_name,
        batch_size,
        device,
        lookback,
        solar_input_folder='solar_data',
        price_input_folder='NEM_annual_data'
    ):
        # read solar or price data 
        if self.solar_or_price == 'solar':
            solar_read_folder = os.path.join(solar_input_folder,'{}'.format(market_year))
            solar_file_name = '{}_{}1_Solar_{}_5min.csv'.format(market_year,state_name,solar_farm_name)
            solar_read_path = os.path.join(solar_read_folder,solar_file_name)
            data = pd.read_csv(solar_read_path)
            data['DateTime'] = data['Unnamed: 0']
            data = data.drop_duplicates(subset=['DateTime']).reset_index(drop=True)['AVAILABILITY']
        else: # price forecast 
            price_file_name = '{}{}.csv'.format(state_name,market_year)
            price_read_path = os.path.join(price_input_folder,price_file_name)
            data = pd.read_csv(price_read_path,usecols=['RRP'])

        
        # split train and test 
        one_month_len = int(60/5*24*31)
        data = data.iloc[:-one_month_len-lookback].to_numpy().flatten()

        train_test_ratio = 0.8
        train_x_arr,train_y_arr,test_x_arr,test_y_arr = split_train_test(arr=list(data),train_test_ratio=train_test_ratio,lookback=lookback)

        train_x_te = torch.FloatTensor(train_x_arr).to(device)
        train_y_te = torch.FloatTensor(train_y_arr).to(device)
        test_x_te = torch.FloatTensor(test_x_arr).to(device)
        test_y_te = torch.FloatTensor(test_y_arr).to(device)

        # get Dataset and Dataloader 
        train_dataset = SolarPriceForecastDataset(x=train_x_te,y=train_y_te)
        test_dataset = SolarPriceForecastDataset(x=test_x_te,y=test_y_te)

        train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
        test_dataloader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=False)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader


    # train and eval 
    def train_and_eval(self):
        epoch_num = 100
        # epoch_num = 1 # debug 
        loss_func = nn.MSELoss()
        lr = 1e-3 
        optimizer = optim.Adam(self.model.parameters(),lr=lr)

        # train 
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
            print('{},Epoch:{},Train Loss:{}'.format(self.solar_or_price,epoch_idx,np.round(train_loss,4))) 
            if epoch_idx % 50 == 0:
                torch.save(self.model.state_dict(),self.model_save_path)
        torch.save(self.model.state_dict(),self.model_save_path)
        np.savetxt(self.train_res_save_path,np.array(train_loss_lst))

        # eval 
        self.model.eval()
        test_loss = 0
        actual_y_lst = []
        pred_y_lst = []
        for idx,(x,y) in enumerate(self.test_dataloader):
            pred_y = self.model(x)
            loss = loss_func(pred_y,y)
            test_loss += loss.detach().cpu().item()

            pred_y_lst.append(pred_y.detach().cpu().numpy().flatten()[0])
            actual_y_lst.append(y.detach().cpu().numpy().flatten()[0])
        res_dict = {'pred_y':pred_y_lst,'actual_y':actual_y_lst}
        res_df = pd.DataFrame(res_dict)
        res_df.to_csv(self.eval_res_save_path)
        print('{},Test Loss:{}'.format(self.solar_or_price,np.round(test_loss,4)))
    
    
    
    # load model 
    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_save_path))
        self.model.eval()
    
    # online inference 
    def inference(self,pred_len,his_x,device,lookback):
        his_x = list(his_x)
        
        pred_lst = []
        for idx in range(pred_len):
            if idx == 0: 
                input_x = his_x
            elif idx <= len(his_x)-1:
                input_x = his_x[idx:len(his_x)] + pred_lst
            else:
                input_x = pred_lst[idx-len(his_x):idx-len(his_x)+lookback]
            pred_y = self.model(torch.FloatTensor(input_x).to(device).unsqueeze(0).unsqueeze(-1)).detach().cpu().numpy().flatten()[0]
            pred_lst.append(pred_y)

        return pred_lst




if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--solar_or_price',default='solar',type=str,help='train solar or price')
    parser.add_argument('--market_year',default=2020,type=int,help='read data begin')
    parser.add_argument('--state_name',default='QLD',type=str,help='state name')
    parser.add_argument('--solar_farm_name',default='RUGBYR1',type=str,help='solar farm name')
    parser.add_argument('--device',default='cuda:6',type=str,help='training device')
    args = parser.parse_args()

    solar_or_price = args.solar_or_price
    market_year = args.market_year
    state_name = args.state_name
    solar_farm_name = args.solar_farm_name
    
    # init forecaster 
    lookback = int(60/5*24)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    batch_size = 256
    forecast_net = LSTMNet(
        input_dim=1,
        output_dim=1
    ).to(device)
    forecaster = SolarPriceForecaster(
        solar_or_price=solar_or_price,
        model=forecast_net
    )


    # load train&test data 
    forecaster.load_data(
        market_year=market_year,
        state_name=state_name,
        solar_farm_name=solar_farm_name,
        batch_size=batch_size,
        device=device,
        lookback=lookback,
    )

    # train 
    forecaster.train_and_eval()

    end_time = time.time()