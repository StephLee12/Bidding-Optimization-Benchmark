import pandas as pd 
import argparse
import time 
import numpy as np 
import os 

import torch  
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader


from benchmark_solarpriceforecast import SolarPriceForecaster
from benchmark_utils import LSTMNetWithSigmoidOutput
from benchmark_utils import LSTMNet
from benchmark_utils import NoLabelDataset
from benchmark_utils import CurtailForecastDataset




class SolarCurtailForecaster:
    def __init__(
        self,
        solar_forecaster,
        model
    ) -> None:

        self.solar_forecaster = solar_forecaster

        self.model = model 

        self.model_save_folder = 'benchmark_forecast_model/curtail'
        os.makedirs(self.model_save_folder,exist_ok=True)
        self.model_file_name = 'curtail_forecast' 
        self.model_save_path = os.path.join(self.model_save_folder,self.model_file_name)

        res_save_folder = 'benchmark_forecast_res/curtail' 
        os.makedirs(res_save_folder,exist_ok=True)
        train_res_file_name = 'curtail_forecast_train.txt' 
        self.train_res_save_path = os.path.join(res_save_folder,train_res_file_name)
        eval_res_file_name = 'curtail_forecast_eval.csv' 
        self.eval_res_save_path = os.path.join(res_save_folder,eval_res_file_name)

    def load_data(
        self,
        market_year,
        state_name,
        solar_farm_name,
        batch_size,
        device,
        lookback,
        solar_input_folder='solar_data',
    ):  
        # read solar 
        solar_read_folder = os.path.join(solar_input_folder,'{}'.format(market_year))
        solar_file_name = '{}_{}1_Solar_{}_5min.csv'.format(market_year,state_name,solar_farm_name)
        solar_read_path = os.path.join(solar_read_folder,solar_file_name)
        data = pd.read_csv(solar_read_path)
        data['DateTime'] = data['Unnamed: 0']
        data = data.drop_duplicates(subset=['DateTime']).reset_index(drop=True)['AVAILABILITY']

        # get predicted solar gen
        one_month_len = int(60/5*24*31)
        data = list(data.iloc[:-one_month_len-lookback].to_numpy().flatten())

        solar_gen_lst = []
        for idx in range(len(data)-lookback+1):
            gen_lst = data[idx:idx+lookback]
            solar_gen_lst.append(gen_lst)
        solar_gen_lst.pop() # remove the last prediction for the next year.
        solar_gen_arr = np.expand_dims(np.array(solar_gen_lst),axis=-1)
        solar_gen_te = torch.FloatTensor(solar_gen_arr).to(device)

        solar_gen_dataset = NoLabelDataset(x=solar_gen_te)
        solar_gen_dataloader = DataLoader(dataset=solar_gen_dataset,batch_size=1,shuffle=False)
        
        pred_solar_lst = []
        for idx,x in enumerate(solar_gen_dataloader):
            pred_y = self.solar_forecaster.model(x).detach().cpu().numpy().flatten()[0]
            pred_solar_lst.append(pred_y)

        # get curtail flag 
        actual_solar_lst = data[lookback:]
        curtail_flag_lst = []
        for pred_solar,actual_solar in zip(pred_solar_lst,actual_solar_lst):
            if actual_solar > pred_solar: curtail_flag_lst.append(1)
            else: curtail_flag_lst.append(0)
        
        # create curtail train set 
        curtail_flag_te = torch.FloatTensor(np.array(curtail_flag_lst)).to(device).unsqueeze(-1).unsqueeze(-1)

        train_test_split_len = int(0.8*len(solar_gen_lst))
        train_x_te = solar_gen_te[:train_test_split_len,:,:]
        train_y_te = curtail_flag_te[:train_test_split_len,:,:]
        test_x_te = solar_gen_te[train_test_split_len:,:,:]
        test_y_te = curtail_flag_te[train_test_split_len:,:,:]

        train_dataset = CurtailForecastDataset(x=train_x_te,y=train_y_te)
        test_dataset = CurtailForecastDataset(x=test_x_te,y=test_y_te)

        train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
        test_dataloader = DataLoader(dataset=test_dataset,batch_size=1,shuffle=False)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader



    def train_and_eval(self):
        # epoch_num = 1 # debug
        epoch_num = 100 
        loss_func = nn.BCELoss()
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
            print('Curtail,Epoch:{},Train Loss:{}'.format(epoch_idx,np.round(train_loss,4))) 
            if epoch_idx % 50 == 0:
                torch.save(self.model.state_dict(),self.model_save_path)
        torch.save(self.model.state_dict(),self.model_save_path)
        np.savetxt(self.train_res_save_path,np.array(train_loss_lst))

        # eval 
        self.model.eval()
        test_acc = 0
        actual_y_lst = []
        pred_y_lst = []
        for idx,(x,y) in enumerate(self.test_dataloader):
            pred_y = self.model(x).detach().cpu().numpy().flatten()[0]
            pred_y = int(np.round(pred_y))
            actual_y = int(y.detach().cpu().numpy().flatten()[0])

            if pred_y == actual_y: test_acc += 1

            pred_y_lst.append(pred_y)
            actual_y_lst.append(actual_y)
        res_dict = {'pred_y':pred_y_lst,'actual_y':actual_y_lst}
        res_df = pd.DataFrame(res_dict)
        res_df.to_csv(self.eval_res_save_path)
        print('Curtail,Accuracy:{}'.format(np.round(test_acc/(idx+1),4)))

    # load model 
    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_save_path))
        self.model.eval()
    
    # online inference 
    def inference(self,pred_len,his_x,pred_x,device,lookback):
        his_x = list(his_x)
        
        pred_lst = []
        for idx in range(pred_len):
            if idx == 0: 
                input_x = his_x
            elif idx <= len(his_x)-1:
                input_x = his_x[idx:len(his_x)] + pred_x[:idx]
            else:
                input_x = pred_x[idx-len(his_x):idx-len(his_x)+lookback]
            pred_y = self.model(torch.FloatTensor(input_x).to(device).unsqueeze(0).unsqueeze(-1)).detach().cpu().numpy().flatten()[0]
            pred_lst.append(int(np.round(pred_y)))

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

    market_year = args.market_year
    state_name = args.state_name
    solar_farm_name = args.solar_farm_name

    # load  solar forecaster 
    solar_or_price = args.solar_or_price
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    solar_forecast_net = LSTMNet(
        input_dim=1,
        output_dim=1
    ).to(device)
    solar_forecaster = SolarPriceForecaster(
        solar_or_price=solar_or_price,
        model=solar_forecast_net
    )
    solar_forecaster.load_model()

    # init curtail forecaster 
    lookback = int(60/5*24)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    batch_size = 256
    curtail_forecast_net = LSTMNetWithSigmoidOutput(
        input_dim=1,
        output_dim=1
    ).to(device)
    curtail_forecaster = SolarCurtailForecaster(
        solar_forecaster=solar_forecaster,
        model=curtail_forecast_net
    )
    curtail_forecaster.load_data(
        market_year=market_year,
        state_name=state_name,
        solar_farm_name=solar_farm_name,
        batch_size=batch_size,
        device=device,
        lookback=lookback
    )

    # train 
    curtail_forecaster.train_and_eval()

    end_time = time.time()

