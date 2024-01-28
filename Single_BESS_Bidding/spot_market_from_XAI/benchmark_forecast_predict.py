import os 
import pandas as pd 
import time
import numpy as np 
import argparse
import torch 

from benchmark_forecast_train import Forecaster
from benchmark_utils import create_seq
from benchmark_forecast_net import LSTMNet


class PredictionGenerator:
    def __init__(
        self,
        state_name,
        market_year
    ) -> None:

        self.state_name = state_name
        self.market_year = market_year
        
        pred_save_folder = 'benchmark/forecast/{}{}'.format(state_name,market_year)
        # os.makedirs(pred_save_folder,exist_ok=True)
        pred_save_name = 'price_pred_gen.csv'
        pred_save_path = os.path.join(pred_save_folder,pred_save_name)
        self.pred_save_path = pred_save_path


    def load_data(
        self,
        input_len,
        data_folder='data_annualNEM'
    ):
        data_file_name = '{}{}.csv'.format(self.state_name,self.market_year) 
        data_path = os.path.join(data_folder,data_file_name)
        data_df = pd.read_csv(data_path,index_col=[0])

        price_cols = ['RRP']

        price_df = data_df[price_cols]
        
        
        last_two_month_len = int((60/5)*24*(31+30+31))
        
        to_forecast_price_df = price_df.iloc[-last_two_month_len-input_len:,:].reset_index(drop=True)
        to_forecast_price_df['RRP'] = (to_forecast_price_df['RRP'] - (-1000)) / (15000 - (-1000))
        eval_price_df = price_df.iloc[-last_two_month_len:,:].reset_index(drop=True)
        to_forecast_price_lst = to_forecast_price_df.values.tolist()
        eval_price_lst = eval_price_df.values.tolist()
        self.to_forecast_price_lst = to_forecast_price_lst
        self.eval_price_lst = eval_price_lst

    
    def load_model(self,model):
        model.load_model()

        self.model = model 


    def make_preds(
        self,
        input_len,
        device,
        pred_len
    ):
        to_forecast_price_seq_arr,_ = create_seq(self.to_forecast_price_lst,seq_len=input_len)

        pred_price4mpc_lst = []
        pred_price_lst = []
        for idx,to_forecast_price_seq in enumerate(to_forecast_price_seq_arr):
            # if idx == 1: break # for debug 
            pred_price4mpc = self.model.inference(
                pred_len=pred_len,
                his_x=to_forecast_price_seq, # input_len * price_dim 
                device=device,
                input_len=input_len
            )
            pred_price4mpc = np.array(pred_price4mpc)
            pred_price4mpc = (pred_price4mpc * (15000 - (-1000))) + (-1000)
            pred_price4mpc = pred_price4mpc.tolist()
            pred_price4mpc_lst.append(pred_price4mpc) # shape of pred_price4mpc: pred_len * price_dim
            pred_price_lst.append(list(np.array(pred_price4mpc[0]).flatten()))

        eval_price_lst = []
        for idx,eval_price in enumerate(self.eval_price_lst):
            # if idx == 1: break # for debug 
            eval_price_lst.append(eval_price)
        
        
        pred_dict = {
            'pred_price4mpc':pred_price4mpc_lst,
            'pred_price':pred_price_lst,
            'actual_price':eval_price_lst,
        }
        pred_df = pd.DataFrame(pred_dict)
        pred_df.to_csv(self.pred_save_path)



if __name__ == "__main__":
    start_time = time.time() 

    parser = argparse.ArgumentParser()
    parser.add_argument('--market_year',default=2021,type=int,help='read data begin')
    parser.add_argument('--state_name',default='VIC',type=str,help='state name')
    parser.add_argument('--device',default='cuda:0',type=str,help='training device')
    parser.add_argument('--input_len',default=31,type=int,help='input series length')
    parser.add_argument('--pred_len',default=4,type=int,help='MPC horizon')
    args = parser.parse_args()


    market_year = args.market_year
    state_name = args.state_name
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    input_len = args.input_len 
    pred_len = args.pred_len


    # build nn
    price_dim = 1
    forecast_net = LSTMNet(
        input_dim=price_dim,
        output_dim=price_dim
    ).to(device)

    # build forecaster
    forecaster = Forecaster(
        model=forecast_net,
        state_name=state_name,
        market_year=market_year,
    )

    # generate predictions
    pred_generator = PredictionGenerator(
        state_name=state_name,
        market_year=market_year
    )
    # load data 
    pred_generator.load_data(
        input_len=input_len,
    )
    # load model 
    pred_generator.load_model(model=forecaster)
    # pred 
    pred_generator.make_preds(
        input_len=input_len,
        device=device,
        pred_len=pred_len
    )
    
    end_time = time.time()

    collpse_time = end_time-start_time

    runtime_save_folder = 'benchmark/forecast/{}{}'.format(state_name,market_year)
    runtime_save_name = 'price_pred_gen_run_time.txt'
    runtime_save_path = os.path.join(runtime_save_folder,runtime_save_name)
    np.savetxt(runtime_save_path,[collpse_time])