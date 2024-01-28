import pandas as pd 
import argparse
import os 
import torch 
import time


from benchmark_utils import create_seq
from benchmark_utils import LSTMNet
from benchmark_utils import LSTMNetWithSigmoidOutput
from benchmark_curtailforecast import SolarCurtailForecaster
from benchmark_solarpriceforecast import SolarPriceForecaster



class DMPC_Preprocess():
    def __init__(self) -> None:
        
        res_save_folder = 'benchmark_res'
        os.makedirs(res_save_folder,exist_ok=True)
        res_file_name = 'solar_price_curtail_pred4mpc.csv'
        self.res_save_path = os.path.join(res_save_folder,res_file_name)
        



    # read data 
    def load_data(
        self,
        state_name,
        market_year,
        solar_farm_name,
        lookback,
        price_input_folder='NEM_annual_data',
        solar_input_folder='solar_data'
    ):

        # read data 
        price_file_name = '{}{}.csv'.format(state_name,market_year)
        price_read_path = os.path.join(price_input_folder,price_file_name)
        price_df = pd.read_csv(price_read_path)
        
        # solar data 
        solar_input_folder = os.path.join(solar_input_folder,'{}'.format(market_year))
        solar_file_name = '{}_{}1_Solar_{}_5min.csv'.format(market_year,state_name,solar_farm_name)
        solar_read_path = os.path.join(solar_input_folder,solar_file_name)
        solar_df = pd.read_csv(solar_read_path)
        solar_df['DateTime'] = solar_df['Unnamed: 0']
        solar_df = solar_df.drop_duplicates(subset=['DateTime']).reset_index(drop=True)

        # get optimization data and eval data (eval data has no lookback)
        one_month_len = int(60/5*24*31)
        to_forecast_price_df = price_df.iloc[-lookback-one_month_len:]
        eval_price_df = price_df.iloc[-one_month_len:]
        to_forecast_solar_df = solar_df.iloc[-lookback-one_month_len:]
        eval_solar_df = solar_df.iloc[-one_month_len:]

        to_forecast_price_arr = to_forecast_price_df['RRP'].to_numpy()
        eval_price_arr = eval_price_df['RRP'].to_numpy()
        to_forecast_solar_arr = to_forecast_solar_df['AVAILABILITY'].to_numpy()
        eval_solar_arr = eval_solar_df['AVAILABILITY'].to_numpy()

        self.to_forecast_price_arr = to_forecast_price_arr
        self.eval_price_arr = eval_price_arr
        self.to_forecast_solar_arr = to_forecast_solar_arr
        self.eval_solar_arr = eval_solar_arr



    # load forecaster 
    def load_forecaster(self,solar_forecaster,price_forecaster,curtail_forecaster):
        solar_forecaster.load_model()
        price_forecaster.load_model()
        curtail_forecaster.load_model()

        self.solar_forecaster = solar_forecaster
        self.price_forecaster = price_forecaster
        self.curtail_forecaster = curtail_forecaster



    # load predicted parameters for MPC
    def make_mpc_prediction(self,lookback,device,mpc_horizon):
        to_forecast_solar_seq_arr = create_seq(list(self.to_forecast_solar_arr),lookback=lookback)
        to_forecast_price_seq_arr = create_seq(list(self.to_forecast_price_arr),lookback=lookback)

        pred_solar_lst = []
        pred_price_lst = []
        for to_forecast_solar_seq,to_forecast_price_seq in zip(to_forecast_solar_seq_arr,to_forecast_price_seq_arr):
        # for idx,(to_forecast_solar_seq,to_forecast_price_seq) in enumerate(zip(to_forecast_solar_seq_arr,to_forecast_price_seq_arr)):
            # if idx >= 1: break
            pred_solar = self.solar_forecaster.inference(pred_len=mpc_horizon,his_x=to_forecast_solar_seq,device=device,lookback=lookback)
            pred_price = self.price_forecaster.inference(pred_len=mpc_horizon,his_x=to_forecast_price_seq,device=device,lookback=lookback)

            pred_solar_lst.append(pred_solar)
            pred_price_lst.append(pred_price)

        pred_curtail_lst = []
        for his_solar,pred_solar in zip(to_forecast_solar_seq_arr,pred_solar_lst):
        # for idx,(his_solar,pred_solar) in enumerate(zip(to_forecast_solar_seq_arr,pred_solar_lst)):
            # if idx >= 1: break 
            pred_curtail = self.curtail_forecaster.inference(pred_len=mpc_horizon,his_x=his_solar,pred_x=pred_solar,device=device,lookback=lookback)
            pred_curtail_lst.append(pred_curtail)

        pred_dict = {'solar':pred_solar_lst,'price':pred_price_lst,'curtail':pred_curtail_lst}
        pred_df = pd.DataFrame(pred_dict)
        pred_df.to_csv(self.res_save_path)



    


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
    mpc_horizon = int(60/5*24)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    solar_forecast_net = LSTMNet(input_dim=1,output_dim=1).to(device)
    price_forecast_net = LSTMNet(input_dim=1,output_dim=1).to(device)
    curtail_forecast_net = LSTMNetWithSigmoidOutput(input_dim=1,output_dim=1).to(device)
    solar_forecaster = SolarPriceForecaster(solar_or_price='solar',model=solar_forecast_net)
    price_forecaster = SolarPriceForecaster(solar_or_price='price',model=price_forecast_net)
    curtail_forecaster = SolarCurtailForecaster(solar_forecaster=solar_forecaster,model=curtail_forecast_net)

    preprocess_model = DMPC_Preprocess()

    preprocess_model.load_data(
        state_name=state_name,
        market_year=market_year,
        solar_farm_name=solar_farm_name,
        lookback=lookback
    )


    preprocess_model.load_forecaster(
        solar_forecaster=solar_forecaster,
        price_forecaster=price_forecaster,
        curtail_forecaster=curtail_forecaster
    )

    preprocess_model.make_mpc_prediction(
        lookback=lookback,
        device=device,
        mpc_horizon=mpc_horizon
    )

    end_time = time.time()