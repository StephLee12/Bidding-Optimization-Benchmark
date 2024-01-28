import numpy as np 
import pandas as pd 

import torch 
import torch.nn as nn 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class NoLabelDataset(Dataset):
    def __init__(self,x) -> None:
        super().__init__()

        self.x = x 
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index):
        return self.x[index,:,:]


class CurtailForecastDataset(Dataset):
    def __init__(self,x,y) -> None:
        super().__init__()

        self.x = x 
        self.y = y
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index,:,:],self.y[index,:,:]

class SolarPriceForecastDataset(Dataset):
    def __init__(self,x,y) -> None:
        super().__init__()
        self.x = x 
        self.y = y
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index,:,:],self.y[index,:,:]

class LSTMNet(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim=100) -> None:
        super(LSTMNet,self).__init__() 
        self.lstm_layer = nn.LSTM(input_dim,hidden_dim,batch_first=True)
        self.linear = nn.Linear(hidden_dim,output_dim)
    
    def forward(self,x):
        lstm_out,(h_n,c_n) = self.lstm_layer(x)
        out = self.linear(h_n)
        return out


class LSTMNetWithSigmoidOutput(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim=100) -> None:
        super(LSTMNetWithSigmoidOutput,self).__init__() 
        self.lstm_layer = nn.LSTM(input_dim,hidden_dim,batch_first=True)
        self.linear = nn.Linear(hidden_dim,output_dim)
    
    def forward(self,x):
        lstm_out,(h_n,c_n) = self.lstm_layer(x)
        out = self.linear(h_n)
        out =  torch.sigmoid(out)
        return out
    

def split_train_test(arr,train_test_ratio,lookback):
    # split x and y (label)
    x_lst = []
    y_lst = []
    for idx in range(len(arr)-lookback):
        x = arr[idx:idx+lookback]
        y = arr[idx+lookback:idx+lookback+1]

        x_lst.append(x)
        y_lst.append(y)
    
    # split train and test 
    split_idx = int(train_test_ratio*len(x_lst))
    train_x_lst = x_lst[:split_idx]
    train_y_lst = y_lst[:split_idx]
    test_x_lst = x_lst[split_idx:]
    test_y_lst = y_lst[split_idx:]

    train_x_arr = np.expand_dims(np.array(train_x_lst),axis=-1)
    train_y_arr = np.expand_dims(np.array(train_y_lst),axis=-1)
    test_x_arr = np.expand_dims(np.array(test_x_lst),axis=-1)
    test_y_arr = np.expand_dims(np.array(test_y_lst),axis=-1)

    return train_x_arr,train_y_arr,test_x_arr,test_y_arr



def create_seq(lst,lookback):
    seq_lst = []
    for idx in range(len(lst)-lookback+1):
        seq = lst[idx:idx+lookback]
        seq_lst.append(seq)

    seq_arr = np.array(seq_lst)

    return seq_arr 


def permute_DMPC_predictions(
    DMPC_pred_solar_lst,
    DMPC_pred_price_lst,
    solar_std,
    price_std,
    scenario_num,
    mpc_horizon,
    solar_cap,
    price_min,
    price_max,
    solar_pred_save_path,
    price_pred_save_path,
):
    pred_solar_lst = [] # 3d-list 
    pred_price_lst = [] # 3d-list: first_dim length->eval_data_len-mpc_horizon; second_dim->scenario_num; third_dim->mpc_horizon
    for pred_solar,pred_price in zip(DMPC_pred_solar_lst,DMPC_pred_price_lst):
    # for idx,(pred_solar,pred_price) in enumerate(zip(DMPC_pred_solar_lst,DMPC_pred_price_lst)): # for debug 
        # if idx >= 2: break 
        scenario_solar_lst = []
        scenario_price_lst = []
        scenario_solar_lst.append(pred_solar)
        scenario_price_lst.append(pred_price)

        # get all scenario for one-step MPC solving
        for scenario_idx in range(scenario_num-1):
            # get solar 
            solar_forecast_error = np.random.normal(loc=0,scale=solar_std,size=mpc_horizon)*solar_cap
            scenario_solar = np.clip(np.array(pred_solar)+solar_forecast_error,0,solar_cap)
            scenario_solar_lst.append(list(scenario_solar))
            # get price
            price_forecast_error = (np.random.normal(loc=0,scale=price_std,size=mpc_horizon)*(price_max-price_min))+price_min
            scenario_price = np.clip(np.array(pred_price)+price_forecast_error,price_min,price_max)
            scenario_price_lst.append(list(scenario_price))

        pred_solar_lst.append(scenario_solar_lst)
        pred_price_lst.append(scenario_price_lst)


    lst = []
    for one_step_solar in pred_solar_lst:
        one_step_solar_arr = np.array(one_step_solar)
        # get the average of all scenarios for one-step MPC solving 
        lst.append(list(np.mean(one_step_solar_arr,axis=0))) 
    
    # save permuted predictions 
    solar_save_dict = {'solar':lst}
    solar_save_df = pd.DataFrame(solar_save_dict)
    solar_save_df.to_csv(solar_pred_save_path)

    price_save_dict = {'scenario{}'.format(scenario_idx):[] for scenario_idx in range(scenario_num)}
    for one_step_all_scenario_price in pred_price_lst:
        for scenario_idx,one_step_one_scenario_price in enumerate(one_step_all_scenario_price):
            price_save_dict['scenario{}'.format(scenario_idx)].append(one_step_one_scenario_price)
    price_save_df = pd.DataFrame(price_save_dict)
    price_save_df.to_csv(price_pred_save_path)

    return lst,pred_price_lst


def load_permuted_predictions(
    solar_pred_save_path,
    price_pred_save_path,
    scenario_num
):
    solar_df = pd.read_csv(solar_pred_save_path,index_col=[0])
    solar_lst = solar_df['solar'].to_list()
    solar_lst = list(map(eval,solar_lst))

    # price_lst = []
    price_df = pd.read_csv(price_pred_save_path,index_col=[0]).iloc[:,:scenario_num]
    for col_name in price_df.columns:
        price_df[col_name] = price_df[col_name].apply(eval)
    price_lst = price_df.values.tolist()

    
    return solar_lst,price_lst



def get_actual_soc_DMPC(
    reserve_bid,
    curtail_solar,
    dispatch_duration,
    ch_eff,
    bat_cap,
    cur_soc
):
    response_bid = min(reserve_bid,curtail_solar)
    ori_soc_change = dispatch_duration*reserve_bid/bat_cap/ch_eff
    real_soc_change = dispatch_duration*response_bid/bat_cap/ch_eff

    cur_soc = cur_soc - ori_soc_change + real_soc_change

    return cur_soc,response_bid


def record_var_DMPC(
    lp_model,
    var_dict,
    optim_step,
    eval_data_len,
    eval_solar,
    pred_solar,
    dispatch_duration,
    ch_eff,
    bat_cap,
    bat_soc_min,
    bat_soc_max
):
    for var in lp_model.variables():
        var_prefix,var_idx = var.name.split('_')
        var_value = var.value()
        # only record the first step 
        if eval(var_idx)==0 and var_prefix!='varSoc':
            var_dict[var_prefix].append(var_value)
        if eval(var_idx)==1 and var_prefix=='varSoc': # for soc, record the changed soc 
            if optim_step==eval_data_len-1: continue
            var_dict[var_prefix].append(var_value)
    

    if var_dict['varBatBinaryDch'][-1] == 1:
        var_dict['curtailResponseBid'].append(0.0)
    else:
        # re-update the soc as we predict the amount of wind curtailment 
        actual_soc,response_bid = get_actual_soc_DMPC(
            reserve_bid=var_dict['varBatReserve'][-1],
            curtail_solar=max(0,eval_solar-pred_solar),
            dispatch_duration=dispatch_duration,
            ch_eff=ch_eff,
            bat_cap=bat_cap,
            cur_soc=var_dict['varSoc'][-1]
        )
        var_dict['varSoc'][-1] = np.clip(actual_soc,bat_soc_min,bat_soc_max)
        var_dict['curtailResponseBid'].append(response_bid)



def record_var_SMPC(
    lp_model,
    scenario_var_dict,
    scenario_var_name_lst
):
    for var in lp_model.variables():
        var_prefix,var_idx = var.name.split('_')
        var_value = var.value()
        # only record the first step 
        if eval(var_idx) == 0 and var_prefix in set(scenario_var_name_lst): 
            scenario_var_dict[var_prefix].append(var_value)


def get_actual_soc_SMPC(
    reserve_bid,
    curtail_solar,
    ch_bid,
    dispatch_duration,
    ch_eff,
    bat_cap,
    cur_soc,
    bat_soc_min,
    bat_soc_max
):
    response_bid = min(reserve_bid,curtail_solar)
    soc_change = (ch_bid+response_bid)*dispatch_duration/bat_cap/ch_eff

    cur_soc = np.clip(cur_soc+soc_change,bat_soc_min,bat_soc_max)

    return cur_soc,response_bid



def get_next_move_SMPC(
    var_dict,
    scenario_var_dict,
    cur_soc,
    dch_eff,
    ch_eff,
    dispatch_duration,
    bat_cap,
    bat_soc_min,
    bat_soc_max,
    curtail_solar
):
    # first determine charge / discharge 
    dch_num = sum(scenario_var_dict['varBatBinaryDch'])
    ch_num = sum(scenario_var_dict['varBatBinaryCh'])

    if dch_num >= ch_num: # discharge
        var_dict['varBatBinaryDch'].append(1)
        var_dict['varBatBinaryCh'].append(0)
        var_dict['varBatReserve'].append(0.0)
        var_dict['varBatBidCh'].append(0.0)
        var_dict['curtailResponseBid'].append(0.0)

        avg_bid_dch = np.mean(scenario_var_dict['varBatBidDch'])
        var_dict['varBatBidDch'].append(avg_bid_dch)

        soc_change = avg_bid_dch*dispatch_duration*dch_eff/bat_cap
        cur_soc = np.clip(cur_soc-soc_change,bat_soc_min,bat_soc_max)
        var_dict['varSoc'].append(cur_soc)


    else: # charge
        var_dict['varBatBinaryDch'].append(0)
        var_dict['varBatBinaryCh'].append(1)
        var_dict['varBatBidDch'].append(0.0)

        avg_bid_ch = np.mean(scenario_var_dict['varBatBidCh'])
        avg_bid_reserve = np.mean(scenario_var_dict['varBatReserve'])
        var_dict['varBatBidCh'].append(avg_bid_ch)
        var_dict['varBatReserve'].append(avg_bid_reserve)

        cur_soc,response_bid = get_actual_soc_SMPC(
            reserve_bid=avg_bid_reserve,
            curtail_solar=curtail_solar,
            ch_bid=avg_bid_ch,
            dispatch_duration=dispatch_duration,
            ch_eff=ch_eff,
            bat_cap=bat_cap,
            cur_soc=cur_soc,
            bat_soc_min=bat_soc_min,
            bat_soc_max=bat_soc_max
        )
        var_dict['varSoc'].append(cur_soc)
        var_dict['curtailResponseBid'].append(response_bid)











def save_revenue_info_DMPC(
    var_df,
    pred_solar_lst,
    eval_solar_lst,
    eval_price_lst,
    dispatch_duration,
    solar_penalty_coeff,
    res_save_path
):
    bat_cum_revenue = 0
    solar_cum_revenue = 0
    bat_revenue_lst = []
    solar_revenue_lst = []
    bat_cum_revenue_lst = []
    solar_cum_revenue_lst = []
    bat_dch_binary_lst = []
    bat_ch_binary_lst = []
    bat_dch_bid_lst = []
    bat_ch_bid_lst = []
    bat_reserve_bid_lst = []
    bat_curtail_response_lst = []
    bat_soc_lst = []

    for idx in range(var_df.shape[0]):
        bat_dch_binary = var_df.loc[idx,'varBatBinaryDch']
        bat_ch_binary = var_df.loc[idx,'varBatBinaryCh']
        bat_dch_bid = var_df.loc[idx,'varBatBidDch']
        bat_ch_bid = var_df.loc[idx,'varBatBidCh']
        bat_reserve_bid = var_df.loc[idx,'varBatReserve']
        bat_curtail_response = var_df.loc[idx,'curtailResponseBid']
        bat_soc = var_df.loc[idx,'varSoc']

        bat_dch_binary_lst.append(bat_dch_binary)
        bat_ch_binary_lst.append(bat_ch_binary)
        bat_dch_bid_lst.append(bat_dch_bid)
        bat_ch_bid_lst.append(bat_ch_bid)
        bat_reserve_bid_lst.append(bat_reserve_bid)
        bat_curtail_response_lst.append(bat_curtail_response)
        bat_soc_lst.append(bat_soc)
        

        # bat revenue calculation 
        if bat_dch_binary: # discharge 
            bat_revenue = bat_dch_bid*dispatch_duration*eval_price_lst[idx]
        else: # charge 
            bat_revenue = -bat_ch_bid*dispatch_duration*eval_price_lst[idx]
        
        # solar revenue calculation 
        solar_revenue = min(eval_solar_lst[idx],pred_solar_lst[idx][0])*dispatch_duration*eval_price_lst[idx]
        # solar penelty 
        solar_revenue -= solar_penalty_coeff*max(0,pred_solar_lst[idx][0]-eval_solar_lst[idx])*dispatch_duration*eval_price_lst[idx]


        # calculate cum revenue 
        bat_cum_revenue += bat_revenue
        solar_cum_revenue += solar_revenue
        
        bat_revenue_lst.append(bat_revenue)
        bat_cum_revenue_lst.append(bat_cum_revenue)
        solar_revenue_lst.append(solar_revenue)
        solar_cum_revenue_lst.append(solar_cum_revenue)

    res_dict = {
        'bat_dch_binary':bat_dch_binary_lst,
        'bat_ch_binary':bat_ch_binary_lst,
        'bat_dch_bid':bat_dch_bid_lst,
        'bat_ch_bid':bat_ch_bid_lst,
        'bat_reserve_bid':bat_reserve_bid_lst,
        'bat_curtail_response':bat_curtail_response_lst,
        'bat_revenue':bat_revenue_lst,
        'bat_cum_revenue':bat_cum_revenue_lst,
        'bat_soc':bat_soc_lst,
        'solar_revenue':solar_revenue_lst,
        'solar_cum_revenue':solar_cum_revenue_lst
    }
    res_df = pd.DataFrame(res_dict)
    res_df.to_csv(res_save_path)
