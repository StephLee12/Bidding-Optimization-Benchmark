import pandas as pd 
import argparse
import os 
import numpy as np 
import pulp
import time

import torch 
import torch.nn as nn
import torch.nn.functional as F

class LSTMNet(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim=100) -> None:
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

class PAO():
    def __init__(
        self,
        mode,
        state_name,
        wind_farm_name,
        market_year,
        price_dim,
        seq_len,
        lstm_save_path,
        eval_save_path
    ) -> None:
        self.mode = mode
        self.state_name = state_name
        self.wind_farm_name = wind_farm_name
        self.market_year = market_year

        self.test_set_size = int((60/5)*24*31)
        self.seq_len = seq_len
        self.price_dim = price_dim
        self.wind_dim = 1
    
        self.load_data()

        self.train_epoch = 300
        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        self.lr = 1e-3
        self.price_predictor = LSTMNet(input_dim=price_dim,output_dim=price_dim).to(self.device)
        self.price_adam_optim = torch.optim.Adam(self.price_predictor.parameters(),lr=self.lr)
        self.wind_predictor = LSTMNet(input_dim=1,output_dim=1).to(self.device)
        self.wind_adam_optim = torch.optim.Adam(self.wind_predictor.parameters(),lr=self.lr)        
        self.curtailment_predictor = LSTMNet(input_dim=1,output_dim=1).to(self.device)
        self.curtailment_adam_optim = torch.optim.Adam(self.curtailment_predictor.parameters(),lr=self.lr)
        self.loss_func = nn.MSELoss()

        self.init_soc = 0.5
        self.min_soc = 0.05
        self.max_soc = 0.95
        self.dch_eff = 0.95
        self.ch_eff = 0.95
        self.energy_capacity = 10
        self.power_capacity = 10
        self.wind_power_capacity = 67
        self.ES_duration = 5/60
        self.Reg_duration = 5/60
        self.penalty_eff = 1.5

        self.rolling_horizon = 100

        self.lstm_save_path = lstm_save_path
        self.eval_save_path = eval_save_path

    # load original data with State and Year
    def load_data(self,price_input_folder='NEM_annual_data',wind_input_folder='VIC_wind_data'):
        self.data = pd.DataFrame()
        price_read_path = os.path.join(price_input_folder,'{}{}.csv'.format(self.state_name,self.market_year))
        if self.mode == 'ES': self.price_columns = ['RRP']
        elif self.mode == 'Reg_FCAS': self.price_columns = ['RAISEREGRRP','LOWERREGRRP']
        else: self.price_columns = ['RRP','RAISEREGRRP','LOWERREGRRP']
        self.price_data = pd.read_csv(price_read_path,usecols=self.price_columns).iloc[1:,:].reset_index(drop=True)

        wind_read_path = os.path.join(wind_input_folder,'VIC_wind_farms_{}'.format(self.market_year),'{}.csv'.format(self.wind_farm_name))
        self.wind_data = pd.read_csv(wind_read_path,index_col=[0])
        self.wind_data['DateTime'] = self.wind_data.index 
        self.wind_data.drop_duplicates(subset=['DateTime'],inplace=True)
        self.wind_data.reset_index(inplace=True)

        self.data_len = self.price_data.shape[0]

        # for the real wind generation and price 
        self.wind_df = self.wind_data.iloc[-self.test_set_size+self.seq_len:,:].reset_index(drop=True)
        self.price_df = self.price_data.iloc[-self.test_set_size+self.seq_len:,:].reset_index(drop=True)

        # for LSTM training and evaluating
        self.wind_data = self.wind_data['AVAILABILITY'].to_numpy().reshape(self.wind_data.shape[0],self.wind_dim)
        self.price_data = self.price_data.to_numpy().reshape(self.price_data.shape[0],self.price_dim)

    # create seq for LSTM
    def create_seq(self,data,seq_len):
        seq_lst = []
        label_lst = []
        length = len(data)
        for i in range(length-seq_len):
            seq = data[i:i+seq_len,:]
            label = data[i+seq_len:i+seq_len+1,:]
            seq_lst.append(seq)
            label_lst.append(label)
        
        return np.array(seq_lst),np.array(label_lst)

    # split dataset for training LSTM
    def split_train_test(self):

        self.price_train_data = self.price_data[:-self.test_set_size-self.seq_len,:]
        self.price_test_data = self.price_data[-self.test_set_size-self.seq_len:,:]

        self.price_train_seq,self.price_train_label = self.create_seq(self.price_train_data,self.seq_len)
        self.price_test_seq,self.price_test_label = self.create_seq(self.price_test_data,self.seq_len)

        self.wind_train_data = self.wind_data[:-self.test_set_size-self.seq_len,:]
        self.wind_test_data = self.wind_data[-self.test_set_size-self.seq_len:,:]

        self.wind_train_seq,self.wind_train_label = self.create_seq(self.wind_train_data,self.seq_len)
        self.wind_test_seq,self.wind_test_label = self.create_seq(self.wind_test_data,self.seq_len)

    # train LSTM and evaluate
    def train_eval_price(self):
        batch_lst = []
        for idx in range(0,self.price_train_seq.shape[0],5000):
            batch_start = idx 
            batch_end = min(idx+5000,self.price_train_seq.shape[0])
            lst = [batch_start,batch_end]
            batch_lst.append(lst)
            
        self.price_predictor.train()
        for i in range(1,self.train_epoch+1):
            for batch_start,batch_end in batch_lst:
                input_tensor = torch.tensor(self.price_train_seq[batch_start:batch_end,:],device=self.device,dtype=torch.float32)
                pred_label = self.price_predictor(input_tensor)
                pred_label = torch.squeeze(pred_label,dim=0)
                actual_label = torch.tensor(self.price_train_label[batch_start:batch_end,:],dtype=torch.float32,device=self.device)
                actual_label = torch.squeeze(actual_label,dim=1)
                loss = self.loss_func(pred_label,actual_label)
                self.price_adam_optim.zero_grad()
                loss.backward()
                self.price_adam_optim.step()
            if i % 10 == 0:
                print('Epoch:{},price loss:{:.2f}'.format(i,loss.detach().cpu().item()))
        torch.save(self.price_predictor.state_dict(),os.path.join(self.lstm_save_path,'{}_{}_{}_price_lstm_model_{}'.format(self.state_name,self.mode,self.market_year,self.train_epoch)))
        
        self.price_predictor.eval()
        input_tensor = torch.tensor(self.price_test_seq,device=self.device,dtype=torch.float32)
        pred_label = self.price_predictor(input_tensor).detach().cpu().numpy()
        pred_label = np.squeeze(pred_label,axis=0)
        label_lst,pred_label_lst = [],[]
        for l,pred_l in zip(self.price_test_label,pred_label):
            l = np.squeeze(l,axis=0)
            label_lst.append(list(l))
            pred_label_lst.append(list(pred_l))
        eval_dict = {
            'pred':pred_label_lst,
            'ground_truth':label_lst
        }
        eval_df = pd.DataFrame.from_dict(eval_dict)
        eval_df.to_csv(os.path.join(self.eval_save_path,'{}_{}_{}_{}_price_prediction.csv'.format(self.state_name,self.mode,self.market_year,self.train_epoch)))  
    
    def train_eval_wind(self):
        batch_lst = []
        for idx in range(0,self.price_train_seq.shape[0],5000):
            batch_start = idx 
            batch_end = min(idx+5000,self.price_train_seq.shape[0])
            lst = [batch_start,batch_end]
            batch_lst.append(lst)

        self.wind_predictor.train()
        for i in range(1,self.train_epoch+1):
            for batch_start,batch_end in batch_lst:
                input_tensor = torch.tensor(self.wind_train_seq[batch_start:batch_end,:],device=self.device,dtype=torch.float32)
                pred_label = self.wind_predictor(input_tensor)
                pred_label = torch.squeeze(pred_label,dim=0)
                actual_label = torch.tensor(self.wind_train_label[batch_start:batch_end,:],device=self.device,dtype=torch.float32)
                actual_label = torch.squeeze(actual_label,dim=1)
                loss = self.loss_func(pred_label,actual_label)
                self.wind_adam_optim.zero_grad()
                loss.backward()
                self.wind_adam_optim.step()
            if i % 10 == 0:
                print('Epoch:{},price loss:{:.2f}'.format(i,loss.detach().cpu().item()))
        torch.save(self.wind_predictor.state_dict(),os.path.join(self.lstm_save_path,'{}_{}_{}_wind_lstm_model_{}'.format(self.state_name,self.mode,self.market_year,self.train_epoch)))
        
        self.wind_predictor.eval()
        input_tensor = torch.tensor(self.wind_test_seq,device=self.device,dtype=torch.float32)
        pred_label = self.wind_predictor(input_tensor).detach().cpu().numpy()
        pred_label = np.squeeze(pred_label,axis=0)
        label_lst,pred_label_lst = [],[]
        for l,pred_l in zip(self.wind_test_label,pred_label):
            l = np.squeeze(l,axis=0)
            label_lst.append(list(l))
            pred_label_lst.append(list(pred_l))
        eval_dict = {
            'pred':pred_label_lst,
            'ground_truth':label_lst
        }
        eval_df = pd.DataFrame.from_dict(eval_dict)
        eval_df.to_csv(os.path.join(self.eval_save_path,'{}_{}_{}_{}_wind_prediction.csv'.format(self.state_name,self.mode,self.market_year,self.train_epoch)))  

    def price_wind_lstm_main(self):
        self.split_train_test()

        self.train_eval_price()
        self.train_eval_wind()

    def curtailment_lstm_main(self,market_year=2019,wind_input_folder='VIC_wind_data'):
        # read extra year data 
        wind_read_path = os.path.join(wind_input_folder,'VIC_wind_farms_{}'.format(market_year),'{}.csv'.format(self.wind_farm_name))
        wind_data = pd.read_csv(wind_read_path,index_col=[0])
        wind_data['DateTime'] = wind_data.index 
        wind_data.drop_duplicates(subset=['DateTime'],inplace=True)
        wind_data.reset_index(inplace=True)
        wind_data = wind_data['AVAILABILITY'].to_numpy().reshape(wind_data.shape[0],self.wind_dim)

        # create seq and label 
        wind_seq,wind_label = self.create_seq(wind_data,self.seq_len)
        wind_label = wind_label.flatten()
        # feed in trained lstm to get predicted wind
        self.wind_predictor.load_state_dict(torch.load(os.path.join(self.lstm_save_path,'{}_{}_{}_wind_lstm_model_{}'.format(self.state_name,self.mode,self.market_year,self.train_epoch))))
        self.wind_predictor.eval()
        input_tensor = torch.tensor(wind_seq,device=self.device,dtype=torch.float32)
        pred_wind_label = self.wind_predictor(input_tensor).detach().cpu().numpy()
        pred_wind_label = pred_wind_label.flatten()
        
        # get curtailment 
        curtailment_train_data = (wind_label - pred_wind_label).reshape(wind_label.shape[0],self.wind_dim)
        curtailment_train_seq,curtailment_train_label = self.create_seq(curtailment_train_data,self.seq_len)
        # train curtailment predictor
        batch_lst = []
        for idx in range(0,curtailment_train_seq.shape[0],5000):
            batch_start = idx 
            batch_end = min(idx+5000,curtailment_train_seq.shape[0])
            lst = [batch_start,batch_end]
            batch_lst.append(lst)
        self.curtailment_predictor.train()
        for i in range(1,self.train_epoch+1):
            for batch_start,batch_end in batch_lst:
                input_tensor = torch.tensor(curtailment_train_seq[batch_start:batch_end,:],device=self.device,dtype=torch.float32)
                pred_label = self.curtailment_predictor(input_tensor)
                pred_label = torch.squeeze(pred_label,dim=0)
                actual_label = torch.tensor(curtailment_train_label[batch_start:batch_end,:],device=self.device,dtype=torch.float32)
                actual_label = torch.squeeze(actual_label,dim=1)
                loss = self.loss_func(pred_label,actual_label)
                self.curtailment_adam_optim.zero_grad()
                loss.backward()
                self.curtailment_adam_optim.step()
            if i % 10 == 0:
                print('Epoch:{},curtailment loss:{:.2f}'.format(i,loss.detach().cpu().item()))
        torch.save(self.curtailment_predictor.state_dict(),os.path.join(self.lstm_save_path,'{}_{}_{}_curtailment_lstm_model_{}'.format(self.state_name,self.mode,self.market_year,self.train_epoch)))

        # get real curtailment data 
        wind_test_data = self.wind_data[-self.test_set_size-self.seq_len*2:,:]
        wind_test_seq,wind_test_label = self.create_seq(wind_test_data,self.seq_len)
        wind_test_label = wind_test_label.flatten()
        self.wind_predictor.eval()
        input_tensor = torch.tensor(wind_test_seq,device=self.device,dtype=torch.float32)
        pred_wind_label = self.wind_predictor(input_tensor).detach().cpu().numpy()
        pred_wind_label = pred_wind_label.flatten()
        curtailment_test_data = (wind_test_label-pred_wind_label).reshape(wind_test_label.shape[0],self.wind_dim)
        curtailment_test_seq,curtailment_test_label = self.create_seq(curtailment_test_data,self.seq_len)
        # eval curtailment predictor 
        self.curtailment_predictor.eval()
        input_tensor = torch.tensor(curtailment_test_seq,device=self.device,dtype=torch.float32)
        pred_label = self.curtailment_predictor(input_tensor).detach().cpu().numpy()
        pred_label = np.squeeze(pred_label,axis=0)
        label_lst,pred_label_lst = [],[]
        for l,pred_l in zip(curtailment_test_label,pred_label):
            l = np.squeeze(l,axis=0)
            label_lst.append(list(l))
            pred_label_lst.append(list(pred_l))
        eval_dict = {
            'pred':pred_label_lst,
            'ground_truth':label_lst
        }
        eval_df = pd.DataFrame.from_dict(eval_dict)
        eval_df.to_csv(os.path.join(self.eval_save_path,'{}_{}_{}_{}_curtailment_prediction.csv'.format(self.state_name,self.mode,self.market_year,self.train_epoch)))  

    # load predicted price and wind 
    def load_wind_price_prediction(self,pred_price_df,pred_wind_df):
        pred_price,actual_price = np.array(list(map(eval,pred_price_df['pred']))),np.array(list(map(eval,pred_price_df['ground_truth'])))
        self.pred_wind,self.actual_wind = np.array(list(map(eval,pred_wind_df['pred']))).squeeze(axis=1),np.array(list(map(eval,pred_wind_df['ground_truth']))).squeeze(axis=1)
        self.curtail_wind = self.actual_wind - self.pred_wind
        self.curtail_wind[self.curtail_wind<0] = 0

        if self.mode == 'ES':
            self.ES_pred_price,self.ES_actual_price = pred_price[:,0],actual_price[:,0]
            self.ES_dch_pred_price4optim = self.ES_pred_price*self.dch_eff
            self.ES_ch_pred_price4optim = self.ES_pred_price/self.ch_eff
        elif self.mode == 'Reg_FCAS':
            self.RR_pred_price,self.RR_actual_price = pred_price[:,0],actual_price[:,0]
            self.RL_pred_price,self.RL_actual_price = pred_price[:,1],actual_price[:,1]
            self.RR_pred_price4optim = self.RR_pred_price*self.dch_eff
            self.RL_pred_price4optim = self.RL_pred_price/self.ch_eff
        else:
            self.ES_pred_price,self.ES_actual_price = pred_price[:,0],actual_price[:,0]
            self.RR_pred_price,self.RR_actual_price = pred_price[:,1],actual_price[:,1]
            self.RL_pred_price,self.RL_actual_price = pred_price[:,2],actual_price[:,2]
            self.ES_dch_pred_price4optim = self.ES_pred_price*self.dch_eff
            self.ES_ch_pred_price4optim = self.ES_pred_price/self.ch_eff
            self.RR_pred_price4optim = self.RR_pred_price*self.dch_eff
            self.RL_pred_price4optim = self.RL_pred_price/self.ch_eff

    # load predicted wind curtailment 
    def load_curtailment_prediction(self,pred_curtailment_df):
        self.pred_curtail_wind = np.array(list(map(eval,pred_curtailment_df['pred'])))
        self.pred_curtail_wind[self.pred_curtail_wind<0] = 0

    # set empty variable dict for saving
    def set_BESS_variable_dict(self):
        var_dict = {}
        if self.mode == 'ES':
            var_prefix_lst = ['DCHBinary','CHBinary','WC','SoC','ESDCH','ESCH','RESPONSE']
        elif self.mode == 'Reg_FCAS': 
            var_prefix_lst = ['DCHBinary','CHBinary','WC','SoC','REGRR','REGRL','RESPONSE']
        else: 
            var_prefix_lst = ['DCHBinary','CHBinary','WC','SoC','ESDCH','ESCH','REGRR','REGRL','RESPONSE']
        
        for prefix in var_prefix_lst:
            var_dict[prefix] = []
        
        return var_dict
    
    def setup_wind_lp_model(self,step):
        pass

    def setup_BESS_lp_model(self,step,soc):
        # build lp model 
        lp_model = pulp.LpProblem(name='BESS',sense=pulp.LpMaximize)
        
        # charge or discharge, decision variable, binary
        horizon = min(self.rolling_horizon,self.horizon-step)
        V_Dch_B = pulp.LpVariable.dicts('DCHBinary',range(horizon),cat='Binary')
        V_Ch_B = pulp.LpVariable.dicts('CHBinary',range(horizon),cat='Binary')
        V_WC = pulp.LpVariable.dicts('WC',range(horizon),lowBound=0,upBound=self.power_capacity,cat='Continuous')
        # battery SoC, free variables
        V_SoC = pulp.LpVariable.dicts('SoC',range(horizon),lowBound=self.min_soc,upBound=self.max_soc,cat='Continuous')
        V_SoC[0].setInitialValue(soc)
        V_SoC[0].fixValue()
        if self.mode == 'ES':
            # variables
            V_ES_Dch = pulp.LpVariable.dicts('ESDCH',range(horizon),lowBound=0,upBound=self.power_capacity,cat='Continuous')
            V_ES_Ch = pulp.LpVariable.dicts('ESCH',range(horizon),lowBound=0,upBound=self.power_capacity,cat='Continuous')   
            # objective
            lp_model += pulp.lpSum([self.ES_dch_pred_price4optim[price_idx] * V_ES_Dch[var_idx] \
                                    - self.ES_ch_pred_price4optim[price_idx] * V_ES_Ch[var_idx] \
                                        for var_idx,price_idx in enumerate(range(step,step+horizon))])
            # constraints
            coeff = self.ES_duration/self.energy_capacity
            for var_idx,price_idx in enumerate(range(step,step+horizon)):
                # charge/discharge constraint 
                lp_model += (V_Dch_B[var_idx]+V_Ch_B[var_idx]==1,'cons_binary_{}'.format(var_idx))
                # power constraint 
                lp_model += (V_ES_Dch[var_idx]-self.power_capacity*V_Dch_B[var_idx]<=0,'cons_ES_dch_binary_{}'.format(var_idx)) 
                lp_model += (V_ES_Ch[var_idx]-self.power_capacity*V_Ch_B[var_idx]<=0,'cons_ES_ch_binary_{}'.format(var_idx))
                lp_model += (V_WC[var_idx]-self.pred_curtail_wind[price_idx]*V_Ch_B[var_idx]<=0,'cons_WC_binary_{}'.format(var_idx))
                lp_model += (V_ES_Ch[var_idx]+V_WC[var_idx]-self.power_capacity<=0,'cons_ES_ch_tot_cap_{}'.format(var_idx))
                # soc constraint 
                lp_model += (coeff*V_ES_Dch[var_idx]+self.min_soc-V_SoC[var_idx]<=0,'cons_ES_dch_soc_{}'.format(var_idx))
                lp_model += (coeff*(V_ES_Ch[var_idx]+V_WC[var_idx])+V_SoC[var_idx]-self.max_soc<=0,'cons_ES_ch_soc_{}'.format(var_idx))
                # update soc 
                if var_idx >= 1 and var_idx <= (horizon - 1):
                    lp_model += (V_SoC[var_idx-1]+coeff*(V_ES_Ch[var_idx-1]+V_WC[var_idx-1])-coeff*V_ES_Dch[var_idx-1]==V_SoC[var_idx],'soc_{}'.format(var_idx))
                    V_SoC[var_idx].fixValue()
        elif self.mode == 'Reg_FCAS':
            # variables
            V_Reg_RR = pulp.LpVariable.dicts('REGRR',range(horizon),lowBound=0,upBound=self.power_capacity,cat='Continuous')
            V_Reg_RL = pulp.LpVariable.dicts('REGRL',range(horizon),lowBound=0,upBound=self.power_capacity,cat='Continuous')
            # objective
            lp_model += pulp.lpSum([self.RR_pred_price4optim[price_idx] * V_Reg_RR[var_idx] \
                                    + self.RL_pred_price4optim[price_idx] * V_Reg_RL[var_idx] \
                                        for var_idx,price_idx in enumerate(range(step,step+horizon))])
            # constraints
            coeff = self.Reg_duration / self.energy_capacity
            for var_idx,price_idx in enumerate(range(step,step+horizon)):
                # charge/discharge constraint
                lp_model += (V_Dch_B[var_idx]+V_Ch_B[var_idx]==1,'cons_binary_{}'.format(var_idx))
                # power constraint
                lp_model += (V_Reg_RR[var_idx]-self.power_capacity*V_Dch_B[var_idx]<=0,'cons_Reg_RR_binary_{}'.format(var_idx))
                lp_model += (V_Reg_RL[var_idx]-self.power_capacity*V_Ch_B[var_idx]<=0,'cons_Reg_RL_binary_{}'.format(var_idx))
                lp_model += (V_WC[var_idx]-self.pred_curtail_wind[price_idx]*V_Ch_B[var_idx]<=0,'cons_WC_binary_{}'.format(var_idx))
                lp_model += (V_Reg_RR[var_idx]+V_WC[var_idx]-self.power_capacity<=0,'cons_Reg_ch_tot_cap_{}'.format(var_idx))
                # soc constraint
                lp_model += (coeff*V_Reg_RR[var_idx]+self.min_soc-V_SoC[var_idx]<=0,'cons_Reg_dch_soc_{}'.format(var_idx))
                lp_model += (coeff*(V_Reg_RL[var_idx]+V_WC[var_idx])+V_SoC[var_idx]-self.max_soc<=0,'cons_Reg_ch_soc_{}'.format(var_idx))
                # update soc 
                if var_idx >= 1 and var_idx <= (horizon - 1):
                    lp_model += (V_SoC[var_idx-1]+coeff*(V_Reg_RL[var_idx-1]+V_WC[var_idx-1])-coeff*V_Reg_RR[var_idx-1]==V_SoC[var_idx],'soc_{}'.format(var_idx))
                    V_SoC[var_idx].fixValue()
        else: # self.mode == 'Join'   
            # variable
            V_ES_Dch = pulp.LpVariable.dicts('ESDCH',range(horizon),lowBound=0,upBound=self.power_capacity,cat='Continuous')
            V_ES_Ch = pulp.LpVariable.dicts('ESCH',range(horizon),lowBound=0,upBound=self.power_capacity,cat='Continuous')   
            V_Reg_RR = pulp.LpVariable.dicts('REGRR',range(horizon),lowBound=0,upBound=self.power_capacity,cat='Continuous')
            V_Reg_RL = pulp.LpVariable.dicts('REGRL',range(horizon),lowBound=0,upBound=self.power_capacity,cat='Continuous') 
            # objective
            lp_model += pulp.lpSum([self.ES_dch_pred_price4optim[price_idx] * V_ES_Dch[var_idx] \
                                    - self.ES_ch_pred_price4optim[price_idx] * V_ES_Ch[var_idx] \
                                    + self.RR_pred_price4optim[price_idx] * V_Reg_RR[var_idx] \
                                    + self.RL_pred_price4optim[price_idx] * V_Reg_RL[var_idx]
                                        for var_idx,price_idx in enumerate(range(step,step+horizon))])
            # constraints
            coeff = self.Reg_duration / self.energy_capacity
            for var_idx,price_idx in enumerate(range(step,step+horizon)):
                # charge/discharge constraint
                lp_model += (V_Dch_B[var_idx]+V_Ch_B[var_idx]==1,'cons_binary_{}'.format(var_idx))
                # power constraint
                lp_model += (V_ES_Dch[var_idx]-self.power_capacity*V_Dch_B[var_idx]<=0,'cons_ES_dch_binary_{}'.format(var_idx)) 
                lp_model += (V_ES_Ch[var_idx]-self.power_capacity*V_Ch_B[var_idx]<=0,'cons_ES_ch_binary_{}'.format(var_idx))
                lp_model += (V_Reg_RR[var_idx]-self.power_capacity*V_Dch_B[var_idx]<=0,'cons_Reg_RR_binary_{}'.format(var_idx))
                lp_model += (V_Reg_RL[var_idx]-self.power_capacity*V_Ch_B[var_idx]<=0,'cons_Reg_RL_binary_{}'.format(var_idx))
                lp_model += (V_WC[var_idx]-self.pred_curtail_wind[price_idx]*V_Ch_B[var_idx]<=0,'cons_WC_binary_{}'.format(var_idx))
                lp_model += (V_ES_Dch[var_idx]+V_Reg_RR[var_idx]-self.power_capacity<=0,'cons_Join_dch_tot_cap_{}'.format(var_idx))
                lp_model += (V_ES_Ch[var_idx]+V_Reg_RL[var_idx]+V_WC[var_idx]-self.power_capacity<=0,'cons_Join_ch_tot_cap_{}'.format(var_idx))
                # soc constraint
                lp_model += (coeff*(V_ES_Dch[var_idx]+V_Reg_RR[var_idx])+self.min_soc-V_SoC[var_idx]<=0,'cons_Join_dch_soc_{}'.format(var_idx))
                lp_model += (coeff*(V_ES_Ch[var_idx]+V_Reg_RL[var_idx]+V_WC[var_idx])+V_SoC[var_idx]-self.max_soc<=0,'cons_Join_ch_soc_{}'.format(var_idx))
                # update soc 
                if var_idx >= 1 and var_idx <= (horizon - 1):
                    lp_model += (V_SoC[var_idx-1]+coeff*(V_ES_Ch[var_idx-1]+V_Reg_RL[var_idx-1]+V_WC[var_idx-1])-coeff*(V_ES_Dch[var_idx-1]+V_Reg_RR[var_idx-1])==V_SoC[var_idx],'soc_{}'.format(var_idx))
                    V_SoC[var_idx].fixValue()

        return lp_model

    def get_actual_soc(self,step,reserve_power,current_soc):
        real_curtail_wind = self.curtail_wind[step] # get acutal curtailment
        response_power = min(reserve_power,real_curtail_wind) # get actual response power
        ori_soc_change = self.ES_duration * reserve_power / self.energy_capacity  # calculate soc change in the optimization 
        real_soc_change = self.ES_duration * response_power / self.energy_capacity # correct the above soc change
        # get real soc 
        current_soc -= ori_soc_change
        current_soc += real_soc_change

        return current_soc,response_power

    # calculate real revenue 
    def cal_actual_revenue(self,var_df):
        BESS_actual_cum_revenue,wind_actual_cum_revenue = 0,0
        BESS_actual_revenue_lst,wind_actual_revenue_lst = [],[]
        BESS_actual_cum_revenue_lst,wind_actual_cum_revenue_lst = [],[]
        for i in range(var_df.shape[0]):
            if self.mode == 'ES':
                dch_b,ch_b,dch,ch = var_df.loc[i,'DCHBinary'],var_df.loc[i,'CHBinary'],var_df.loc[i,'ESDCH'],var_df.loc[i,'ESCH']
                if dch_b: 
                    BESS_actual_revenue = dch * self.ES_actual_price[i] * self.ES_duration * self.dch_eff
                else: 
                    BESS_actual_revenue = -ch * self.ES_actual_price[i] * self.ES_duration / self.ch_eff

                wind_actual_revenue = min(self.pred_wind[i],self.actual_wind[i]) * self.ES_actual_price[i] * self.ES_duration
                wind_actual_revenue -= self.penalty_eff * self.ES_actual_price[i] * abs(self.pred_wind[i]-self.actual_wind[i]) * self.ES_duration
            elif self.mode == 'Reg_FCAS': 
                dch_b,ch_b,bid_RR,bid_RL = var_df.loc[i,'DCHBinary'],var_df.loc[i,'CHBinary'],var_df.loc[i,'REGRR'],var_df.loc[i,'REGRL']
                if dch_b:
                    BESS_actual_revenue = bid_RR * self.RR_actual_price[i] * self.Reg_duration * self.dch_eff
                else:
                    BESS_actual_revenue = bid_RL * self.RL_actual_price[i] * self.Reg_duration / self.ch_eff

                wind_actual_revenue = min(self.pred_wind[i],self.actual_wind[i]) * self.RR_actual_price[i] * self.Reg_duration
                wind_actual_revenue -= self.penalty_eff * self.RR_actual_price[i] * abs(self.pred_wind[i]-self.actual_wind[i]) * self.Reg_duration
            else: 
                dch_b,ch_b,dch,ch,bid_RR,bid_RL = var_df.loc[i,'DCHBinary'],var_df.loc[i,'CHBinary'],var_df.loc[i,'ESDCH'],var_df.loc[i,'ESCH'],var_df.loc[i,'REGRR'],var_df.loc[i,'REGRL']
                if dch_b:
                    BESS_actual_revenue = dch * self.ES_actual_price[i] * self.ES_duration * self.dch_eff
                    BESS_actual_revenue += bid_RR * self.RR_actual_price[i] * self.Reg_duration * self.dch_eff
                else:
                    BESS_actual_revenue = -ch * self.ES_actual_price[i] * self.ES_duration / self.ch_eff
                    BESS_actual_revenue += bid_RL * self.RL_actual_price[i] * self.Reg_duration / self.ch_eff
                
                if self.ES_pred_price[i] <= self.RR_pred_price[i]:
                    wind_actual_revenue = min(self.pred_wind[i],self.actual_wind[i]) * self.RR_actual_price[i] * self.Reg_duration
                    wind_actual_revenue -= self.penalty_eff * self.RR_actual_price[i] * abs(self.pred_wind[i]-self.actual_wind[i]) * self.Reg_duration
                else:
                    wind_actual_revenue = min(self.pred_wind[i],self.actual_wind[i]) * self.ES_actual_price[i] * self.ES_duration
                    wind_actual_revenue -= self.penalty_eff * self.ES_actual_price[i] * abs(self.pred_wind[i]-self.actual_wind[i]) * self.ES_duration
            
            BESS_actual_cum_revenue += BESS_actual_revenue
            wind_actual_cum_revenue += wind_actual_revenue

            BESS_actual_revenue_lst.append(BESS_actual_revenue)
            BESS_actual_cum_revenue_lst.append(BESS_actual_cum_revenue)
            wind_actual_revenue_lst.append(wind_actual_revenue)
            wind_actual_cum_revenue_lst.append(wind_actual_cum_revenue)
        
        return BESS_actual_revenue_lst,wind_actual_revenue_lst,BESS_actual_cum_revenue_lst,wind_actual_cum_revenue_lst

    def main(self):
        # check if there are trained models 
        pred_file_name = '{}_{}_{}_{}_wind_prediction.csv'.format(self.state_name,self.mode,self.market_year,self.train_epoch)
        if pred_file_name not in os.listdir(self.eval_save_path):
            self.price_wind_lstm_main()
        # load trained wind model to train model to predict curtailment
        curtailment_file_name = '{}_{}_{}_{}_curtailment_prediction.csv'.format(self.state_name,self.mode,self.market_year,self.train_epoch)
        if curtailment_file_name not in os.listdir(self.eval_save_path):
            self.curtailment_lstm_main(market_year=2019)

        # load predicted price
        pred_price_read_path = os.path.join(self.eval_save_path,'{}_{}_{}_{}_price_prediction.csv'.format(self.state_name,self.mode,self.market_year,self.train_epoch))
        pred_price_df = pd.read_csv(pred_price_read_path,index_col=[0])
        pred_wind_read_path = os.path.join(self.eval_save_path,'{}_{}_{}_{}_wind_prediction.csv'.format(self.state_name,self.mode,self.market_year,self.train_epoch))
        pred_wind_df = pd.read_csv(pred_wind_read_path,index_col=[0])
        self.load_wind_price_prediction(pred_price_df,pred_wind_df)
        # load predicited wind curtailment 
        pred_curtailment_read_path = os.path.join(self.eval_save_path,'{}_{}_{}_{}_curtailment_prediction.csv'.format(self.state_name,self.mode,self.market_year,self.train_epoch))
        pred_curtailment_df = pd.read_csv(pred_curtailment_read_path,index_col=[0])
        self.load_curtailment_prediction(pred_curtailment_df)

        # build lp model and solve in a rolling horizon way
        self.horizon = self.pred_wind.shape[0]
        # self.horizon = 50
        BESS_var_dict = self.set_BESS_variable_dict()
        BESS_var_dict['SoC'].append(self.init_soc)
        for step in range(0,self.horizon):
            # build MILP model
            BESS_lp_model = self.setup_BESS_lp_model(step,BESS_var_dict['SoC'][step])
            # wind_lp_model = self.setup_wind_lp_model(step)
            # build solver
            if self.mode == 'Join':
                solver = pulp.GUROBI_CMD(keepFiles=True,msg=False,gapRel=0.01) # for Joint market optimization 
            else:
                solver = pulp.GUROBI_CMD(keepFiles=True,msg=False,gapRel=0.005) # gapRel as 0.005 makes the joint market optimization stuck 
            # solver = pulp.GLPK_CMD(path='glpsol',keepFiles=False,msg=True)
            # solve
            BESS_lp_model.solve(solver)
            # wind_lp_model.solve(solver)
            # save BESS info
            for var in BESS_lp_model.variables():
                var_name,idx = var.name.split('_')
                var_value = var.value()
                if eval(idx) == 0 and var_name != 'SoC':
                    BESS_var_dict[var_name].append(var_value)
                if eval(idx) == 1 and var_name == 'SoC':
                    if step == self.horizon - 1: continue
                    BESS_var_dict[var_name].append(var_value)   
            # re-update the soc since we predict the amount of wind curtailment
            actual_soc,response_power = self.get_actual_soc(step,reserve_power=BESS_var_dict['WC'][-1],current_soc=BESS_var_dict['SoC'][-1])
            BESS_var_dict['SoC'][-1] = np.clip(actual_soc,self.min_soc,self.max_soc)
            BESS_var_dict['RESPONSE'].append(response_power)

            if (step + 1) % 2000 == 0:
                os.system('rm -rf /dev/shm')

            print('Step {} finished'.format(step))
        
        var_df = pd.DataFrame.from_dict(BESS_var_dict)
        # calculate real revenue 
        BESS_actual_revenue_lst,wind_actual_revenue_lst,BESS_actual_cum_revenue_lst,wind_actual_cum_revenue_lst = self.cal_actual_revenue(var_df)
        var_df['BESS_actual_revenue'] = BESS_actual_revenue_lst
        var_df['BESS_actual_cum_revenue'] = BESS_actual_cum_revenue_lst
        var_df['wind_actual_revenue'] = wind_actual_revenue_lst
        var_df['wind_actual_cum_revenue'] = wind_actual_cum_revenue_lst
        var_df['wind_curtail_power'] = self.curtail_wind
        
        var_df.to_csv(os.path.join(self.eval_save_path,'{}_{}_{}_res.csv'.format(self.state_name,self.mode,self.market_year)))

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',default='ES',type=str,help='mode')
    parser.add_argument('--state_name',default='VIC',type=str,help='state name')
    parser.add_argument('--market_year',default=2018,type=int,help='read data begin')
    parser.add_argument('--seq_len',default=16,type=int,help='seq_len')
    args = parser.parse_args()

    lstm_model_save_folder = 'predict_then_optimize/lstm_models'
    os.makedirs(lstm_model_save_folder,exist_ok=True)
    eval_save_folder = 'predict_then_optimize/eval_results'
    os.makedirs(eval_save_folder,exist_ok=True)

    if args.mode == 'ES': price_dim = 1
    elif args.mode == 'Reg_FCAS': price_dim = 2
    else: price_dim = 3
    pao_model = PAO(
        mode=args.mode,
        state_name=args.state_name,
        wind_farm_name='OAKLAND1',
        market_year=args.market_year,
        price_dim=price_dim,
        seq_len=args.seq_len,
        lstm_save_path=lstm_model_save_folder,
        eval_save_path=eval_save_folder
    )
    pao_model.main()
    end_time = time.time()
    print('PAO Running time:{}'.format(end_time-start_time))
    running_time = np.array([end_time-start_time])
    np.savetxt('predict_then_optimize/eval_results/{}_{}_{}_runningTime.txt'.format(args.state_name,args.mode,args.market_year),running_time)