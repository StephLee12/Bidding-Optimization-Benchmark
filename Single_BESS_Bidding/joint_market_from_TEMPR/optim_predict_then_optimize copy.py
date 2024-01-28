import pandas as pd 
import argparse
import os 
import numpy as np 
import pulp
import time

import torch 
import torch.nn as nn
import torch.nn.functional as F

INPUT_PATH = 'NEM_annual_data'

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


class TransformerNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    

    def forward(self,x):
        pass 



class TrainForecaster:
    def __init__(self) -> None:
        pass

    def load_data(self):
        



class PredictThenOptimize():
    def __init__(
        self,
        mode,
        state_name,
        market_year,
        price_dim,
        seq_len,
        lstm_save_path,
        eval_save_path
    ) -> None:
        self.mode = mode
        self.state_name = state_name
        self.market_year = market_year

        self.seq_len = seq_len
        self.price_dim = price_dim
        self.lstm_save_path = lstm_save_path
        self.eval_save_path = eval_save_path
        self.load_data()

        self.device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
        self.predictor = LSTMNet(input_dim=price_dim,output_dim=price_dim).to(self.device)
        self.lr = 1e-3
        self.loss_func = nn.MSELoss()
        self.optim = torch.optim.Adam(self.predictor.parameters(),lr=self.lr)
        self.epochs = 200

        self.init_soc = 0.5
        self.min_soc = 0.0
        self.max_soc = 1.0
        self.dch_coeff = 0.95
        self.ch_coeff = 0.95
        self.energy_cap = 10
        self.rated_power = 2
        self.max_CFCAS_cap = 1
        self.spot_duration = 5/60
        self.fast_duration = 5/60
        self.slow_duration = 5/60
        self.delay_duration = 5/60
    
    # create seq for LSTM
    def create_seq(self,data):
        seq_lst = []
        label_lst = []
        length = len(data)
        for i in range(length-self.seq_len):
            seq = data[i:i+self.seq_len]
            label = data[i+self.seq_len:i+self.seq_len+1]
            seq_lst.append(seq)
            label_lst.append(label)
        
        return np.array(seq_lst),np.array(label_lst)

    # load original data with State and Year
    def load_data(self):
        self.data_df = pd.DataFrame()
        read_path = os.path.join(INPUT_PATH,self.state_name+str(self.market_year)+'.csv')
        last_month_length = int((60/5)*24*31) + int((60/5)*24*30)
        self.data_df = pd.read_csv(read_path,index_col=[0])

        if self.mode == 'ES':
            self.price_columns = ['RRP']
        elif self.mode == 'Contingency_FCAS':
            self.price_columns = ['RAISE6SECRRP','LOWER6SECRRP','RAISE60SECRRP','LOWER60SECRRP','RAISE5MINRRP','LOWER5MINRRP']

            # get contingency info
            self.FR_flag_arr = self.data_df.iloc[-last_month_length+self.seq_len:,-6].values
            self.FL_flag_arr = self.data_df.iloc[-last_month_length+self.seq_len:,-5].values
            self.SR_flag_arr = self.data_df.iloc[-last_month_length+self.seq_len:,-4].values
            self.SL_flag_arr = self.data_df.iloc[-last_month_length+self.seq_len:,-3].values
            self.DR_flag_arr = self.data_df.iloc[-last_month_length+self.seq_len:,-2].values
            self.DL_flag_arr = self.data_df.iloc[-last_month_length+self.seq_len:,-1].values
        else:
            self.price_columns = ['RRP','RAISE6SECRRP','LOWER6SECRRP','RAISE60SECRRP','LOWER60SECRRP','RAISE5MINRRP','LOWER5MINRRP']

            self.FR_flag_arr = self.data_df.iloc[-last_month_length+self.seq_len:,-6].values
            self.FL_flag_arr = self.data_df.iloc[-last_month_length+self.seq_len:,-5].values
            self.SR_flag_arr = self.data_df.iloc[-last_month_length+self.seq_len:,-4].values
            self.SL_flag_arr = self.data_df.iloc[-last_month_length+self.seq_len:,-3].values
            self.DR_flag_arr = self.data_df.iloc[-last_month_length+self.seq_len:,-2].values
            self.DL_flag_arr = self.data_df.iloc[-last_month_length+self.seq_len:,-1].values
        
        self.price_arr = self.data_df[self.price_columns].to_numpy()
        self.price_train_arr = self.price_arr[:-last_month_length,:]
        self.price_test_arr = self.price_arr[-last_month_length:,:]
        self.price_train_seq,self.price_train_label = self.create_seq(self.price_train_arr)
        self.price_test_seq,self.price_test_label = self.create_seq(self.price_test_arr)
    
    # train LSTM and evaluate
    def train_eval_predictor(self):
        batch_lst = []
        for idx in range(0,self.price_train_seq.shape[0],5000):
            batch_start = idx 
            batch_end = min(idx+5000,self.price_train_seq.shape[0])
            lst = [batch_start,batch_end]
            batch_lst.append(lst)
        
        self.predictor.train()
        for i in range(1,self.epochs+1):
            for batch_start,batch_end in batch_lst:
                input_tensor = torch.tensor(self.price_train_seq[batch_start:batch_end,:],device=self.device,dtype=torch.float32)
                pred_label = self.predictor(input_tensor)
                pred_label = torch.squeeze(pred_label,dim=0)
                actual_label = torch.tensor(self.price_train_label[batch_start:batch_end,:],dtype=torch.float32,device=self.device)
                actual_label = torch.squeeze(actual_label,dim=1)
                loss = self.loss_func(pred_label,actual_label)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            if i % 10 == 0:
                print('Epoch:{},loss:{:.2f}'.format(i,loss.detach().cpu().item()))
        torch.save(self.predictor.state_dict(),os.path.join(self.lstm_save_path,'{}_{}_{}_lstm_model_{}'.format(self.state_name,self.mode,self.market_year,self.epochs)))
        
        self.predictor.eval()
        input_tensor = torch.tensor(self.price_test_seq,device=self.device,dtype=torch.float32)
        pred_label = self.predictor(input_tensor).detach().cpu().numpy()
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
        eval_df.to_csv(os.path.join(self.eval_save_path,'{}_{}_{}_prediction.csv'.format(self.state_name,self.mode,self.market_year)))        

    # load predicted price
    def load_price(self,eval_price):
        pred_price,actual_price = np.array(list(map(eval,eval_price['pred']))),np.array(list(map(eval,eval_price['ground_truth'])))
        if self.mode == 'ES':
            self.ES_pred_price,self.ES_actual_price = pred_price[:,0],actual_price[:,0]
            self.ES_dch_price4optim = self.ES_pred_price*self.dch_coeff
            self.ES_ch_price4optim = self.ES_pred_price/self.ch_coeff
        elif self.mode == 'Contingency_FCAS':
            self.FR_pred_price,self.FR_actual_price = pred_price[:,0],actual_price[:,0]
            self.FR_price4optim = self.FR_pred_price*self.dch_coeff
            self.FL_pred_price,self.FL_actual_price = pred_price[:,1],actual_price[:,1]
            self.FL_price4optim = self.FR_pred_price/self.ch_coeff
            self.SR_pred_price,self.SR_actual_price = pred_price[:,2],actual_price[:,2]
            self.SR_price4optim = self.SR_pred_price*self.dch_coeff
            self.SL_pred_price,self.SL_actual_price = pred_price[:,3],actual_price[:,3]
            self.SL_price4optim = self.SL_pred_price/self.ch_coeff
            self.DR_pred_price,self.DR_actual_price = pred_price[:,4],actual_price[:,4]
            self.DR_price4optim = self.DR_pred_price*self.dch_coeff
            self.DL_pred_price,self.DL_actual_price = pred_price[:,5],actual_price[:,5]
            self.DL_price4optim = self.DL_pred_price/self.ch_coeff
        else:
            self.ES_pred_price,self.ES_actual_price = pred_price[:,0],actual_price[:,0]
            self.ES_dch_price4optim = self.ES_pred_price*self.dch_coeff
            self.ES_ch_price4optim = self.ES_pred_price/self.ch_coeff
            self.FR_pred_price,self.FR_actual_price = pred_price[:,1],actual_price[:,1]
            self.FR_price4optim = self.FR_pred_price*self.dch_coeff
            self.FL_pred_price,self.FL_actual_price = pred_price[:,2],actual_price[:,2]
            self.FL_price4optim = self.FR_pred_price/self.ch_coeff
            self.SR_pred_price,self.SR_actual_price = pred_price[:,3],actual_price[:,3]
            self.SR_price4optim = self.SR_pred_price*self.dch_coeff
            self.SL_pred_price,self.SL_actual_price = pred_price[:,4],actual_price[:,4]
            self.SL_price4optim = self.SL_pred_price/self.ch_coeff
            self.DR_pred_price,self.DR_actual_price = pred_price[:,5],actual_price[:,5]
            self.DR_price4optim = self.DR_pred_price*self.dch_coeff
            self.DL_pred_price,self.DL_actual_price = pred_price[:,6],actual_price[:,6]
            self.DL_price4optim = self.DL_pred_price/self.ch_coeff
        
        self.pred_price = pred_price
        self.actual_price = actual_price

    # set up optimization model 
    def setup_lp_model(self):

        # build lp model 
        lp_model = pulp.LpProblem(name='bidding',sense=pulp.LpMaximize)

        self.horizon = self.pred_price.shape[0]
        # self.horizon = 100 # for test 
        # charge or discharge, decision variable, binary
        V_DCH_B = pulp.LpVariable.dicts('DCHBinary',range(self.horizon),cat='Binary')
        V_CH_B = pulp.LpVariable.dicts('CHBinary',range(self.horizon),cat='Binary')
        # battery SoC, free variables
        V_SoC = pulp.LpVariable.dicts('SoC',range(self.horizon),lowBound=self.min_soc,upBound=self.max_soc,cat='Continuous')
        V_SoC[0].setInitialValue(self.init_soc)
        V_SoC[0].fixValue()
        if self.mode == 'ES':
            # variables
            V_ES_DCH = pulp.LpVariable.dicts('ESDCH',range(self.horizon),lowBound=0,upBound=self.rated_power,cat='Continuous')
            V_ES_CH = pulp.LpVariable.dicts('ESCH',range(self.horizon),lowBound=0,upBound=self.rated_power,cat='Continuous')            

            # objective
            lp_model += pulp.lpSum([self.ES_dch_price4optim[i] * V_ES_DCH[i] \
                                    - self.ES_ch_price4optim[i] * V_ES_CH[i] \
                                        for i in range(self.horizon)])
            
            # constraints
            coeff = self.spot_duration/self.energy_cap
            for i in range(self.horizon):
                lp_model += (V_DCH_B[i]+V_CH_B[i]==1,'cons_binary_{}'.format(i))
                lp_model += (V_ES_DCH[i]-self.rated_power*V_DCH_B[i]<=0,'cons_ES_dch_binary_{}'.format(i)) 
                lp_model += (V_ES_CH[i]-self.rated_power*V_CH_B[i]<=0,'cons_ES_ch_binary_{}'.format(i))
                lp_model += (coeff*V_ES_DCH[i]+self.min_soc-V_SoC[i]<=0,'cons_ES_dch_soc_{}'.format(i))
                lp_model += (coeff*V_ES_CH[i]+V_SoC[i]-self.max_soc<=0,'cons_ES_ch_soc_{}'.format(i))
                # update soc 
                if i >= 1 and i <= (self.horizon - 1):
                    lp_model += (V_SoC[i-1]+coeff*V_ES_CH[i-1]-coeff*V_ES_DCH[i-1]==V_SoC[i],'soc_{}'.format(i))
                    V_SoC[i].fixValue()

        elif self.mode == 'Contingency_FCAS':
            V_FR = pulp.LpVariable.dicts('FR',range(self.horizon),lowBound=0,upBound=self.max_CFCAS_cap,cat='Continuous')
            V_FL = pulp.LpVariable.dicts('FL',range(self.horizon),lowBound=0,upBound=self.max_CFCAS_cap,cat='Continuous')
            V_SR = pulp.LpVariable.dicts('SR',range(self.horizon),lowBound=0,upBound=self.max_CFCAS_cap,cat='Continuous')
            V_SL = pulp.LpVariable.dicts('SL',range(self.horizon),lowBound=0,upBound=self.max_CFCAS_cap,cat='Continuous')
            V_DR = pulp.LpVariable.dicts('DR',range(self.horizon),lowBound=0,upBound=self.max_CFCAS_cap,cat='Continuous')
            V_DL = pulp.LpVariable.dicts('DL',range(self.horizon),lowBound=0,upBound=self.max_CFCAS_cap,cat='Continuous')

            lp_model += pulp.lpSum([self.FR_price4optim[i] * V_FR[i] + self.FL_price4optim[i] * V_FL[i] \
                                    + self.SR_price4optim[i] * V_SR[i] + self.SL_price4optim[i] * V_SL[i] \
                                    + self.DR_price4optim[i] * V_DR[i] + self.DL_price4optim[i] * V_DL[i] \
                                        for i in range(self.horizon)])
            fast_coeff,slow_coeff,delay_coeff = self.fast_duration/self.energy_cap,self.slow_duration/self.energy_cap,self.delay_duration/self.energy_cap
            for i in range(self.horizon):
                lp_model += (V_DCH_B[i]+V_CH_B[i]==1,'cons_binary_{}'.format(i))
                lp_model += (V_FR[i]-self.max_CFCAS_cap*V_DCH_B[i]<=0,'cons_FCAS_FR_binary_{}'.format(i))
                lp_model += (V_FL[i]-self.max_CFCAS_cap*V_CH_B[i]<=0,'cons_FCAS_FL_binary_{}'.format(i))
                lp_model += (V_SR[i]-self.max_CFCAS_cap*V_DCH_B[i]<=0,'cons_FCAS_SR_binary_{}'.format(i))
                lp_model += (V_SL[i]-self.max_CFCAS_cap*V_CH_B[i]<=0,'cons_FCAS_SL_binary_{}'.format(i))
                lp_model += (V_DR[i]-self.max_CFCAS_cap*V_DCH_B[i]<=0,'cons_FCAS_DR_binary_{}'.format(i))
                lp_model += (V_DL[i]-self.max_CFCAS_cap*V_CH_B[i]<=0,'cons_FCAS_DL_binary_{}'.format(i))
                lp_model += (V_FR[i]+V_SR[i]+V_DR[i]-self.rated_power<=0,'cons_FCAS_dch_tot_capacity_{}'.format(i))
                lp_model += (V_FL[i]+V_SL[i]+V_DL[i]-self.rated_power<=0,'cons_FCAS_ch_tot_capacity_{}'.format(i))
                lp_model += (fast_coeff*V_FR[i]+slow_coeff*V_SR[i]+delay_coeff*V_DR[i]+self.min_soc-V_SoC[i]<=0,'cons_FCAS_dch_soc_{}'.format(i))
                lp_model += (fast_coeff*V_FL[i]+slow_coeff*V_SL[i]+delay_coeff*V_DL[i]+V_SoC[i]-self.max_soc<=0,'cons_FCAS_ch_soc_{}'.format(i))

                FR_coeff_with_flag = self.FR_flag_arr[i]*self.fast_duration/self.energy_cap
                FL_coeff_with_flag = self.FL_flag_arr[i]*self.fast_duration/self.energy_cap
                SR_coeff_with_flag = self.SR_flag_arr[i]*self.slow_duration/self.energy_cap
                SL_coeff_with_flag = self.SL_flag_arr[i]*self.slow_duration/self.energy_cap
                DR_coeff_with_flag = self.DR_flag_arr[i]*self.delay_duration/self.energy_cap
                DL_coeff_with_flag = self.DL_flag_arr[i]*self.delay_duration/self.energy_cap
                if i >= 1 <= (self.horizon - 1):
                    lp_model += (V_SoC[i-1]-FR_coeff_with_flag*V_FR[i-1]+FL_coeff_with_flag*V_FL[i-1]\
                                -SR_coeff_with_flag*V_SR[i-1]+SL_coeff_with_flag*V_SL[i-1]\
                                -DR_coeff_with_flag*V_DR[i-1]+DL_coeff_with_flag*V_DL[i-1] == V_SoC[i],'soc_{}'.format(i))
                    V_SoC[i].fixValue()
        else:
            V_ES_DCH = pulp.LpVariable.dicts('ESDCH',range(self.horizon),lowBound=0,upBound=self.rated_power,cat='Continuous')
            V_ES_CH = pulp.LpVariable.dicts('ESCH',range(self.horizon),lowBound=0,upBound=self.rated_power,cat='Continuous')
            V_FR = pulp.LpVariable.dicts('FR',range(self.horizon),lowBound=0,upBound=self.max_CFCAS_cap,cat='Continuous')
            V_FL = pulp.LpVariable.dicts('FL',range(self.horizon),lowBound=0,upBound=self.max_CFCAS_cap,cat='Continuous')
            V_SR = pulp.LpVariable.dicts('SR',range(self.horizon),lowBound=0,upBound=self.max_CFCAS_cap,cat='Continuous')
            V_SL = pulp.LpVariable.dicts('SL',range(self.horizon),lowBound=0,upBound=self.max_CFCAS_cap,cat='Continuous')
            V_DR = pulp.LpVariable.dicts('DR',range(self.horizon),lowBound=0,upBound=self.max_CFCAS_cap,cat='Continuous')
            V_DL = pulp.LpVariable.dicts('DL',range(self.horizon),lowBound=0,upBound=self.max_CFCAS_cap,cat='Continuous')

            lp_model += pulp.lpSum([self.ES_dch_price4optim[i] * V_ES_DCH[i] - self.ES_ch_price4optim[i] * V_ES_CH[i] \
                                    + self.FR_price4optim[i] * V_FR[i] + self.FL_price4optim[i] * V_FL[i] \
                                    + self.SR_price4optim[i] * V_SR[i] + self.SL_price4optim[i] * V_SL[i] \
                                    + self.DR_price4optim[i] * V_DR[i] + self.DL_price4optim[i] * V_DL[i] \
                                        for i in range(self.horizon)])
            
            ES_coeff = self.spot_duration/self.energy_cap
            fast_coeff,slow_coeff,delay_coeff = self.fast_duration/self.energy_cap,self.slow_duration/self.energy_cap,self.delay_duration/self.energy_cap
            for i in range(self.horizon):
                lp_model += (V_DCH_B[i]+V_CH_B[i]==1,'cons_binary_{}'.format(i))
                lp_model += (V_ES_DCH[i]-self.rated_power*V_DCH_B[i]<=0,'cons_ES_dch_binary_{}'.format(i)) 
                lp_model += (V_ES_CH[i]-self.rated_power*V_CH_B[i]<=0,'cons_ES_ch_binary_{}'.format(i))
                lp_model += (V_FR[i]-self.max_CFCAS_cap*V_DCH_B[i]<=0,'cons_FCAS_FR_binary_{}'.format(i))
                lp_model += (V_FL[i]-self.max_CFCAS_cap*V_CH_B[i]<=0,'cons_FCAS_FL_binary_{}'.format(i))
                lp_model += (V_SR[i]-self.max_CFCAS_cap*V_DCH_B[i]<=0,'cons_FCAS_SR_binary_{}'.format(i))
                lp_model += (V_SL[i]-self.max_CFCAS_cap*V_CH_B[i]<=0,'cons_FCAS_SL_binary_{}'.format(i))
                lp_model += (V_DR[i]-self.max_CFCAS_cap*V_DCH_B[i]<=0,'cons_FCAS_DR_binary_{}'.format(i))
                lp_model += (V_DL[i]-self.max_CFCAS_cap*V_CH_B[i]<=0,'cons_FCAS_DL_binary_{}'.format(i))
                lp_model += (V_FR[i]+V_SR[i]+V_DR[i]-self.rated_power<=0,'cons_FCAS_dch_tot_capacity_{}'.format(i))
                lp_model += (V_FL[i]+V_SL[i]+V_DL[i]-self.rated_power<=0,'cons_FCAS_ch_tot_capacity_{}'.format(i))
                lp_model += (fast_coeff*V_FR[i]+slow_coeff*V_SR[i]+delay_coeff*V_DR[i]+self.min_soc-V_SoC[i]<=0,'cons_FCAS_dch_soc_{}'.format(i))
                lp_model += (fast_coeff*V_FL[i]+slow_coeff*V_SL[i]+delay_coeff*V_DL[i]+V_SoC[i]-self.max_soc<=0,'cons_FCAS_ch_soc_{}'.format(i))
                
                FR_coeff_with_flag = self.FR_flag_arr[i]*self.fast_duration/self.energy_cap
                FL_coeff_with_flag = self.FL_flag_arr[i]*self.fast_duration/self.energy_cap
                SR_coeff_with_flag = self.SR_flag_arr[i]*self.slow_duration/self.energy_cap
                SL_coeff_with_flag = self.SL_flag_arr[i]*self.slow_duration/self.energy_cap
                DR_coeff_with_flag = self.DR_flag_arr[i]*self.delay_duration/self.energy_cap
                DL_coeff_with_flag = self.DL_flag_arr[i]*self.delay_duration/self.energy_cap
                if i>= 1 and i <= (self.horizon - 1):
                    lp_model += (V_SoC[i-1]-ES_coeff*V_ES_DCH[i-1]+ES_coeff*V_ES_CH[i-1]\
                                -FR_coeff_with_flag*V_FR[i-1]+FL_coeff_with_flag*V_FL[i-1]\
                                -SR_coeff_with_flag*V_SR[i-1]+SL_coeff_with_flag*V_SL[i-1]\
                                -DR_coeff_with_flag*V_DR[i-1]+DL_coeff_with_flag*V_DL[i-1] == V_SoC[i],'soc_{}'.format(i))
                    V_SoC[i].fixValue()

        return lp_model

    # set empty variable dict for saving
    def set_variable_dict(self):
        var_dict = {}
        if self.mode == 'ES':
            var_prefix_lst = ['DCHBinary','CHBinary','SoC','ESDCH','ESCH']
        elif self.mode == 'Arbitrage':
            var_prefix_lst = ['DCHBinary','CHBinary','SoC','FR','FL','SR','SL','DR','DL']
        else:
            var_prefix_lst = ['DCHBinary','CHBinary','SoC','ESDCH','ESCH','FR','FL','SR','SL','DR','DL']
        
        for prefix in var_prefix_lst:
            var_dict[prefix] = [0 for _ in range(self.horizon)]
            # var_dict[prefix] = []
        
        return var_dict

    # calculate real revenue 
    def cal_real_revenue(self,var_df):
        real_cum_revenue = 0
        real_cum_revenue_lst = []
        # 'RAISE6SECRRP','LOWER6SECRRP','RAISE60SECRRP','LOWER60SECRRP','RAISE5MINRRP','LOWER5MINRRP']
        for i in range(var_df.shape[0]):
            if self.mode == 'ES':
                dch_b,ch_b,dch,ch = var_df.loc[i,'DCHBinary'],var_df.loc[i,'CHBinary'],var_df.loc[i,'ESDCH'],var_df.loc[i,'ESCH']
                if dch_b: 
                    # pred_cum_revenue += dch * self.ES_dch_price_dict[i]
                    real_cum_revenue += dch * self.ES_actual_price[i]*self.spot_duration*self.dch_coeff
                else: 
                    # pred_cum_revenue -= ch * self.ES_ch_price_dict[i]
                    real_cum_revenue -= ch * self.ES_actual_price[i]*self.spot_duration/self.ch_coeff
            elif self.mode == 'Contingency_FCAS':
                dch_b,ch_b = var_df.loc[i,'DCHBinary'],var_df.loc[i,'CHBinary']
                FR_cap,SR_cap,DR_cap = var_df.loc[i,'FR'],var_df.loc[i,'SR'],var_df.loc[i,'DR']
                FL_cap,SL_cap,DL_cap = var_df.loc[i,'FL'],var_df.loc[i,'SL'],var_df.loc[i,'DL']
                if dch_b: 
                    # pred_cum_revenue += (FR_cap*self.FR_price_dict[i]+SR_cap*self.SR_price_dict[i]+DR_cap*self.DR_price_dict[i])
                    real_cum_revenue += FR_cap*self.FR_actual_price[i]*self.fast_duration*self.dch_coeff
                    real_cum_revenue += SR_cap*self.SR_actual_price[i]*self.slow_duration*self.dch_coeff
                    real_cum_revenue += DR_cap*self.DR_actual_price[i]*self.delay_duration*self.dch_coeff
                else: 
                    # pred_cum_revenue += (FL_cap*self.FL_price_dict[i]+SL_cap*self.SL_price_dict[i]+DL_cap*self.DL_price_dict[i])
                    real_cum_revenue += FL_cap*self.FL_actual_price[i]*self.fast_duration/self.ch_coeff
                    real_cum_revenue += SL_cap*self.SL_actual_price[i]*self.slow_duration/self.ch_coeff
                    real_cum_revenue += DL_cap*self.DL_actual_price[i]*self.delay_duration/self.ch_coeff
            else: 
                dch_b,ch_b,dch,ch = var_df.loc[i,'DCHBinary'],var_df.loc[i,'CHBinary'],var_df.loc[i,'ESDCH'],var_df.loc[i,'ESCH']
                FR_cap,SR_cap,DR_cap = var_df.loc[i,'FR'],var_df.loc[i,'SR'],var_df.loc[i,'DR']
                FL_cap,SL_cap,DL_cap = var_df.loc[i,'FL'],var_df.loc[i,'SL'],var_df.loc[i,'DL']
                if dch_b: 
                    # pred_cum_revenue += (dch*self.ES_dch_price_dict[i]+FR_cap*self.FR_price_dict[i]+SR_cap*self.SR_price_dict[i]+DR_cap*self.DR_price_dict[i])
                    real_cum_revenue += dch * self.ES_actual_price[i]*self.spot_duration*self.dch_coeff
                    real_cum_revenue += FR_cap*self.FR_actual_price[i]*self.fast_duration*self.dch_coeff
                    real_cum_revenue += SR_cap*self.SR_actual_price[i]*self.slow_duration*self.dch_coeff
                    real_cum_revenue += DR_cap*self.DR_actual_price[i]*self.delay_duration*self.dch_coeff
                else: 
                    # pred_cum_revenue += (-ch*self.ES_ch_price_dict[i]+FL_cap*self.FL_price_dict[i]+SL_cap*self.SL_price_dict[i]+DL_cap*self.DL_price_dict[i])
                    real_cum_revenue -= ch * self.ES_actual_price[i]*self.spot_duration/self.ch_coeff
                    real_cum_revenue += FL_cap*self.FL_actual_price[i]*self.fast_duration/self.ch_coeff
                    real_cum_revenue += SL_cap*self.SL_actual_price[i]*self.slow_duration/self.ch_coeff
                    real_cum_revenue += DL_cap*self.DL_actual_price[i]*self.delay_duration/self.ch_coeff
            
            # pred_cum_revenue_lst.append(pred_cum_revenue)
            real_cum_revenue_lst.append(real_cum_revenue)
        
        return real_cum_revenue_lst

    def main(self):
        # check if there are trained models 
        model_name = '{}_{}_{}_lstm_model_{}'.format(self.state_name,self.mode,self.market_year,self.epochs)
        if model_name not in os.listdir(self.lstm_save_path):
            self.train_eval_predictor()
        
        # load predicted price
        eval_price = pd.read_csv(os.path.join(self.eval_save_path,'{}_{}_{}_prediction.csv'.format(self.state_name,self.mode,self.market_year)),index_col=[0])
        self.load_price(eval_price)

        # build lp model and solve 
        lp_model = self.setup_lp_model()
        solver = pulp.GUROBI_CMD(keepFiles=True,msg=True)
        # solver = pulp.GLPK_CMD(path='glpsol',keepFiles=False,msg=True)
        lp_model.solve(solver)
        
        # save variables
        var_dict = self.set_variable_dict()
        for var in lp_model.variables():
            var_name,idx = var.name.split('_')
            var_value = var.value()
            # var_dict[var_name].append(var_value)
            var_dict[var_name][eval(idx)] = var_value
        var_df = pd.DataFrame.from_dict(var_dict)
        # calculate real revenue 
        real_cum_revenue_lst = self.cal_real_revenue(var_df)
        # var_df['pred_cum_revenue'] = pred_cum_revenue_lst
        var_df['real_cum_revenue'] = real_cum_revenue_lst
        
        var_df.to_csv(os.path.join(self.eval_save_path,'{}_{}_{}_res.csv'.format(self.state_name,self.mode,self.market_year)))
        
        # if (idx + 1) % 2000 == 0:
        #     os.system('rm -rf /dev/shm')


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',default='ES',type=str,help='mode')
    parser.add_argument('--state_name',default='VIC',type=str,help='state name')
    parser.add_argument('--market_year',default=2016,type=int,help='market_year')
    parser.add_argument('--seq_len',default=31,type=int,help='seq_len')
    args = parser.parse_args()

    lstm_model_save_folder = 'predict_then_optimize/lstm_models/{}{}'.format(args.state_name,args.market_year)
    os.makedirs(lstm_model_save_folder,exist_ok=True)
    eval_save_folder = 'predict_then_optimize/eval_results/{}{}'.format(args.state_name,args.market_year)
    os.makedirs(eval_save_folder,exist_ok=True)

    if args.mode == 'ES': price_dim = 1
    elif args.mode == 'Contingency_FCAS': price_dim = 6
    else: price_dim = 7
    pao_model = PredictThenOptimize(
        mode=args.mode,
        state_name=args.state_name,
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
    np.savetxt('predict_then_optimize/eval_results/{}{}/{}_{}_{}_runningTime.txt'.format(args.state_name,args.market_year,args.state_name,args.mode,args.market_year),running_time)

