import pandas as pd 
import argparse
import os 
import numpy as np 
import pulp
import time


from optim_predict_then_optimize_utils import create_seq_no_label,record_var



class PredictThenOptimize():
    def __init__(
        self,
        lstm_or_trans,
        mode,
        state_name,
        market_year,
    ) -> None:
        self.lstm_or_trans = lstm_or_trans

        self.mode = mode
        self.state_name = state_name
        self.market_year = market_year

        solver_log_save_folder = 'predict_then_optimize/eval_results/{}{}'.format(state_name,market_year)
        solver_log_save_name = 'optim_solver_with_{}_{}.log'.format(lstm_or_trans,mode)
        solver_log_save_path = os.path.join(solver_log_save_folder,solver_log_save_name)
        res_save_folder = 'predict_then_optimize/eval_results/{}{}'.format(state_name,market_year)
        res_file_name = 'optimization_res_with_{}_{}.csv'.format(lstm_or_trans,mode)
        res_save_path = os.path.join(res_save_folder,res_file_name)
        self.solver_log_save_path = solver_log_save_path
        self.res_save_path = res_save_path

    # load original data with State and Year
    def load_data(
        self,
        mpc_horizon,
        data_folder='NEM_annual_data',
        pred_data_folder='predict_then_optimize'
    ):
        data_file_name = '{}{}.csv'.format(self.state_name,self.market_year)
        data_path = os.path.join(data_folder,data_file_name)
        data_df = pd.read_csv(data_path,index_col=[0])

        last_two_month_len = int((60/5)*24*31) + int((60/5)*24*30)
        if self.mode!='ES':
            FR_flag_lst = data_df.iloc[-last_two_month_len:,-6].values.tolist()
            FL_flag_lst = data_df.iloc[-last_two_month_len:,-5].values.tolist()
            SR_flag_lst = data_df.iloc[-last_two_month_len:,-4].values.tolist()
            SL_flag_lst = data_df.iloc[-last_two_month_len:,-3].values.tolist()
            DR_flag_lst = data_df.iloc[-last_two_month_len:,-2].values.tolist()
            DL_flag_lst = data_df.iloc[-last_two_month_len:,-1].values.tolist()

            FR_flag_seq = create_seq_no_label(data=FR_flag_lst,seq_len=mpc_horizon)
            FL_flag_seq = create_seq_no_label(data=FL_flag_lst,seq_len=mpc_horizon)
            SR_flag_seq = create_seq_no_label(data=SR_flag_lst,seq_len=mpc_horizon)
            SL_flag_seq = create_seq_no_label(data=SL_flag_lst,seq_len=mpc_horizon)
            DR_flag_seq = create_seq_no_label(data=DR_flag_lst,seq_len=mpc_horizon)
            DL_flag_seq = create_seq_no_label(data=DL_flag_lst,seq_len=mpc_horizon)

            self.FR_flag_lst = FR_flag_lst
            self.FL_flag_lst = FL_flag_lst
            self.SR_flag_lst = SR_flag_lst
            self.SL_flag_lst = SL_flag_lst
            self.DR_flag_lst = DR_flag_lst
            self.DL_flag_lst = DL_flag_lst
            
            self.FR_flag4mpc_arr = FR_flag_seq
            self.FL_flag4mpc_arr = FL_flag_seq
            self.SR_flag4mpc_arr = SR_flag_seq
            self.SL_flag4mpc_arr = SL_flag_seq
            self.DR_flag4mpc_arr = DR_flag_seq
            self.DL_flag4mpc_arr = DL_flag_seq


        pred_data_file_name = 'price_pred_gen_{}.csv'.format(self.mode)
        pred_data_path = os.path.join(pred_data_folder,'{}_models'.format(self.lstm_or_trans),'{}{}'.format(self.state_name,self.market_year),pred_data_file_name)
        pred_data_df = pd.read_csv(pred_data_path,index_col=[0])
        pred_data_df['pred_price4mpc'] = pred_data_df['pred_price4mpc'].apply(lambda x: eval(x))
        pred_data_df['pred_price'] = pred_data_df['pred_price'].apply(lambda x: eval(x))
        pred_data_df['actual_price'] = pred_data_df['actual_price'].apply(lambda x: eval(x))

        pred_price4mpc_lst = pred_data_df['pred_price4mpc'].values.tolist()
        pred_price_lst = pred_data_df['pred_price'].values.tolist()
        actual_price_lst = pred_data_df['actual_price'].values.tolist()


        self.pred_price4mpc_lst = pred_price4mpc_lst
        self.pred_price_lst = pred_price_lst
        self.actual_price_lst = actual_price_lst

        self.data_len = len(actual_price_lst)
        

    def build_optim(
        self,
        mpc_step,
        mpc_horizon,
        cur_soc,
        soc_min,
        soc_max,
        rated_power,
        bat_cap,
        max_CFCAS_power,
        spot_duration,
        fast_duration,
        slow_duration,
        delayed_duration,
        dch_coeff,
        ch_coeff
    ):
        lp_model = pulp.LpProblem(name='{}'.format(self.mode),sense=pulp.LpMaximize) 

        var_binary_dch = pulp.LpVariable.dicts('varBinaryDch',range(mpc_horizon),cat='Binary')
        var_binary_ch = pulp.LpVariable.dicts('varBinaryCh',range(mpc_horizon),cat='Binary')
        var_soc = pulp.LpVariable.dicts('varSoC',range(mpc_horizon),lowBound=soc_min,upBound=soc_max)
        
        if self.mode == 'ES':
            # decision variables 
            var_spot_dch = pulp.LpVariable.dicts('varSpotDch',range(mpc_horizon),lowBound=0,upBound=rated_power)
            var_spot_ch = pulp.LpVariable.dicts('varSpotCh',range(mpc_horizon),lowBound=0,upBound=rated_power)

            # objective 
            lp_model += pulp.lpSum([self.pred_price4mpc_lst[mpc_step][idx][0]*var_spot_dch[idx] \
                                    - self.pred_price4mpc_lst[mpc_step][idx][0]*var_spot_ch[idx] for idx in range(mpc_horizon)])

            # constraints 
            for idx in range(mpc_horizon):
                lp_model += var_binary_ch[idx] + var_binary_dch[idx] <= 1, 'cons_dchChBinary_{}'.format(idx)
                lp_model += var_spot_dch[idx] - rated_power*var_binary_dch[idx] <= 0, 'cons_spotDch_{}'.format(idx)
                lp_model += var_spot_ch[idx] - rated_power*var_binary_ch[idx] <= 0, 'cons_spotCh_{}'.format(idx)
                if idx == 0: 
                    lp_model += cur_soc - var_spot_dch[idx]*spot_duration*dch_coeff/bat_cap + var_spot_ch[idx]*spot_duration/ch_coeff/bat_cap == var_soc[idx], 'cons_soc_{}'.format(idx)
                else:
                    lp_model += var_soc[idx-1] - var_spot_dch[idx]*spot_duration*dch_coeff/bat_cap + var_spot_ch[idx]*spot_duration/ch_coeff/bat_cap == var_soc[idx], 'cons_soc_{}'.format(idx)


        elif self.mode == 'Contingency_FCAS':
            # decision variables 
            var_bid_FR = pulp.LpVariable.dicts('varBidFR',range(mpc_horizon),lowBound=0,upBound=max_CFCAS_power)
            var_bid_FL = pulp.LpVariable.dicts('varBidFL',range(mpc_horizon),lowBound=0,upBound=max_CFCAS_power)
            var_bid_SR = pulp.LpVariable.dicts('varBidSR',range(mpc_horizon),lowBound=0,upBound=max_CFCAS_power)
            var_bid_SL = pulp.LpVariable.dicts('varBidSL',range(mpc_horizon),lowBound=0,upBound=max_CFCAS_power)
            var_bid_DR = pulp.LpVariable.dicts('varBidDR',range(mpc_horizon),lowBound=0,upBound=max_CFCAS_power)
            var_bid_DL = pulp.LpVariable.dicts('varBidDL',range(mpc_horizon),lowBound=0,upBound=max_CFCAS_power)

            # objective 
            lp_model += pulp.lpSum(
                [
                    self.pred_price4mpc_lst[mpc_step][idx][0]*var_bid_FR[idx] + self.pred_price4mpc_lst[mpc_step][idx][1]*var_bid_FL[idx]\
                    + self.pred_price4mpc_lst[mpc_step][idx][2]*var_bid_SR[idx] + self.pred_price4mpc_lst[mpc_step][idx][3]*var_bid_SL[idx]\
                    + self.pred_price4mpc_lst[mpc_step][idx][4]*var_bid_DR[idx] + self.pred_price4mpc_lst[mpc_step][idx][5]*var_bid_DL[idx] \
                    for idx in range(mpc_horizon)
                ]
            )

            # constraints
            for idx in range(mpc_horizon):
                lp_model += var_binary_ch[idx] + var_binary_dch[idx] <= 1, 'cons_dchChBinary_{}'.format(idx)
                lp_model += var_bid_FR[idx] - max_CFCAS_power*var_binary_dch[idx] <= 0, 'cons_FR_{}'.format(idx)
                lp_model += var_bid_FL[idx] - max_CFCAS_power*var_binary_ch[idx] <= 0, 'cons_FL_{}'.format(idx)
                lp_model += var_bid_SR[idx] - max_CFCAS_power*var_binary_dch[idx] <= 0, 'cons_SR_{}'.format(idx)
                lp_model += var_bid_SL[idx] - max_CFCAS_power*var_binary_ch[idx] <= 0, 'cons_SL_{}'.format(idx)
                lp_model += var_bid_DR[idx] - max_CFCAS_power*var_binary_dch[idx] <= 0, 'cons_DR_{}'.format(idx)
                lp_model += var_bid_DL[idx] - max_CFCAS_power*var_binary_ch[idx] <= 0, 'cons_DL_{}'.format(idx)
                if idx == 0: 
                    lp_model += cur_soc - var_bid_FR[idx]*self.FR_flag4mpc_arr[mpc_step][idx]*fast_duration*dch_coeff/bat_cap + var_bid_FL[idx]*self.FL_flag4mpc_arr[mpc_step][idx]*fast_duration/ch_coeff/bat_cap\
                                    - var_bid_SR[idx]*self.SR_flag4mpc_arr[mpc_step][idx]*slow_duration*dch_coeff/bat_cap + var_bid_SL[idx]*self.SL_flag4mpc_arr[mpc_step][idx]*slow_duration/ch_coeff/bat_cap\
                                    - var_bid_DR[idx]*self.DR_flag4mpc_arr[mpc_step][idx]*delayed_duration*dch_coeff/bat_cap + var_bid_DL[idx]*self.DL_flag4mpc_arr[mpc_step][idx]*delayed_duration/ch_coeff/bat_cap\
                                    == var_soc[idx], 'var_soc_{}'.format(idx)
                else:
                    lp_model += var_soc[idx-1] - var_bid_FR[idx]*self.FR_flag4mpc_arr[mpc_step][idx]*fast_duration*dch_coeff/bat_cap + var_bid_FL[idx]*self.FL_flag4mpc_arr[mpc_step][idx]*fast_duration/ch_coeff/bat_cap\
                                    - var_bid_SR[idx]*self.SR_flag4mpc_arr[mpc_step][idx]*slow_duration*dch_coeff/bat_cap + var_bid_SL[idx]*self.SL_flag4mpc_arr[mpc_step][idx]*slow_duration/ch_coeff/bat_cap\
                                    - var_bid_DR[idx]*self.DR_flag4mpc_arr[mpc_step][idx]*delayed_duration*dch_coeff/bat_cap + var_bid_DL[idx]*self.DL_flag4mpc_arr[mpc_step][idx]*delayed_duration/ch_coeff/bat_cap\
                                    == var_soc[idx], 'var_soc_{}'.format(idx)

        else: 
            # decision variables
            var_spot_dch = pulp.LpVariable.dicts('varSpotDch',range(mpc_horizon),lowBound=0,upBound=rated_power)
            var_spot_ch = pulp.LpVariable.dicts('varSpotCh',range(mpc_horizon),lowBound=0,upBound=rated_power)
            var_bid_FR = pulp.LpVariable.dicts('varBidFR',range(mpc_horizon),lowBound=0,upBound=max_CFCAS_power)
            var_bid_FL = pulp.LpVariable.dicts('varBidFL',range(mpc_horizon),lowBound=0,upBound=max_CFCAS_power)
            var_bid_SR = pulp.LpVariable.dicts('varBidSR',range(mpc_horizon),lowBound=0,upBound=max_CFCAS_power)
            var_bid_SL = pulp.LpVariable.dicts('varBidSL',range(mpc_horizon),lowBound=0,upBound=max_CFCAS_power)
            var_bid_DR = pulp.LpVariable.dicts('varBidDR',range(mpc_horizon),lowBound=0,upBound=max_CFCAS_power)
            var_bid_DL = pulp.LpVariable.dicts('varBidDL',range(mpc_horizon),lowBound=0,upBound=max_CFCAS_power)

            # objective 
            lp_model += pulp.lpSum(
                [   self.pred_price4mpc_lst[mpc_step][idx][0]*var_spot_dch[idx] - self.pred_price4mpc_lst[mpc_step][idx][0]*var_spot_ch[idx]\
                    + self.pred_price4mpc_lst[mpc_step][idx][1]*var_bid_FR[idx] + self.pred_price4mpc_lst[mpc_step][idx][2]*var_bid_FL[idx]\
                    + self.pred_price4mpc_lst[mpc_step][idx][3]*var_bid_SR[idx] + self.pred_price4mpc_lst[mpc_step][idx][4]*var_bid_SL[idx]\
                    + self.pred_price4mpc_lst[mpc_step][idx][5]*var_bid_DR[idx] + self.pred_price4mpc_lst[mpc_step][idx][6]*var_bid_DL[idx] \
                    for idx in range(mpc_horizon)
                ]
            )

            # constraints
            for idx in range(mpc_horizon):
                lp_model += var_binary_ch[idx] + var_binary_dch[idx] <= 1, 'cons_dchChBinary_{}'.format(idx)
                lp_model += var_spot_dch[idx] - rated_power*var_binary_dch[idx] <= 0, 'cons_spotDch_{}'.format(idx)
                lp_model += var_spot_ch[idx] - rated_power*var_binary_ch[idx] <= 0, 'cons_spotCh_{}'.format(idx)
                lp_model += var_bid_FR[idx] - max_CFCAS_power*var_binary_dch[idx] <= 0, 'cons_FR_{}'.format(idx)
                lp_model += var_bid_FL[idx] - max_CFCAS_power*var_binary_ch[idx] <= 0, 'cons_FL_{}'.format(idx)
                lp_model += var_bid_SR[idx] - max_CFCAS_power*var_binary_dch[idx] <= 0, 'cons_SR_{}'.format(idx)
                lp_model += var_bid_SL[idx] - max_CFCAS_power*var_binary_ch[idx] <= 0, 'cons_SL_{}'.format(idx)
                lp_model += var_bid_DR[idx] - max_CFCAS_power*var_binary_dch[idx] <= 0, 'cons_DR_{}'.format(idx)
                lp_model += var_bid_DL[idx] - max_CFCAS_power*var_binary_ch[idx] <= 0, 'cons_DL_{}'.format(idx)
                if idx == 0: 
                    lp_model += cur_soc - var_spot_dch[idx]*spot_duration*dch_coeff/bat_cap + var_spot_ch[idx]*spot_duration/ch_coeff/bat_cap\
                                - var_bid_FR[idx]*self.FR_flag4mpc_arr[mpc_step][idx]*fast_duration*dch_coeff/bat_cap + var_bid_FL[idx]*self.FL_flag4mpc_arr[mpc_step][idx]*fast_duration/ch_coeff/bat_cap\
                                - var_bid_SR[idx]*self.SR_flag4mpc_arr[mpc_step][idx]*slow_duration*dch_coeff/bat_cap + var_bid_SL[idx]*self.SL_flag4mpc_arr[mpc_step][idx]*slow_duration/ch_coeff/bat_cap\
                                - var_bid_DR[idx]*self.DR_flag4mpc_arr[mpc_step][idx]*delayed_duration*dch_coeff/bat_cap + var_bid_DL[idx]*self.DL_flag4mpc_arr[mpc_step][idx]*delayed_duration/ch_coeff/bat_cap\
                                == var_soc[idx], 'var_soc_{}'.format(idx)
                else:
                    lp_model += var_soc[idx-1] - var_spot_dch[idx]*spot_duration*dch_coeff/bat_cap + var_spot_ch[idx]*spot_duration/ch_coeff/bat_cap\
                                - var_bid_FR[idx]*self.FR_flag4mpc_arr[mpc_step][idx]*fast_duration*dch_coeff/bat_cap + var_bid_FL[idx]*self.FL_flag4mpc_arr[mpc_step][idx]*fast_duration/ch_coeff/bat_cap\
                                - var_bid_SR[idx]*self.SR_flag4mpc_arr[mpc_step][idx]*slow_duration*dch_coeff/bat_cap + var_bid_SL[idx]*self.SL_flag4mpc_arr[mpc_step][idx]*slow_duration/ch_coeff/bat_cap\
                                - var_bid_DR[idx]*self.DR_flag4mpc_arr[mpc_step][idx]*delayed_duration*dch_coeff/bat_cap + var_bid_DL[idx]*self.DL_flag4mpc_arr[mpc_step][idx]*delayed_duration/ch_coeff/bat_cap\
                                == var_soc[idx], 'var_soc_{}'.format(idx)


        return lp_model 


    def solve_optim(
        self,
        mpc_horizon,
        cur_soc,
        soc_min,
        soc_max,
        rated_power,
        bat_cap,
        max_CFCAS_power,
        spot_duration,
        fast_duration,
        slow_duration,
        delayed_duration,
        dch_coeff,
        ch_coeff
    ):
        if self.mode == 'ES': var_name_lst = ['varBinaryDch','varBinaryCh','varSoC','varSpotDch','varSpotCh']
        elif self.mode == 'Contingency_FCAS': var_name_lst = ['varBinaryDch','varBinaryCh','varSoC','varBidFR','varBidFL','varBidSR','varBidSL','varBidDR','varBidDL']
        else: var_name_lst = ['varBinaryDch','varBinaryCh','varSoC','varSpotDch','varSpotCh','varBidFR','varBidFL','varBidSR','varBidSL','varBidDR','varBidDL']
        append_var_name_lst = ['actual_price','cur_revenue']
        var_name_lst.extend(append_var_name_lst)
        var_dict = {var_name: [] for var_name in var_name_lst}


        for mpc_step in range(self.data_len-mpc_horizon):
        # for mpc_step in range(5): # for debug 
            lp_model = self.build_optim(
                mpc_step=mpc_step,
                mpc_horizon=mpc_horizon,
                cur_soc=cur_soc,
                soc_min=soc_min,
                soc_max=soc_max,
                rated_power=rated_power,
                bat_cap=bat_cap,
                max_CFCAS_power=max_CFCAS_power,
                spot_duration=spot_duration,
                fast_duration=fast_duration,
                slow_duration=slow_duration,
                delayed_duration=delayed_duration,
                dch_coeff=dch_coeff,
                ch_coeff=ch_coeff
            )
            solver = pulp.GUROBI(msg=False,gapRel=0.05,timeLimit=60,logPath=self.solver_log_save_path)
            solver_status = lp_model.solve(solver)

            if solver_status != 1:
                print('debug')

            # get var value 
            cur_soc = record_var(
                lp_model=lp_model,
                var_dict=var_dict,
                actual_price=self.actual_price_lst[mpc_step],
                spot_duration=spot_duration,
                fast_duration=fast_duration,
                slow_duration=slow_duration,
                delayed_duration=delayed_duration,
                mode=self.mode
            )

            # avoid stack overflow 
            if (mpc_step + 1) % 1000 == 0:
                os.system('rm -rf /dev/shm')

            # log 
            print('Step {} finished'.format(mpc_step))

        
        var_df = pd.DataFrame(var_dict)
        var_df['real_cum_revenue'] = var_df['cur_revenue'].cumsum()
        var_df.to_csv(self.res_save_path)






if __name__ == "__main__":
    start_time = time.time()


    parser = argparse.ArgumentParser()
    parser.add_argument('--lstm_or_trans',default='lstm',type=str,help='choose model: lstm or transformer')
    parser.add_argument('--mode',default='Contingency_FCAS',type=str,help='market type')
    parser.add_argument('--market_year',default=2016,type=int,help='read data begin')
    parser.add_argument('--state_name',default='VIC',type=str,help='state name')
    parser.add_argument('--mpc_horizon',default=48,type=int,help='input series length')
    parser.add_argument('--cur_soc',default=0.5,type=float,help='initial soc')
    parser.add_argument('--soc_min',default=0.05,type=float,help='soc minimum')
    parser.add_argument('--soc_max',default=0.95,type=float,help='soc maximum')
    parser.add_argument('--rated_power',default=2,type=float,help='rated power')
    parser.add_argument('--bat_cap',default=10,type=float,help='battery capacity')
    parser.add_argument('--max_CFCAS_power',default=1,type=float,help='max contingency FCAS power')
    parser.add_argument('--spot_duration',default=5/60,type=float,help='spot market dispatch duration')
    parser.add_argument('--fast_duration',default=5/60/5/15,type=float,help='fast CFCAS dispatch duration')
    parser.add_argument('--slow_duration',default=5/60/5,type=float,help='slow CFCAS dispatch interval')
    parser.add_argument('--delayed_duration',default=5/60,type=float,help='delayed CFCAS disptach duration')
    parser.add_argument('--dch_coeff',default=0.95,type=float,help='discharging coeff')
    parser.add_argument('--ch_coeff',default=0.95,type=float,help='charging coeff')

    args = parser.parse_args()

    lstm_or_trans = args.lstm_or_trans
    mode = args.mode 
    market_year = args.market_year
    state_name = args.state_name
    mpc_horizon = args.mpc_horizon 
    cur_soc = args.cur_soc
    soc_min = args.soc_min
    soc_max = args.soc_max 
    rated_power = args.rated_power 
    bat_cap = args.bat_cap 
    max_CFCAS_power = args.max_CFCAS_power
    spot_duration = args.spot_duration
    fast_duration = args.fast_duration 
    slow_duration = args.slow_duration 
    delayed_duration = args.delayed_duration 
    dch_coeff = args.dch_coeff 
    ch_coeff = args.ch_coeff 


    pao_obj = PredictThenOptimize(
        lstm_or_trans=lstm_or_trans,
        mode=mode,
        state_name=state_name,
        market_year=market_year
    )

    pao_obj.load_data(
        mpc_horizon=mpc_horizon
    )

    pao_obj.solve_optim(
        mpc_horizon=mpc_horizon,
        cur_soc=cur_soc,
        soc_min=soc_min,
        soc_max=soc_max,
        rated_power=rated_power,
        bat_cap=bat_cap,
        max_CFCAS_power=max_CFCAS_power,
        spot_duration=spot_duration,
        fast_duration=fast_duration,
        slow_duration=slow_duration,
        delayed_duration=delayed_duration,
        dch_coeff=dch_coeff,
        ch_coeff=ch_coeff
    )


    end_time = time.time()


    collpse_time = end_time - start_time 


    runtime_save_folder = 'predict_then_optimize/eval_results/{}{}'.format(state_name,market_year)
    runtime_save_name = 'optimization_run_time_{}.txt'.format(mode)
    runtime_save_path = os.path.join(runtime_save_folder,runtime_save_name)
    np.savetxt(runtime_save_path,[collpse_time])

