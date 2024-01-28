import pandas as pd 
import argparse
import os 
import numpy as np 
import pulp
import time


from optim_predict_then_optimize_utils import record_var_perfect_info



class PerfectInfoBenchmark():
    def __init__(
        self,
        mode,
        state_name,
        market_year,
    ) -> None:

        self.mode = mode
        self.state_name = state_name
        self.market_year = market_year

        solver_log_save_folder = 'predict_then_optimize/eval_results/{}{}'.format(state_name,market_year)
        solver_log_save_name = 'optim_solver_with_perfect_info_{}.log'.format(mode)
        solver_log_save_path = os.path.join(solver_log_save_folder,solver_log_save_name)
        res_save_folder = 'predict_then_optimize/eval_results/{}{}'.format(state_name,market_year)
        res_file_name = 'optimization_res_with_perfect_info_{}.csv'.format(mode)
        res_save_path = os.path.join(res_save_folder,res_file_name)
        self.solver_log_save_path = solver_log_save_path
        self.res_save_path = res_save_path

    # load original data with State and Year
    def load_data(
        self,
        data_folder='NEM_annual_data',
    ):
        data_file_name = '{}{}.csv'.format(self.state_name,self.market_year)
        data_path = os.path.join(data_folder,data_file_name)
        data_df = pd.read_csv(data_path,index_col=[0])

        last_two_month_len = int((60/5)*24*31) + int((60/5)*24*30)
        if self.mode!='ES':
            FR_flag_arr = data_df.iloc[-last_two_month_len:,-6].values
            FL_flag_arr = data_df.iloc[-last_two_month_len:,-5].values
            SR_flag_arr = data_df.iloc[-last_two_month_len:,-4].values
            SL_flag_arr = data_df.iloc[-last_two_month_len:,-3].values
            DR_flag_arr = data_df.iloc[-last_two_month_len:,-2].values
            DL_flag_arr = data_df.iloc[-last_two_month_len:,-1].values

            self.FR_flag_arr = FR_flag_arr
            self.FL_flag_arr = FL_flag_arr
            self.SR_flag_arr = SR_flag_arr
            self.SL_flag_arr = SL_flag_arr
            self.DR_flag_arr = DR_flag_arr
            self.DL_flag_arr = DL_flag_arr

        if self.mode == 'ES': price_cols = ['RRP']
        elif self.mode == 'Contingency_FCAS': price_cols = ['RAISE6SECRRP','LOWER6SECRRP','RAISE60SECRRP','LOWER60SECRRP','RAISE5MINRRP','LOWER5MINRRP']
        else: price_cols = ['RRP','RAISE6SECRRP','LOWER6SECRRP','RAISE60SECRRP','LOWER60SECRRP','RAISE5MINRRP','LOWER5MINRRP']

        price_df = data_df[price_cols]
        eval_price_df = price_df.iloc[-last_two_month_len:,:].reset_index(drop=True)
        eval_price_arr = eval_price_df.values 

        self.eval_price_arr = eval_price_arr

        self.data_len = eval_price_df.shape[0]
        # self.data_len = 5 # for debug 
            

        

    def build_optim(
        self,
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

        var_binary_dch = pulp.LpVariable.dicts('varBinaryDch',range(self.data_len),cat='Binary')
        var_binary_ch = pulp.LpVariable.dicts('varBinaryCh',range(self.data_len),cat='Binary')
        var_soc = pulp.LpVariable.dicts('varSoC',range(self.data_len),lowBound=soc_min,upBound=soc_max)
        
        if self.mode == 'ES':
            # decision variables 
            var_spot_dch = pulp.LpVariable.dicts('varSpotDch',range(self.data_len),lowBound=0,upBound=rated_power)
            var_spot_ch = pulp.LpVariable.dicts('varSpotCh',range(self.data_len),lowBound=0,upBound=rated_power)

            # objective 
            lp_model += pulp.lpSum([self.eval_price_arr[idx][0]*var_spot_dch[idx] \
                                    - self.eval_price_arr[idx][0]*var_spot_ch[idx] for idx in range(self.data_len)])

            # constraints 
            for idx in range(self.data_len):
                lp_model += var_binary_ch[idx] + var_binary_dch[idx] <= 1, 'cons_dchChBinary_{}'.format(idx)
                lp_model += var_spot_dch[idx] - rated_power*var_binary_dch[idx] <= 0, 'cons_spotDch_{}'.format(idx)
                lp_model += var_spot_ch[idx] - rated_power*var_binary_ch[idx] <= 0, 'cons_spotCh_{}'.format(idx)
                if idx == 0: 
                    lp_model += cur_soc - var_spot_dch[idx]*spot_duration*dch_coeff/bat_cap + var_spot_ch[idx]*spot_duration/ch_coeff/bat_cap == var_soc[idx], 'cons_soc_{}'.format(idx)
                else:
                    lp_model += var_soc[idx-1] - var_spot_dch[idx]*spot_duration*dch_coeff/bat_cap + var_spot_ch[idx]*spot_duration/ch_coeff/bat_cap == var_soc[idx], 'cons_soc_{}'.format(idx)


        elif self.mode == 'Contingency_FCAS':
            # decision variables 
            var_bid_FR = pulp.LpVariable.dicts('varBidFR',range(self.data_len),lowBound=0,upBound=max_CFCAS_power)
            var_bid_FL = pulp.LpVariable.dicts('varBidFL',range(self.data_len),lowBound=0,upBound=max_CFCAS_power)
            var_bid_SR = pulp.LpVariable.dicts('varBidSR',range(self.data_len),lowBound=0,upBound=max_CFCAS_power)
            var_bid_SL = pulp.LpVariable.dicts('varBidSL',range(self.data_len),lowBound=0,upBound=max_CFCAS_power)
            var_bid_DR = pulp.LpVariable.dicts('varBidDR',range(self.data_len),lowBound=0,upBound=max_CFCAS_power)
            var_bid_DL = pulp.LpVariable.dicts('varBidDL',range(self.data_len),lowBound=0,upBound=max_CFCAS_power)

            # objective 
            lp_model += pulp.lpSum(
                [
                    self.eval_price_arr[idx][0]*var_bid_FR[idx] + self.eval_price_arr[idx][1]*var_bid_FL[idx]\
                    + self.eval_price_arr[idx][2]*var_bid_SR[idx] + self.eval_price_arr[idx][3]*var_bid_SL[idx]\
                    + self.eval_price_arr[idx][4]*var_bid_DR[idx] + self.eval_price_arr[idx][5]*var_bid_DL[idx] \
                    for idx in range(self.data_len)
                ]
            )

            # constraints
            for idx in range(self.data_len):
                lp_model += var_binary_ch[idx] + var_binary_dch[idx] <= 1, 'cons_dchChBinary_{}'.format(idx)
                lp_model += var_bid_FR[idx] - max_CFCAS_power*var_binary_dch[idx] <= 0, 'cons_FR_{}'.format(idx)
                lp_model += var_bid_FL[idx] - max_CFCAS_power*var_binary_ch[idx] <= 0, 'cons_FL_{}'.format(idx)
                lp_model += var_bid_SR[idx] - max_CFCAS_power*var_binary_dch[idx] <= 0, 'cons_SR_{}'.format(idx)
                lp_model += var_bid_SL[idx] - max_CFCAS_power*var_binary_ch[idx] <= 0, 'cons_SL_{}'.format(idx)
                lp_model += var_bid_DR[idx] - max_CFCAS_power*var_binary_dch[idx] <= 0, 'cons_DR_{}'.format(idx)
                lp_model += var_bid_DL[idx] - max_CFCAS_power*var_binary_ch[idx] <= 0, 'cons_DL_{}'.format(idx)
                if idx == 0: 
                    lp_model += cur_soc - var_bid_FR[idx]*self.FR_flag_arr[idx]*fast_duration*dch_coeff/bat_cap + var_bid_FL[idx]*self.FL_flag_arr[idx]*fast_duration/ch_coeff/bat_cap\
                                    - var_bid_SR[idx]*self.SR_flag_arr[idx]*slow_duration*dch_coeff/bat_cap + var_bid_SL[idx]*self.SL_flag_arr[idx]*slow_duration/ch_coeff/bat_cap\
                                    - var_bid_DR[idx]*self.DR_flag_arr[idx]*delayed_duration*dch_coeff/bat_cap + var_bid_DL[idx]*self.DL_flag_arr[idx]*delayed_duration/ch_coeff/bat_cap\
                                    == var_soc[idx], 'var_soc_{}'.format(idx)
                else:
                    lp_model += var_soc[idx-1] - var_bid_FR[idx]*self.FR_flag_arr[idx]*fast_duration*dch_coeff/bat_cap + var_bid_FL[idx]*self.FL_flag_arr[idx]*fast_duration/ch_coeff/bat_cap\
                                    - var_bid_SR[idx]*self.SR_flag_arr[idx]*slow_duration*dch_coeff/bat_cap + var_bid_SL[idx]*self.SL_flag_arr[idx]*slow_duration/ch_coeff/bat_cap\
                                    - var_bid_DR[idx]*self.DR_flag_arr[idx]*delayed_duration*dch_coeff/bat_cap + var_bid_DL[idx]*self.DL_flag_arr[idx]*delayed_duration/ch_coeff/bat_cap\
                                    == var_soc[idx], 'var_soc_{}'.format(idx)

        else: 
            # decision variables
            var_spot_dch = pulp.LpVariable.dicts('varSpotDch',range(self.data_len),lowBound=0,upBound=rated_power)
            var_spot_ch = pulp.LpVariable.dicts('varSpotCh',range(self.data_len),lowBound=0,upBound=rated_power)
            var_bid_FR = pulp.LpVariable.dicts('varBidFR',range(self.data_len),lowBound=0,upBound=max_CFCAS_power)
            var_bid_FL = pulp.LpVariable.dicts('varBidFL',range(self.data_len),lowBound=0,upBound=max_CFCAS_power)
            var_bid_SR = pulp.LpVariable.dicts('varBidSR',range(self.data_len),lowBound=0,upBound=max_CFCAS_power)
            var_bid_SL = pulp.LpVariable.dicts('varBidSL',range(self.data_len),lowBound=0,upBound=max_CFCAS_power)
            var_bid_DR = pulp.LpVariable.dicts('varBidDR',range(self.data_len),lowBound=0,upBound=max_CFCAS_power)
            var_bid_DL = pulp.LpVariable.dicts('varBidDL',range(self.data_len),lowBound=0,upBound=max_CFCAS_power)

            # objective 
            lp_model += pulp.lpSum(
                [   self.eval_price_arr[idx][0]*var_spot_dch[idx] - self.eval_price_arr[idx][0]*var_spot_ch[idx]\
                    + self.eval_price_arr[idx][1]*var_bid_FR[idx] + self.eval_price_arr[idx][2]*var_bid_FL[idx]\
                    + self.eval_price_arr[idx][3]*var_bid_SR[idx] + self.eval_price_arr[idx][4]*var_bid_SL[idx]\
                    + self.eval_price_arr[idx][5]*var_bid_DR[idx] + self.eval_price_arr[idx][6]*var_bid_DL[idx] \
                    for idx in range(self.data_len)
                ]
            )

            # constraints
            for idx in range(self.data_len):
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
                                - var_bid_FR[idx]*self.FR_flag_arr[idx]*fast_duration*dch_coeff/bat_cap + var_bid_FL[idx]*self.FL_flag_arr[idx]*fast_duration/ch_coeff/bat_cap\
                                - var_bid_SR[idx]*self.SR_flag_arr[idx]*slow_duration*dch_coeff/bat_cap + var_bid_SL[idx]*self.SL_flag_arr[idx]*slow_duration/ch_coeff/bat_cap\
                                - var_bid_DR[idx]*self.DR_flag_arr[idx]*delayed_duration*dch_coeff/bat_cap + var_bid_DL[idx]*self.DL_flag_arr[idx]*delayed_duration/ch_coeff/bat_cap\
                                == var_soc[idx], 'var_soc_{}'.format(idx)
                else:
                    lp_model += var_soc[idx-1] - var_spot_dch[idx]*spot_duration*dch_coeff/bat_cap + var_spot_ch[idx]*spot_duration/ch_coeff/bat_cap\
                                - var_bid_FR[idx]*self.FR_flag_arr[idx]*fast_duration*dch_coeff/bat_cap + var_bid_FL[idx]*self.FL_flag_arr[idx]*fast_duration/ch_coeff/bat_cap\
                                - var_bid_SR[idx]*self.SR_flag_arr[idx]*slow_duration*dch_coeff/bat_cap + var_bid_SL[idx]*self.SL_flag_arr[idx]*slow_duration/ch_coeff/bat_cap\
                                - var_bid_DR[idx]*self.DR_flag_arr[idx]*delayed_duration*dch_coeff/bat_cap + var_bid_DL[idx]*self.DL_flag_arr[idx]*delayed_duration/ch_coeff/bat_cap\
                                == var_soc[idx], 'var_soc_{}'.format(idx)


        return lp_model 


    def solve_optim(
        self,
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
        var_dict = {var_name: [0 for _ in range(self.data_len)] for var_name in var_name_lst}


        lp_model = self.build_optim(
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
        solver = pulp.GUROBI(msg=True,gapRel=0.05,logPath=self.solver_log_save_path)
        solver_status = lp_model.solve(solver)

        # if solver_status != 1:
        #     print('debug')

        # get var value 
        record_var_perfect_info(
            lp_model=lp_model,
            var_dict=var_dict,
            actual_price_lst=self.eval_price_arr.tolist(),
            spot_duration=spot_duration,
            fast_duration=fast_duration,
            slow_duration=slow_duration,
            delayed_duration=delayed_duration,
            mode=self.mode,
            data_len=self.data_len
        )
        
        var_df = pd.DataFrame(var_dict)
        var_df.to_csv(self.res_save_path)






if __name__ == "__main__":
    start_time = time.time()


    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',default='Contingency_FCAS',type=str,help='market type')
    parser.add_argument('--market_year',default=2016,type=int,help='read data begin')
    parser.add_argument('--state_name',default='VIC',type=str,help='state name')
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

    mode = args.mode 
    market_year = args.market_year
    state_name = args.state_name
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

    per_info_optim_obj = PerfectInfoBenchmark(
        mode=mode,
        state_name=state_name,
        market_year=market_year
    )

    per_info_optim_obj.load_data()

    per_info_optim_obj.solve_optim(
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
    runtime_save_name = 'optimization_run_time_perfect_info_{}.txt'.format(mode)
    runtime_save_path = os.path.join(runtime_save_folder,runtime_save_name)
    np.savetxt(runtime_save_path,[collpse_time])

