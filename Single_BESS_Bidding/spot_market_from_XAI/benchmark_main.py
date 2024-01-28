import pandas as pd 
import argparse
import os 
import numpy as np 
import pulp
import time

from benchmark_utils import record_var


class PredictThenOptimize():
    def __init__(
        self,
        state_name,
        market_year,
    ) -> None:

        self.state_name = state_name
        self.market_year = market_year

        solver_log_save_folder = 'benchmark/eval_res/{}{}'.format(state_name,market_year)
        os.makedirs(solver_log_save_folder, exist_ok=True)
        solver_log_save_name = 'optim_solver.log'
        solver_log_save_path = os.path.join(solver_log_save_folder,solver_log_save_name)
        res_save_folder = 'benchmark/eval_res/{}{}'.format(state_name,market_year)
        res_file_name = 'optimization_res.csv'
        res_save_path = os.path.join(res_save_folder,res_file_name)
        self.solver_log_save_path = solver_log_save_path
        self.res_save_path = res_save_path


    # load original data with State and Year
    def load_data(
        self,
        pred_data_folder='benchmark'
    ):
        pred_data_file_name = 'price_pred_gen.csv'
        pred_data_path = os.path.join(pred_data_folder,'forecast','{}{}'.format(self.state_name,self.market_year),pred_data_file_name)
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
        spot_duration,
        dch_coeff,
        ch_coeff
    ):
        lp_model = pulp.LpProblem(name='bat',sense=pulp.LpMaximize) 

        var_binary_dch = pulp.LpVariable.dicts('varBinaryDch',range(mpc_horizon),cat='Binary')
        var_binary_ch = pulp.LpVariable.dicts('varBinaryCh',range(mpc_horizon),cat='Binary')
        var_soc = pulp.LpVariable.dicts('varSoC',range(mpc_horizon),lowBound=soc_min,upBound=soc_max)
        
        # decision variables 
        var_bid_dch = pulp.LpVariable.dicts('varBidDch',range(mpc_horizon),lowBound=0,upBound=rated_power)
        var_bid_ch = pulp.LpVariable.dicts('varBidCh',range(mpc_horizon),lowBound=0,upBound=rated_power)

        # objective 
        lp_model += pulp.lpSum([self.pred_price4mpc_lst[mpc_step][idx][0]*var_bid_dch[idx] \
                                - self.pred_price4mpc_lst[mpc_step][idx][0]*var_bid_ch[idx] for idx in range(mpc_horizon)])

        # constraints 
        for idx in range(mpc_horizon):
            lp_model += var_binary_ch[idx] + var_binary_dch[idx] <= 1, 'cons_dchChBinary_{}'.format(idx)
            lp_model += var_bid_dch[idx] - rated_power*var_binary_dch[idx] <= 0, 'cons_bidDch_{}'.format(idx)
            lp_model += var_bid_ch[idx] - rated_power*var_binary_ch[idx] <= 0, 'cons_bidCh_{}'.format(idx)
            if idx == 0: 
                lp_model += cur_soc - var_bid_dch[idx]*spot_duration/dch_coeff/bat_cap + var_bid_ch[idx]*spot_duration*ch_coeff/bat_cap == var_soc[idx], 'cons_soc_{}'.format(idx)
            else:
                lp_model += var_soc[idx-1] - var_bid_dch[idx]*spot_duration/dch_coeff/bat_cap + var_bid_ch[idx]*spot_duration*ch_coeff/bat_cap == var_soc[idx], 'cons_soc_{}'.format(idx)


        return lp_model 


    def solve_optim(
        self,
        mpc_horizon,
        cur_soc,
        soc_min,
        soc_max,
        rated_power,
        bat_cap,
        spot_duration,
        dch_coeff,
        ch_coeff
    ):
        var_name_lst = ['varBinaryDch','varBinaryCh','varSoC','varBidDch','varBidCh']
        append_var_name_lst = ['pred_price','actual_price','cur_revenue']
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
                spot_duration=spot_duration,
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
                pred_price=self.pred_price_lst[mpc_step],
                spot_duration=spot_duration,
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
    parser.add_argument('--market_year',default=2021,type=int,help='read data begin')
    parser.add_argument('--state_name',default='VIC',type=str,help='state name')
    parser.add_argument('--mpc_horizon',default=4,type=int,help='input series length')
    parser.add_argument('--cur_soc',default=0.5,type=float,help='initial soc')
    parser.add_argument('--soc_min',default=0.05,type=float,help='soc minimum')
    parser.add_argument('--soc_max',default=0.95,type=float,help='soc maximum')
    parser.add_argument('--rated_power',default=5,type=float,help='rated power')
    parser.add_argument('--bat_cap',default=5,type=float,help='battery capacity')
    parser.add_argument('--dch_coeff',default=0.95,type=float,help='discharging coeff')
    parser.add_argument('--ch_coeff',default=0.95,type=float,help='charging coeff')

    args = parser.parse_args()

    market_year = args.market_year
    state_name = args.state_name
    mpc_horizon = args.mpc_horizon 
    cur_soc = args.cur_soc
    soc_min = args.soc_min
    soc_max = args.soc_max 
    rated_power = args.rated_power 
    bat_cap = args.bat_cap 
    spot_duration = 5/60
    dch_coeff = args.dch_coeff 
    ch_coeff = args.ch_coeff 


    pao_obj = PredictThenOptimize(
        state_name=state_name,
        market_year=market_year
    )

    pao_obj.load_data()

    pao_obj.solve_optim(
        mpc_horizon=mpc_horizon,
        cur_soc=cur_soc,
        soc_min=soc_min,
        soc_max=soc_max,
        rated_power=rated_power,
        bat_cap=bat_cap,
        spot_duration=spot_duration,
        dch_coeff=dch_coeff,
        ch_coeff=ch_coeff
    )


    end_time = time.time()

    collpse_time = end_time - start_time 

    runtime_save_folder = 'benchmark/eval_res/{}{}'.format(state_name,market_year)
    runtime_save_name = 'optimization_run_time.txt'
    runtime_save_path = os.path.join(runtime_save_folder,runtime_save_name)
    np.savetxt(runtime_save_path,[collpse_time])