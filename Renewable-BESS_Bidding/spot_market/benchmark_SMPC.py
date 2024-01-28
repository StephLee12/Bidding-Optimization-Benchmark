import pandas as pd 
import argparse
import os 
import pulp
import numpy as np 
import time
import logging  

from benchmark_DMPC import DMPC
from benchmark_utils import record_var_SMPC
from benchmark_utils import get_next_move_SMPC
from benchmark_utils import save_revenue_info_DMPC
from benchmark_utils import permute_DMPC_predictions
from benchmark_utils import load_permuted_predictions


class SMPC(DMPC):
    def __init__(
        self, 
        state_name, 
        market_year, 
        solar_farm_name
    ) -> None:
        super().__init__(state_name, market_year, solar_farm_name)

        res_save_folder = 'benchmark_res'
        res_file_name = 'benchmark_SMPC_res.csv'
        self.res_save_path = os.path.join(res_save_folder,res_file_name)

    
    def load_permuted_predictions(
        self,
        solar_std,
        price_std,
        gen_scenario_num,
        load_scenario_num,
        mpc_horizon,
        solar_cap,
        price_min,
        price_max,
        load_flag
    ):  
        # we assume knowing the distribution of forecast error,solar_std and price_std 

        save_folder = 'benchmark_res'
        solar_file_name = 'solar_pred4smpc.csv'
        price_file_name = 'price_pred4smpc.csv'
        solar_pred_save_path = os.path.join(save_folder,solar_file_name)
        price_pred_save_path = os.path.join(save_folder,price_file_name)


        if load_flag == 0: # need permute DMPC predictions 
            pred_solar_lst,pred_price_lst = permute_DMPC_predictions(
                DMPC_pred_solar_lst=self.pred_solar_lst,
                DMPC_pred_price_lst=self.pred_price_lst,
                solar_std=solar_std,
                price_std=price_std,
                scenario_num=gen_scenario_num,
                mpc_horizon=mpc_horizon,
                solar_cap=solar_cap,
                price_min=price_min,
                price_max=price_max,
                solar_pred_save_path=solar_pred_save_path,
                price_pred_save_path=price_pred_save_path
            )

            self.pred_solar_lst = pred_solar_lst
            self.pred_price_lst = pred_price_lst

        else: # directly load permuted predictions 
            pred_solar_lst,pred_price_lst = load_permuted_predictions(
                solar_pred_save_path=solar_pred_save_path,
                price_pred_save_path=price_pred_save_path,
                scenario_num=load_scenario_num
            )

            self.pred_solar_lst = pred_solar_lst
            self.pred_price_lst = pred_price_lst




    def build_optim(
        self,
        optim_step,
        mpc_horizon,
        cur_soc,
        bat_soc_min,
        bat_soc_max,
        bat_rated_power,
        dispatch_duration,
        bat_cap,
        dch_eff,
        ch_eff,
        scenario_idx
    ):
        lp_model = pulp.LpProblem(name='bat',sense=pulp.LpMaximize)

        # binary variables for battery discharge/charge
        var_bat_binary_dch = pulp.LpVariable.dicts('varBatBinaryDch',range(mpc_horizon),cat='Binary')
        var_bat_binary_ch = pulp.LpVariable.dicts('varBatBinaryCh',range(mpc_horizon),cat='Binary')
        # continuous variable for battery soc 
        var_bat_soc = pulp.LpVariable.dicts('varSoc',range(mpc_horizon),lowBound=bat_soc_min,upBound=bat_soc_max,cat='Continuous')
        var_bat_soc[0].setInitialValue(cur_soc)
        var_bat_soc[0].fixValue()
        # continuous variables for battery bid to discharge and charge 
        var_bat_bid_dch = pulp.LpVariable.dicts('varBatBidDch',range(mpc_horizon),lowBound=0,upBound=bat_rated_power,cat='Continuous')
        var_bat_bid_ch = pulp.LpVariable.dicts('varBatBidCh',range(mpc_horizon),lowBound=0,upBound=bat_rated_power,cat='Continuous')
        # continuous variable for reserve bid for curtailment 
        var_bat_reserve = pulp.LpVariable.dicts('varBatReserve',range(mpc_horizon),lowBound=0,upBound=bat_rated_power,cat='Continuous')


        cur_pred_price_lst = self.pred_price_lst[optim_step][scenario_idx]
        cur_pred_curtail_lst = self.pred_curtail_lst[optim_step]


        # objective 
        obj = pulp.lpSum([cur_pred_price_lst[idx]*var_bat_bid_dch[idx] - cur_pred_price_lst[idx]*var_bat_bid_ch[idx] for idx in range(mpc_horizon)])
        lp_model += obj, 'objective' 

        # constraints 
        for idx in range(mpc_horizon):
            # charge and discharge cannot happen simultaneously 
            lp_model += var_bat_binary_dch[idx]+var_bat_binary_ch[idx]<=1, 'cons_chDchBinary_{}'.format(idx)
            # bid power constraints 
            lp_model += var_bat_bid_dch[idx]-bat_rated_power*var_bat_binary_dch[idx]<=0, 'cons_dchBid_{}'.format(idx)
            lp_model += var_bat_bid_ch[idx]-bat_rated_power*var_bat_binary_ch[idx]<=0, 'cons_chBid_{}'.format(idx)
            lp_model += var_bat_reserve[idx]-cur_pred_curtail_lst[idx]*bat_rated_power*var_bat_binary_ch[idx]<=0, 'cons_curtailBid_{}'.format(idx)
            lp_model += var_bat_bid_ch[idx]+var_bat_reserve[idx]-bat_rated_power<=0, 'cons_chTotBid_{}'.format(idx)
            # soc constraint 
            dch_coeff = dispatch_duration*dch_eff/bat_cap
            ch_coeff = dispatch_duration/(bat_cap*ch_eff)
            lp_model += dch_coeff*var_bat_bid_dch[idx]+bat_soc_min-var_bat_soc[idx]<=0, 'cons_dchSoc_{}'.format(idx)
            lp_model += ch_coeff*(var_bat_bid_ch[idx]+var_bat_reserve[idx])+var_bat_soc[idx]-bat_soc_max<=0, 'cons_chSoc_{}'.format(idx) 
            # update soc 
            if idx>=1 and idx<=mpc_horizon-1:
                lp_model += var_bat_soc[idx-1]+ch_coeff*(var_bat_bid_ch[idx-1]+var_bat_reserve[idx-1])-dch_coeff*var_bat_bid_dch[idx-1]==var_bat_soc[idx], 'cons_soc_{}'.format(idx)
                var_bat_soc[idx].fixValue()
        
        return lp_model 
    

    def solve_optim(
        self,
        mpc_horizon,
        init_soc,
        bat_soc_min,
        bat_soc_max,
        bat_rated_power,
        dispatch_duration,
        bat_cap,
        dch_eff,
        ch_eff,
        solar_penalty_coeff,
        logger,
        scenario_num
    ):
        var_name_lst = ['varBatBinaryDch','varBatBinaryCh','varSoc','varBatBidDch','varBatBidCh','varBatReserve']
        var_name_lst.append('curtailResponseBid') 
        var_dict = {var_name:[] for var_name in var_name_lst}
        var_dict['varSoc'].append(init_soc)

        for optim_step in range(self.eval_data_len-mpc_horizon):
        # for optim_step in range(2): # for debug 
            scenario_var_name_lst = ['varBatBinaryDch','varBatBinaryCh','varBatBidDch','varBatBidCh','varBatReserve']
            scenario_var_dict = {var_name:[] for var_name in scenario_var_name_lst}
            for scenario_idx in range(scenario_num):
                lp_model = self.build_optim(
                    optim_step=optim_step,
                    mpc_horizon=mpc_horizon,
                    cur_soc=var_dict['varSoc'][optim_step],
                    bat_soc_min=bat_soc_min,
                    bat_soc_max=bat_soc_max,
                    bat_rated_power=bat_rated_power,
                    dispatch_duration=dispatch_duration,
                    bat_cap=bat_cap,
                    dch_eff=dch_eff,
                    ch_eff=ch_eff,
                    scenario_idx=scenario_idx
                )
                solver = pulp.GUROBI(msg=False,gapRel=0.05,timeLimit=60,logPath='SMPC_gurobi.log')
                # solver = pulp.GUROBI_CMD(msg=True,gapRel=0.005,timeLimit=60,logPath='SMPC_gurobi.log')
                solve_status = lp_model.solve(solver)

                record_var_SMPC(
                    lp_model=lp_model,
                    scenario_var_dict=scenario_var_dict,
                    scenario_var_name_lst=scenario_var_name_lst
                )
            
            # get next move 
            get_next_move_SMPC(
                var_dict=var_dict,
                scenario_var_dict=scenario_var_dict,
                cur_soc=var_dict['varSoc'][optim_step],
                dch_eff=dch_eff,
                ch_eff=ch_eff,
                dispatch_duration=dispatch_duration,
                bat_cap=bat_cap,
                bat_soc_min=bat_soc_min,
                bat_soc_max=bat_soc_max,
                curtail_solar=max(0,self.eval_solar_lst[optim_step]-self.pred_solar_lst[optim_step][0])
            )

            # avoid stack overflow 
            # if (optim_step + 1) % 1 == 0:
            os.system('rm -rf /dev/shm')   


            # log 
            print('Step {} finished'.format(optim_step))
            logger.info('Step {} finished'.format(optim_step))    


        # save info 
        var_dict['varSoc'].pop()
        var_df = pd.DataFrame(var_dict)   
        save_revenue_info_DMPC(
            var_df=var_df,
            pred_solar_lst=self.pred_solar_lst,
            eval_solar_lst=self.eval_solar_lst,
            eval_price_lst=self.eval_price_lst,
            dispatch_duration=dispatch_duration,
            solar_penalty_coeff=solar_penalty_coeff,
            res_save_path=self.res_save_path
        )   


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--market_year',default=2020,type=int,help='read data begin')
    parser.add_argument('--state_name',default='QLD',type=str,help='state name')
    parser.add_argument('--solar_farm_name',default='RUGBYR1',type=str,help='solar farm name')
    parser.add_argument('--gen_scenario_num',default=100,type=int,help='the number of scenarios to generate')
    parser.add_argument('--load_scenario_num',default=10,type=int,help='the number of scenarios to load')
    parser.add_argument('--load_flag',default=1,type=int,help='0:generate scenario; 1: load generated scenario')
    args = parser.parse_args() 

    market_year = args.market_year
    state_name = args.state_name
    solar_farm_name = args.solar_farm_name
    gen_scenario_num = args.gen_scenario_num
    load_scenario_num = args.load_scenario_num
    load_flag = args.load_flag 

    # init model 
    mpc_model = SMPC(
        state_name=state_name,
        market_year=market_year,
        solar_farm_name=solar_farm_name
    )

    # scenario generation 
    solar_std = 0.1
    price_std = 0.05
    mpc_horizon = int(60/5*24)
    solar_cap = 65 
    price_min = -1000
    price_max = 15000
    mpc_model.load_permuted_predictions(
        solar_std=solar_std,
        price_std=price_std,
        gen_scenario_num=gen_scenario_num,
        load_scenario_num=load_scenario_num,
        mpc_horizon=mpc_horizon,
        solar_cap=solar_cap,
        price_min=price_min,
        price_max=price_max,
        load_flag=load_flag
    )

    # mpc solving 
    log_save_folder = 'benchmark_log'
    os.makedirs(log_save_folder,exist_ok=True)
    log_file_name = 'SMPC.log'
    log_save_path = os.path.join(log_save_folder,log_file_name)
    logging.basicConfig(
        filename=log_save_path,
        filemode='w',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )
    logger = logging.getLogger('SMPC')


    init_soc = 0.5
    bat_soc_min = 0.05
    bat_soc_max = 0.95
    bat_rated_power = 10
    dispatch_duration = 5/60
    bat_cap = 10
    dch_eff = 0.95
    ch_eff = 0.95
    solar_penalty_coeff = 1.5
    mpc_model.solve_optim(
        mpc_horizon=mpc_horizon,
        init_soc=init_soc,
        bat_soc_min=bat_soc_min,
        bat_soc_max=bat_soc_max,
        bat_rated_power=bat_rated_power,
        dispatch_duration=dispatch_duration,
        bat_cap=bat_cap,
        dch_eff=dch_eff,
        ch_eff=ch_eff,
        solar_penalty_coeff=solar_penalty_coeff,
        logger=logger,
        scenario_num=load_scenario_num
    )

    end_time = time.time()
    logger.info('SMPC Running Time:{} secs'.format(np.round(end_time-start_time,1)))
