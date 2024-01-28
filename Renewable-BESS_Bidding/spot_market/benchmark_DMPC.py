import pandas as pd 
import argparse
import os 
import pulp
import numpy as np 
import time
import logging 


from benchmark_utils import record_var_DMPC
from benchmark_utils import save_revenue_info_DMPC


class DMPC():
    def __init__(
        self,
        state_name,
        market_year,
        solar_farm_name,
    ) -> None:
        self.load_data(
            state_name=state_name,
            market_year=market_year,
            solar_farm_name=solar_farm_name,
        )
        self.load_predictions()

        res_save_folder = 'benchmark_res'
        res_file_name = 'benchmark_DMPC_res.csv'
        self.res_save_path = os.path.join(res_save_folder,res_file_name)

    
    # load data for evaluation 
    def load_data(
        self,
        state_name,
        market_year,
        solar_farm_name,
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
        eval_price_df = price_df.iloc[-one_month_len:]
        eval_solar_df = solar_df.iloc[-one_month_len:]

        eval_price_lst = eval_price_df['RRP'].to_list()
        eval_solar_lst = eval_solar_df['AVAILABILITY'].to_list()

        self.eval_price_lst = eval_price_lst
        self.eval_solar_lst = eval_solar_lst

        self.eval_data_len = len(self.eval_price_lst)



    # load predictions for solar, price, curtail with the length of mpc_horizon
    def load_predictions(self,load_folder='benchmark_res',file_name='solar_price_curtail_pred4mpc.csv'):
        pred_df = pd.read_csv(os.path.join(load_folder,file_name),index_col=[0])
        
        pred_solar = pred_df['solar'].to_list()
        pred_curtail = pred_df['curtail'].to_list()
        pred_price = pred_df['price'].to_list()

        pred_solar_lst = list(map(eval,pred_solar))
        pred_solar_lst.pop()
        pred_curtail_lst = list(map(eval,pred_curtail))
        pred_curtail_lst.pop()
        pred_price_lst = list(map(eval,pred_price))
        pred_price_lst.pop()

        self.pred_solar_lst = pred_solar_lst
        self.pred_curtail_lst = pred_curtail_lst
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
        ch_eff
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


        cur_pred_price_lst = self.pred_price_lst[optim_step]
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
        logger
    ):
        var_name_lst = ['varBatBinaryDch','varBatBinaryCh','varSoc','varBatBidDch','varBatBidCh','varBatReserve']
        var_name_lst.append('curtailResponseBid')
        var_dict = {var_name:[] for var_name in var_name_lst}
        var_dict['varSoc'].append(init_soc)


        # start solving 
        for optim_step in range(self.eval_data_len-mpc_horizon):
        # for optim_step in range(5): # for debug
            # solve once 
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
                ch_eff=ch_eff
            )
            # solver = pulp.GUROBI_CMD(msg=True,gapRel=0.005,timeLimit=60,logPath='DMPC_gurobi.log')
            solver = pulp.GUROBI(msg=False,gapRel=0.05,timeLimit=60,logPath='DMPC_gurobi.log')
            solve_status = lp_model.solve(solver)

            # get var val
            record_var_DMPC(
                lp_model=lp_model,
                var_dict=var_dict,
                optim_step=optim_step,
                eval_data_len=self.eval_data_len-mpc_horizon,
                # eval_data_len=5, # for debug 
                eval_solar=self.eval_solar_lst[optim_step],
                pred_solar=self.pred_solar_lst[optim_step][0],
                dispatch_duration=dispatch_duration,
                ch_eff=ch_eff,
                bat_cap=bat_cap,
                bat_soc_min=bat_soc_min,
                bat_soc_max=bat_soc_max
            )
                
            # avoid stack overflow 
            if (optim_step + 1) % 100 == 0:
                os.system('rm -rf /dev/shm')


            # log 
            print('Step {} finished'.format(optim_step))
            logger.info('Step {} finished'.format(optim_step))


        # save info 
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
    args = parser.parse_args() 

    market_year = args.market_year
    state_name = args.state_name
    solar_farm_name = args.solar_farm_name

    mpc_model = DMPC(
        state_name=state_name,
        market_year=market_year,
        solar_farm_name=solar_farm_name
    )

    log_save_folder = 'benchmark_log'
    os.makedirs(log_save_folder,exist_ok=True)
    log_file_name = 'DMPC.log'
    log_save_path = os.path.join(log_save_folder,log_file_name)
    logging.basicConfig(
        filename=log_save_path,
        filemode='w',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )
    logger = logging.getLogger('DMPC')

    mpc_horizon = int(60/5*24)
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
        logger=logger
    )



    end_time = time.time()
    logger.info('DMPC Running Time:{} secs'.format(np.round(end_time-start_time,1)))














