import numpy as np 

from torch.utils.data import Dataset

# create time series 
def create_seq(data,seq_len):
    seq_lst = []
    label_lst = []
    length = len(data)
    for i in range(length-seq_len):
        seq = data[i:i+seq_len]
        label = data[i+seq_len:i+seq_len+1]
        seq_lst.append(seq)
        label_lst.append(label)
    
    return np.array(seq_lst),np.array(label_lst)


def create_seq_no_label(data,seq_len):
    seq_lst = []
    length = len(data)
    for i in range(length-seq_len):
        seq = data[i:i+seq_len]
        seq_lst.append(seq)
    
    return np.array(seq_lst)

def record_var(
    lp_model,
    var_dict,
    actual_price,
    spot_duration,
    fast_duration,
    slow_duration,
    delayed_duration,
    mode
):  
    var_dict['actual_price'].append(actual_price)
    for var in lp_model.variables():
        var_prefix,var_idx = var.name.split('_')
        var_value = var.value()
        # only record the first step 
        if eval(var_idx)==0: 
            var_dict[var_prefix].append(var_value)
            if var_prefix == 'varSpotDch': var_spot_dch = var_value
            elif var_prefix == 'varSpotCh': var_spot_ch = var_value
            elif var_prefix == 'varBidFR': var_bid_FR = var_value
            elif var_prefix == 'varBidFL': var_bid_FL = var_value
            elif var_prefix == 'varBidSR': var_bid_SR = var_value 
            elif var_prefix == 'varBidSL': var_bid_SL = var_value 
            elif var_prefix == 'varBidDR': var_bid_DR = var_value
            elif var_prefix == 'varBidDL': var_bid_DL = var_value 
            elif var_prefix == 'varSoC': var_soc = var_value

    
    if mode == 'ES':
        cur_revenue = actual_price[0]*spot_duration*(var_spot_dch-var_spot_ch)
    elif mode == 'Contingency_FCAS':
        cur_revenue = actual_price[0]*fast_duration*var_bid_FR +\
                        actual_price[1]*fast_duration*var_bid_FL +\
                        actual_price[2]*slow_duration*var_bid_SR +\
                        actual_price[3]*slow_duration*var_bid_SL +\
                        actual_price[4]*delayed_duration*var_bid_DR +\
                        actual_price[5]*delayed_duration*var_bid_DL
    else:
        cur_revenue = actual_price[0]*spot_duration*(var_spot_dch-var_spot_ch) +\
                        actual_price[1]*fast_duration*var_bid_FR +\
                        actual_price[2]*fast_duration*var_bid_FL +\
                        actual_price[3]*slow_duration*var_bid_SR +\
                        actual_price[4]*slow_duration*var_bid_SL +\
                        actual_price[5]*delayed_duration*var_bid_DR +\
                        actual_price[6]*delayed_duration*var_bid_DL

    var_dict['cur_revenue'].append(cur_revenue)


    return var_soc




def record_var_perfect_info(
    lp_model,
    var_dict,
    actual_price_lst,
    spot_duration,
    fast_duration,
    slow_duration,
    delayed_duration,
    mode,
    data_len
):  
    var_dict['actual_price'] = actual_price_lst
    for var in lp_model.variables():
        var_prefix,var_idx = var.name.split('_')
        var_value = var.value()
        var_dict[var_prefix][eval(var_idx)] = var_value 

    cur_revenue_lst = []
    if mode == 'ES':
        for idx in range(data_len):
            cur_revenue = actual_price_lst[idx][0]*spot_duration*(var_dict['varSpotDch'][idx]-var_dict['varSpotCh'][idx])
            cur_revenue_lst.append(cur_revenue)
    elif mode == 'Contingency_FCAS':
        for idx in range(data_len):
            cur_revenue = actual_price_lst[idx][0]*fast_duration*var_dict['varBidFR'][idx] + actual_price_lst[idx][1]*fast_duration*var_dict['varBidFL'][idx] \
                            + actual_price_lst[idx][2]*slow_duration*var_dict['varBidSR'][idx] + actual_price_lst[idx][3]*slow_duration*var_dict['varBidSL'][idx] \
                            + actual_price_lst[idx][4]*delayed_duration*var_dict['varBidDR'][idx] + actual_price_lst[idx][5]*delayed_duration*var_dict['varBidDL'][idx]
            cur_revenue_lst.append(cur_revenue)
    else:
        for idx in range(data_len):
            cur_revenue = actual_price_lst[idx][0]*spot_duration*(var_dict['varSpotDch'][idx]-var_dict['varSpotCh'][idx]) \
                            + actual_price_lst[idx][1]*fast_duration*var_dict['varBidFR'][idx] + actual_price_lst[idx][2]*fast_duration*var_dict['varBidFL'][idx] \
                            + actual_price_lst[idx][3]*slow_duration*var_dict['varBidSR'][idx] + actual_price_lst[idx][4]*slow_duration*var_dict['varBidSL'][idx] \
                            + actual_price_lst[idx][5]*delayed_duration*var_dict['varBidDR'][idx] + actual_price_lst[idx][6]*delayed_duration*var_dict['varBidDL'][idx]
            cur_revenue_lst.append(cur_revenue)

    var_dict['cur_revenue'] = cur_revenue_lst
    var_dict['real_cum_revenue'] = np.cumsum(cur_revenue_lst)






class PriceForecastDataset(Dataset):
    def __init__(self,x,y) -> None:
        super().__init__()


        self.x = x 
        self.y = y  


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index,:,:], self.y[index,:,:]