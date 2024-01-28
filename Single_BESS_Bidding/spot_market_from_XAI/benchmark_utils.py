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


def record_var(
    lp_model,
    var_dict,
    actual_price,
    pred_price,
    spot_duration
):  
    var_dict['actual_price'].append(actual_price)
    var_dict['pred_price'].append(pred_price)
    for var in lp_model.variables():
        var_prefix,var_idx = var.name.split('_')
        var_value = var.value()
        # only record the first step 
        if eval(var_idx)==0: 
            var_dict[var_prefix].append(var_value)
            if var_prefix == 'varBidDch': var_spot_dch = var_value
            elif var_prefix == 'varBidCh': var_spot_ch = var_value
            elif var_prefix == 'varSoC': var_soc = var_value
    
    cur_revenue = actual_price[0]*spot_duration*(var_spot_dch-var_spot_ch)

    var_dict['cur_revenue'].append(cur_revenue)


    return var_soc




class PriceForecastDataset(Dataset):
    def __init__(self,x,y) -> None:
        super().__init__()


        self.x = x 
        self.y = y  


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index,:,:], self.y[index,:,:]