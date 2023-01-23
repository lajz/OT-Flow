import pandas as pd
import numpy as np
import datasets
from sklearn.preprocessing import MinMaxScaler

class PROSUMER:
    
    class Data:
        
        def __init__(self, data):
             
            self.x = data.astype(np.float32)
            self.N = self.x.shape[0] # num examples
            
    def __init__(self, path, use_num_days_data):
        
        # file = datasets.root + 'prosumer/data_flat.npy'
        file = datasets.root + path # 'prosumer/data.npy'
        
        d = load_data_normalised(file, use_num_days_data)
        trn = d["data"]["trn"]
        val = d["data"]["val"]
        tst = d["data"]["tst"]
        
        self.scalers = d["scalers"]
        
        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)
        
        self.n_dims = self.trn.x.shape[1]
        
def load_data_normalised(root_path, use_num_days_data):
    print(f"loading data from {root_path}")
    raw_df = pd.read_csv(root_path)
    df = raw_df.head(use_num_days_data)
    cols = df.columns.to_list()
    
    # Extract and normalize the buy, sell, and prosumer cols
    agent_buy_cols = [x for x in cols if "agent_buy" in x]
    agent_sell_cols = [x for x in cols if "agent_sell" in x]
    pro_cols = [x for x in cols if "prosumer_response" in x] 
    
    agent_buy_responses = df[agent_buy_cols].values.reshape(-1, 1)
    agent_sell_responses = df[agent_sell_cols].values.reshape(-1, 1)
    pro_responses = df[pro_cols].values.reshape(-1, 1)
    days = df['day'].values.reshape(-1, 1)
    
    agent_buy_scaler = MinMaxScaler()
    agent_sell_scaler = MinMaxScaler()
    pro_scaler = MinMaxScaler()
    day_scaler = MinMaxScaler()
    
    agent_buy_scaler.fit(agent_buy_responses)
    agent_sell_scaler.fit(agent_sell_responses)
    pro_scaler.fit(pro_responses)
    day_scaler.fit(days)
    
    # reconstruct normalized df using the same column order
    normalized_d = {}
    for col in df.columns:
        if "agent_buy" in col:
            normalized_d[col] = agent_buy_scaler.transform(df[col].values.reshape(-1, 1)).reshape(-1)
        elif "agent_sell" in col:
            normalized_d[col] = agent_sell_scaler.transform(df[col].values.reshape(-1, 1)).reshape(-1)
        elif "prosumer_response" in col:
            normalized_d[col] = pro_scaler.transform(df[col].values.reshape(-1, 1)).reshape(-1)
        elif "day" in col:
            normalized_d[col] = day_scaler.transform(df[col].values.reshape(-1, 1)).reshape(-1)
    
    new_df = pd.DataFrame(normalized_d)
    print(new_df.columns)

    # construct data splits
    data = new_df.values
    N_test = int(0.1 * data.shape[0])
    data_test = data[-N_test:]
    data = data[0:-N_test]
    N_validate = int(0.1 * data.shape[0])
    data_validate = data[-N_validate:]
    data_train = data[0:-N_validate]
    
    return {"data": 
                {"trn": data_train, 
                 "val": data_validate, 
                 "tst": data_test}, 
            "scalers": 
                {"agent_buy": agent_buy_scaler,
                 "agent_sell": agent_sell_scaler, 
                 "pro": pro_scaler,
                 "day": day_scaler,
                 }
           }