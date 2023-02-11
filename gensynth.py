import os
from pathlib import Path
import argparse
import numpy as np
from src.OTFlowProblem import *
from src.Phi import *
import datasets

import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--experiment_folder_path', default="experiments/cnf/random_prices_o25_s25", type=str)
parser.add_argument('--raw_data_folder_name', default="random_prices_o25_s25", type=str)
parser.add_argument('--use_num_days_data', default=365, type=int)
parser.add_argument('--generate_num_days_data', default=5000, type=int)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device {device}")

def batch_iter(X, batch_size=64, shuffle=False):
    """
    X: feature tensor (shape: num_instances x num_features)
    """
    if shuffle:
        idxs = torch.randperm(X.shape[0])
    else:
        idxs = torch.arange(X.shape[0])
    if X.is_cuda:
        idxs = idxs.cuda()
    for batch_idxs in idxs.split(batch_size):
        yield X[batch_idxs]

df_cols = [f"agent_buy_{i}" for i in range(24)] + [f"agent_sell_{i}" for i in range(24)] + [f"prosumer_response_{i}" for i in range(24)] + ["day"]
df_cols_dict = {i: df_cols[i] for i in range(73)}

# models = ["2022_05_10_12_09_30_prosumer_alph100_15_m256_checkpt.pth",
# "2022_05_10_12_09_33_prosumer_alph100_15_m256_checkpt.pth",
# "2022_05_10_12_23_16_prosumer_alph100_15_m256_checkpt.pth",
# "2022_05_10_12_23_17_prosumer_alph100_15_m256_checkpt.pth",
# "2022_05_10_12_36_39_prosumer_alph100_15_m256_checkpt.pth",
# "2022_05_10_12_36_42_prosumer_alph100_15_m256_checkpt.pth",
# "2022_05_10_12_49_50_prosumer_alph100_15_m256_checkpt.pth",
# "2022_05_10_12_49_53_prosumer_alph100_15_m256_checkpt.pth",
# "2022_05_10_13_03_10_prosumer_alph100_15_m256_checkpt.pth",
# "2022_05_10_13_03_14_prosumer_alph100_15_m256_checkpt.pth"]

experiment_folder_path = args.experiment_folder_path + f"_{args.use_num_days_data}"

experiment_files = [f for f in os.listdir(experiment_folder_path) if os.path.isfile(os.path.join(experiment_folder_path, f))]
prosumer_names = [
    "Hog_services_Kerrie",
	"Hog_assembly_Colette",
	"Hog_assembly_Dona",
	"Hog_assembly_Jasmine",
	"Hog_education_Casandra",
	"Hog_education_Donnie",
	"Hog_education_Jewel",
	"Hog_education_Jordan",
	"Hog_education_Madge",
	"Hog_education_Rachael",
	"Hog_food_Morgan",
	"Hog_industrial_Jeremy",
	"Hog_industrial_Joanne",
	"Hog_industrial_Mariah",
	"Hog_industrial_Quentin",
	"Hog_lodging_Francisco",
	"Hog_lodging_Hal",
	"Hog_lodging_Nikki",
	"Hog_lodging_Ora",
	"Hog_lodging_Shanti",
	"Hog_office_Almeda",
	"Hog_office_Bessie",
	"Hog_office_Betsy",
	"Hog_office_Bill",
	"Hog_office_Denita",
	"Hog_office_Gustavo",
	"Hog_office_Lavon",
	"Hog_office_Lizzie",
	"Hog_office_Mary",
	"Hog_office_Merilyn",
	"Hog_office_Mike",
	"Hog_office_Miriam",
	"Hog_office_Myles",
	"Hog_office_Napoleon",
	"Hog_office_Shawna",
	"Hog_office_Shawnna",
	"Hog_office_Sherrie",
	"Hog_office_Shon",
	"Hog_office_Sydney",
	"Hog_office_Valda",
	"Hog_other_Noma",
	"Hog_other_Tobias",
	"Hog_parking_Jean",
	"Hog_parking_Shannon",
	"Hog_public_Crystal",
	"Hog_public_Gerard",
	"Hog_public_Kevin",
	"Hog_public_Octavia",
	"Hog_services_Adrianna", 
]
models = {}
for prosumer_name in prosumer_names:
    matching = [s for s in experiment_files if prosumer_name in s]
    if len(matching) > 0:
        matching.sort(reverse=True)
        models[prosumer_name] = os.path.join(experiment_folder_path, matching[0])

def compute_loss(net, x, nt):
    Jc, cs = OTFlowProblem(x, net, [0, 1], nt=nt, stepper="rk4", alph=net.alph)
    return Jc, cs

def unnorm(arr, split): # util function to unnormalize data
    return data.scalers[split].inverse_transform(arr)
    
def run_prosumer(prosumer_name):
    print(f"creating data for {prosumer_name} with model from {models[prosumer_name]}")
    model_path = models[prosumer_name]
    data = datasets.PROSUMER(f"{args.raw_data_folder_name}/{prosumer_name}.csv", args.use_num_days_data) # real data
    print(f"using real data from pi_b_logs/{prosumer_name}.csv")

    trainData = torch.from_numpy(data.trn.x)  # x sampled from unknown rho_0
    nSamples = args.generate_num_days_data # ! modify number of samples to generate
    normSamples = torch.randn(nSamples, trainData.shape[1])  # y ~ rho_1 : (nSamples, features)

    d = trainData.shape[1] # pure stochasticity is 73
    nt_test = 24  # args.nt

    checkpt = torch.load(model_path, map_location=lambda storage, loc: storage)
    print(checkpt["args"])
    m = checkpt["args"].m
    alph = checkpt["args"].alph
    nTh = checkpt["args"].nTh
    net = Phi(nTh=nTh, m=m, d=d, alph=alph)
    argPrec = checkpt["state_dict"]["A"].dtype
    net = net.to(argPrec)
    net.load_state_dict(checkpt["state_dict"])
    net = net.to(device)
    cvt = lambda x: x.type(argPrec).to(device, non_blocking=True)

    net.eval()
    with torch.no_grad():
        nGen = normSamples.shape[0]
        modelFx = np.zeros(trainData.shape)  # each row is a different sample from the testData (8022, 72)
        modelFinvfx = np.zeros(trainData.shape) # (8022, 72)
        modelGen = np.zeros((nSamples, d)) # (10, 72)

        print("skipping modelFx")
        # idx = 0
        # for i, x0 in enumerate(batch_iter(trainData, batch_size=10000)):  # model Fx (approx normal), Finv(Fx)
        #     x0 = cvt(x0)  # (8022, 72) normalized data input dimension 25 (1 + [7+1]*[3])
        #     fx = integrate(x0[:, 0:d], net, [0.0, 1.0], nt_test, stepper="rk4", alph=net.alph)  # (8022, 75) -- added 3 dims is for [L, C, R] losses
        #     finvfx = integrate(fx[:, 0:d], net, [1.0, 0.0], nt_test, stepper="rk4", alph=net.alph)  # (8022, 75)

        #     # consolidate fx and finvfx into one spot
        #     batchSz = x0.shape[0]
        #     modelFx[idx : idx + batchSz, 0:d] = fx[:, 0:d].detach().cpu().numpy()
        #     modelFinvfx[idx : idx + batchSz, 0:d] = finvfx[:, 0:d].detach().cpu().numpy()
        #     idx = idx + batchSz

        # generate samples
        print("generating samples")
        idx = 0
        iterations = 0
        while idx < nSamples:
        
            for i, y in tqdm(enumerate(batch_iter(normSamples, batch_size=64))):
                y = cvt(y)  # (nGen, 73) put on device with proper precision
                iterations += 1
                if iterations > nSamples * 10:
                    return False
                # finvy is our generated sample from gaussian y
                finvy = integrate(y[:, 0:d], net, [1.0, 0.0], nt_test, stepper="rk4", alph=net.alph) # (nGen 76) -- includes [L, C, R]
                batchSz = y.shape[0]
                assert (y.shape[1] == d), f"normSample has {y.shape[1]}-dim gaussian instead of d={d}"  # no reason for it not to equal the same

                synthetic = finvy[:, 0:d].detach().cpu().numpy()  # synthetic data

                valid_gens = []
                for i in range(synthetic.shape[0]):
                    if idx + len(valid_gens) >= nSamples:
                        break
                    synth_gen = synthetic[i, 0:d]
                    if np.min(synth_gen) > -1.5 and  np.max(synth_gen) < 1.5:
                        valid_gens.append(synth_gen)
                if len(valid_gens) > 0:
                    modelGen[idx : idx + len(valid_gens), 0:d] = np.stack(valid_gens, axis=0)  # populate synthetic in one place
                    idx = idx + len(valid_gens)

    
    # data_path is pi_b, model spits out 72-dim

    # todo: comment out modelFx since this is just forward pass of flow
    # modelGen this is generatd samples (normalized) --> need to unnormalize before saving them

    unnormalized_d = {}
    for i in range(73):
        col_scaler = None
        col_name = df_cols_dict[i] # {col_i: col_name}
        if "agent_buy" in col_name:
            col_scaler = "agent_buy"
        elif "agent_sell" in col_name:
            col_scaler = "agent_sell"
        elif "prosumer_response" in col_name:
            col_scaler = "pro"
            
        elif "day" in col_name:
            col_scaler = "day"
        
        print(f"Min: {np.min(modelGen[:, i])} | Max {np.max(modelGen[:, i])}")
        # check these scales make sense (some of the actions are out of scale)
        new_col = unnorm(modelGen[:, i].reshape(-1, 1), col_scaler).reshape(-1)
        
        if "day" in col_name:
            full_digit_day = np.round(new_col, decimals=0)
            # if more than 20,000 rows are out of range try normalizing to [0,365]
            print(f"{(full_digit_day < 0).sum()} days < 0, and {(full_digit_day > 364).sum()} > 364")
            # full_digit_day[full_digit_day < 0] = 0
            # full_digit_day[full_digit_day > 364] = 364
            new_col = full_digit_day
        
        unnormalized_d[col_name] = new_col
    
    print(f"Completed {prosumer_name}")
    new_df = pd.DataFrame(unnormalized_d)
    out_dir = Path("synth_data").joinpath(args.raw_data_folder_name).joinpath(f"days_{args.use_num_days_data}")
    out_dir.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(out_dir.joinpath(f"{prosumer_name}.csv"), index=False)
    return True

prosumers_completed = []
prosumers_not_completed = []
for prosumer_name in models.keys():
    if (run_prosumer(prosumer_name)):
        prosumers_completed.append(prosumer_name)
    else:
        prosumers_not_completed.append(prosumer_name)
    
# todo visualize prices somehow
print("Prosumers Completed")
print(prosumers_completed)
print("Prosumers Not Completed")
print(prosumers_not_completed)
