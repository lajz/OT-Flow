# trainLargeOTflow.py
# train OT-Flow for the large density estimation data sets
import wandb
import argparse
import os
import time
import datetime
import json
import torch.optim as optim
import numpy as np
import math
import lib.toy_data as toy_data
import lib.utils as utils
from lib.utils import count_parameters

from src.mmd import mmd

from src.plotter import plot4
from src.OTFlowProblem import *
from src.Phi import *
import config
import datasets

wandb.login(key="0c79d5a0c295ca9d1bac8a08de98f9f0196e7b2e")
wandb.init(project="otflow", entity="market-maker")

cf = config.getconfig()

if cf.gpu: # default sizes
    def_viz_freq = 8000
    def_batch    = 2000 # ? batch size
    def_niter    = 8000
    def_m        = 256 # ? hidden dim size
    def_val_freq = 100
else: # if no gpu on platform, assume debugging on a local cpu
    def_viz_freq = 20
    def_val_freq = 20
    def_batch    = 200
    def_niter    = 2000
    def_m        = 16

parser = argparse.ArgumentParser('OT-Flow')
parser.add_argument(
    '--data', choices=['power', 'gas', 'hepmass', 'miniboone', 'bsds300','mnist', 'prosumer'], type=str, default='miniboone'
)
parser.add_argument('--data_split_path', default='prosumer/data.npy', type=str)
parser.add_argument('--use_num_days_data', default=365, type=int)
parser.add_argument('--prosumer_name', default='', type=str)

parser.add_argument("--nt"    , type=int, default=6, help="number of time steps")
parser.add_argument("--nt_val", type=int, default=10, help="number of time steps for validation")
parser.add_argument('--alph_C'  , type=float, default=100)
parser.add_argument('--alph_R'  , type=float, default=15)
parser.add_argument('--m'     , type=int, default=def_m)
parser.add_argument('--nTh'   , type=int, default=2)

parser.add_argument('--lr'       , type=float, default=0.01)
parser.add_argument("--drop_freq", type=int  , default=0, help="how often to decrease learning rate; 0 lets the mdoel choose")
parser.add_argument("--lr_drop"  , type=float, default=10.0, help="how much to decrease learning rate (divide by)")
parser.add_argument('--weight_decay', type=float, default=0.0)

parser.add_argument('--prec'      , type=str, default='single', choices=['single','double'], help="single or double precision")
parser.add_argument('--niters'    , type=int, default=def_niter) # num iterations
parser.add_argument('--batch_size', type=int, default=def_batch)
parser.add_argument('--test_batch_size', type=int, default=def_batch)

parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--early_stopping', type=int, default=20)

parser.add_argument('--save', type=str, default='experiments/cnf/large')
parser.add_argument('--viz_freq', type=int, default=def_viz_freq)
parser.add_argument('--val_freq', type=int, default=def_val_freq) # validation frequency needs to be less than viz_freq or equal to viz_freq
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

# args.alph = [float(item) for item in args.alph.split(',')]
args.alph = [1, args.alph_C, args.alph_R]
args.save = args.save + f"_{args.use_num_days_data}"

# add timestamp to save path
start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

wandb.config = {
    "learning_rate": args.lr,
    "nt": args.nt,
    "nt_val": args.nt_val,
    "alph": args.alph,
    "m": args.m
    }

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info("start time: " + start_time)
logger.info(args)

test_batch_size = args.test_batch_size if args.test_batch_size else args.batch_size

device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

if args.prec =='double':
    prec = torch.float64
else:
    prec = torch.float32



def batch_iter(X, batch_size=args.batch_size, shuffle=False):
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


# decrease the learning rate based on validation
ndecs = 0
n_vals_wo_improve=0
def update_lr(optimizer, n_vals_without_improvement):
    global ndecs
    if ndecs == 0 and n_vals_without_improvement > args.early_stopping:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop
        ndecs = 1
    elif ndecs == 1 and n_vals_without_improvement > args.early_stopping:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop**2
        ndecs = 2
    else:
        ndecs += 1
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop**ndecs


def load_data(name):

    if name == 'bsds300':
        return datasets.BSDS300()

    elif name == 'power':
        return datasets.POWER()

    elif name == 'gas':
        return datasets.GAS()

    elif name == 'hepmass':
        return datasets.HEPMASS()

    elif name == 'miniboone':
        return datasets.MINIBOONE()
    
    elif name == 'prosumer':
        # get the file format as well
        return datasets.PROSUMER(args.data_split_path, args.use_num_days_data)

    else:
        raise ValueError('Unknown dataset')


def compute_loss(net, x, nt):
    Jc , cs = OTFlowProblem(x, net, [0,1], nt=nt, stepper="rk4", alph=net.alph)
    return Jc, cs



if __name__ == '__main__':

    cvt = lambda x: x.type(prec).to(device, non_blocking=True)

    data = load_data(args.data)
    data.trn.x = torch.from_numpy(data.trn.x)
    print(data.trn.x.shape)
    data.val.x = torch.from_numpy(data.val.x)

    # hyperparameters of model
    d   = data.trn.x.shape[1]
    nt  = args.nt
    nt_val = args.nt_val
    nTh = args.nTh
    m   = args.m

    # set up neural network to model potential function Phi
    net = Phi(nTh=nTh, m=m, d=d, alph=args.alph)
    net = net.to(prec).to(device)


    # resume training on a model that's already had some training
    if args.resume is not None: # todo: model reloading
        # reload model
        checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        m       = checkpt['args'].m
        alph    = args.alph # overwrite saved alpha
        nTh     =  checkpt['args'].nTh
        args.hutch = checkpt['args'].hutch
        net     = Phi(nTh=nTh, m=m, d=d, alph=alph)  # the phi aka the value function
        prec = checkpt['state_dict']['A'].dtype
        net     = net.to(prec)
        net.load_state_dict(checkpt["state_dict"])
        net     = net.to(device)

    if args.val_freq == 0:
        # if val_freq set to 0, then validate after every epoch
        args.val_freq = math.ceil(data.trn.x.shape[0]/args.batch_size)

    # ADAM optimizer
    optim = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    logger.info(net)
    logger.info("-------------------------")
    logger.info("DIMENSION={:}  m={:}  nTh={:}   alpha={:}".format(d,m,nTh,net.alph))
    logger.info("nt={:}   nt_val={:}".format(nt,nt_val))
    logger.info("Number of trainable parameters: {}".format(count_parameters(net)))
    logger.info("-------------------------")
    logger.info(str(optim)) # optimizer info
    logger.info("data={:} batch_size={:} gpu={:}".format(args.data, args.batch_size, args.gpu))
    logger.info("maxIters={:} val_freq={:} viz_freq={:}".format(args.niters, args.val_freq, args.viz_freq))
    logger.info("saveLocation = {:}".format(args.save))
    logger.info("-------------------------\n")

    begin = time.time()
    end = begin
    best_loss = float('inf')
    best_cs = [0.0]*3
    bestParams = None

    log_msg = (
        '{:5s}  {:6s}  {:7s}   {:9s}  {:9s}  {:9s}  {:9s}     {:9s}  {:9s}  {:9s}  {:9s} '.format(
            'iter', ' time','lr','loss', 'L (L2)', 'C (loss)', 'R (HJB)', 'valLoss', 'valL', 'valC', 'valR',
        )
    )
    logger.info(log_msg)

    timeMeter = utils.AverageMeter()

    # box constraints / acceptable range for parameter values
    clampMax = 1.5
    clampMin = -1.5

    net.train()
    itr = 1
    while itr < args.niters:
        # train
        wandb.log({"iteration": itr})
        for x0 in batch_iter(data.trn.x, shuffle=True):   
            x0 = cvt(x0)
            optim.zero_grad()

            # clip parameters
            for p in net.parameters():
                p.data = torch.clamp(p.data, clampMin, clampMax)

            currParams = net.state_dict()
            loss,cs  = compute_loss(net, x0, nt=nt)
            loss.backward()
          
            optim.step()
            timeMeter.update(time.time() - end)
            
            wandb.log({"loss": loss})
            wandb.log({"L": cs[0], "C": cs[1], "R": cs[2]})
            wandb.log({"absL": abs(cs[0])}) # want this to stabilize = flat gradient
            log_message = (
                '{:05d}  {:6.3f}  {:7.1e}   {:9.3e}  {:9.3e}  {:9.3e}  {:9.3e} '.format(
                    itr, timeMeter.val, optim.param_groups[0]['lr'], loss, cs[0], cs[1], cs[2]
                )
            )

            if torch.isnan(loss): # catch NaNs when hyperparameters are poorly chosen
                logger.info(log_message)
                logger.info("NaN encountered....exiting prematurely")
                logger.info("Training Time: {:} seconds".format(timeMeter.sum))
                logger.info('File: ' + start_time + '_{:}_{:}_m{:}_checkpt.pth'.format(args.data, args.prosumer_name, m))
                exit(1)

            # validation
            if itr % args.val_freq == 0 or itr == args.niters:
                net.eval()
                with torch.no_grad():

                    valLossMeter = utils.AverageMeter()
                    valAlphMeterL = utils.AverageMeter()
                    valAlphMeterC = utils.AverageMeter()
                    valAlphMeterR = utils.AverageMeter()

                    for x0 in batch_iter(data.val.x, batch_size=test_batch_size):
                        x0 = cvt(x0)
                        nex = x0.shape[0]
                        val_loss, val_cs = compute_loss(net, x0, nt=nt_val)
                        valLossMeter.update(val_loss.item(), nex)
                        valAlphMeterL.update(val_cs[0].item(), nex)
                        valAlphMeterC.update(val_cs[1].item(), nex)
                        valAlphMeterR.update(val_cs[2].item(), nex)
                        
                        wandb.log({"val_loss": val_loss})
                        wandb.log({"val_L": val_cs[0], "val_C": val_cs[1], "val_R": cs[2]})
                        wandb.log({"absval_L": abs(val_cs[0])})


                    # add to print message
                    log_message += '    {:9.3e}  {:9.3e}  {:9.3e}  {:9.3e} '.format(
                        valLossMeter.avg, valAlphMeterL.avg, valAlphMeterC.avg, valAlphMeterR.avg
                    )

                    # save best set of parameters
                    if valLossMeter.avg < best_loss:
                        n_vals_wo_improve = 0
                        best_loss = valLossMeter.avg
                        best_cs = [  valAlphMeterL.avg, valAlphMeterC.avg, valAlphMeterR.avg ]
                        utils.makedirs(args.save)
                        bestParams = net.state_dict()
                        save_path = os.path.join(args.save, start_time + '_{:}_{:}_m{:}_checkpt.pth'.format(args.data, args.prosumer_name, m))
                        torch.save({
                            'args': args,
                            'state_dict': bestParams,
                        }, save_path)
                        prosumer_checkpoints_dict_path = os.path.join(args.save, "prosumer_checkpoints_dict.json")
                        if os.path.isfile(prosumer_checkpoints_dict_path):
                            with open(prosumer_checkpoints_dict_path, "r") as jsonFile:
                                prosumer_data = json.load(jsonFile)
                        else:
                            prosumer_data = {}
                        
                        prosumer_data.setdefault(args.prosumer_name, {})["latest"] = save_path
                        
                        with open(prosumer_checkpoints_dict_path, "w") as jsonFile:
                            prosumer_data = json.dump(prosumer_data, jsonFile)
                    else:
                        n_vals_wo_improve+=1

                    net.train()
                    log_message += ' no improve: {:d}/{:d}'.format(n_vals_wo_improve, args.early_stopping)
            logger.info(log_message) # print iteration

            # create plots for assessment mid-training
            if itr % args.viz_freq == 0:
                with torch.no_grad():
                    net.eval()
                    currState = net.state_dict()
                    if bestParams: # was failing b/c no bestParams
                        logger.info("bestParams found")
                        net.load_state_dict(bestParams)

                    # plot one batch 
                    p_samples = cvt(data.val.x[0:test_batch_size,:])
                    nSamples = p_samples.shape[0]
                    y = cvt(torch.randn(nSamples,d)) # sampling from rho_1 / standard normal

                    sPath = os.path.join(args.save, 'figs', start_time + '_{:04d}.png'.format(itr))
                    plot4(net, p_samples, y, nt_val, sPath, sTitle='loss {:.2f}  ,  C {:.2f}'.format(best_loss, best_cs[1] ), doPaths=False)

                    net.load_state_dict(currState)
                    net.train()

            if args.drop_freq == 0: # if set to the code setting 0 , the lr drops based on validation
                if n_vals_wo_improve > args.early_stopping:
                    if ndecs>2:
                        logger.info("early stopping engaged")
                        logger.info("Training Time: {:} seconds".format(timeMeter.sum))
                        logger.info('File: ' + start_time + '_{:}_{:}_m{:}_checkpt.pth'.format(args.data, args.prosumer_name, m)
                          )
                        exit(0)
                    else:
                        update_lr(optim, n_vals_wo_improve)
                        n_vals_wo_improve = 0
            else:
                # shrink step size
                if itr % args.drop_freq == 0:
                    for p in optim.param_groups:
                        p['lr'] /= args.lr_drop
                    print("lr: ", p['lr'])

            itr += 1
            end = time.time()
            # end batch_iter
            
    logger.info("Training Time: {:} seconds".format(timeMeter.sum))
    logger.info('Training has finished.  ' + start_time + '_{:}_{:}_m{:}_checkpt.pth'.format(args.data, args.prosumer_name, m))
    
    print("evaluating model")
    net.eval()
    data = load_data(args.data)
    testData = torch.from_numpy(data.tst.x) # x sampled from unknown rho_0
    nSamples = testData.shape[0] # 100000 -- modify number of samples to generate
    normSamples = torch.randn(nSamples, testData.shape[1]) # y sampled from rho_1 (nSamples, features)
    
    logger.info("test data shape: {:}".format(testData.shape))

    nex = testData.shape[0]
    d   = testData.shape[1]
    nt_test = args.nt

    logger.info(net)
    logger.info("----------TESTING---------------")
    logger.info("DIMENSION={:}  m={:}  nTh={:}   alpha={:}".format(d,m,nTh,net.alph))
    logger.info("nt_test={:}".format(nt_test))
    logger.info("Number of trainable parameters: {}".format(count_parameters(net)))
    logger.info("Number of testing examples: {}".format(nex))
    logger.info("-------------------------")
    logger.info("data={:} batch_size={:} gpu={:}".format(args.data, args.batch_size, args.gpu))
    logger.info("saveLocation = {:}".format(args.save))
    logger.info("-------------------------\n")
    
    log_msg = (
        '{:4s}        {:9s}  {:9s}  {:11s}  {:9s}'.format(
            'itr', 'loss', 'L (L_2)', 'C (loss)', 'R (HJB)'
        )
    )
    logger.info(log_msg)
    
    with torch.no_grad():

        # meters to hold testing results
        testLossMeter  = utils.AverageMeter()
        testAlphMeterL = utils.AverageMeter()
        testAlphMeterC = utils.AverageMeter()
        testAlphMeterR = utils.AverageMeter()


        itr = 1
        for x0 in batch_iter(testData, batch_size=args.batch_size):

            x0 = cvt(x0)
            nex = x0.shape[0]
            test_loss, test_cs = compute_loss(net, x0, nt=nt_test)
            testLossMeter.update(test_loss.item(), nex)
            testAlphMeterL.update(test_cs[0].item(), nex)
            testAlphMeterC.update(test_cs[1].item(), nex)
            testAlphMeterR.update(test_cs[2].item(), nex)
            log_message = 'batch {:4d}: {:9.3e}  {:9.3e}  {:11.5e}  {:9.3e}'.format(
                itr, test_loss, test_cs[0], test_cs[1], test_cs[2]
            )
            logger.info(log_message)  # print batch
            itr+=1

        # add to print message
        log_message = '[TEST]      {:9.3e}  {:9.3e}  {:11.5e}  {:9.3e} '.format(
            testLossMeter.avg, testAlphMeterL.avg, testAlphMeterC.avg, testAlphMeterR.avg
        )

        logger.info(log_message) # print total
        logger.info("Testing Time:          {:.2f} seconds with {:} parameters".format( time.time() - end, count_parameters(net) ))

        # computing inverse flow (uniform -> data)
        logger.info("computing inverse...")
        nGen = normSamples.shape[0]

        modelFx     = np.zeros(testData.shape)
        modelFinvfx = np.zeros(testData.shape)
        modelGen    = np.zeros(normSamples.shape)

        idx = 0
        for i , x0 in enumerate(batch_iter(testData, batch_size=args.batch_size)):
            x0     = cvt(x0)
            fx     = integrate(x0[:, 0:d], net, [0.0, 1.0], nt_test, stepper="rk4", alph=net.alph)
            finvfx = integrate(fx[:, 0:d], net, [1.0, 0.0], nt_test, stepper="rk4", alph=net.alph)

            # consolidate fx and finvfx into one spot
            batchSz = x0.shape[0]
            modelFx[ idx:idx+batchSz , 0:d ]     = fx[:,0:d].detach().cpu().numpy()
            modelFinvfx[ idx:idx+batchSz , 0:d ] = finvfx[:,0:d].detach().cpu().numpy()
            idx = idx + batchSz

        # logger.info("model inv error:  {:.3e}".format(np.linalg.norm(testData.numpy() - modelFinvfx) / nex)) # initial bug
        logger.info("model inv error:  {:.3e}".format( np.mean(np.linalg.norm(testData.numpy() - modelFinvfx, ord=2, axis=1))))

        # this portion can take a long time
        # generate samples
        logger.info("generating samples...")
        idx = 0
        for i, y in enumerate(batch_iter(normSamples, batch_size=args.batch_size)):
            y = cvt(y) # put on device with proper precision
            
            # finvy is our generated sample from uniform y
            finvy = integrate(y[:, 0:d], net, [1.0, 0.0], nt_test, stepper="rk4",alph=net.alph)
            batchSz = y.shape[0]
            assert y.shape[1] == d, f"normSample has {y.shape[1]}-dim gaussian instead of d={d}" # no reason for it not to equal the same
            # assert finvy.shape[1] == d, f"synthetic has {y.shape[1]}-dim instead of d={d}" # ! not sure why this fails
            
            synthetic = finvy[:,0:d].detach().cpu().numpy() # synthetic data
            
            modelGen[ idx:idx+batchSz , 0:d ] = synthetic
            idx = idx + batchSz

        testData = testData.detach().cpu().numpy()  # make to numpy
        normSamples = normSamples.detach().cpu().numpy()

        # when running abbreviated style, use smaller sample sizes to compute mmd so its quicker
        nSamples = testData.shape[0]  # number of samples for the MMD
        testSamps = testData[0:nSamples, :]
        modelSamps = modelGen[0:nSamples, 0:d]
        
        # Only run MMD on the tuple we are generating!
        # mmd_evaluation = mmd(modelSamps[:, :3]  , testSamps[:, :3] )
        # wandb.log({"mmd_abbreviated": mmd_evaluation})
        mmd_full = mmd(modelSamps, testSamps)
        wandb.log({"mmd_full": mmd_full})
        print("MMD( ourGen   , rho_0 ),  num(ourGen)={:d}    , num(rho_0)={:d} : {:.5e}".format( modelSamps.shape[0]  , testSamps.shape[0] , mmd_full))
