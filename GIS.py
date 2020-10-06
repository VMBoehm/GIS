from SIT import *
from load_data import * 
import argparse

parser = argparse.ArgumentParser()

# data
parser.add_argument('--dataset', type=str, default='power',
                    choices=['power', 'gas', 'hepmass', 'miniboone', 'bsds300', 'mnist', 'fmnist', 'cifar10'],
                    help='Name of dataset to use.')

parser.add_argument('--train_size', type=int, default=-1,
                    help='Size of training data. Negative or zero means all the training data.') 

parser.add_argument('--validate_size', type=int, default=-1,
                    help='Size of validation data. Negative or zero means all the validation data.') 

parser.add_argument('--seed', type=int, default=738,
                    help='Random seed for PyTorch and NumPy.')

parser.add_argument('--root', type=str, default ='./', help='root directory for data')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

assert torch.cuda.is_available()
device = torch.device('cuda')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

if args.dataset == 'power':
    data_train, data_validate, data_test = load_data_power()
elif args.dataset == 'gas':
    data_train, data_validate, data_test = load_data_gas()
elif args.dataset == 'hepmass':
    data_train, data_validate, data_test = load_data_hepmass()
elif args.dataset == 'miniboone':
    data_train, data_validate, data_test = load_data_miniboone()
elif args.dataset == 'bsds300':
    data_train, data_validate, data_test = load_data_bsds300()
elif args.dataset == 'mnist':
    data_train, data_test = load_data_mnist(args.root)
elif args.dataset == 'fmnist':
    data_train, data_test = load_data_fmnist(args.root)
elif args.dataset == 'cifar10':
    data_train, data_test = load_data_cifar10()

if args.dataset in ['power', 'gas', 'hepmass', 'miniboone', 'bsds300']:
    data_train = torch.tensor(data_train).float().to(device)
    data_validate = torch.tensor(data_validate).float().to(device)
    data_test = torch.tensor(data_test).float().to(device)
else:
    data_train = torch.tensor(data_train).float().reshape(len(data_train), -1).to(device)
    data_train = (data_train + torch.rand_like(data_train)) / 256.
    data_test = torch.tensor(data_test).float().reshape(len(data_test), -1).to(device)
    data_test = (data_test + torch.rand_like(data_test)) / 256.
    
    data_validate = data_train[-10000:]
    data_train = data_train[:-10000]

if args.train_size > 0:
    assert args.train_size <= len(data_train)
    data_train = data_train[torch.randperm(len(data_train))][:args.train_size]

if args.validate_size > 0:
    assert args.validate_size <= len(data_validate)
    data_validate = data_validate[torch.randperm(len(data_validate))][:args.validate_size]

#hyperparameters
ndim = data_train.shape[1] 
if ndim <= 8 or len(data_train) / float(ndim) < 20:
    n_component = ndim 
else:
    n_component = 8 
interp_nbin = min(200, int(len(data_train)**0.5)) 
KDE = True 
bw_factor_data = 1 
alpha = (0.9, 0.99) 
edge_bins = min(int(len(data_train)/200./interp_nbin), 5) 
batchsize = 2**15 
derivclip = None 
noise_threshold = 0
verbose = True
ndata_wT = min(len(data_train), int(math.log(ndim)*50000)) 
ndata_spline = len(data_train) 
MSWD_max_iter = min(len(data_train) // ndim, 200)
shape = [1,28,28] 
t_total = time.time()

#define the model
model = SIT(ndim=ndim).requires_grad_(False).cuda()
#model = torch.load('/global/scratch/biwei/model/GIS/GIS_%s_train%d_validate%d_seed%d_defaultparam' % (args.dataset, len(data_train), len(data_validate), args.seed))

logp_train = []
logp_validate = []
logp_test = []
SWD = []

logj_train = torch.zeros(len(data_train))
logj_validate = torch.zeros(len(data_validate))
logj_test = torch.zeros(len(data_test))

best_validate_logp = -1e10
best_Nlayer = 0

if args.dataset in ['mnist', 'fmnist', 'cifar10']:
    #logit transform
    layer = logit(lambd=1e-5)
    data_train, logj_train = layer(data_train)
    logp_train.append((torch.mean(logj_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_train**2,  dim=1)/2)).item())

    data_validate, logj_validate = layer(data_validate)
    logp_validate.append((torch.mean(logj_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_validate**2,  dim=1)/2)).item())

    data_test, logj_test = layer(data_test)
    logp_test.append((torch.mean(logj_test) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_test**2,  dim=1)/2)).item())

    model.add_layer(layer)
    print ('logp:', logp_train[-1], logp_validate[-1], logp_test[-1])
    
    best_validate_logp = logp_validate[-1]
    best_Nlayer = 1 

wait = 0
print(data_train.shape)
#sliced transport layer
while True:
    t = time.time()
    i = len(model.layer)
    if args.dataset in ['power', 'gas', 'hepmass', 'miniboone', 'bsds300']:
        layer = SlicedTransport(ndim=ndim, n_component=n_component, interp_nbin=interp_nbin).requires_grad_(False).cuda()
    elif i % 2 == 0:
        kernel = [4, 4]
        n_component = 8
        shift = torch.randint(4, (2,)).tolist()
        layer = PatchSlicedTransport(shape=shape, kernel_size=kernel, shift=shift, n_component=n_component, interp_nbin=interp_nbin).requires_grad_(False).cuda()
    else:
        kernel = [2, 2]
        n_component = 4
        shift = torch.randint(2, (2,)).tolist()
        layer = PatchSlicedTransport(shape=shape, kernel_size=kernel, shift=shift, n_component=n_component, interp_nbin=interp_nbin).requires_grad_(False).cuda()
    
    if ndata_wT < len(data_train) and ndim > 1:
        order = torch.randperm(data_train.shape[0])
        layer.fit_wT(data=data_train[order][:ndata_wT], MSWD_max_iter=MSWD_max_iter, verbose=verbose)
    else:
        layer.fit_wT(data=data_train, MSWD_max_iter=MSWD_max_iter, verbose=verbose)
    
    SWD1 = layer.fit_spline(data=data_train, edge_bins=edge_bins, derivclip=derivclip, alpha=alpha, noise_threshold=noise_threshold, KDE=KDE, bw_factor=bw_factor_data, batchsize=batchsize, verbose=verbose)

    if (SWD1>noise_threshold).any():

        j = 0
        while j * batchsize < len(data_train):
            data_train[j*batchsize:(j+1)*batchsize], logj_train1 = layer(data_train[j*batchsize:(j+1)*batchsize])
            logj_train[j*batchsize:(j+1)*batchsize] = logj_train[j*batchsize:(j+1)*batchsize] + logj_train1
            j += 1
        logp_train.append((torch.mean(logj_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_train**2,  dim=1)/2)).item())

        j = 0
        while j * batchsize < len(data_validate):
            data_validate[j*batchsize:(j+1)*batchsize], logj_validate1 = layer(data_validate[j*batchsize:(j+1)*batchsize])
            logj_validate[j*batchsize:(j+1)*batchsize] = logj_validate[j*batchsize:(j+1)*batchsize] + logj_validate1
            j += 1
        logp_validate.append((torch.mean(logj_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_validate**2,  dim=1)/2)).item())

        j = 0
        while j * batchsize < len(data_test):
            data_test[j*batchsize:(j+1)*batchsize], logj_test1 = layer(data_test[j*batchsize:(j+1)*batchsize])
            logj_test[j*batchsize:(j+1)*batchsize] = logj_test[j*batchsize:(j+1)*batchsize] + logj_test1
            j += 1
        logp_test.append((torch.mean(logj_test) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(data_test**2,  dim=1)/2)).item())

        SWD.append(SWD1)
        model.add_layer(layer)

    if logp_validate[-1] > best_validate_logp:
        best_validate_logp = logp_validate[-1]
        best_Nlayer = len(model.layer) 
        wait = 0
    else:
        wait += 1

    print ('logp:', logp_train[-1], logp_validate[-1], logp_test[-1], 'time:', time.time()-t, 'iteration:', len(model.layer), 'best:', best_Nlayer)
    print ()

    if wait == 100:
        break

model.layer = model.layer[:best_Nlayer]
print ('best logp:', logp_train[best_Nlayer-1], logp_validate[best_Nlayer-1], logp_test[best_Nlayer-1], 'time:', time.time()-t_total, 'iteration:', len(model.layer))
#torch.save(model, '/global/scratch/biwei/model/GIS/GIS_%s_train%d_validate%d_seed%d_defaultparam' % (args.dataset, len(data_train), len(data_validate), args.seed))
