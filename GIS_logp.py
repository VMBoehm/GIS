from SIT import *
from load_data import * 
import argparse
import os
import pickle
from sklearn.decomposition import PCA

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

# convert to dictionary
params = vars(args)


class Data():
    def __init__(self,params,device):

        self.params = params
        
        if params['dataset'] in ['mnist','fmnist','cifar10','celeba']:
            self.type = 'image'
        else:
            self.type = 'flat'

            data_train, data_val, data_test = globals()["load_data_%s"%params['dataset']]() 

        if self.type =='image':

            data_val   = data_train[-10000:]
            data_train = data_train[:-10000]
    
            data_train = (data_train+np.random.rand(np.prod(data_train.shape)).reshape(data_train.shape))/256.
            data_val   = (data_val+np.random.rand(np.prod(data_val.shape)).reshape(data_val.shape))/256.
            data_test  = (data_test+np.random.rand(np.prod(data_test)).reshape(data_test.shape))/256.

        if params['pca']:
            pca = PCA()
            pca.fit(data_train)
            assert(params['pca_cut']<0.2)
            print(pca.explained_variance_ratio_)
            num = np.where(pca.explained_variance_ratio_>params['pca_cut'])[0][-1]
            print('using ', num, ' components')
            pca = PCA(n_components=num)
            pca.fit(data_train)
            
            data_train = pca.transform(data_train)
            data_val   = pca.transform(data_val)
            data_test  = pca.transform(data_test)

            self.type  = 'flat'
        
        if self.type=='flat':
            self.train = torch.tensor(data_train).float().to(device)
            self.val   = torch.tensor(data_val).float().to(device)
            self.test  = torch.tensor(data_test).float().to(device)
        else:
            self.train = torch.tensor(data_train).float().reshape(len(data_train), -1).to(device)
            self.test  = torch.tensor(data_test).float().reshape(len(data_test), -1).to(device)
            self.val   = torch.tensor(data_val).float().reshape(len(data_val), -1).to(device)
    

    
        if params['train_size'] > 0:
            assert params['train_size'] <= len(self.train)
            self.train = self.train[torch.randperm(len(self.train))][:params['train_size']]

        if params['validate_size'] > 0:
            assert params['validate_size'] <= len(self.val)
            self.val = self.val[torch.randperm(len(self.val))][:params['validate_size']]

        # not sure what this is doing
        self.ndim = data_train.shape[1] 
        if self.ndim <= 8 or len(self.train) / float(self.ndim) < 20:
            self.n_component = self.ndim 
        else:
            self.n_component = 8 
        
        self.shape = [1]+list(self.val.shape[1:])


class GIS():

    def __init__(self,params,data):

        self.params = params
        self.data = data
        #define the model
        if self.params['load'] == False:
            self.model  = SIT(ndim=data.ndim).requires_grad_(False).cuda()
        else:
            model = torch.load(params['model_file'])

        self.logp_train = []
        self.logp_validate = []
        self.logp_test = []
        self.SWD = []

    def set_rand(self):
        torch.manual_seed(params['seed'])
        np.random.seed(params['seed'])
        torch.cuda.manual_seed_all(params['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train(self):
        logj_train = torch.zeros(len(data.train))
        logj_validate = torch.zeros(len(data.val))
        logj_test = torch.zeros(len(data.test))

        self.best_validate_logp = -1e10
        self.best_Nlayer = 0

        if self.data.type== 'image':
            #logit transform
            layer = logit(lambd=1e-5)
            self.data.train, logj_train = layer(data.train)
            self.logp_train.append((torch.mean(logj_train) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(self.data.train**2,  dim=1)/2)).item())
    
    

            self.data.val, logj_validate = layer(self.data.val)
            self.logp_validate.append((torch.mean(logj_validate) - ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(self.data.val**2,  dim=1)/2)).item())

            self.model.add_layer(layer)
            print ('logp:', logp_train[-1], logp_validate[-1], logp_test[-1])
    
            self.best_validate_logp = logp_validate[-1]
            self.best_Nlayer = 1 

        wait = 0
        i = len(self.model.layer) 
        #sliced transport layer
        while i<20:
            print('i %d \n'%d)
            t = time.time()
            i = len(self.model.layer)
            if self.data.type=='flat':
                layer = SlicedTransport(ndim=self.data.ndim, n_component=self.data.n_component, interp_nbin=params['interp_nbin']).requires_grad_(False).cuda()
            elif i % 2 == 0:
                kernel = [4, 4]
                n_component = 8
                shift = torch.randint(4, (2,)).tolist()
                layer = PatchSlicedTransport(shape=self.data.shape, kernel_size=params['kernel'], shift=params['shift'], n_component=self.data.n_component, interp_nbin=params['interp_nbin']).requires_grad_(False).cuda()
            else:
                kernel = [2, 2]
                n_component = 4
                shift = torch.randint(2, (2,)).tolist()
                layer = PatchSlicedTransport(shape=self.shape, kernel_size=params['kernel'], shift=params['shift'], n_component=self.data.n_component, interp_nbin=params['interp_nbin']).requires_grad_(False).cuda()
        
            if params['ndata_wT'] < len(data.train) and data.ndim > 1:
                order = torch.randperm(data.train.shape[0])
                layer.fit_wT(data=self.data.train[order][:params['ndata_wT']], MSWD_max_iter=params['MSWD_max_iter'], verbose=params['verbose'])
            else:
                layer.fit_wT(data=self.data.train, MSWD_max_iter=params['MSWD_max_iter'], verbose=params['verbose'])
        
            SWD1 = layer.fit_spline(data=self.data.train, edge_bins=params['edge_bins'], derivclip=params['derivclip'], alpha=params['alpha'], noise_threshold=params['noise_threshold'], KDE=params['KDE'], bw_factor=params['bw_factor_data'], batchsize=params['batchsize'], verbose=params['verbose'])
            batchsize = params['batchsize']
            if (SWD1>params['noise_threshold']).any():
                j = 0
                while j * batchsize < len(self.data.train):
                    self.data.train[j*batchsize:(j+1)*batchsize], logj_train1 = layer(self.data.train[j*batchsize:(j+1)*batchsize])
                    logj_train[j*batchsize:(j+1)*batchsize] = logj_train[j*batchsize:(j+1)*batchsize] + logj_train1
                    j += 1
                self.logp_train.append((torch.mean(logj_train) - data.ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(self.data.train**2,  dim=1)/2)).item())

                j = 0
                while j * batchsize < len(self.data.val):
                    self.data.val[j*batchsize:(j+1)*batchsize], logj_validate1 = layer(self.data.val[j*batchsize:(j+1)*batchsize])
                    logj_validate[j*batchsize:(j+1)*batchsize] = logj_validate[j*batchsize:(j+1)*batchsize] + logj_validate1
                    j += 1
                self.logp_validate.append((torch.mean(logj_validate) - data.ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(self.data.val**2,  dim=1)/2)).item())

                j = 0
                while j * batchsize < len(self.data.test):
                    self.data.test[j*batchsize:(j+1)*batchsize], logj_test1 = layer(self.data.test[j*batchsize:(j+1)*batchsize])
                    logj_test[j*batchsize:(j+1)*batchsize] = logj_test[j*batchsize:(j+1)*batchsize] + logj_test1
                    j += 1
                self.logp_test.append((torch.mean(logj_test) - data.ndim/2*torch.log(torch.tensor(2*math.pi)) - torch.mean(torch.sum(self.data.test**2,  dim=1)/2)).item())
                self.SWD.append(SWD1)
                self.model.add_layer(layer)

            if self.logp_validate[-1] > self.best_validate_logp:
                self.best_validate_logp = self.logp_validate[-1]
                self.best_Nlayer = len(self.model.layer) 
                wait = 0
            else:
                wait += 1

            print ('logp:', self.logp_train[-1], self.logp_validate[-1], self.logp_test[-1], 'time:', time.time()-t, 'iteration:', len(self.model.layer), 'best:', self.best_Nlayer)

            if wait == 100:
                break

        self.model.layer = self.model.layer[:best_Nlayer]
        print ('best logp:', self.logp_train[best_Nlayer-1], self.logp_validate[best_Nlayer-1], self.logp_test[best_Nlayer-1], 'time:', time.time()-t_total, 'iteration:', len(self.model.layer))

    def save_model(self):
        torch.save(model, params('model_file'))

    def fwd(self,x):
        return self.model.forward(x)

    def logp(self,x):
        return model.evaluate_density(x)

if __name__=='__main__':

    ROOT = '/global/scratch/vboehm/gis/'
    LOC  = './params/'

    if not os.path.isdir(ROOT):
        os.makedirs(ROOT)

    if not os.path.isdir(LOC):
        os.makedirs(LOC)

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    assert torch.cuda.is_available()
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


    params['pca']         = True
    params['pca_cut']     = 0.001
 
    data = Data(params,device)

    params['run_name']    = 'test'
    params['load']        = False
    params['interp_nbin'] = min(200, int(len(data.train)**0.5)) 
    params['KDE']         = True 
    params['bw_factor_data'] = 1 
    params['alpha']       = (0.9, 0.99) 
    params['edge_bins']   = min(int(len(data.train)/200./params['interp_nbin']), 5) 
    params['batchsize']   = 2**15 
    params['derivclip']   = None 
    params['noise_threshold'] = 0
    params['verbose']     = True
    params['ndata_wT']    = min(len(data.train), int(math.log(data.ndim)*50000)) 
    params['ndata_spline']  = len(data.train) 
    params['MSWD_max_iter'] = min(len(data.train) // data.ndim, 200)
    params['savio_root']    = ROOT
    params['model_file']    = os.path.join(ROOT, '%s'%params['run_name'])
    params['params_file']   = os.path.join(LOC,'%s.pkl'%params['run_name'])

    pickle.dump(params,open(params['params_file'],'wb'))

    gis = GIS(params, data)
    gis.set_rand()
    gis.train()
    gis.save()
    print(gis.fwd(data.train[0:1]))
    print(torch.mean(gis.logp(data.train)))
