import time
import numpy as np
from .sampler import Basic_Sampler
from .sampler.parameters_sampler import *
from scipy.sparse import coo_matrix

class spgbn():
    def __init__(self, data, args, device='cpu'):
        self.args = args
        self.K = [args.initial_nodes_num] * args.layers_num
        self.basic_sampler = Basic_Sampler(device)
        self.param_sampler = Param_Sampler(device)

        self.Phi = {}
        self.Theta = {}
        self.c_j = {}
        self.p_j = {}

        self.Ukk = {}
        self.Vkk = {}
        self.Lam = {}
        self.M = {}
        self.Z = {}
        self.D = {}
        self.A = {}
        self.xi = {}
        self.zeta = {} 

        # latent count for phi
        self.WSZS = {} 
        # latent count for theta
        self.Xt_to_t1 = {}        

        self.a0 = 0.01
        self.b0 = 0.01
        self.e0 = 0.01
        self.f0 = 0.01
        self.epsilon=0.01

        self.alpha=1
        self.beta=1
        self.L=10
        self.realmin = 2.2204e-16

        self.r_k = None
        self.gamma0 = None
        self.c0 = None

        self.ZS = None
        self.DS = None
        self.WS = None
        self.ZSDS = None
        self.ZSWS = None
        self.n_dot_k = None

        #N = args.rows
        #V = args.cols
        self.X_all = data
        self.X_train = data[args.train_indices, :]
        self.args.rows = self.X_train.shape[0]
        self.args.cols = self.X_train.shape[1]
        self.args.total_counts = np.sum(self.X_train)
        self.find_WSDS(self.X_train)
        ...

    def find_WSDS(self, data):
        # find the location of each nonzero element in the data matrix: di(col index), wi(row index), cc(nonzero element)
        """
        di, wi, cc = find(data.T)

        ii = np.zeros(self.num_total_words, dtype = np.int32)
        jj = np.zeros(self.num_total_words, dtype = np.int32)

        # ii, jj are the row and column index of the each word in the bag of words
        count = 0
        for i in range(len(cc)):
            ii[count:(count + cc[i])] = wi[i]
            jj[count:(count + cc[i])] = di[i]
            count += cc[i]

        WS = ii
        DS = jj
        """
        coo_data = coo_matrix(data)
        wi = coo_data.row
        di = coo_data.col
        cc = coo_data.data

        # repeat the row and column indices according to the count of each word
        self.DS = np.repeat(wi, cc.astype(np.int32))
        self.WS = np.repeat(di, cc.astype(np.int32))
    
    def initialize_para(self, t):

        if t == 0:
            # assign each word to latent node randomly
            self.ZS = np.random.randint(0, self.K[0], size = self.args.total_counts) 
            self.ZSDS = coo_matrix((np.ones(self.args.total_counts), (self.ZS, self.DS)), shape = (self.K[0], self.args.rows)).toarray().astype(np.int32)
            self.ZSWS = coo_matrix((np.ones(self.args.total_counts), (self.ZS, self.WS)), shape = (self.K[0], self.args.cols)).toarray().astype(np.int32)
            
            self.WSZS[0] = self.ZSWS.T
            self.Xt_to_t1[0] = self.ZSDS
            self.n_dot_k = np.sum(self.ZSDS, axis = 1)
            self.c_j = {t: np.ones(self.args.rows) for t in range(self.args.layers_num + 1)}
            self.p_j = self.param_sampler.calculate_pj(self.c_j, t)
            self.r_k = 1 / self.K[0] * np.ones(self.K[0])
            self.gamma0 = 1
            self.c0 = 1

            self.Ukk[t] = np.ones((self.args.cols, self.L)) 
            self.Vkk[t] = np.ones((self.K[t], self.L)) 
            self.Lam[t] = 1 / self.L * np.ones(self.L)
            self.M[t] = np.random.poisson(self.Ukk[t] @ np.diag(self.Lam[t]) @ (self.Vkk[t]).T)
            
            # sparse prior for first layer
            self.D[t] = np.random.rand(self.args.cols, self.K[t])
            mask = np.random.rand(self.args.cols, self.K[t])
            # sampling sparse: here just set a  random initial value for Z
            self.Z[t] = (mask > 0.0).astype(int)
            # sparse：construct graph structure in A
            self.A[t] = self.D[t] * self.Z[t] 

        else:
            self.K[t] = self.K[t - 1]
            if self.K[t] <= 4: return 0

            self.Phi[t] = np.random.rand(self.K[t-1], self.K[t])
            self.Phi[t] = self.Phi[t] / np.maximum(self.realmin, self.Phi[t].sum(axis = 0))
            self.Theta[t] = np.ones((self.K[t], self.args.rows)) / self.K[t]
            self.p_j = self.param_sampler.calculate_pj(self.c_j, t)
            self.r_k = 1 / self.K[t] * np.ones(self.K[t])
            self.gamma0 = self.K[t] / self.K[0]
            self.c0 = 1

            self.Ukk[t] = np.ones((self.K[t-1], self.L)) 
            self.Vkk[t] = np.ones((self.K[t], self.L)) 
            self.Lam[t] = 1 / self.L * np.ones(self.L)

            self.M[t] = np.random.poisson(self.Ukk[t] @ np.diag(self.Lam[t]) @ (self.Vkk[t]).T)
            self.Z[t] = (self.M[t] >= 1).astype(int)
            self.D[t] = np.random.rand(self.K[t-1], self.K[t])

    def train(self, t):
        self.t = t
        self.initialize_para(t)
        time_records = []
        for iter in range(self.args.burnin + self.args.collection):
            start_time = time.time()
            # trim nodes
            if iter == self.args.burnin:
                self.Trim()

            # layerwise training
            for tc in range(self.t+1):
                # sample latent counts: Xt_to_t1, WSZS
                if tc == 0:
                    # random permutation
                    dex111 = np.random.permutation(self.args.total_counts)
                    self.ZS = self.ZS[dex111]
                    self.DS = self.DS[dex111]
                    self.WS = self.WS[dex111]

                    # sampling latent counts according to collapsed Gibbs sampling
                    if self.t == 0:
                        shape = np.tile(self.r_k, (self.args.rows,1)).T
                    else:
                        shape = np.dot(self.Phi[1], self.Theta[1])
                    self.collapsed_gibbs(self.ZSDS, self.ZSWS, self.n_dot_k, self.ZS, 
                                        self.WS, self.DS, shape, self.A[0])

                else:
                    self.latent_counts_allocation(tc, self.Xt_to_t1[tc-1], self.Phi[tc], self.Theta[tc])

                # sample sparse graph structure
                self.sample_graph_structure(tc)
            
                # sample Phi
                if tc > 0:
                    self.Phi[tc] = self.param_sampler.sample_phi(self.WSZS[tc], self.A[tc])
                    if (iter + 1) % 10 == 0:
                        sparse_rate = (np.count_nonzero(self.Phi[tc] == 0))/(self.Phi[tc].shape[0] * self.Phi[tc].shape[1])
                        print('The num of zero links of the {}th layer: {}, Sparse rate: {:.5f}'
                              .format(tc, np.count_nonzero(self.Phi[tc] == 0), sparse_rate))

            # sample r_k, gamma0, c_0
            XTp1 = self.basic_sampler.crt_sum_matrix((self.Xt_to_t1[self.t]).T, self.r_k)
            self.r_k, self.gamma0, self.c0 = self.param_sampler.sample_rk(XTp1, self.r_k, self.p_j[self.t+1], self.gamma0, self.c0) 

            # sample c_j and p_j
            if iter > 9:
                if self.t > 0:
                    self.p_j[1] = self.basic_sampler.beta(np.sum(self.Xt_to_t1[0], axis=0) + self.a0, np.sum(self.Theta[1], axis=0) + self.b0)
                else:
                    self.p_j[1] = self.basic_sampler.beta(np.sum(self.Xt_to_t1[0], axis=0) + self.a0, np.sum(self.r_k, axis=0) + self.b0)
                self.p_j[1] = np.minimum(np.maximum(self.p_j[1], self.realmin), 1 - self.realmin)
                self.c_j[1] = (1 - self.p_j[1]) / self.p_j[1]
                for t in range(2, self.t + 2):
                    if t == self.t + 1:
                        self.c_j[t] = self.basic_sampler.gamma(np.sum(self.r_k) * np.ones(self.args.rows) + self.e0, 1)/ \
                                        (np.sum(self.Theta[t-1], axis=0) + self.f0)
                    else:
                        self.c_j[t] = self.basic_sampler.gamma(np.sum(self.Theta[t], axis=0) + self.e0, 1)/ \
                                        (np.sum(self.Theta[t-1], axis=0) + self.f0)
                if self.t >= 1:
                    p_j_temp = self.param_sampler.calculate_pj(self.c_j, self.t)
                    for t in range(2, self.t + 2):
                        self.p_j[t] = p_j_temp[t]

                # sample theta
                self.sample_theta(tc)

            end_time = time.time()
            time_records.append(end_time - start_time)
            if (iter + 1) % 10 == 0:
                avg_time = sum(time_records) / 10
                time_records = []
                print(f'JointTrain Layer: {self.t + 1}, Iter: {iter + 1}, K: {np.count_nonzero(XTp1)}, Average time per iter: {avg_time:.2f} seconds')
        
        # sample final Phi after training
        print('*'*50)
        print('Sample parameters Phi')
        
        self.sample_final_phi(tc)
        sparse_rate = (np.count_nonzero(self.Phi[tc] == 0))/(self.Phi[tc].shape[0] * self.Phi[tc].shape[1])

        print(f'Zero links num of the {tc+1}th layer: {np.count_nonzero(self.Phi[tc] == 0)}')
        print(f'Sparse rate: {sparse_rate: .5f}') 
        print(f'Finish the training of the {self.t + 1}-th layer')
        print('*'*50)


    def test(self):
        """Extract the latent features from the test data"""
        # num of total docs
        N = self.X_all.shape[0]

        # Initialize c_j and p_j with median value of c_j from training results
        c_jmean = np.zeros(self.t + 2)
        for t in range(self.t + 2):
            c_jmean[t] = np.median(self.c_j[t])
        test_c_j = {}
        for t in range(self.t + 2):
            test_c_j[t] = np.ones(N) * c_jmean[t]
        test_p_j = self.param_sampler.calculate_pj(test_c_j, self.t)

        # Initialize theta from top to bottom
        test_Theta = {}
        for t in range(self.t, -1, -1):
            if t == self.t:
                shape = np.tile(self.r_k, (N,1)).T
            else:
                shape = np.dot(self.Phi[t+1], test_Theta[t+1])
                shape[np.isnan(shape)] = 0
            temp = self.basic_sampler.gamma(shape) / test_c_j[t+1]
            test_Theta[t] = np.maximum(temp, 1e-2)

        # Latent count for theta    
        test_Xt_to_t1 = {}

        # Intialize average value of c_j p_j and theta
        ThetaFreqAver = {}
        c_jAver = {}
        p_jAver = {}
        for t in range(self.t + 2):
            if t <= self.t:
                ThetaFreqAver[t] = 0
            c_jAver[t] = 0
            p_jAver[t] = 0
        #--------------------------------------------------------------------------------------------------------------
        time_records = []
        for iter in range(self.args.burnin + self.args.collection):
            
            start_time = time.time()
            # Upward pass latent count
            for t in range(self.t+1):
                if t == 0:
                    test_Xt_to_t1[t] = self.basic_sampler.mult_matrix_fast((self.X_all).T, self.Phi[t], test_Theta[t])
                else:
                    test_Xt_to_t1[t], _ = self.basic_sampler.crt_mult_matrix(test_Xt_to_t1[t-1], self.Phi[t], test_Theta[t])
        
            # Sample c_j p_j
            if iter >= 10:
                if self.t > 0:
                    test_p_j[1] = self.basic_sampler.beta( np.sum(test_Xt_to_t1[0], axis=0) + self.a0, np.sum(test_Theta[1], axis=0) + self.b0)
                else:
                    test_p_j[1] = self.basic_sampler.beta( np.sum(test_Xt_to_t1[0], axis=0) + self.a0, np.sum(self.r_k, axis=0) + self.b0)
                test_p_j[1] = np.minimum(np.maximum(test_p_j[1], self.realmin), 1 - self.realmin)
                test_c_j[1] = (1 - test_p_j[1]) / test_p_j[1]
                for t in range(2, self.t + 1):
                    if t == self.t:
                        test_c_j[t] = self.basic_sampler.gamma(np.sum(self.r_k) * np.ones(N) + self.e0, 1) / (np.sum(test_Theta[t-1], axis=0) + self.f0)
                    else:
                        test_c_j[t] = self.basic_sampler.gamma(np.sum(test_Theta[t], axis=0) + self.e0, 1) / (np.sum(test_Theta[t-1], axis=0) + self.f0)
                test_p_j_temp = self.param_sampler.calculate_pj(test_c_j, self.t)
                for t in range(2, self.t + 1):
                    test_p_j[t] = test_p_j_temp[t]

            # Sample Theta (Downward)
            for t in range(self.t, -1, -1):
                if t == self.t:
                    shape = np.tile(self.r_k, (N,1)).T
                else:
                    shape = np.dot(self.Phi[t+1], test_Theta[t+1])
                    shape[np.isnan(shape)] = 0
                num_current_nodes = shape.shape[0]
                temp = test_c_j[t+1] - np.log(np.maximum(1 - test_p_j[t], self.realmin))
                test_Theta[t] = self.basic_sampler.gamma(shape + test_Xt_to_t1[t]) / np.tile(temp, (num_current_nodes, 1))

                if (np.isnan(test_Theta[t])).any():
                    print('Warning: Theta Nan')
                    test_Theta[t][np.isnan(test_Theta[t])]=0

            end_time = time.time()
            time_records.append(end_time - start_time)

            if (iter + 1) % 10 == 0:
                avg_time = sum(time_records) / 10
                time_records = []
                print(f'Testing Layer: {self.t + 1}, Iter: {iter + 1}, Average time per iter: {avg_time:.2f} seconds')

            # Average
            if iter >= self.args.burnin:
                for t in range(self.t + 2):
                    if t <= self.t:
                        ThetaFreqAver[t] = ThetaFreqAver[t] + (test_Theta[t] / (np.maximum(np.sum(test_Theta[t], axis=0), self.realmin))) / self.args.collection
                    c_jAver[t] = c_jAver[t] + test_c_j[t] / self.args.collection
                    p_jAver[t] = p_jAver[t] + test_p_j[t] / self.args.collection
        #--------------------------------------------------------------------------------------------------------------
        return ThetaFreqAver

    def collapsed_gibbs(self, ZSDS, ZSWS, n_dot_k, ZS, WS, DS, shape, A):
        updated_res = self.basic_sampler.gnbp_collapsed_deep_sparse_mp(ZSDS, ZSWS, n_dot_k, ZS,
                                                         WS, DS, shape, A) 
        self.ZSDS, self.ZSWS, self.n_dot_k, self.ZS = updated_res
        self.WSZS[0] = self.ZSWS.T
        self.Xt_to_t1[0] = self.ZSDS

    def latent_counts_allocation(self, t, Xt_to_t1, Phi, Theta):
        update_res = self.basic_sampler.crt_mult_matrix(Xt_to_t1, Phi, Theta)
        self.Xt_to_t1[t], self.WSZS[t] = update_res

    def sample_graph_structure(self, t):
        """Sample hierarchical parameters Z and D from posterior distribution"""
        # sample Z
        self.xi[t] = self.basic_sampler.crt(self.WSZS[t], self.Z[t] * self.D[t])  # auxiliary variable which is likelihood of binary parameter Z
        self.zeta[t] = self.basic_sampler.beta(np.sum(self.WSZS[t], axis = 0) + self.realmin, np.sum(self.Z[t] * self.D[t], axis = 0) + self.realmin)
        dex = (self.xi[t] >= 1)  # shape: v*k_1 or k_{t-1}*k_t
        self.Z[t][dex] = 1 # if the likelihood is not zero, directly set Z to 1
        zero_dex_row, zero_dex_col = np.where(dex.astype(int) == 0) # if the likelihood is zero, we need to sample
        num_zero_in_dex = len(zero_dex_row)
        temp = np.einsum('vl, kl, l -> vk', self.Ukk[t], self.Vkk[t], self.Lam[t])
        temp = np.exp(-temp)
        pij1 = (1-temp[zero_dex_row, zero_dex_col]) * np.exp(self.D[t][zero_dex_row, zero_dex_col] * np.log(1-self.zeta[t][zero_dex_col] + self.realmin))
        pij0 = temp[zero_dex_row, zero_dex_col]
        rs = (np.random.rand(num_zero_in_dex) < (pij1/ (pij1 + pij0))).astype(int)
        self.Z[t][zero_dex_row, zero_dex_col] = rs

        # sample D
        temp_ln_1_zeta = np.tile(np.log(1-self.zeta[t] + self.realmin), (self.Z[t].shape[0],1))
        self.D[t] = self.basic_sampler.gamma(0.1 + self.xi[t]) / (0.1 - self.Z[t] * temp_ln_1_zeta)
        self.D[t] = np.ones((self.D[t]).shape) * 0.1

        # compute A
        self.A[t] = self.Z[t] * self.D[t]

        # sample U, V, Lam
        ULV = self.Ukk[t] @ np.diag(self.Lam[t]) @ (self.Vkk[t]).T
        self.M[t] = self.Z[t] * self.basic_sampler.Po_plus(ULV)
        M_K1L, M_K2L = self.basic_sampler.Mult_3Mat(self.M[t], self.Ukk[t], self.Vkk[t], self.Lam[t])
        self.Ukk[t] = self.basic_sampler.gamma(self.a0 + M_K1L) / (self.b0 + (self.Vkk[t] @ np.diag(self.Lam[t])).sum(axis = 0))
        self.Vkk[t] = self.basic_sampler.gamma(self.a0 + M_K2L) / (self.b0 + (self.Ukk[t] @ np.diag(self.Lam[t])).sum(axis = 0))
        temp_uv = np.einsum('ul, vl -> l', self.Ukk[t], self.Vkk[t])
        self.Lam[t] = self.basic_sampler.gamma(self.alpha/self.L + M_K1L.sum(axis = 0)) / (self.beta + temp_uv)

        if (((self.A[t]).sum(axis = 0) == 0).any()):
            dex = ((self.A[t]).sum(axis=0)) == 0
            self.A[t][:, dex] = 1 / (self.A[t].shape[0])

    def sample_theta(self, t):
        for t in range(self.t, -1, -1):
            if t == self.t:
                shape = np.tile(self.r_k, (self.args.rows,1)).T
            else:
                shape = np.dot(self.Phi[t+1], self.Theta[t+1])
                shape[np.isnan(shape)] = 0
            if t > 0:
                num_current_nodes = shape.shape[0]
                temp = self.c_j[t+1] - np.log(np.maximum(1 - self.p_j[t], self.realmin))
                self.Theta[t] = self.basic_sampler.gamma(shape + self.Xt_to_t1[t]) / np.tile(temp, (num_current_nodes,1))
    
    def sample_final_phi(self, t):
        """Sample final structure parameters Phi after training"""
        for t in range(self.t+1):
            self.A[t] = self.Z[t] * self.D[t]
            if (((self.A[t]).sum(axis = 0) == 0).any()):
                dex = ((self.A[t]).sum(axis=0)) == 0
                (self.A[t])[:, dex] = 1/((self.A[t]).shape[0])
            self.Phi[t] = self.param_sampler.sample_phi(self.WSZS[t], self.A[t], False)

    def Trim(self):
        """Trim the inactive factors in the current layer"""
        if self.t == 0:
            kk, kki, kkj = np.unique(self.ZS, return_index=True, return_inverse=True)
            self.gamma0 = self.gamma0 * len(kk) / self.K[0]
            self.r_k = self.r_k[kk.astype(int)] # prune r_k 
            self.K[0] = len(kk)
            self.ZS = kkj
            self.ZSDS = coo_matrix((np.ones(self.args.total_counts), (self.ZS, self.DS)), shape = (self.K[0], self.args.rows)).toarray()
            self.ZSWS = coo_matrix((np.ones(self.args.total_counts), (self.ZS, self.WS)), shape = (self.K[0], self.args.cols)).toarray()
            self.n_dot_k = np.sum(self.ZSDS, axis = 1)
            self.WSZS[0] = self.ZSWS.T
            self.Xt_to_t1[0] = self.ZSDS

            # Trim Z,D,Vkk
            self.Z[self.t] = self.Z[self.t][:,kk]
            self.D[self.t] = self.D[self.t][:,kk]
            self.Vkk[self.t] = self.Vkk[self.t][kk,:]

        else:
            dexK = np.where(np.sum(self.Xt_to_t1[self.t], axis = 1) == 0)[0]
            # if there exist inactive factors, then prune them
            if len(dexK) > 0:
                self.gamma0 = self.gamma0 * len(dexK) / self.K[self.t]
                self.r_k = np.delete(self.r_k, dexK, axis = 0)
                self.K[self.t] = self.K[self.t] - len(dexK)
                self.Xt_to_t1[self.t] = np.delete(self.Xt_to_t1[self.t], dexK, axis = 0)
                self.Theta[self.t] = np.delete(self.Theta[self.t], dexK, axis = 0)
                self.WSZS[self.t] = np.delete(self.WSZS[self.t], dexK, axis = 1)
                self.Phi[self.t] = np.delete(self.Phi[self.t], dexK, axis = 1)
                self.Z[self.t] = np.delete(self.Z[self.t], dexK, axis = 1)
                self.D[self.t] = np.delete(self.D[self.t], dexK, axis = 1)
                self.Vkk[self.t] = np.delete(self.Vkk[self.t], dexK, axis = 0)