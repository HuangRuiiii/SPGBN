import numpy as np
from .basic_sampler import Basic_Sampler

class Param_Sampler(object):
    
    def __init__(self, device = 'cpu'):
        self.realmin = np.finfo(float).tiny
        self._sampler = Basic_Sampler(device)

    def calculate_pj(self, c_j: dict, T: int):
        '''
        calculate p_j from layer 1 to T+1 according to c_j
        Inputs:
            c_j  : [dict] T+1 1*N vector, the variables in the scale parameter in the Theta
            T    : [int] network depth
        Outputs:
            p_j  : [dict] T+1 1*N vector, the variables in the scale parameter in the Theta

        '''
        p_j = {}
        N = len(c_j[1])
        p_j[0] = (1 - np.exp(-1)) * np.ones(N)
        p_j[1] = 1 / (1 + c_j[1])

        for t in range(2, T+2):
            temp = -np.log(np.maximum(1 - p_j[t-1], self.realmin))
            p_j[t] = temp / (temp + c_j[t])

            if np.any(np.isnan(p_j[t])):
                print('Warning: pj Nan')
                p_j[t][np.isnan(p_j[t])] = self.realmin
        return p_j

    def sample_phi(self, WSZS, At, IsNoSample = False):
        '''
        update Phi_t at layer t
        Inputs:
            WSZS_t  : [np.ndarray]  (K_t-1)*(K_t) count matrix appearing in the likelihood of Phi_t
            Eta_t   : [np.ndarray]  scalar, the variables in the prior of Phi_t
        Outputs:
            Phi_t   : [np.ndarray]  (K_t-1)*(K_t), topic matrix at layer t

        '''
        if not IsNoSample:
            Phi = self._sampler.gamma(WSZS + At)
            temp = np.sum(Phi, axis = 0)
            tempdex = temp > 0
            Phi[:, tempdex] = Phi[:,tempdex] / temp[tempdex]
            Phi[:, ~tempdex] = 0
        else:
            Phi = At + WSZS 
            temp = np.sum(Phi, axis = 0)
            Phi = Phi/ temp
            if np.isnan(Phi).any():
                print('Warning: Phi Nan')
                tempdex = temp > 0
                Phi[:, ~tempdex] = 0

        return Phi

    def sample_rk(self, XTp1, rk, pj, gamma0, c0):
        e0 = 1
        f0 = 1
        a0 = 0.01
        b0 = 0.01
        KT = len(rk)

        c0 = self._sampler.gamma((e0 + gamma0) / (f0 + np.sum(rk)))
        sumlogpi = np.sum(np.log(np.maximum(1-pj, self.realmin)))
        p_prime = -sumlogpi / (c0 - sumlogpi)

        crt_sum = self._sampler.crt_sum(XTp1, gamma0/KT)
        gamma0 = self._sampler.gamma(a0 + crt_sum) / (b0 - np.log(np.maximum(1-p_prime, self.realmin)))
        rk = self._sampler.gamma(gamma0/KT + XTp1) / (c0 - sumlogpi)

        return rk, gamma0, c0