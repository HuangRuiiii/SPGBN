"""
===========================================
Model Sampler implemented on CPU
===========================================
"""

import numpy as np
import numpy.ctypeslib as npct
import ctypes  
from ctypes import *
from scipy.sparse import csc_matrix, coo_matrix
import os

class model_sampler_cpu(object):

    def __init__(self, system_type='Windows', seed=0):
        """
        The basic class for sampling distribution on cpu
        """
        super(model_sampler_cpu, self).__init__()

        self.system_type = system_type
        self.seed = seed

        array_2d_double = npct.ndpointer(dtype=np.double, ndim=2, flags='C')
        array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='C')
        array_2d_int = npct.ndpointer(dtype=np.int32, ndim=2, flags='C')
        array_1d_int = npct.ndpointer(dtype=np.int32, ndim=1, flags='C')
        ll = ctypes.cdll.LoadLibrary

        if system_type == "Windows":
            self.GNBP_Collapsed_Deep_lib = ll(os.path.dirname(__file__) + r"\_compact\gnbp_collapsed_deep.dll")
            self.GNBP_Collapsed_Deep_Sparse_lib = ll(os.path.dirname(__file__) + r"\_compact\gnbp_collapsed_deep_sparse.dll")
            self.GNBP_Collapsed_Deep_Sparse_MP_lib = ll(os.path.dirname(__file__) + r"\_compact\gnbp_collapsed_deep_sparse_mp.dll")
            self.Crt_Sum_Matrix_lib = ll(os.path.dirname(__file__) + r"\_compact\crt_sum_matrix.dll")
            self.Crt_Sum_lib = ll(os.path.dirname(__file__) + r"\_compact\crt_sum.dll")
            self.Crt_Mult_Matrix_lib = ll(os.path.dirname(__file__) + r"\_compact\crt_mult_matrix.dll")
            self.Mult_Matrix_Fast_lib = ll(os.path.dirname(__file__) + r"\_compact\mult_matrix_fast.dll")
            self.Crt_lib = ll(os.path.dirname(__file__) + r"\_compact\crt.dll")
            self.Mult_3Mat_lib = ll(os.path.dirname(__file__) + r"\compact\mult_3mat.dll")
            self.Mult_Sparse_lib = ll(os.path.dirname(__file__) + r"\_compact\mult_sparse.dll")
        else:
            self.GNBP_Collapsed_Deep_lib = ll(os.path.dirname(__file__) + "/_compact/gnbp_collapsed_deep.so")
            self.GNBP_Collapsed_Deep_Sparse_lib = ll(os.path.dirname(__file__) + "/_compact/gnbp_collapsed_deep_sparse.so")
            self.GNBP_Collapsed_Deep_Sparse_MP_lib = ll(os.path.dirname(__file__) + "/_compact/gnbp_collapsed_deep_sparse_mp.so")
            self.Crt_Sum_Matrix_lib = ll(os.path.dirname(__file__) + "/_compact/crt_sum_matrix.so")
            self.Crt_Sum_lib = ll(os.path.dirname(__file__) + "/_compact/crt_sum.so")
            self.Crt_Mult_Matrix_lib = ll(os.path.dirname(__file__) + "/_compact/crt_mult_matrix.so")
            self.Mult_Matrix_Fast_lib = ll(os.path.dirname(__file__) + "/_compact/mult_matrix_fast.so")
            self.Crt_lib = ll(os.path.dirname(__file__) + "/_compact/crt.so")
            self.Mult_3Mat_lib = ll(os.path.dirname(__file__) + "/_compact/mult_3mat.so")
            self.Mult_Sparse_lib = ll(os.path.dirname(__file__) + "/_compact/mult_sparse.so")
            
        self.Crt_lib.Crt_Sample.restype = None
        self.Crt_lib.Crt_Sample.argtypes = [array_1d_int, array_1d_int, array_1d_int, array_1d_double, c_int, c_int, c_int, array_1d_int]

        self.Crt_Sum_lib.Crt_Sum_Sample.restype = None
        self.Crt_Sum_lib.Crt_Sum_Sample.argtypes = [c_int, array_1d_int, c_double, array_1d_double]

        self.Crt_Sum_Matrix_lib.Crt_Sum_Matrix.restype = None
        self.Crt_Sum_Matrix_lib.Crt_Sum_Matrix.argtypes = [array_1d_double, array_1d_int, array_1d_int,
                                                           array_1d_int, c_int, c_int, array_1d_int]
    
        self.Crt_Mult_Matrix_lib.Crt_Mult_Matrix_Sample.restype = None
        self.Crt_Mult_Matrix_lib.Crt_Mult_Matrix_Sample.argtypes = [array_2d_int, array_2d_int, array_2d_double, array_2d_double,
                                                                    array_1d_int, array_1d_int, array_1d_int, c_int, c_int, c_int, array_1d_double]

        self.Mult_Matrix_Fast_lib.Mult_Matrix_Fast_Sample.restype = None
        self.Mult_Matrix_Fast_lib.Mult_Matrix_Fast_Sample.argtypes = [array_2d_int, array_2d_int, array_2d_double, array_2d_double,
                                                                      array_1d_int, array_1d_int, array_1d_int, c_int, c_int, c_int, array_1d_double]
        
        self.Mult_3Mat_lib.Mult_3Mat_Sample.restype = None
        self.Mult_3Mat_lib.Mult_3Mat_Sample.argtypes = [array_2d_int, array_2d_int, array_2d_double, array_2d_double, array_1d_double,
                                                        array_1d_int, array_1d_int, array_1d_int, c_int, c_int, c_int, array_1d_double]
        
        self.GNBP_Collapsed_Deep_lib.GNBP_Collapsed_Deep_Sample.restype = None
        self.GNBP_Collapsed_Deep_lib.GNBP_Collapsed_Deep_Sample.argtypes = [array_2d_int, array_2d_int, array_1d_int, 
                                                                            array_1d_int, array_1d_int, array_1d_int,
                                                                            c_int, c_int, c_int, c_int,
                                                                            array_2d_double, c_double, array_1d_double]

        self.GNBP_Collapsed_Deep_Sparse_lib.GNBP_Collapsed_Deep_Sample.restype = None
        self.GNBP_Collapsed_Deep_Sparse_lib.GNBP_Collapsed_Deep_Sample.argtypes = [array_2d_int, array_2d_int, array_1d_int, 
                                                                                   array_1d_int, array_1d_int, array_1d_int,
                                                                                   c_int, c_int, c_int, c_int,
                                                                                   array_2d_double, array_2d_double, array_1d_double]
        self.GNBP_Collapsed_Deep_Sparse_MP_lib.GNBP_Collapsed_Deep_Sample.restype = None
        self.GNBP_Collapsed_Deep_Sparse_MP_lib.GNBP_Collapsed_Deep_Sample.argtypes = [array_2d_int, array_2d_int, array_1d_int, 
                                                                                   array_1d_int, array_1d_int, array_1d_int,
                                                                                   c_int, c_int, c_int, c_int,
                                                                                   array_2d_double, array_2d_double, array_1d_double]
        
        self.Mult_Sparse_lib.Mult_Sparse_Sample.restype = None
        self.Mult_Sparse_lib.Mult_Sparse_Sample.argtypes = [array_2d_double, array_1d_int, array_1d_int, array_1d_int, 
                                                            array_2d_double, array_2d_double, c_int, c_int, c_int]

    def crt(self, X, R):

        nz_X_dex = np.nonzero(X)
        row_dex = nz_X_dex[0]
        col_dex = nz_X_dex[1]
        nz_X = X[nz_X_dex]
        nz_R = R[nz_X_dex]

        row_dex = np.array(row_dex, order = 'C').astype(np.int32)
        col_dex = np.array(col_dex, order = 'C').astype(np.int32)
        nz_X = np.array(nz_X, order = 'C').astype(np.int32)
        nz_R = np.array(nz_R, order = 'C').astype('double')

        Num = len(nz_X)
        K_tm1 = X.shape[0]
        K_t = X.shape[1]
        L = np.zeros(Num, order = 'C').astype(np.int32)

        self.Crt_lib.Crt_Sample(row_dex, col_dex, nz_X, nz_R, Num, K_tm1, K_t, L)
        L = coo_matrix((L, nz_X_dex), shape = (K_tm1, K_t)).toarray().astype(np.int32)

        return L


    def crt_sum(self, XTp1, r):

        Lenx = len(XTp1)
        x = np.array(XTp1, order = 'C').astype(np.int32)
        Lsum = np.zeros(1, order = 'C').astype('double')

        self.Crt_Sum_lib.Crt_Sum_Sample(Lenx, x, r, Lsum)

        return Lsum

    def crt_sum_matrix(self, Xt_to_t1, r_k):

        X = csc_matrix(Xt_to_t1)
        pr = X.data
        ir = X.indices
        jc = X.indptr

        pr = np.array(pr, order = 'C').astype(np.int32)
        ir = np.array(ir, order = 'C').astype(np.int32)
        jc = np.array(jc, order = 'C').astype(np.int32)
        Xt_to_t1 = np.array(Xt_to_t1, order = 'C').astype(np.int32)
        r_k = np.array(r_k, order = 'C').astype('double')

        Ksize = Xt_to_t1.shape[0]
        Nsize = Xt_to_t1.shape[1]
        Lsum = np.zeros(Nsize, order = 'C').astype(np.int32)

        self.Crt_Sum_Matrix_lib.Crt_Sum_Matrix(r_k, pr, ir, jc, Ksize, Nsize, Lsum)
        
        return Lsum

    def crt_mult_matrix(self, Xt_to_t1, Phi, Theta):

        X = csc_matrix(Xt_to_t1)
        pr = X.data
        ir = X.indices
        jc = X.indptr

        Vsize = Xt_to_t1.shape[0]
        Nsize = Xt_to_t1.shape[1]
        Ksize = Phi.shape[1]
        pr = np.array(pr, order = 'C').astype(np.int32)
        ir = np.array(ir, order = 'C').astype(np.int32)
        jc = np.array(jc, order = 'C').astype(np.int32)
        Phi = np.array(Phi, order = 'C').astype('double')
        Theta = np.array(Theta, order = 'C').astype('double')

        ZSDS = np.zeros((Ksize, Nsize), order = 'C').astype(np.int32)
        WSZS = np.zeros((Vsize, Ksize), order = 'C').astype(np.int32)
        prob_cumsum = np.zeros(Ksize, order = 'C').astype('double')

        self.Crt_Mult_Matrix_lib.Crt_Mult_Matrix_Sample(ZSDS, WSZS, Phi, Theta, pr, ir, jc, Vsize, Nsize, Ksize, prob_cumsum)

        return ZSDS, WSZS

    def mult_matrix_fast(self, Xt_to_t1, Phi, Theta):

        X = csc_matrix(Xt_to_t1)
        pr = X.data
        ir = X.indices
        jc = X.indptr

        Vsize = Xt_to_t1.shape[0]
        Nsize = Xt_to_t1.shape[1]
        Ksize = Phi.shape[1]
        pr = np.array(pr, order = 'C').astype(np.int32)
        ir = np.array(ir, order = 'C').astype(np.int32)
        jc = np.array(jc, order = 'C').astype(np.int32)
        Phi = np.array(Phi, order = 'C').astype('double')
        Theta = np.array(Theta, order = 'C').astype('double')

        ZSDS = np.zeros((Ksize, Nsize), order = 'C').astype(np.int32)
        WSZS = np.zeros((Vsize, Ksize), order = 'C').astype(np.int32)
        prob_cumsum = np.zeros(Ksize, order = 'C').astype('double')

        self.Mult_Matrix_Fast_lib.Mult_Matrix_Fast_Sample(ZSDS, WSZS, Phi, Theta, pr, ir, jc, Vsize, Nsize, Ksize, prob_cumsum)

        return ZSDS
    
    def Mult_3Mat(self, M, Ukk, Vkk, Lam):

        X = csc_matrix(M)
        pr = X.data
        ir = X.indices
        jc = X.indptr

        K1size = M.shape[0]
        K2size = M.shape[1]
        Lsize = len(Lam)
        pr = np.array(pr, order = 'C').astype(np.int32)
        ir = np.array(ir, order = 'C').astype(np.int32)
        jc = np.array(jc, order = 'C').astype(np.int32)
        Ukk = np.array(Ukk, order = 'C').astype('double')
        Vkk = np.array(Vkk, order = 'C').astype('double')
        Lam = np.array(Lam, order = 'C').astype('double')

        M_K1L = np.zeros((K1size, Lsize), order = 'C').astype(np.int32)
        M_K2L = np.zeros((K2size, Lsize), order = 'C').astype(np.int32)
        prob_cumsum = np.zeros(Lsize,order = 'C').astype('double')
        self.Mult_3Mat_lib.Mult_3Mat_Sample(M_K1L, M_K2L, Ukk, Vkk, Lam, pr, ir, jc, K1size, K2size, Lsize, prob_cumsum)

        return M_K1L, M_K2L
    
    def mult_sparse(self, X, Phi_0, Theta_0):
        
        Xmask = csc_matrix(X)
        pr = Xmask.data
        ir = Xmask.indices
        jc = Xmask.indptr

        Vsize = X.shape[0]
        Nsize = X.shape[1]
        Ksize = Phi_0.shape[1]
        pr = np.array(pr, order = 'C').astype(np.int32)
        ir = np.array(ir, order = 'C').astype(np.int32)
        jc = np.array(jc, order = 'C').astype(np.int32)
        Phi_0 = np.array(Phi_0, order = 'C').astype('double')
        Theta_0 = np.array(Theta_0, order = 'C').astype('double')

        X1 = np.zeros((Vsize, Nsize), order = 'C').astype('double')
        self.Mult_Sparse_lib.Mult_Sparse_Sample(X1, pr, ir, jc, Phi_0, Theta_0, Vsize, Nsize, Ksize)

        return X1

    def gnbp_collapsed_deep(self, ZSDS, ZSWS, n_dot_k, ZS, WS, DS, shape, eta):

        Vsize = ZSWS.shape[1]
        Nsize = ZSDS.shape[1]
        Ksize = ZSDS.shape[0]
        WordNum = len(WS)

        ZSDS = np.array(ZSDS, order = 'C').astype(np.int32)
        ZSWS = np.array(ZSWS, order = 'C').astype(np.int32)
        n_dot_k = np.array(n_dot_k, order = 'C').astype(np.int32)
        ZS = np.array(ZS, order = 'C').astype(np.int32)
        WS = np.array(WS, order = 'C').astype(np.int32)
        DS = np.array(DS, order = 'C').astype(np.int32)
        shape = np.array(shape, order = 'C').astype('double')
        prob_cum_sum = np.zeros(Ksize, order = 'C').astype('double')

        self.GNBP_Collapsed_Deep_lib.GNBP_Collapsed_Deep_Sample(ZSDS, ZSWS, n_dot_k, ZS, WS, DS,
                                                                Vsize, Nsize, Ksize, WordNum,
                                                                shape, eta, prob_cum_sum)

        return ZSDS, ZSWS, n_dot_k, ZS
    
    def gnbp_collapsed_deep_sparse(self, ZSDS, ZSWS, n_dot_k, ZS, WS, DS, shape, At):

        Vsize = ZSWS.shape[1]
        Nsize = ZSDS.shape[1]
        Ksize = ZSDS.shape[0]
        WordNum = len(WS)

        ZSDS = np.array(ZSDS, order = 'C').astype(np.int32)
        ZSWS = np.array(ZSWS, order = 'C').astype(np.int32)
        n_dot_k = np.array(n_dot_k, order = 'C').astype(np.int32)
        ZS = np.array(ZS, order = 'C').astype(np.int32)
        WS = np.array(WS, order = 'C').astype(np.int32)
        DS = np.array(DS, order = 'C').astype(np.int32)
        At = np.array(At, order = 'C').astype('double')
        shape = np.array(shape, order = 'C').astype('double')
        prob_cum_sum = np.zeros(Ksize, order = 'C').astype('double')

        self.GNBP_Collapsed_Deep_Sparse_lib.GNBP_Collapsed_Deep_Sample(ZSDS, ZSWS, n_dot_k, ZS, WS, DS,
                                                                Vsize, Nsize, Ksize, WordNum,
                                                                shape, At, prob_cum_sum)

        return ZSDS, ZSWS, n_dot_k, ZS

    def gnbp_collapsed_deep_sparse_mp(self, ZSDS, ZSWS, n_dot_k, ZS, WS, DS, shape, At):

        Vsize = ZSWS.shape[1]
        Nsize = ZSDS.shape[1]
        Ksize = ZSDS.shape[0]
        WordNum = len(WS)

        ZSDS = np.array(ZSDS, order = 'C').astype(np.int32)
        ZSWS = np.array(ZSWS, order = 'C').astype(np.int32)
        n_dot_k = np.array(n_dot_k, order = 'C').astype(np.int32)
        ZS = np.array(ZS, order = 'C').astype(np.int32)
        WS = np.array(WS, order = 'C').astype(np.int32)
        DS = np.array(DS, order = 'C').astype(np.int32)
        At = np.array(At, order = 'C').astype('double')
        shape = np.array(shape, order = 'C').astype('double')
        prob_cum_sum = np.zeros(Ksize, order = 'C').astype('double')

        self.GNBP_Collapsed_Deep_Sparse_MP_lib.GNBP_Collapsed_Deep_Sample(ZSDS, ZSWS, n_dot_k, ZS, WS, DS,
                                                                Vsize, Nsize, Ksize, WordNum,
                                                                shape, At, prob_cum_sum)

        return ZSDS, ZSWS, n_dot_k, ZS

    def Po_plus(self, rate):
        r1 = rate[rate >= 1]
        r2 = rate[rate < 1]
        m = np.zeros_like(rate)
        m1 = np.zeros_like(r1)
        m2 = np.zeros_like(r2)

        while True:
            dex = np.nonzero(m1 == 0)  
            if dex[0].size == 0: 
                break
            else:
                r_dex = r1[dex]
                temp = np.random.poisson(r_dex)
                idex = temp > 0
                m1[dex] = np.where(idex, temp, m1[dex])
        m[rate >= 1] = m1

        while True:
            dex = np.nonzero(m2 == 0)
            if dex[0].size == 0:
                break
            else:
                r_dex = r2[dex]
                temp = 1 + np.random.poisson(r_dex)
                idex = np.random.rand(*temp.shape) < (1 / temp)
                m2[dex] = np.where(idex, temp, m2[dex])
        m[rate < 1] = m2

        return m

    def truncated_poisson_rnd(self, lambdas):
        """
        Draw random samples from a truncated Poisson distribution.
        Args:
        - lambdas: array of lambda values for the Poisson distribution

        Returns:
        - x: array of samples from the truncated Poisson distribution
        """
        lambdas = np.asarray(lambdas)
        lambda1 = lambdas[lambdas > 1]
        lambda2 = lambdas[lambdas <= 1]
        
        x = np.zeros(lambdas.shape, dtype=int)
        x1 = np.zeros(lambda1.shape, dtype=int)
        x2 = np.zeros(lambda2.shape, dtype=int)
        
        while True:
            dex = np.where(x1 == 0)[0]
            if dex.size == 0:
                break
            lambdadex = lambda1[dex]
            temp = np.random.poisson(lambdadex)
            idex = temp > 0
            x1[dex[idex]] = temp[idex]
            
        x[lambdas > 1] = x1
        
        while True:
            dex = np.where(x2 == 0)[0]
            if dex.size == 0:
                break
            lambdadex = lambda2[dex]
            temp = 1 + np.random.poisson(lambdadex)
            idex = np.random.rand(len(temp)) < 1.0 / temp
            x2[dex[idex]] = temp[idex]
            
        x[lambdas <= 1] = x2
        
        return x
