#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>

int BinarySearch(double probrnd, double *prob_cumsum, int Ksize) {
    int k, kstart, kend;
    if (probrnd <= prob_cumsum[0])
        return 0;
    else {
        for (kstart = 1, kend = Ksize - 1;;) {
            if (kstart >= kend) {
                return kend;
            } else {
                k = kstart + (kend - kstart) / 2;
                if (prob_cumsum[k - 1] > probrnd && prob_cumsum[k] > probrnd)
                    kend = k - 1;
                else if (prob_cumsum[k - 1] < probrnd && prob_cumsum[k] < probrnd)
                    kstart = k + 1;
                else
                    return k;
            }
        }
    }
    return k;
}

void GNBP_Collapsed_Deep_Sample(int *ZSDS, int *ZSWS, int *n_dot_k,
                                int *ZS, int *WS, int *DS, 
                                int Vsize, int Nsize, int Ksize, int WordNum,
                                double *shape, double eta, double *prob_cumsum) {
    double cum_sum, probrnd;
    int i, k, j, v;
    
    for (i = 0; i < WordNum; i++) {
        v = WS[i];
        j = DS[i];
        k = ZS[i];

        ZSDS[k * Nsize + j]--;
        ZSWS[k * Vsize + v]--;
        n_dot_k[k]--;

        for (cum_sum = 0, k = 0; k < Ksize; k++) {
            cum_sum += (eta + ZSWS[k * Vsize + v]) / (Vsize * eta + n_dot_k[k]) * (ZSDS[k * Nsize + j] + shape[k * Nsize + j]);
            prob_cumsum[k] = cum_sum;
        }

        probrnd = (double)rand() / (double)RAND_MAX * cum_sum;
        k = BinarySearch(probrnd, prob_cumsum, Ksize);
        ZS[i] = k;
        ZSDS[k * Nsize + j]++;
        ZSWS[k * Vsize + v]++;
        n_dot_k[k]++;
    }
}