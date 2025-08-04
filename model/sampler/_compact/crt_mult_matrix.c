#include<stdio.h>
#include<stdlib.h>
#include<math.h>

int BinarySearch(double probrnd, double *prob_cumsum, int Ksize) {
    int k, kstart, kend;
    if (probrnd <=prob_cumsum[0])
        return(0);
    else {
        for (kstart=1, kend=Ksize-1; ; ) {
            if (kstart >= kend) {
                return(kend);
            }
            else {
                k = kstart+ (kend-kstart)/2;
                if (prob_cumsum[k-1]>probrnd && prob_cumsum[k]>probrnd)
                    kend = k-1;
                else if (prob_cumsum[k-1]<probrnd && prob_cumsum[k]<probrnd)
                    kstart = k+1;
                else
                    return(k);
            }
        }
    }
    return(k);
}

void Crt_Mult_Matrix_Sample(int *ZSDS, int *WSZS, double *Phi, double *Theta, int *pr, int *ir, int *jc, int Vsize, int Nsize, int Ksize,  double *prob_cumsum)
{
    
    double cum_sum, probrnd;
    int k, j, v, token, total=0, table;
    int starting_row_index, stopping_row_index, current_row_index;
    
    for (j=0;j<Nsize;j++) {
        starting_row_index = jc[j];
        stopping_row_index = jc[j+1];
        if (starting_row_index == stopping_row_index)
            continue;
        else {
            for (current_row_index =  starting_row_index; current_row_index<stopping_row_index; current_row_index++) {
                v = ir[current_row_index];
                for (cum_sum=0,k=0; k<Ksize; k++) {
                    cum_sum += Phi[v * Ksize + k]*Theta[k * Nsize + j];
                    prob_cumsum[k] = cum_sum;
                }
                if (pr[total]<0.5)
                    table=0;
                else {
                    for (token=1, table=1;token< (int) pr[total];token++) {
                        if  (((double) rand() / RAND_MAX) <= (cum_sum/(cum_sum+ token)))
                            table++;
                    }
                }
                
                for (token=0;token< table;token++) {
                    probrnd = (double) rand() / RAND_MAX *cum_sum;
                    k = BinarySearch(probrnd, prob_cumsum, Ksize);
                    
                    ZSDS[k * Nsize + j]++;
                    WSZS[v * Ksize + k]++;
                }
                total++;
            }
        }
    }
}
