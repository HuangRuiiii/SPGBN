#include<stdio.h>
#include<stdlib.h>
#include<math.h>

int BinarySearch(double probrnd, double *prob_cumsum, int Ksize){
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


void Mult_3Mat_Sample(int *M_K1L, int *M_K2L, double *Ukk, double *Vkk, double *Lam, 
               int *pr, int *ir, int *jc, int K1size, int K2size, int Lsize, double *prob_cumsum){

    double cum_sum, probrnd;
    int k1, k2, l, token, total=0;
    int starting_row_index, stopping_row_index, current_row_index;

    for(k2=0; k2<K2size; k2++){
        starting_row_index = jc[k2];
        stopping_row_index = jc[k2+1];
        if (starting_row_index == stopping_row_index)
            continue;
        else{
            for(current_row_index=starting_row_index; current_row_index<stopping_row_index; current_row_index++){
                k1 = ir[current_row_index];
                for(cum_sum=0, l=0; l<Lsize; l++){
                    cum_sum += Ukk[k1 * Lsize + l] * Vkk[k2 * Lsize + l] * Lam[l];
                    prob_cumsum[l] = cum_sum;
                }
                for(token=0; token < pr[total]; token++){
                    probrnd = (double) rand()/RAND_MAX * cum_sum;

                    l = BinarySearch(probrnd, prob_cumsum, Lsize);

                    M_K1L[k1 * Lsize + l]++;
                    M_K2L[k2 * Lsize + l]++;
                }
                total++;
            }
        }
    }
}