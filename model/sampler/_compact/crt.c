#include<stdio.h>
#include<stdlib.h>
#include<math.h>


void Crt_Sample(int *row_dex, int *col_index, int *nz_X, double *nz_R, int Num, int K_tm1, int K_t, int *L){
    int i,j,M;
    double prob;

    for (i = 0; i < Num; i++){

        M = nz_X[i];
        //prob = (double *)calloc(M, sizeof(double));
        for (j = 0; j < M; j++){
            prob = nz_R[i] / (nz_R[i] + j);
            if ((double) rand()/RAND_MAX <= prob)
                L[i]++;
        }
    }
}