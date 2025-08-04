#include<stdio.h>
#include<stdlib.h>
#include<math.h>

void Crt_Sum_Sample(int Lenx, int *x, double r, double *Lsum){
    int i, j;
    double maxx;
    double *prob;

    for(i = 0, maxx = 0; i < Lenx; i++){
        if (maxx < x[i]) 
            maxx = x[i];
    }
        
    prob = (double *) calloc(maxx, sizeof(double));

    for(i=0; i<maxx; i++){
        prob[i] = r/(r+i);
    }

    for(i=0, Lsum[0]; i<Lenx; i++)
        for(j=0; j<x[i]; j++){
            if((double) rand() <= prob[j]*RAND_MAX) Lsum[0]++;
        }
}