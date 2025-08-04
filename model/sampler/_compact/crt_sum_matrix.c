#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define MAX(a,b) ((a)>(b) ? a : b)

void Crt_Sum_Matrix(double *r_k, int *pr, int *ir, int *jc, int Ksize, int Nsize, int *Lsum){
    int j, i;
    double maxx;
    double *prob;
    int starting_row_index, stopping_row_index, current_row_index;

    for ( j=0; j<Nsize; j++){
        starting_row_index = jc[j];
        stopping_row_index = jc[j+1];
        if (starting_row_index == stopping_row_index)
            continue;
        else {
            for (current_row_index = starting_row_index; current_row_index<stopping_row_index; current_row_index++) {
                maxx = MAX(maxx, pr[current_row_index]);
            }
            prob = (double *)calloc(maxx, sizeof(double));
            for ( i=0; i<maxx; i++)
                prob[i] = r_k[j]/(r_k[j] + i);
            
            for ( Lsum[j]=0, current_row_index = starting_row_index; current_row_index<stopping_row_index; current_row_index++)
                for( i=0; i<pr[current_row_index]; i++) {
                    if ((double) rand()/RAND_MAX <= prob[i])
                        Lsum[j]++;
                }
        }
    }
}