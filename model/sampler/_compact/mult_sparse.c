#include<stdio.h>
#include<stdlib.h>
#include<math.h>

void Mult_Sparse_Sample(double *X1, int *pr, int *ir, int *jc, double *Phi_0, double *Theta_0, int Vsize, int Nsize, int Ksize)
{
    int k, j, v , total = 0;
    int starting_row_index, stopping_row_index, current_row_index;
    double cum_sum;

    for (j = 0; j < Nsize; j++) {
        starting_row_index = jc[j];
        stopping_row_index = jc[j + 1];
        if (starting_row_index == stopping_row_index)
            continue;
        else {
            // 逐个遍历一列的所有非零元素
            for (current_row_index = starting_row_index; current_row_index < stopping_row_index; current_row_index++) {
                v = ir[current_row_index]; // v is the word index （row index of Bow）
                for (cum_sum = 0, k = 0; k < Ksize; k++) {
                    cum_sum += Phi_0[v * Ksize + k] * Theta_0[k * Nsize + j];
                }
                //printf("cum_sum: %f\n", cum_sum);
                //pr[total++] = cum_sum;
                X1[v*Nsize+j] = cum_sum;
            }
        }
    }
}