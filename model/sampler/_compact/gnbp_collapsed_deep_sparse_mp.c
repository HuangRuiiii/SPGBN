#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<omp.h>  
#include<time.h> 

int BinarySearch(double probrnd, double *prob_cumsum, int Ksize) {
    int left = 0;
    int right = Ksize - 1;
    
    if (probrnd <= prob_cumsum[0])
        return 0;

    while (left < right) {
        int mid = left + (right - left) / 2;
        
        if (prob_cumsum[mid] < probrnd) {
            left = mid + 1;
        } else {
            if (mid == 0 || prob_cumsum[mid-1] < probrnd)
                return mid;
            right = mid - 1;
        }
    }
    
    return left;
}

void GNBP_Collapsed_Deep_Sample(int *ZSDS, int *ZSWS, int *n_dot_k,
                               int *ZS, int *WS, int *DS, 
                               int Vsize, int Nsize, int Ksize, int WordNum,
                               double *shape, double *At, double *prob_cumsum) {

    unsigned int seed = time(NULL);
    int max_threads = omp_get_max_threads();
    double **local_prob_cumsum = (double**)malloc(max_threads * sizeof(double*));

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        local_prob_cumsum[thread_id] = (double*)malloc(Ksize * sizeof(double));

        //#pragma omp master
        //printf("Using %d threads for sampling\n", omp_get_num_threads());
    }
    
    #pragma omp parallel for schedule(dynamic, 1000)
    for (int i = 0; i < WordNum; i++) {
        int thread_id = omp_get_thread_num();
        double *my_prob_cumsum = local_prob_cumsum[thread_id];
        
        int v = WS[i];
        int j = DS[i];
        int k = ZS[i];
        double cum_sum = 0.0;
        
        #pragma omp atomic
        ZSDS[k * Nsize + j]--;
        
        #pragma omp atomic
        ZSWS[k * Vsize + v]--;
        
        #pragma omp atomic
        n_dot_k[k]--;
        
        // compute cumulative probabilities
        for (k = 0; k < Ksize; k++) {
            double denominator = Vsize * At[v*Ksize + k] + n_dot_k[k];
            if (denominator <= 0) denominator = 1e-10; 
            
            double term1 = (At[v * Ksize + k] + ZSWS[k * Vsize + v]) / denominator;
            double term2 = (ZSDS[k * Nsize + j] + shape[k * Nsize + j]);
            
            cum_sum += term1 * term2;
            my_prob_cumsum[k] = cum_sum;
        }
        
        if (cum_sum <= 0) {
            ZS[i] = rand() % Ksize; 
        } else {
            unsigned int my_seed = seed + i + thread_id;
            srand(my_seed);
            double probrnd = ((double)rand() / RAND_MAX) * cum_sum;

            k = BinarySearch(probrnd, my_prob_cumsum, Ksize);
            ZS[i] = k;
        }
        
        k = ZS[i];
        #pragma omp atomic
        ZSDS[k * Nsize + j]++;
        
        #pragma omp atomic
        ZSWS[k * Vsize + v]++;
        
        #pragma omp atomic
        n_dot_k[k]++;
    }
    
    for (int t = 0; t < max_threads; t++) {
        free(local_prob_cumsum[t]);
    }
    free(local_prob_cumsum);
}