CC = gcc
CFLAGS = -fPIC -shared -fopenmp -O3
SRCS = gnbp_collapsed_deep.c crt_sum_matrix.c crt_sum.c crt_mult_matrix.c \
	   mult_matrix_fast.c crt.c mult_3mat.c gnbp_collapsed_deep_sparse.c \
	   mult_sparse.c gnbp_collapsed_deep_sparse_mp.c

%.so: %.c
	$(CC) $(CFLAGS) -o $@ $< 

all: $(SRCS:.c=.so)
	@echo "All .so files generated successfully!"