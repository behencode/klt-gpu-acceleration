######################################################################
# Choose your favorite C compiler
CC = gcc

NVCC ?= nvcc
CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= -arch=sm_75
CFLAGS_GPU = $(CFLAGS) -DKLT_USE_CUDA
NVCCFLAGS = $(CUDA_ARCH) -O3 -DKLT_USE_CUDA $(CFLAGS)
NVCCLDFLAGS = $(CUDA_ARCH)
GPU_LIBS = -L$(CUDA_HOME)/lib64 -lcudart

######################################################################
# OpenACC compiler configuration (V4)
# Supports: GCC with -fopenacc, PGI/NVHPC with -acc, Clang with -fopenacc
ACC_COMPILER ?= gcc
OPENACC_FLAGS = -fopenacc -O3 -DKLT_USE_OPENACC
OPENACC_LIBS = -lm

######################################################################
# -DNDEBUG prevents the assert() statements from being included in 
# the code.  If you are having problems running the code, you might 
# want to comment this line to see if an assert() statement fires.
FLAG1 = -DNDEBUG

######################################################################
# -DKLT_USE_QSORT forces the code to use the standard qsort() 
# routine.  Otherwise it will use a quicksort routine that takes
# advantage of our specific data structure to greatly reduce the
# running time on some machines.  Uncomment this line if for some
# reason you are unhappy with the special routine.
# FLAG2 = -DKLT_USE_QSORT

######################################################################
# Add your favorite C flags here.
CFLAGS = $(FLAG1) $(FLAG2)
CFLAGS_OPENACC = $(OPENACC_FLAGS) $(FLAG1) $(FLAG2)


######################################################################
# There should be no need to modify anything below this line (but
# feel free to if you want).

EXAMPLES = example1.c example2.c example3.c example4.c example5.c
ARCH = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
       storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c
LIB = -L/usr/local/lib -L/usr/lib

ARCH_OBJS = $(ARCH:.c=.o)
GPU_ARCH_OBJS = $(ARCH:.c=.gpu.o)
ACC_ARCH_OBJS = $(ARCH:.c=.acc.o)
GPU_EXAMPLES = example3_gpu
ACC_EXAMPLES = example3_acc

.SUFFIXES:  .c .o .cu .gpu.o .acc.o

all:  lib acc $(EXAMPLES:.c=)

.c.o:
	$(CC) -c $(CFLAGS) $<

%.gpu.o: %.c
	$(CC) -c $(CFLAGS_GPU) $< -o $@

%.acc.o: %.c
	$(CC) -c $(CFLAGS_OPENACC) $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

lib: $(ARCH_OBJS)
	rm -f libklt.a
	ar ruv libklt.a $(ARCH_OBJS)
	rm -f $(ARCH_OBJS)

example1: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example2: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example3: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example4: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example5: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

gpu: libklt_gpu.a $(GPU_EXAMPLES)

acc: libklt_acc.a $(ACC_EXAMPLES)

libklt_gpu.a: $(GPU_ARCH_OBJS) convolve_gpu.o
	rm -f libklt_gpu.a
	ar ruv libklt_gpu.a $(GPU_ARCH_OBJS) convolve_gpu.o
	rm -f $(GPU_ARCH_OBJS) convolve_gpu.o

libklt_acc.a: $(ACC_ARCH_OBJS)
	rm -f libklt_acc.a
	ar ruv libklt_acc.a $(ACC_ARCH_OBJS)
	rm -f $(ACC_ARCH_OBJS)

%_gpu: %.c libklt_gpu.a
	$(NVCC) $(NVCCLDFLAGS) -O3 -DKLT_USE_CUDA -o $@ $< -L. -lklt_gpu $(LIB) -lm $(GPU_LIBS)

%_acc: %.c libklt_acc.a
	$(CC) $(OPENACC_FLAGS) -O3 -o $@ $< -L. -lklt_acc $(LIB) $(OPENACC_LIBS)

run_cpu: example3
	./example3

run_gpu: example3_gpu
	./example3_gpu

run_acc: example3_acc
	./example3_acc

depend:
	makedepend $(ARCH) $(EXAMPLES)

clean:
	rm -f *.o *.a $(EXAMPLES:.c=) $(GPU_EXAMPLES) $(ACC_EXAMPLES) *.tar *.tar.gz libklt.a \
	      feat*.ppm features.ft features.txt




######################################################################
# Profiling targets
######################################################################
libklt_prof.a: $(ARCH)
	$(CC) -c -pg -O2 $(CFLAGS) $(ARCH)
	ar ruv libklt_prof.a $(ARCH:.c=.o)
	rm -f $(ARCH:.c=.o)

example3_prof: libklt_prof.a
	$(CC) -pg -O2 $(CFLAGS) -o example3_prof example3.c -L. -lklt_prof $(LIB) -lm

profile: example3_prof
	@echo "========================================"
	@echo "Running profiling..."
	@echo "========================================"
	./example3_prof
	@echo ""
	@echo "Generating profile report..."
	gprof ./example3_prof gmon.out > profile.txt
	@echo ""
	@echo "Top 50 lines of profile:"
	@head -n 50 profile.txt
	@echo ""
	@echo "Full profile saved to: profile.txt"

clean-profile:
	rm -f gmon.out profile*.txt example3_prof libklt_prof.a
