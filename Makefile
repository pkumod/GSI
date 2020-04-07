# nvprof
# https://docs.nvidia.com/cuda/profiler-users-guide/index.html#metrics-reference-3x
# nvprof -m gst_transactions ./GSI.exe data/triangle.g data/triangle.g >& prof.log
# nvprof -m gld_transactions ./GSI.exe data/triangle.g data/triangle.g >& prof.log
# nvprof -m gld_efficiency
# nvprof -m gld_throughput
# nvprof -m branch_efficiency
# nvprof --print-gpu-trace
#
# https://www.researchgate.net/post/How_do_you_get_a_detailed_profile_of_CUDA_kernel
# https://devtalk.nvidia.com/default/topic/536277/visual-profiler/-calculating-gst_throughput-and-gld_throughput-with-nvprof/

# cuda-gdb
# According to the doc: http://docs.nvidia.com/cuda/cuda-memcheck/index.html#compilation-options
#
# "The stack backtrace feature of the CUDA-MEMCHECK tools is more useful when the application contains function symbol names. For the host backtrace, this varies based on the host OS. On Linux, the host compiler must be given the -rdynamic option to retain function symbols." 
#
# single stepping
# https://devtalk.nvidia.com/default/topic/1046228/cuda-gdb/cuda-single-stepping-and-threads/

#默认情况下，L1是被开启的，-Xptxas -dlcm=cg可以用来禁用L1
#after Maxwell, L1 is replaced by read-only cache(also called texture cache), whose unit is 32B
# To open L1: -Xptxas -dlcm=ca
# https://blog.csdn.net/langb2014/article/details/51348636
# https://www.cnblogs.com/neopenx/p/4643705.html
# Formal Reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

#compile parameters
# -ggdb3
#  https://blog.csdn.net/luotuo44/article/details/38090623/
#  debug macros: info macro, macro
#  debug thrust: https://github.com/thrust/thrust/wiki/Debugging

CC = g++
#opt-in to caching of global memory accesses in L1 via the -Xptxas -dlcm=ca option to nvcc
#intermediate files, using --keep for nvcc
#for PTX codes, using --ptx for nvcc
NVCC = nvcc -arch=sm_35 -lcudadevrt -rdc=true -G --ptxas-options=-v -lineinfo -Xcompiler -rdynamic -I ~/cudaToolkit/cub-1.8.0/
#NVCC = nvcc -arch=sm_35 -lcudadevrt -rdc=true -G -Xcompiler -rdynamic -lineinfo
#CFLAGS = -g -c #-fprofile-arcs -ftest-coverage -coverage #-pg
#EXEFLAG = -g #-fprofile-arcs -ftest-coverage -coverage #-pg #-O2
#CFLAGS = -g -c #-fprofile-arcs -ftest-coverage -coverage #-pg
#EXEFLAG = -g #-fprofile-arcs -ftest-coverage -coverage #-pg #-O2
#NVCC = nvcc -arch=sm_35 -lcudadevrt -rdc=true 
#CFLAGS = -c #-fprofile-arcs -ftest-coverage -coverage #-pg
CFLAGS = -c -O2 #-fprofile-arcs -ftest-coverage -coverage #-pg
EXEFLAG = -O2 #-fprofile-arcs -ftest-coverage -coverage #-pg #-O2
# BETTER: try -fno-builtin-strlen -funswitch-loops -finline-functions

#add -lreadline -ltermcap if using readline or objs contain readline
library = #-lgcov -coverage

objdir = ./objs/
objfile = $(objdir)Util.o $(objdir)IO.o $(objdir)Match.o $(objdir)Graph.o

all: GSI.exe

GSI.exe: $(objfile) main/run.cpp
	$(NVCC) $(EXEFLAG) -o GSI.exe main/run.cpp $(objfile)

$(objdir)Util.o: util/Util.cpp util/Util.h
	$(CC) $(CFLAGS) util/Util.cpp -o $(objdir)Util.o

$(objdir)Graph.o: graph/Graph.cpp graph/Graph.h
	$(CC) $(CFLAGS) graph/Graph.cpp -o $(objdir)Graph.o

$(objdir)IO.o: io/IO.cpp io/IO.h
	$(CC) $(CFLAGS) io/IO.cpp -o $(objdir)IO.o

$(objdir)Match.o: match/Match.cu match/Match.h
	$(NVCC) $(CFLAGS) match/Match.cu -o $(objdir)Match.o

.PHONY: clean dist tarball test sumlines doc

clean:
	rm -f $(objdir)*
dist: clean
	rm -f *.txt *.exe
	rm -f *.g
	rm -f cuda-memcheck.*

tarball:
	tar -czvf gsi.tar.gz main util match io graph Makefile README.md objs

test: main/test.o $(objfile)
	$(CC) $(EXEFLAG) -o test main/test.cpp $(objfile) $(library)

sumline:
	bash script/sumline.sh

doc: 
	doxygen 
	#cd document/latex/;make

