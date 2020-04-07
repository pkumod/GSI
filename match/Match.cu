/*=============================================================================
# Filename: Match.cpp
# Author: Bookug Lobert 
# Mail: 1181955272@qq.com
# Last Modified: 2016-12-15 01:38
# Description: 
=============================================================================*/

#include <cub/cub.cuh> 
#include "Match.h"

using namespace std;

//on Titan XP the pointer consumes 8 bytes
//it uses little-endian byte order

//Contsant memory and cache
//The constant memory size is 64 KB for compute capability 1.0-3.0 devices. The cache working set is only 8KB
//part of constant memory is used for compiling and kernel executation
//https://stackoverflow.com/questions/10256402/why-is-the-constant-memory-size-limited-in-cuda
//Local Memory  (no limit other than the capacity of global memory)
//https://stackoverflow.com/questions/28810365/amount-of-local-memory-per-cuda-thread
//Shared Memory
//https://devblogs.nvidia.com/using-shared-memory-cuda-cc/
//Read-only cache: 48KB, no L1 cache in Titan XP
//But we should not occupy the whole cache with __ldg()
//When the compiler detects that
//the read-only condition is satisfied for some data, it will use __ldg() to read it. The
//compiler might not always be able to detect that the read-only condition is satisfied
//for some data. Marking pointers used for loading such data with both the const and
//__restrict__ qualifiers increases the likelihood that the compiler will detect the readonly condition.

//NOTICE: if we use too many checkCudaErrors in the program, then it may report error:
//too many resources requested for launch cudaGetLastError()
/*#define DEBUG 1*/
#define MAXIMUM_SCORE 100000000.0f
//! the maximum degree in the query graph, used for linking structures
#define MAX_DEGREE 20
#define SUMMARY_SIZE 2*1024  //2048 unsigneds=8KB
#define SUMMARY_BYTES SUMMARY_SIZE*4 //8KB
#define SUMMARY_BITS SUMMARY_BYTES*8 //8KB=1024*64bits

//GPU上用new/delete大量处理小内存的性能会比较差
//如果中间表实在太大(可能最终结果本身就很多)，那么需要考虑分段或者中间表结构的压缩(是否可以按列存?)
//block或warp内部考虑用共享内存来去重排序并使负载均衡，根据两表大小决定用哈希表还是二分，
//同一时刻只要一个哈希表且可动态生成，并在shared memory中加缓存来优化显存访问

//Constant memory has 64KB cache for each SM
//constant variable is not a pointer, and can not be declared in a function
//no need to alloc or free for constant variables, it is only readable for kernel functions
__constant__ unsigned* c_row_offset;
__constant__ unsigned* c_column_index;
__constant__ unsigned c_key_num;
__constant__ unsigned* c_result_tmp_pos;
__constant__ unsigned* c_result;
__constant__ unsigned* c_candidate;
/*__constant__ unsigned c_candidate_num;*/
__constant__ unsigned c_result_row_num;
__constant__ unsigned c_result_col_num;
/*__constant__ unsigned c_link_num;*/
/*__constant__ unsigned c_link_pos[MAX_DEGREE];*/
/*__constant__ unsigned c_link_edge[MAX_DEGREE];*/
__constant__ unsigned c_link_pos;
__constant__ unsigned c_link_edge;
__constant__ unsigned c_signature[SIGNUM];

void 
Match::initGPU(int dev)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    cudaSetDevice(dev);
	//NOTE: 48KB shared memory per block, 1024 threads per block, 30 SMs and 128 cores per SM
    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %luB; compute v%d.%d; clock: %d kHz; shared mem: %dB; block threads: %d; SM count: %d\n",
               devProps.name, devProps.totalGlobalMem, 
               (int)devProps.major, (int)devProps.minor, 
               (int)devProps.clockRate,
			   devProps.sharedMemPerBlock, devProps.maxThreadsPerBlock, devProps.multiProcessorCount);
    }
	cout<<"GPU selected"<<endl;
	//GPU initialization needs several seconds, so we do it first and only once
	//https://devtalk.nvidia.com/default/topic/392429/first-cudamalloc-takes-long-time-/
	int* warmup = NULL;
	/*unsigned long bigg = 0x7fffffff;*/
	/*cudaMalloc(&warmup, bigg);*/
	/*cout<<"warmup malloc"<<endl;*/
    //NOTICE: if we use nvprof to time the API calls, we will find the time of cudaMalloc() is very long.
    //The reason is that we do not add cudaDeviceSynchronize() here, so it is asynchronously and will include other instructions' time.
    //However, we do not need to add this synchronized function if we do not want to time the API calls
	cudaMalloc(&warmup, sizeof(int));
	cudaFree(warmup);
	cout<<"GPU warmup finished"<<endl;
    //heap corruption for 3 and 4
	/*size_t size = 0x7fffffff;*/    //size_t is unsigned long in x64
    unsigned long size = 0x7fffffff;   //approximately 2G
    /*size *= 3;   */
    size *= 4;
	/*size *= 2;*/
	//NOTICE: the memory alloced by cudaMalloc is different from the GPU heap(for new/malloc in kernel functions)
	/*cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);*/
	cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
	cout<<"check heap limit: "<<size<<endl;

	// Runtime API
	// cudaFuncCachePreferShared: shared memory is 48 KB
	// cudaFuncCachePreferEqual: shared memory is 32 KB
	// cudaFuncCachePreferL1: shared memory is 16 KB
	// cudaFuncCachePreferNone: no preference
	/*cudaFuncSetCacheConfig(MyKernel, cudaFuncCachePreferShared)*/
	//The initial configuration is 48 KB of shared memory and 16 KB of L1 cache
	//The maximum L2 cache size is 3 MB.
	//also 48 KB read-only cache: if accessed via texture/surface memory, also called texture cache;
	//or use _ldg() or const __restrict__
	//4KB constant memory, ? KB texture memory. cache size?
	//CPU的L1 cache是根据时间和空间局部性做出的优化，但是GPU的L1仅仅被设计成针对空间局部性而不包括时间局部性。频繁的获取L1不会导致某些数据驻留在cache中，只要下次用不到，直接删。
	//L1 cache line 128B, L2 cache line 32B, notice that load is cached while store not
	//mmeory read/write is in unit of a cache line
	//the word size of GPU is 32 bits
    //Titan XP uses little-endian byte order
}

//the seed is a prime, which can be well chosed to yield good performance(low conflicts)
__device__ uint32_t 
MurmurHash2(const void * key, int len, uint32_t seed) 
{
    // 'm' and 'r' are mixing constants generated offline.
    // They're not really 'magic', they just happen to work well.
    const uint32_t m = 0x5bd1e995;
    const int r = 24;
    // Initialize the hash to a 'random' value
    uint32_t h = seed ^ len;
    // Mix 4 bytes at a time into the hash
    const unsigned char * data = (const unsigned char *) key;
    while (len >= 4) 
    {
        uint32_t k = *(uint32_t*) data;
        k *= m;
        k ^= k >> r;
        k *= m;
        h *= m;
        h ^= k;
        data += 4;
        len -= 4;
    }
    // Handle the last few bytes of the input array
    switch (len) 
    {
        case 3:
            h ^= data[2] << 16;
        case 2:
            h ^= data[1] << 8;
        case 1:
          h ^= data[0];
          h *= m;
    };
    // Do a few final mixes of the hash to ensure the last few
    // bytes are well-incorporated.
    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;
    return h;
}


//NOTICE: below is for smid, detecting running on which SM.
#define DEVICE_INTRINSIC_QUALIFIERS   __device__ __forceinline__

DEVICE_INTRINSIC_QUALIFIERS
unsigned int
smid()
{
  unsigned int r;
  asm("mov.u32 %0, %%smid;" : "=r"(r));
  return r;
}

DEVICE_INTRINSIC_QUALIFIERS
unsigned int
nsmid()
{
#if (__CUDA_ARCH__ >= 200)
  unsigned int r;
  asm("mov.u32 %0, %%nsmid;" : "=r"(r));
  return r;
#else
  return 30;
#endif
}


void 
Match::copyHtoD(unsigned*& d_ptr, unsigned* h_ptr, unsigned bytes)
{
    unsigned* p = NULL;
    cudaMalloc(&p, bytes);
    cudaMemcpy(p, h_ptr, bytes, cudaMemcpyHostToDevice);
    d_ptr = p;
    checkCudaErrors(cudaGetLastError());
}

void Match::exclusive_sum(unsigned* d_array, unsigned size)
{
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL; //must be set to distinguish two phase
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_array, d_array, size);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_array, d_array, size);
    cudaFree(d_temp_storage);
}


Match::Match(Graph* _query, Graph* _data)
{
	this->query = _query;
	this->data = _data;
	id2pos = pos2id = NULL;
}

Match::~Match()
{
	delete[] this->id2pos;
}

inline void 
Match::add_mapping(int _id)
{
	pos2id[current_pos] = _id;
	id2pos[_id] = current_pos;
	this->current_pos++;
}

//! \fn get_minimum_idx(float* score, int qsize)
//! \brief
//! \note nodes are selected based on connectivity(the next node should be linked to at least one mapped node)
int
Match::get_minimum_idx(float* score, int qsize)
{
    float* min_ptr = NULL;
    float minscore = FLT_MAX;
    //choose the start node based on score
    if(this->current_pos == 0)
    {
        min_ptr = min_element(score, score+qsize);
        minscore = *min_ptr;
    }

    for(int i = 0; i < this->current_pos; ++i)
    {
        int id = this->pos2id[i];
        int insize = this->query->vertices[id].in.size(), outsize = this->query->vertices[id].out.size();
        for(int j = 0; j < insize; ++j)
        {
            int id2 = this->query->vertices[id].in[j].vid;
            if(score[id2] < minscore)
            {
                minscore = score[id2];
                min_ptr = score+id2;
            }
        }
        for(int j = 0; j < outsize; ++j)
        {
            int id2 = this->query->vertices[id].out[j].vid;
            if(score[id2] < minscore)
            {
                minscore = score[id2];
                min_ptr = score+id2;
            }
        }
    }
	int min_idx = min_ptr - score;
    //set this ID to maximum so it will not be chosed again
	memset(min_ptr, 0x7f, sizeof(float));
	/*thrust::device_ptr<float> dev_ptr(d_score);*/
	/*float* min_ptr = thrust::raw_pointer_cast(thrust::min_element(dev_ptr, dev_ptr+qsize));*/
	/*int min_idx = min_ptr - d_score;*/
	/*//set this node's score to maximum so it won't be chosed again*/
	/*cudaMemset(min_ptr, 0x7f, sizeof(float));*/

	//NOTICE: memset is used per-byte, so do not set too large value, otherwise it will be negative
	//http://blog.csdn.net/Vmurder/article/details/46537613
	/*cudaMemset(d_score+min_idx, 1000.0f, sizeof(float));*/
	/*float tmp = 0.0f;*/
	/*cout<<"to check the score: ";*/
	/*for(int i = 0; i < qsize; ++i)*/
	/*{*/
		/*cudaMemcpy(&tmp, d_score+i, sizeof(float), cudaMemcpyDeviceToHost);*/
		/*cout<<tmp<<" ";*/
	/*}cout<<endl;*/
#ifdef DEBUG
	checkCudaErrors(cudaGetLastError());
#endif

	this->add_mapping(min_idx);
	return min_idx;
}

void
Match::copyGraphToGPU()
{
    //BETTER: we may not include this time for final comparison because it only needs once

	/*cout<<"to copy graph"<<endl;*/
	//cudaMemcpyFromSymbol    cudaMemcpy + cudaGetSymbolAddress
	/*cudaMemcpyToSymbol(c_data_row_offset_in, &d_data_row_offset_in, sizeof(unsigned*));*/
	/*cudaMemcpyToSymbol(c_data_edge_value_in, &d_data_edge_value_in, sizeof(unsigned*));*/
	/*cudaMemcpyToSymbol(c_data_edge_offset_in, &d_data_edge_offset_in, sizeof(unsigned*));*/
	/*cudaMemcpyToSymbol(c_data_column_index_in, &d_data_column_index_in, sizeof(unsigned*));*/
	/*cudaMemcpyToSymbol(c_data_row_offset_out, &d_data_row_offset_out, sizeof(unsigned*));*/
	/*cudaMemcpyToSymbol(c_data_edge_value_out, &d_data_edge_value_out, sizeof(unsigned*));*/
	/*cudaMemcpyToSymbol(c_data_edge_offset_out, &d_data_edge_offset_out, sizeof(unsigned*));*/
	/*cudaMemcpyToSymbol(c_data_column_index_out, &d_data_column_index_out, sizeof(unsigned*));*/
#ifdef DEBUG
	/*cout<<"data graph already in GPU"<<endl;*/
	checkCudaErrors(cudaGetLastError());
#endif
}

__host__ unsigned
binary_search_cpu(unsigned _key, unsigned* _array, unsigned _array_num)
{
	//MAYBE: return the right border, or use the left border and the right border as parameters
    if (_array_num == 0 || _array == NULL)
    {
		return INVALID;
    }

    unsigned _first = _array[0];
    unsigned _last = _array[_array_num - 1];

    if (_last == _key)
    {
        return _array_num - 1;
    }

    if (_last < _key || _first > _key)
    {
		return INVALID;
    }

    unsigned low = 0;
    unsigned high = _array_num - 1;
    unsigned mid;
    while (low <= high)
    {
        mid = (high - low) / 2 + low;   //same to (low+high)/2
        if (_array[mid] == _key)
        {
            return mid;
        }
        if (_array[mid] > _key)
        {
            high = mid - 1;
        }
        else
        {
            low = mid + 1;
        }
    }
	return INVALID;
}

//BETTER: maybe we can use dynamic parallism here
__device__ unsigned
binary_search(unsigned _key, unsigned* _array, unsigned _array_num)
{
//http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#host
/*#if defined(__CUDA_ARCH__)*/
	//MAYBE: return the right border, or use the left border and the right border as parameters
    if (_array_num == 0 || _array == NULL)
    {
		return INVALID;
    }

    unsigned _first = _array[0];
    unsigned _last = _array[_array_num - 1];

    if (_last == _key)
    {
        return _array_num - 1;
    }

    if (_last < _key || _first > _key)
    {
		return INVALID;
    }

    unsigned low = 0;
    unsigned high = _array_num - 1;
    unsigned mid;
    while (low <= high)
    {
        mid = (high - low) / 2 + low;   //same to (low+high)/2
        if (_array[mid] == _key)
        {
            return mid;
        }
        if (_array[mid] > _key)
        {
            high = mid - 1;
        }
        else
        {
            low = mid + 1;
        }
    }
	return INVALID;
/*#else*/
/*#endif*/
}


__host__ float
compute_score(int size)
{
	return 0.0f +size;
}

bool
Match::score_node(float* _score, int* _qnum)
{
	bool success = true;
	for(int i = 0; i < this->query->vertex_num; ++i)
	{
		//BETTER: consider degree and substructure in the score
		if(_qnum[i] == 0)  // not found
		{
			/*d_score[i] = -1.0f;*/
			success = false;
            break;
		}
		else
		{
			_score[i] = compute_score(_qnum[i]);
			/**d_success = true;*/
		}
	}
	return success;
}

__global__ void
filter_kernel(unsigned* d_signature_table, unsigned* d_status, unsigned dsize)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= dsize)
	{
		return; 
	}
    unsigned flag = 1;
    //TODO+DEBUG: the first vertex label, should be checked via a==b
    for(int j = 0; j < SIGNUM; ++j)
    {
        unsigned usig = c_signature[j];
        unsigned vsig = d_signature_table[dsize*j+i];
        //BETTER: reduce memory access here?
        if(flag)
        {
            //WARN: usig&vsig==usig is not right because the priority of == is higher than bitwise operation
            flag = ((usig & vsig) == usig)?1:0;
            //WARN: below is wrong because usig may have many 1s
            /*flag = ((usig & vsig) != 0)?1:0;*/
        }
    }
    d_status[i] = flag;
}

__global__ void
scatter_kernel(unsigned* d_status, unsigned* d_cand, unsigned dsize)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= dsize)
	{
		return; 
	}
    int pos = d_status[i];
    if(pos != d_status[i+1])
    {
        d_cand[pos] = i;
    }
}

//NOTICE: the performance of this function varies sharply under the same configuration.
//The reason may be the terrible implementation of exclusive_scan in Thrust library!
bool
Match::filter(float* _score, int* _qnum)
{
    int qsize = this->query->vertex_num, dsize = this->data->vertex_num;
    this->candidates = new unsigned*[qsize];
    int bytes = dsize * SIGBYTE;
    unsigned* d_signature_table = NULL;
    cudaMalloc(&d_signature_table, bytes);
    cudaMemcpy(d_signature_table, this->data->signature_table, bytes, cudaMemcpyHostToDevice);

    unsigned* d_status = NULL;
    cudaMalloc(&d_status, sizeof(unsigned)*(dsize+1));
    int BLOCK_SIZE = 1024;
	int GRID_SIZE = (dsize+BLOCK_SIZE-1)/BLOCK_SIZE;
    for(int i = 0; i < qsize; ++i)
    {
        //store the signature of query graph there
        cudaMemcpyToSymbol(c_signature, this->query->signature_table+SIGNUM*i, SIGBYTE);
        filter_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_signature_table, d_status, dsize);
	checkCudaErrors(cudaGetLastError());
        cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
    //NOTICE: the speed of CUB is much better than Thrust: single-pass, shared mem, multiple schemes(device,block,warp-wide)
    //while Thrust has problems: register spills, little usage of shared mem, low occupancy, low scalability
    /*long t1 = Util::get_cur_time();*/
        /*thrust::device_ptr<unsigned> dev_ptr(d_status);*/
        /*thrust::exclusive_scan(dev_ptr, dev_ptr+dsize+1, dev_ptr);*/
    exclusive_sum(d_status, dsize+1);
	checkCudaErrors(cudaGetLastError());
    /*long t2 = Util::get_cur_time();*/
    /*cout<<"prefix sum scan used: "<<t2-t1<<" ms"<<endl;*/

        cudaMemcpy(&_qnum[i], d_status+dsize, sizeof(unsigned), cudaMemcpyDeviceToHost);
        if(_qnum[i] == 0)
        {
            break;
        }
        unsigned* d_cand = NULL;
        cudaMalloc(&d_cand, sizeof(unsigned)*_qnum[i]);
        scatter_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_status, d_cand, dsize);
	checkCudaErrors(cudaGetLastError());
        cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
        this->candidates[i] = d_cand;
    }
    cudaFree(d_status);
	cudaFree(d_signature_table);

	//get the num of candidates and compute scores
	bool success = score_node(_score, _qnum);
	if(!success)
	{
#ifdef DEBUG
		cout<<"query already fail after filter"<<endl;
#endif
		return false;
	}

	return true;
}

//BETTER: use shared memory to reduce conflicts
//However, the existing atomic operations are already fast here
__global__ void
candidate_kernel(unsigned* d_candidate, unsigned* d_candidate_tmp, unsigned candidate_num)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    //a program in kernel function to test if the given GPU architecture is little endian
    //the lowest byte(not bit) is in the lowest memory address
    /*if(i == 0)*/
    /*{*/
        /*unsigned t = 0x01020304;*/
        /*char* p = (char*)&t;*/
        /*printf("check endian: %d %d %d %d\n", *p, *(p+1), *(p+2), *(p+3));*/
    /*}*/
	if(i >= candidate_num)
	{
		return; 
	}
    //int atomicOr(int* address, int val);
    unsigned ele = d_candidate_tmp[i];
    unsigned num = ele >> 5;
    ele &= 0x1f;
    //NOTICE: we assume that the architecture is little-endian
    //in big-endian it should be 1<<(32-ele)
    //ANALYSIS: it may be ok no matter the machine is little-endian or not, if we use the same type of read/write data
    ele = 1 << ele;
    atomicOr(d_candidate+num, ele);
}

void
Match::update_score(float* _score, int qsize, int _idx)
{
	//BETTER: acquire it from edge label frequence: p = (P2num)/T, divide in and out edge?
	//score the node or the edge?? how about m*n*p, cost of the current step and the join result's size(cost of the next step)
	float p = 0.9f;
	/*float p = 0.1f;*/
	/*float p = 0.5f;*/
    int insize = this->query->vertices[_idx].in.size(), outsize = this->query->vertices[_idx].out.size();
	int i, j;
	for(i = 0; i < insize; ++i)  //in neighbor
	{
		j = this->query->vertices[_idx].in[i].vid;
		_score[j] *= p;
	}
	for(i = 0; i < outsize; ++i)  //out neighbor
	{
		j = this->query->vertices[_idx].out[i].vid;
		_score[j] *= p;
	}
}

void
Match::acquire_linking(int*& link_pos, int*& link_edge, int& link_num, int idx)
{
	vector<int> tmp_vertex, tmp_edge;
	int i, qsize = this->query->vertex_num;
    int insize = this->query->vertices[idx].in.size(), outsize = this->query->vertices[idx].out.size();

	//BETTER: deal with parallel edge
    //WARN: currently, for parallel edge(general meaning) only the last is dealed
	int* edge2value = new int[qsize];
	memset(edge2value, -1, sizeof(int)*qsize);
	for(i = 0; i < insize; ++i)
	{
		int label = this->query->vertices[idx].in[i].elb;
		int vid = this->query->vertices[idx].in[i].vid;
        edge2value[vid] = label;
	}
	for(i = 0; i < this->current_pos; ++i)
	{
		int id = this->pos2id[i];
		int label = edge2value[id];
		if(label != -1)
		{
			tmp_vertex.push_back(i);
			tmp_edge.push_back(label);
		}
	}

	memset(edge2value, -1, sizeof(int)*qsize);
	for(i = 0; i < outsize; ++i)
	{
		int label = this->query->vertices[idx].out[i].elb;
		int vid = this->query->vertices[idx].out[i].vid;
        edge2value[vid] = label;
	}
	for(i = 0; i < this->current_pos; ++i)
	{
		int id = this->pos2id[i];
		int label = edge2value[id];
		if(label != -1)
		{
			tmp_vertex.push_back(i);
			tmp_edge.push_back(0 - label);
		}
	}

	delete[] edge2value;
	link_num = tmp_vertex.size();
	link_pos = new int[link_num];
	link_edge = new int[link_num];
	for(i = 0; i <link_num; ++i)
	{
		link_pos[i] = tmp_vertex[i];
		link_edge[i] = tmp_edge[i];
	}
}

//int *result = new int[1000];
/*int *result_end = thrust::set_intersection(A1, A1 + size1, A2, A2 + size2, result, thrust::less<int>());*/
//
//BETTER: choose between merge-join and bianry-search, or using multiple threads to do intersection
//or do inetrsection per-element, use compact operation finally to remove invalid elements
__device__ void
intersect(unsigned*& cand, unsigned& cand_num, unsigned* list, unsigned list_num)
{
	int i, cnt = 0;
	for(i = 0; i < cand_num; ++i)
	{
        unsigned key = cand[i];
		unsigned found = binary_search(key, list, list_num);
		if(found != INVALID)
		{
			cand[cnt++] = key;
		}
	}
	cand_num = cnt;
}

__device__ void
subtract(unsigned*& cand, unsigned& cand_num, unsigned* record, unsigned result_col_num)
{
    //DEBUG: this will cause error when using dynamic allocation, gowalla with q0
	int i, j, cnt = 0;
    for(j = 0; j < cand_num; ++j)
    {
        unsigned key = cand[j];
        for(i = 0; i < result_col_num; ++i)
        {
            if(record[i] == key)
            {
                break;
            }
        }
        if(i == result_col_num)
        {
            cand[cnt++] = key;
        }
    }
	cand_num = cnt;
}

//WARN: in case of 2-node loops like: A->B and B->A (this can be called generalized parallel edge)
//BETTER: implement warp-binary-search method
__global__ void
first_kernel(unsigned* d_result_tmp_pos)
{
    //NOTICE: if a shared structure has not be used really, the compiler(even without optimization options) will not assign space for it on SM
    //the three pools request 12KB for each block
    __shared__ unsigned s_pool1[1024];
    __shared__ unsigned s_pool2[1024];
    /*__shared__ unsigned s_pool3[1024];*/
    //WARN: for unsigned variable, never use >0 and -- to judge!(overflow)
    //NOTICE: for signed type, right shift will add 1s in the former!
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned idx = i & 0x1f;
    i = i >> 5; //group ID
	/*printf("compare %d and %d\n", i, result_row_num);*/
	if(i >= c_result_row_num)
	{
		return; 
	}

	/*printf("thread id %d\n", i);*/
	unsigned* record = c_result+i*c_result_col_num;
    /*unsigned test = __ldg(&d_result_tmp_num[0]);*/
    //NOTICE: we use this function to verify that on Titan XP the pointer consumes 8 bytes
    /*if(i == 0)*/
    /*{*/
        /*printf("check pointer %u\n", sizeof(record));*/
    /*}*/

    unsigned id = record[c_link_pos];
    //NOTICE: here we should use the group number within the block instead of grid
    unsigned bgroup = threadIdx.x & 0xffffffe0;  //equal to (x/32)*32

    //BETTER: control the whole block, share the inputs
    //find the same ID within the block

    unsigned bucket = MurmurHash2(&id, 4, HASHSEED) % c_key_num;
    s_pool1[bgroup+idx] = c_row_offset[32*bucket+idx];
    if(idx == 0)
    {
        s_pool2[bgroup] = INVALID;
        s_pool2[bgroup+1] = INVALID;
    }
    if(idx < 30 && (idx&1)==0)
    {
        if(s_pool1[bgroup+idx] == id)
        {
            s_pool2[bgroup] = s_pool1[bgroup+idx+1];
            s_pool2[bgroup+1] = s_pool1[bgroup+idx+3];
        }
    }
    /*if(pool2[bgroup*32] == INVALID && pool1[32*bgroup+30] != INVALID)*/
    /*{*/
        /*//TODO:multiple groups*/
    /*}*/
    //NOTICE: not use all threads to write, though the conflicts do not matter
    //(experiments show the gst number is the same, and the performance is similar)
    if(idx == 0)
    {
        d_result_tmp_pos[i] = s_pool2[bgroup+1] - s_pool2[bgroup];
    }
}

/*__device__ unsigned d_maxTaskLen;*/
/*__device__ unsigned d_minTaskLen;*/
//NOTICE: though the registers reported by compiler are reduced when using constant memory, 
//the real registers used when running may not be reduced so much.
//when using constants, they are kept for a whole block instead of occupying registers for each thread
//NOTICE: we can hack the mechanism by comparing with first_kernel, compiler and running(nvprof --print-gpu-trace)
__global__ void
second_kernel(unsigned* d_result_tmp, unsigned* d_result_tmp_num)
/*second_kernel(unsigned* d_result_tmp, unsigned* d_result_tmp_num, const unsigned* __restrict__ d_summary)*/
{
    __shared__ unsigned s_pool1[1024];
    __shared__ unsigned s_pool2[1024];
    __shared__ unsigned s_pool3[1024];
    __shared__ unsigned s_pool4[32];
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned idx = i & 0x1f;
    i = i >> 5; //group ID(warp index) within the whole kernel
	if(i >= c_result_row_num)
	{
		return; 
	}

    unsigned bgroup = threadIdx.x & 0xffffffe0;  //equal to (x/32)*32
    unsigned gidx = threadIdx.x >> 5;  //warp index within this block
    //NOTICE:we assume the size of record <= 32
    //cache the record of this row in shared mem
    if(idx < c_result_col_num)
    {
        s_pool2[bgroup+idx] = c_result[i*c_result_col_num+idx];
    }
    unsigned bucket = MurmurHash2(&s_pool2[bgroup+c_link_pos], 4, HASHSEED) % c_key_num;
    s_pool1[bgroup+idx] = c_row_offset[32*bucket+idx];
    if(idx == 0)
    {
        s_pool3[bgroup] = INVALID;
    }
    if(idx < 30 && (idx&1)==0)
    {
        if(s_pool1[bgroup+idx] == s_pool2[bgroup+c_link_pos])
        {
            s_pool3[bgroup] = s_pool1[bgroup+idx+1];
            s_pool3[bgroup+1] = s_pool1[bgroup+idx+3];
        }
    }
    /*if(pool2[bgroup*32] == INVALID && pool1[32*bgroup+30] != INVALID)*/
    /*{*/
        /*//TODO:multiple groups*/
    /*}*/
    if(s_pool3[bgroup] == INVALID)  // not found
    {
        if(idx == 0)
        {
            d_result_tmp_num[i] = 0;
        }
        return;
    }

    //BETTER:we may set d_result_tmp_num+=i here and use i for other usage later
    d_result_tmp += c_result_tmp_pos[i];
    unsigned size = s_pool3[bgroup+1] - s_pool3[bgroup];
    unsigned* list = c_column_index + s_pool3[bgroup];
    unsigned pos = 0;
    unsigned loop = size >> 5;
    size = size & 0x1f;
    unsigned pred, presum;
    unsigned cand_num = 0;
    s_pool4[gidx] = 0;
    //BETTER: include the remainings in the loop body by a simple judgement
    for(int j = 0; j < loop; ++j, pos+=32)
    {
        s_pool1[bgroup+idx] = list[pos+idx];
        unsigned k;
        for(k = 0; k < c_result_col_num; ++k)
        {
            if(s_pool2[bgroup+k] == s_pool1[bgroup+idx])
            {
                break;
            }
        }
        pred = 0;
        if(k == c_result_col_num)
        {
            unsigned num = s_pool1[bgroup+idx] >> 5;
            unsigned res = s_pool1[bgroup+idx] & 0x1f;
            res = 1 << res;
            if((c_candidate[num] & res) == res)
            {
                pred = 1;
            }
        }
        //BETTER: use shared mem for reduce/prefix-sum to save registers
        presum = pred;
        //prefix sum in a warp to find positions
        for(unsigned stride = 1; stride < 32; stride <<= 1)
        {
            //NOTICE: this must be called by the whole warp, not placed in the judgement
            unsigned tmp = __shfl_up(presum, stride);
            if(idx >= stride)
            {
                presum += tmp;
            }
        }
        //this must be called first, only in inclusive-scan the 31-th element is the sum
        unsigned total = __shfl(presum, 31);  //broadcast to all threads in the warp
        //transform inclusive prefixSum to exclusive prefixSum
        presum = __shfl_up(presum, 1);
        //NOTICE: for the first element, the original presum value is copied
        if(idx == 0)
        {
            presum = 0;
        }
        //write to corresponding position
        //NOTICE: warp divergence exists(even we use compact, the divergence also exists in the compact operation)
        if(pred == 1)
        {
            if(s_pool4[gidx]+presum < 32)
            {
                s_pool3[bgroup+s_pool4[gidx]+presum] = s_pool1[bgroup+idx];
            }
        }
        //flush 128B: one 4-segment writes is better than four 1-segment writes
        if(s_pool4[gidx]+total >= 32)
        {
            d_result_tmp[cand_num+idx] = s_pool3[bgroup+idx];
            cand_num += 32;
            if(pred == 1)
            {
                unsigned pos = s_pool4[gidx] + presum;
                if(pos>=32)
                {
                    s_pool3[bgroup+pos-32] = s_pool1[bgroup+idx];
                }
            }
            s_pool4[gidx] = s_pool4[gidx] + total - 32;
        }
        else
        {
            //NOTICE:for a warp this is ok due to SIMD feature: sync read and sync write
            s_pool4[gidx] += total;
        }
    }
    presum = pred = 0; //init all threads to 0s first because later there is a judgement
    if(idx < size)
    {
        s_pool1[bgroup+idx] = list[pos+idx];
        unsigned k;
        for(k = 0; k < c_result_col_num; ++k)
        {
            if(s_pool2[bgroup+k] == s_pool1[bgroup+idx])
            {
                break;
            }
        }
        if(k == c_result_col_num)
        {
            unsigned num = s_pool1[bgroup+idx] >> 5;
            unsigned res = s_pool1[bgroup+idx] & 0x1f;
            res = 1 << res;
            if((c_candidate[num] & res) == res)
            {
                pred = 1;
            }
        }
        presum = pred;
    }
    for(unsigned stride = 1; stride < 32; stride <<= 1)
    {
        unsigned tmp = __shfl_up(presum, stride);
        if(idx >= stride)
        {
            presum += tmp;
        }
    }
    unsigned total = __shfl(presum, 31);  //broadcast to all threads in the warp
    presum = __shfl_up(presum, 1);
    if(idx == 0)
    {
        presum = 0;
    }
    if(pred == 1)
    {
        if(s_pool4[gidx]+presum < 32)
        {
            s_pool3[bgroup+s_pool4[gidx]+presum] = s_pool1[bgroup+idx];
        }
    }
    unsigned newsize = s_pool4[gidx] + total;
    if(newsize >= 32)
    {
        d_result_tmp[cand_num+idx] = s_pool3[bgroup+idx];
        cand_num += 32;
        if(pred == 1)
        {
            unsigned pos = s_pool4[gidx] + presum;
            if(pos>=32)
            {
                d_result_tmp[cand_num+pos-32] = s_pool1[bgroup+idx];
            }
        }
        cand_num += (newsize - 32);
    }
    else
    {
        if(idx < newsize)
        {
            d_result_tmp[cand_num+idx] = s_pool3[bgroup+idx];
        }
        cand_num += newsize;
    }

    if(idx == 0)
    {
        d_result_tmp_num[i] = cand_num;
    }
}

__global__ void
join_kernel(unsigned* d_result_tmp, unsigned* d_result_tmp_num)
{
    __shared__ unsigned s_pool1[1024];
    __shared__ unsigned s_pool2[1024];
    __shared__ unsigned s_pool3[1024];
    __shared__ unsigned s_pool4[32];
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned idx = i & 0x1f;
    i = i >> 5; //group ID
	if(i >= c_result_row_num)
	{
		return; 
	}

	unsigned res_num = d_result_tmp_num[i];
    //NOTICE: though invalid rows exist, but a warp will end directly here and not occupy resource any more(no divergence)
    if(res_num == 0)   //early termination
    {
        return;
    }
    unsigned bgroup = threadIdx.x & 0xffffffe0;  //equal to (x/32)*32
    unsigned gidx = threadIdx.x >> 5;
    if(idx == 0)
    {
        s_pool2[bgroup+c_link_pos] = c_result[i*c_result_col_num+c_link_pos];
    }
    unsigned bucket = MurmurHash2(&s_pool2[bgroup+c_link_pos], 4, HASHSEED) % c_key_num;
    s_pool1[bgroup+idx] = c_row_offset[32*bucket+idx];
    if(idx == 0)
    {
        s_pool3[bgroup] = INVALID;
    }
    if(idx < 30 && (idx&1)==0)
    {
        if(s_pool1[bgroup+idx] == s_pool2[bgroup+c_link_pos])
        {
            s_pool3[bgroup] = s_pool1[bgroup+idx+1];
            s_pool3[bgroup+1] = s_pool1[bgroup+idx+3];
        }
    }
    /*if(pool2[bgroup*32] == INVALID && pool1[32*bgroup+30] != INVALID)*/
    /*{*/
        /*//TODO:multiple groups*/
    /*}*/
    if(s_pool3[bgroup] == INVALID)  // not found
    {
        if(idx == 0)
        {
            d_result_tmp_num[i] = 0;
        }
        return;
    }

    unsigned list_num = s_pool3[bgroup+1] - s_pool3[bgroup];
    unsigned* list = c_column_index + s_pool3[bgroup];
    d_result_tmp += c_result_tmp_pos[i];
    //BETTER: select strategy here, (m,n)  n < 32mlog2(n)
    unsigned pos1 = 0, pos2 = 0;
    unsigned pred, presum;
    unsigned cand_num = 0;
    int choice = 0;
    s_pool4[gidx] = 0;
    //NOTICE: we need to store results in d_result_tmp, so we can not search in d_result_tmp because it is destroyed and rebuilt.(on the other hand, d_result_tmp may be smaller than list)
    //WARN: not use res_num-31 here, because it may be overflow for unsigned to be negative
    while(pos1 < res_num && pos2 < list_num)
    {
        if(choice <= 0)
        {
            s_pool1[bgroup+idx] = INVALID;
            if(pos1 + idx < res_num)
            {
                s_pool1[bgroup+idx] = d_result_tmp[pos1+idx];
            }
        }
        if(choice >= 0)
        {
            s_pool2[bgroup+idx] = INVALID;
            if(pos2 + idx < list_num)
            {
                s_pool2[bgroup+idx] = list[pos2+idx];
            }
        }
        pred = 0;  //some threads may fail in the judgement below
        unsigned valid1 = (pos1+32<res_num)?32:(res_num-pos1);
        unsigned valid2 = (pos2+32<list_num)?32:(list_num-pos2);
        if(pos1 + idx < res_num)
        {
            pred = binary_search(s_pool1[bgroup+idx], s_pool2+bgroup, valid2);
            if(pred != INVALID)
            {
                pred = 1;
            }
            else
            {
                pred = 0;
            }
        }
        presum = pred;
        for(unsigned stride = 1; stride < 32; stride <<= 1)
        {
            unsigned tmp = __shfl_up(presum, stride);
            if(idx >= stride)
            {
                presum += tmp;
            }
        }
        unsigned total = __shfl(presum, 31);  //broadcast to all threads in the warp
        presum = __shfl_up(presum, 1);
        if(idx == 0)
        {
            presum = 0;
        }
        if(pred == 1)
        {
            if(s_pool4[gidx]+presum < 32)
            {
                s_pool3[bgroup+s_pool4[gidx]+presum] = s_pool1[bgroup+idx];
            }
        }
        if(s_pool4[gidx]+total >= 32)
        {
            d_result_tmp[cand_num+idx] = s_pool3[bgroup+idx];
            cand_num += 32;
            if(pred == 1)
            {
                unsigned pos = s_pool4[gidx] + presum;
                if(pos>=32)
                {
                    s_pool3[bgroup+pos-32] = s_pool1[bgroup+idx];
                }
            }
            s_pool4[gidx] = s_pool4[gidx] + total - 32;
        }
        else
        {
            s_pool4[gidx] += total;
        }

        //set the next movement
        choice = s_pool1[bgroup+valid1-1] - s_pool2[bgroup+valid2-1];
        if(choice <= 0)
        {
            pos1 += 32;
        }
        if(choice >= 0)
        {
            pos2 += 32;
        }
    }
    if(idx < s_pool4[gidx])
    {
        d_result_tmp[cand_num+idx] = s_pool3[bgroup+idx];
    }
    cand_num += s_pool4[gidx];

    if(idx == 0)
    {
        d_result_tmp_num[i] = cand_num;
    }
}

//NOTICE: the load balance strategies of Merrill is wonderful, but it may fail when comparing with
//natural-balanced strategies because they do not need the work of preparation(which must be done to ensure balance)
__global__ void
link_kernel(unsigned* d_result_tmp, unsigned* d_result_tmp_pos, unsigned* d_result_tmp_num, unsigned* d_result_new)
{
    //BETTER:consider bank conflicts here, should we use column-oriented table for global memory and shared memory?
    //In order to keep in good occupancy(>=50%), the shared mem usage should <= 24KB for 1024-threads block
    __shared__ unsigned cache[1024];
    /*__shared__ unsigned s_pool[1024*5];  //the work poll*/
    //NOTICE: though a block can be synchronized, we should use volatile to ensure data is not cached in private registers
    //If shared mem(or global mem) is used by a warp, then volatile is not needed.
    //http://www.it1352.com/539600.html
    /*volatile __shared__ unsigned swpos[1024];*/
    __shared__ unsigned swpos[32];

	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    i >>= 5;
    //NOTICE: we should not use this if we want to control the whole block
    //(another choice is to abandon the border block)
	/*if(i >= c_result_row_num)*/
	/*{*/
		/*return; */
	/*}*/
    unsigned bgroup = threadIdx.x & 0xffffffe0;  //equal to (x/32)*32
    unsigned idx = threadIdx.x & 0x1f;  //thread index within the warp
    unsigned gidx = threadIdx.x >> 5; //warp ID within the block

    unsigned tmp_begin = 0, start = 0, size = 0;
    if(i < c_result_row_num)
    {
        tmp_begin = d_result_tmp_pos[i];
        start = d_result_tmp_num[i];
        //NOTICE: the size is ok to be 0 here
        size = d_result_tmp_num[i+1] - start;
        start *= (c_result_col_num+1);
    }

    //Usage of Shared Memory: cache records only when size > 0
    if(idx == 0)
    {
        if(size > 0)
        {
            //NOTICE: we use a single thread to read a batch a time
            memcpy(cache+gidx*32, c_result+i*c_result_col_num, sizeof(unsigned)*c_result_col_num);
        }
    }
    unsigned curr = 0;
    unsigned* record = cache + gidx * 32;

    //Usage of Load Balance
    __syncthreads();
    //use a block to deal with tasks >=1024
    while(true)
    {
        if(threadIdx.x == 0)
        {
            swpos[0] = INVALID;
        }
        //NOTICE: the sync function is needed, but had better not use it too much 
        //It is costly, which may stop the running warps and replace them with other warps
        __syncthreads();
        if(size >= curr+1024)
        {
            swpos[0] = gidx;
        }
        __syncthreads();
        if(swpos[0] == INVALID)
        {
            break;
        }
        //WARN:output info within kernel function will degrade perfomance heavily
        //printf("FOUND: use a block!\n");
        unsigned* ptr = cache + 32 * swpos[0];
        if(swpos[0] == gidx)
        {
            swpos[1] = tmp_begin;
            swpos[2] = start;
            swpos[3] = curr;
            swpos[4] = size;
        }
        __syncthreads();
        //NOTICE: here we use a block to handle the task as much as possible
        //(this choice may save the work of preparation)
        //Another choice is only do 1024 and set size-=1024, later vote again
        while(swpos[3]+1023 <swpos[4])
        {
            unsigned pos = (c_result_col_num+1)*(swpos[3]+threadIdx.x);
            memcpy(d_result_new+swpos[2]+pos, ptr, sizeof(unsigned)*c_result_col_num);
            d_result_new[swpos[2]+pos+c_result_col_num] = d_result_tmp[swpos[1]+swpos[3]+threadIdx.x];
            if(threadIdx.x == 0)
            {
                swpos[3] += 1024;
            }
            __syncthreads();
        }
        if(swpos[0] == gidx)
        {
            curr = swpos[3];
        }
        __syncthreads();
    }
    __syncthreads();

    //combine the tasks of rows and divide equally
    //NOTICE: though we can combine even when the tasks of some row is very small, it is not good.
    //(the time of combining may be consuming compared to using exactly a warp for each row, when the size is nearly 32)

    while(curr < size)
    {
        //this judgement is fine, only causes divergence in the end
        if(curr+idx < size)
        {
            unsigned pos = (c_result_col_num+1)*(curr+idx);
            memcpy(d_result_new+start+pos, record, sizeof(unsigned)*c_result_col_num);
            d_result_new[start+pos+c_result_col_num] = d_result_tmp[tmp_begin+curr+idx];
        }
        curr += 32;
    }
    //BETTER: the implementation of memcpy() may be optimized for single thread with batch read/write
    //using a struct representing more bytes? or use vload4
}

//BETTER: use async memcpy and event?
bool
Match::join(unsigned* d_summary, int* link_pos, int* link_edge, int link_num, unsigned*& d_result, unsigned* d_candidate, unsigned num, unsigned& result_row_num, unsigned& result_col_num)
{
	/*int *d_link_pos, *d_link_edge;*/
	/*cudaMalloc(&d_link_pos, sizeof(int)*link_num);*/
	/*cudaMemcpy(d_link_pos, link_pos, sizeof(int)*link_num, cudaMemcpyHostToDevice);*/
	/*cudaMalloc(&d_link_edge, sizeof(int)*link_num);*/
	/*cudaMemcpy(d_link_edge, link_edge, sizeof(int)*link_num, cudaMemcpyHostToDevice);*/

	unsigned sum;
	unsigned* d_result_tmp = NULL;
	unsigned* d_result_tmp_pos = NULL;
	unsigned* d_result_tmp_num = NULL;
	cudaMalloc(&d_result_tmp_pos, sizeof(unsigned)*(result_row_num+1));
	cudaMalloc(&d_result_tmp_num, sizeof(unsigned)*(result_row_num+1));
#ifdef DEBUG
	checkCudaErrors(cudaGetLastError());
#endif

	cudaMemcpyToSymbol(c_result, &d_result, sizeof(unsigned*));
	cudaMemcpyToSymbol(c_candidate, &d_candidate, sizeof(char*));
	/*cudaMemcpyToSymbol(c_candidate_num, &num, sizeof(unsigned));*/
	cudaMemcpyToSymbol(c_result_row_num, &result_row_num, sizeof(unsigned));
	cudaMemcpyToSymbol(c_result_col_num, &result_col_num, sizeof(unsigned));

    /*int BLOCK_SIZE = 256;*/
    int BLOCK_SIZE = 1024;
    int GRID_SIZE = (result_row_num*32+BLOCK_SIZE-1)/BLOCK_SIZE;
    //NOTICE: too large block, and too many registers per block, may cause this kernel function unable to start
    //In addition, we should adjust the size of blocks to let as many blocks as possible residing on a SM
#ifdef DEBUG
        cout<<"now to do join kernel "<<result_row_num<<" "<<result_col_num<<" "<<GRID_SIZE<<" "<<BLOCK_SIZE<<endl;
#endif
    //NOTICE: we ensure that link_num > 0
	long begin = Util::get_cur_time();
    for(int i = 0; i < link_num; ++i)
    {
        cudaMemcpyToSymbol(c_link_pos, link_pos+i, sizeof(unsigned));
        int label = link_edge[i];
        unsigned *d_row_offset = NULL, *d_column_index = NULL;
        PCSR* tcsr;
        if(label < 0)
        {
            label = -label;
            tcsr = &(this->data->csrs_in[label]);
        }
        else
        {
            tcsr = &(this->data->csrs_out[label]);
        }
        copyHtoD(d_row_offset, tcsr->row_offset, sizeof(unsigned)*(tcsr->key_num*32));
        copyHtoD(d_column_index, tcsr->column_index, sizeof(unsigned)*(tcsr->getEdgeNum()));
        cudaMemcpyToSymbol(c_row_offset, &d_row_offset, sizeof(unsigned*));
        cudaMemcpyToSymbol(c_column_index, &d_column_index, sizeof(unsigned*));
        cudaMemcpyToSymbol(c_key_num, &(tcsr->key_num), sizeof(unsigned));
        cudaMemcpyToSymbol(c_link_edge, &label, sizeof(unsigned));
        cudaMemcpyToSymbol(c_result_tmp_pos, &d_result_tmp_pos, sizeof(unsigned*));
        cout<<"the "<<i<<"-th edge"<<endl;

        //BETTER: handle infrequent edge first to lower the size of d_result_tmp
        if(i == 0)
        {
            first_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_result_tmp_pos);
            cudaDeviceSynchronize();
            checkCudaErrors(cudaGetLastError());
            cout<<"first kernel finished"<<endl;

            /*thrust::device_ptr<unsigned> dev_ptr(d_result_tmp_pos);*/
            /*thrust::exclusive_scan(dev_ptr, dev_ptr+result_row_num+1, dev_ptr);*/
            exclusive_sum(d_result_tmp_pos, result_row_num+1);

            cudaMemcpy(&sum, &d_result_tmp_pos[result_row_num], sizeof(unsigned), cudaMemcpyDeviceToHost);
            cout<<"To malloc on GPU: "<<sizeof(unsigned)*sum<<endl;
            assert(sum < 2000000000);  //keep the bytes < 8GB
            cudaMalloc(&d_result_tmp, sizeof(unsigned)*sum);
            checkCudaErrors(cudaGetLastError());
            second_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_result_tmp, d_result_tmp_num);
            /*unsigned* h_result_tmp = new unsigned[sum];*/
            /*cudaMemcpy(h_result_tmp, d_result_tmp, sizeof(unsigned)*sum, cudaMemcpyDeviceToHost);*/
            /*for(int p = 0; p < sum; ++p)*/
            /*{*/
                /*cout<<h_result_tmp[p]<<" ";*/
            /*}*/
            /*cout<<endl;*/
        }
        else
        {
            join_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_result_tmp, d_result_tmp_num);
        }
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
        cout<<"iteration kernel finished"<<endl;
        cudaFree(d_row_offset);
        cudaFree(d_column_index);
    }
	long end = Util::get_cur_time();
	cerr<<"join_kernel used: "<<(end-begin)<<"ms"<<endl;
#ifdef DEBUG
	cout<<"join kernel finished"<<endl;
#endif

	/*thrust::device_ptr<unsigned> dev_ptr(d_result_tmp_num);*/
	/*//link the temp result into a new table*/
	/*thrust::exclusive_scan(dev_ptr, dev_ptr+result_row_num+1, dev_ptr);*/
    exclusive_sum(d_result_tmp_num, result_row_num+1);
#ifdef DEBUG
	checkCudaErrors(cudaGetLastError());
#endif
	/*sum = thrust::reduce(dev_ptr, dev_ptr+result_row_num);*/
	cudaMemcpy(&sum, d_result_tmp_num+result_row_num, sizeof(unsigned), cudaMemcpyDeviceToHost);
	//BETTER: judge if success here
	/*cout<<"new table num: "<<sum<<endl;*/
	/*int tmp = 0;*/
	/*for(int i = 0; i < result_row_num; ++i)*/
	/*{*/
		/*cudaMemcpy(&tmp, d_result_tmp_num+i, sizeof(int), cudaMemcpyDeviceToHost);*/
		/*cout<<"check tmp: "<<tmp<<endl;*/
	/*}*/
#ifdef DEBUG
	checkCudaErrors(cudaGetLastError());
#endif

	unsigned* d_result_new = NULL; 
	if(sum > 0)
	{
		cudaMalloc(&d_result_new, sizeof(unsigned)*sum*(result_col_num+1));
#ifdef DEBUG
		checkCudaErrors(cudaGetLastError());
#endif
		/*BLOCK_SIZE = 512;*/
		/*GRID_SIZE = (result_row_num+BLOCK_SIZE-1)/BLOCK_SIZE;*/
		//BETTER?: combine into a large array(value is the record id) and link per element
		long begin = Util::get_cur_time();
		link_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_result_tmp, d_result_tmp_pos, d_result_tmp_num, d_result_new);
		checkCudaErrors(cudaGetLastError());
		cudaDeviceSynchronize();
		long end = Util::get_cur_time();
#ifdef DEBUG
		cerr<<"link_kernel used: "<<(end-begin)<<"ms"<<endl;
#endif
#ifdef DEBUG
		checkCudaErrors(cudaGetLastError());
#endif
	}
    //if the original result table is exactly the first candidate set, then here also delete it
    cudaFree(d_result);
#ifdef DEBUG
	checkCudaErrors(cudaGetLastError());
#endif
	d_result = d_result_new;

	cudaFree(d_result_tmp);  
	cudaFree(d_result_tmp_pos);  
	cudaFree(d_result_tmp_num);  
#ifdef DEBUG
	checkCudaErrors(cudaGetLastError());
#endif
	result_col_num++;
	result_row_num = sum;

	if(result_row_num == 0)
	{
		return false;
	}
	else
	{
		return true;
	}
}

__global__ void
bloom_kernel(unsigned* d_array, unsigned candidate_num, unsigned* d_summary)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= candidate_num)
	{
		return; 
	}
    unsigned id = d_array[i];

    unsigned pos = MurmurHash2(&id, 4, HASHSEED) % SUMMARY_BITS;
    unsigned num = pos >> 5;
    pos &= 0x1f;
    pos = 1 << pos;
    atomicOr(d_summary+num, pos);

    pos = MurmurHash2(&id, 4, HASHSEED2) % SUMMARY_BITS;
    num = pos >> 5;
    pos &= 0x1f;
    pos = 1 << pos;
    atomicOr(d_summary+num, pos);
}

void 
Match::match(IO& io, unsigned*& final_result, unsigned& result_row_num, unsigned& result_col_num, int*& id_map)
{
//NOTICE: device variables can not be assigned and output directly on Host
/*unsigned maxTaskLen = 0, minTaskLen = 1000000;*/
/*cudaMemcpyToSymbol(d_maxTaskLen, &maxTaskLen, sizeof(unsigned));*/
/*cudaMemcpyToSymbol(d_minTaskLen, &minTaskLen, sizeof(unsigned));*/
//CUDA device variable (can only de declared on Host, not in device/global functions), can be used in all kernel functions like constant variables
/*https://blog.csdn.net/rong_toa/article/details/78664902*/
/*cudaGetSymbolAddress((void**)&dp,devData);*/
/*cudaMemcpy(dp,&value,sizeof(float),cudaMemcpyHostToDevice);*/

	long t0 = Util::get_cur_time();
	copyGraphToGPU();
	long t1 = Util::get_cur_time();
	cerr<<"copy graph used: "<<(t1-t0)<<"ms"<<endl;
#ifdef DEBUG
	cout<<"graph copied to GPU"<<endl;
#endif

	int qsize = this->query->vertex_num;
    assert(qsize <= 12);
	float* score = new float[qsize];
	/*float* d_score = NULL;*/
	/*cudaMalloc(&d_score, sizeof(float)*qsize);*/
	/*checkCudaErrors(cudaGetLastError());*/
	/*cout<<"assign score"<<endl;*/

	int* qnum = new int[qsize+1];
	/*int* d_qnum = NULL;*/
	/*cudaMalloc(&d_qnum, sizeof(int)*(qsize+1));*/
	/*checkCudaErrors(cudaGetLastError());*/
	/*cout<<"assign d_qnum"<<endl;*/

	/*cout<<"to filter"<<endl;*/
	bool success = filter(score, qnum);
	long t2 = Util::get_cur_time();
	cout<<"filter used: "<<(t2-t1)<<"ms"<<endl;
    for(int i = 0; i < qsize; ++i)
    {
        cout<<qnum[i]<<" ";
    }cout<<endl;

#ifdef DEBUG
	cout<<"filter finished"<<endl;
#endif
	if(!success)
	{
		delete[] score;
		delete[] qnum;
		result_row_num = 0;
		result_col_num = qsize;
		final_result = NULL;
		release();
		return; 
	}

    unsigned bitset_size = sizeof(unsigned) * Util::RoundUpDivision(this->data->vertex_num, sizeof(unsigned)*8);
    cout<<"data vertex num: "<<this->data->vertex_num<<" bitset size: "<<bitset_size<<"B"<<endl;
    //NOTICE: the bitset is very large, we should only keep one set at a time
    unsigned* d_candidate = NULL;  //candidate bitset
    cudaMalloc(&d_candidate, bitset_size);
    checkCudaErrors(cudaGetLastError());

    //Below is for summary of candidates to be placed in low-latency cache
    //However, when trying to place them in read-only cache, no performance gain.
    //BETTER: try constant cache?(<8KB cache)
    unsigned* d_summary = NULL;
    /*cudaMalloc(&d_summary, SUMMARY_BYTES);  //alloc on bytes*/
    /*checkCudaErrors(cudaGetLastError());*/

	/*cudaFree(d_data_inverse_label);*/
	/*cudaFree(d_data_inverse_offset);*/
	/*cudaFree(d_data_inverse_vertex);*/
#ifdef DEBUG
	checkCudaErrors(cudaGetLastError());
	cout<<"candidates prepared"<<endl;
#endif
	long t3 = Util::get_cur_time();
	cerr<<"build candidates used: "<<(t3-t2)<<"ms"<<endl;

	//initialize the mapping structure
	this->id2pos = new int[qsize];
	this->pos2id = new int[qsize];
	this->current_pos = 0;
	memset(id2pos, -1, sizeof(int)*qsize);
	memset(pos2id, -1, sizeof(int)*qsize);
	//select the minium score and fill the table
	int idx = this->get_minimum_idx(score, qsize);
	cout<<"start node found: "<<idx<<" "<<this->query->vertex_value[idx]<<" candidate size: "<<qnum[idx]<<endl;

	//intermediate table of join results
	result_row_num = qnum[idx];
	result_col_num = 1;
	unsigned* d_result = this->candidates[idx];  
	cout<<"intermediate table built"<<endl;

	//NOTICE: the query graph is not so large, so we can analyse the join order in CPU(or use GPU for help)
	//each step build a new one and release the older
	for(int step = 1; step < qsize; ++step)
	{
/*#ifdef DEBUG*/
		cout<<"this is the "<<step<<" round"<<endl;
/*#endif*/

        long t4 = Util::get_cur_time();
        // update the scores of query nodes
        update_score(score, qsize, idx);
        long t5 = Util::get_cur_time();
        cerr<<"update score used: "<<(t5-t4)<<"ms"<<endl;
        int idx2 = this->get_minimum_idx(score, qsize);
        long t6 = Util::get_cur_time();
        cerr<<"get minimum idx used: "<<(t6-t5)<<"ms"<<endl;
    /*#ifdef DEBUG*/
        cout<<"next node to join: "<<idx2<<" "<<this->query->vertex_value[idx2]<<" candidate size: "<<qnum[idx2]<<endl;
    /*#endif*/

		//acquire the edge linkings on CPU, and pass to GPU
		int *link_pos, *link_edge, link_num;
		this->acquire_linking(link_pos, link_edge, link_num, idx2);
        long t7 = Util::get_cur_time();
        cerr<<"acquire linking used: "<<(t7-t6)<<"ms"<<endl;

        long tmp1 = Util::get_cur_time();
        //build the bitset
		checkCudaErrors(cudaGetLastError());
        cudaMemset(d_candidate, 0, bitset_size);
		checkCudaErrors(cudaGetLastError());
		int candidate_num = qnum[idx2];
        candidate_kernel<<<Util::RoundUpDivision(candidate_num, 1024), 1024>>>(d_candidate, this->candidates[idx2], candidate_num);
		checkCudaErrors(cudaGetLastError());
        cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
        long tmp2 = Util::get_cur_time();
        cout<<"candidate kernel used: "<<(tmp2-tmp1)<<"ms"<<endl;

        //build summary which is placed in read-only cache: the summary is groups of 8B=64 bits
        /*cudaMemset(d_summary, 0, SUMMARY_BYTES);   //NOTICE: this is needed for each iteration*/
        //METHOD 1: bloom filter, two hash functions using MurmurHash2 with two seeds
        /*bloom_kernel<<<Util::RoundUpDivision(candidate_num, 1024), 1024>>>(this->candidates[idx2], candidate_num, d_summary);*/
        /*cudaDeviceSynchronize();*/
        //METHOD 2: compressed bitmap
        /*unsigned width = Util::RoundUpDivision(bitset_size, SUMMARY_BYTES);*/
        /*bitmap_kernel<<<Util::RoundUpDivision(bitset_size, 1024), 1024>>>(d_candidate, bitset_size, d_summary);*/
        //METHOD 3: Interval Summary
		/*checkCudaErrors(cudaGetLastError());*/
        /*long tmp3 = Util::get_cur_time();*/
        /*cout<<"build summary used: "<<(tmp3-tmp2)<<"ms"<<endl;*/

        cudaFree(this->candidates[idx2]);
		checkCudaErrors(cudaGetLastError());
		//join the intermediate table with a candidate list
        //BETTER: use segmented join if the table is too large!
		success = this->join(d_summary, link_pos, link_edge, link_num, d_result, d_candidate, candidate_num, result_row_num, result_col_num);

		delete[] link_pos;
		delete[] link_edge;
#ifdef DEBUG
		checkCudaErrors(cudaGetLastError());
#endif
		if(!success)
		{
			break;
		}
		idx = idx2;
		cout<<"intermediate table: "<<result_row_num<<" "<<result_col_num<<endl;
	}

#ifdef DEBUG
	cout<<"join process finished"<<endl;
#endif
	//NOTICE: new int[] pointer in GPU is different from cudaMalloc and can not be released by cudaFree
	//We choose to use one-dimension array instead of two dimension
    cudaFree(d_candidate);
    /*checkCudaErrors(cudaFree(d_summary));*/

	long t8 = Util::get_cur_time();
	//transfer the result to CPU and output
	if(success)
	{
		final_result = new unsigned[result_row_num * result_col_num];
		cudaMemcpy(final_result, d_result, sizeof(unsigned)*result_col_num*result_row_num, cudaMemcpyDeviceToHost);
	}
	else
	{
		final_result = NULL;
		result_row_num = 0;
		result_col_num = qsize;
	}
#ifdef DEBUG
	checkCudaErrors(cudaGetLastError());
#endif
	checkCudaErrors(cudaFree(d_result));
	long t9 = Util::get_cur_time();
	cerr<<"copy result used: "<<(t9-t8)<<"ms"<<endl;
#ifdef DEBUG
	checkCudaErrors(cudaGetLastError());
#endif
	id_map = this->id2pos;

	delete[] score;
	delete[] qnum;
	release();

    //NOTICE: device variables can not be assigned and output directly on Host
    /*cudaMemcpyFromSymbol(&maxTaskLen, d_maxTaskLen, sizeof(unsigned));*/
    /*cudaMemcpyFromSymbol(&minTaskLen, d_minTaskLen,  sizeof(unsigned));*/
    /*cudaDeviceSynchronize();*/
    /*cout<<"Maximum and Minimum task size: "<<maxTaskLen<<" "<<minTaskLen<<endl;*/
}

void
Match::release()
{
	delete[] this->pos2id;
    delete[] this->candidates;
#ifdef DEBUG
	checkCudaErrors(cudaGetLastError());
#endif
}

