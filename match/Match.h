/*=============================================================================
# Filename: Match.h
# Author: Bookug Lobert 
# Mail: 1181955272@qq.com
# Last Modified: 2016-10-24 22:55
# Description: find all subgraph-graph mappings between query graph and data graph
=============================================================================*/

//HELP:
//nvcc compile: http://blog.csdn.net/wzk6_3_8/article/details/15501931
//cuda-memcheck:  http://docs.nvidia.com/cuda/cuda-memcheck/#about-cuda-memcheck
//nvprof:  http://blog.163.com/wujiaxing009@126/blog/static/71988399201701310151777?ignoreua
//
//Use 2D array on GPU:
//http://blog.csdn.net/lavorange/article/details/42125029
//
//http://blog.csdn.net/langb2014/article/details/51348523
//to see the memory frequency of device
//nvidia-smi -a -q -d CLOCK | fgrep -A 3 "Max Clocks" | fgrep "Memory"
//
//GPU cache:
//http://blog.csdn.net/langb2014/article/details/51348616

//the visit array of query, can be placed in CPU directly
//选择性过滤候选集, 查询图的搜索顺序应是动态调整的
//要能快速估算每个点的候选集大小：标签频繁度，度数，图结构，或者用gpu启发式地训练、过滤
//BETTER: how about combine BFS and DFS, according to computing resources and GPU memory

#ifndef _MATCH_MATCH_H
#define _MATCH_MATCH_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cassert>

#include "gputimer.h"

#include "../util/Util.h"
#include "../graph/Graph.h"
#include "../io/IO.h"

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}


class Match
{
public:
	Match(Graph* _query, Graph* _data);
	void match(IO& io, unsigned*& final_result, unsigned& result_row_num, unsigned& result_col_num, int*& id_map);
	~Match();

	static void initGPU(int dev);

private:
	Graph* query;
	Graph* data;

    unsigned** candidates;

	int current_pos;
	int* id2pos;
	int* pos2id;
	void add_mapping(int _id);

	int get_minimum_idx(float* score, int qsize);

	unsigned *d_query_vertex_num, *d_query_label_num, *d_query_vertex_value, *d_query_row_offset_in, *d_query_row_offset_out, *d_query_edge_value_in, *d_query_edge_offset_in, *d_query_edge_value_out, *d_query_edge_offset_out, *d_query_column_index_in, *d_query_column_index_out, *d_query_inverse_label, *d_query_inverse_offset, *d_query_inverse_vertex;
	unsigned *d_data_vertex_num, *d_data_label_num, *d_data_vertex_value, *d_data_row_offset_in, *d_data_row_offset_out, *d_data_edge_value_in, *d_data_edge_offset_in, *d_data_edge_value_out, *d_data_edge_offset_out, *d_data_column_index_in, *d_data_column_index_out, *d_data_inverse_label, *d_data_inverse_offset, *d_data_inverse_vertex;

	void copyGraphToGPU();
	void release();
	//candidates placed in GPU, only copy the num back
	bool filter(float* _score, int* _qnum);
	void acquire_linking(int*& link_pos, int*& link_edge, int& link_num, int idx);
	bool join(unsigned* d_summary, int* link_pos, int* link_edge, int link_num, unsigned*& d_result, unsigned* d_candidate, unsigned num, unsigned& result_row_num, unsigned& result_col_num);

	bool score_node(float* _score, int* _qnum);
	void update_score(float* _score, int qsize, int _idx);

    //utilities
    void copyHtoD(unsigned*& d_ptr, unsigned* h_ptr, unsigned bytes);
    void exclusive_sum(unsigned* d_array, unsigned size);
};

#endif //_MATCH_MATCH_H

