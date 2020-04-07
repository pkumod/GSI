/*=============================================================================
# Filename: Graph.h
# Author: Bookug Lobert 
# Mail: 1181955272@qq.com
# Last Modified: 2016-10-24 23:00
# Description: 
=============================================================================*/

//Data Structure:  CSR with 4 arrays, we should use structure of array instead of array of structure
//row offsets
//column indices
//values(label of edge)
//flags(label of vertice)

#ifndef _GRAPH_GRAPH_H
#define _GRAPH_GRAPH_H

#include "../util/Util.h"

class Neighbor
{
public:
	VID vid;
	LABEL elb;
	Neighbor()
	{
		vid = -1;
		elb = -1;
	}
	Neighbor(int _vid, int _elb)
	{
		vid = _vid;
		elb = _elb;
	}
	bool operator<(const Neighbor& _nb) const
	{
		if(this->elb == _nb.elb)
		{
			return this->vid < _nb.vid;
		}
		else
		{
			return this->elb < _nb.elb;
		}
	}
};

class Element
{
public:
	int label;
	int id;
	bool operator<(const Element& _ele) const
	{
		if(this->label == _ele.label)
		{
			return this->id <_ele.id;
		}
		else
		{
			return this->label < _ele.label;
		}
	}
};

class Vertex
{
public:
	//VID id;
	LABEL label;
	//NOTICE:VID and EID is just used in this single graph
	std::vector<Neighbor> in;
	std::vector<Neighbor> out;
	Vertex()
	{
		label = -1;
	}
	Vertex(LABEL lb):label(lb)
	{
	}
};

class PCSR
{
public:
    unsigned* row_offset;  //the size is 32*key_num
    unsigned* column_index;
    unsigned key_num;  //also the group number
    unsigned edge_num;
    PCSR()
    {
        row_offset = NULL;
        column_index = NULL;
        key_num = 0;
        edge_num = 0;
    }
    ~PCSR()
    {
        delete[] row_offset;
        delete[] column_index;
    }
    inline unsigned getEdgeNum() const
    {
        return this->edge_num;
    }
};

class Graph
{
public:
	std::vector<Vertex> vertices;
	void addVertex(LABEL _vlb);
	void addEdge(VID _from, VID _to, LABEL _elb);

    unsigned vertexLabelNum, edgeLabelNum;
	//CSR format: 4 pointers
	unsigned vertex_num;
	unsigned* vertex_value;

    PCSR* csrs_in;
    PCSR* csrs_out;
	//unsigned* row_offset_in;  //range is 0~vertex_num, the final is a border(not valid vertex)
	//unsigned* edge_value_in;
	//unsigned* edge_offset_in;
	//unsigned* column_index_in;
	//unsigned* row_offset_out;
	//unsigned* edge_value_out;
	//unsigned* edge_offset_out;
	//unsigned* column_index_out;
	//Inverse Label List
	//unsigned label_num;
	//unsigned* inverse_label;
	//unsigned* inverse_offset;
	//unsigned* inverse_vertex;

    //signature table
    //column oriented for data graph
    //row oriented for query graph
    unsigned* signature_table;

	Graph() 
	{
		//row_offset_in = row_offset_out = vertex_value = column_index_in = column_index_out = edge_value_in = edge_value_out = NULL;
		//edge_offset_in = edge_offset_out = NULL;
		vertex_num = 0;
		//inverse_label = inverse_offset = inverse_vertex = NULL;
		//label_num = 0;
        signature_table = NULL;
        csrs_in = csrs_out = NULL;
	}
	~Graph() 
	{ 
		delete[] vertex_value;
		//delete[] row_offset_in;
		//delete[] row_offset_out;
		//delete[] column_index_in;
		//delete[] column_index_out;
		//delete[] edge_value_in;
		//delete[] edge_value_out;
		//delete[] edge_offset_in;
		//delete[] edge_offset_out;

		//delete[] inverse_label;
		//delete[] inverse_offset;
		//delete[] inverse_vertex;
        delete[] signature_table;
        delete[] csrs_in;
        delete[] csrs_out;
	}

    void buildPCSR(PCSR* pcsr, std::vector<unsigned>& keys, int label, bool incoming);
	void transformToCSR();
	void preprocessing(bool column_oriented=false);
    void buildSignature(bool column_oriented);
	void printGraph();

	//inline unsigned eSizeIn() const
	//{
		//unsigned in_label_num = this->row_offset_in[vertex_num];
		//return this->edge_offset_in[in_label_num];
	//}
	//inline unsigned eSizeOut() const
	//{
		//unsigned out_label_num = this->row_offset_out[vertex_num];
		//return this->edge_offset_out[out_label_num];
	//}
	//inline unsigned eSize() const
	//{
		//return eSizeIn() + eSizeOut();
	//}
	inline unsigned vSize() const
	{
		return vertex_num;
	}
	//inline void getInNeighbor(unsigned id, unsigned& in_neighbor_num, unsigned& in_neighbor_offset)
	//{
		//in_neighbor_offset = this->edge_offset_in[row_offset_in[id]];
		//in_neighbor_num = this->edge_offset_in[row_offset_in[id+1]] - in_neighbor_offset;
	//}
	//inline void getOutNeighbor(unsigned id, unsigned& out_neighbor_num, unsigned& out_neighbor_offset)
	//{
		//out_neighbor_offset = this->edge_offset_out[row_offset_out[id]];
		//out_neighbor_num = this->edge_offset_out[row_offset_out[id+1]] - out_neighbor_offset;
	//}
    unsigned countMaxDegree();
};

#endif

