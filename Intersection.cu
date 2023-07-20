#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "kernel.cuh"
#include "until.cpp"
#include <vector>
#include <time.h>

#define IDX2C(i,j,ld)  (((j)*(ld))+(i))
//将x的第y位置1
#define setbit(x,y) x|=(1<<y)
//将x的第y位置0
#define clrbit(x,y) x&=!(1<<y)
#define getbit(x,y) (x>>(y))&1
class Intersection {
public:
	// intersection num
	int list_num;
	// pointers pointing to each array
	int** arrptr;
	// size of each list
	int* list_size;
	// order of each list
	int* order;
	// inverted hash
	std::vector<std::vector<int>*>  inverted_map;
	// use two long int as hash bins.
	EncodeVector* ev;
	// matrix
//	bool* A_vector;
//	bool** A_matrix;
//	bool** B_matrix;
//	bool** R_matrix;
	//

	Intersection (std::vector<std::vector<int>* >* p_arr, int* vector_size,int num, int* index){
		list_num = num;
		arrptr = new int* [num];
		for (int i = 0; i < list_num;++i){
			arrptr[i] =  p_arr->at(i)->data();
		}
		list_size = vector_size;
		order = index;
	}

	void MemoryCPU2GPU ();
	int CPU_buildVectors();

	// cuBLAS
	int blasIntersection ();
	// WMMA
	int WMMAIntersection();
	// binary search
	int binaryIntersection();
	// bitwise join cpu-version
	int CPUBitwiseIntersection();

	int CPUSTLintersection();

};

int Intersection:: blasIntersection(){
	 // none binary data type
	return 0;
}

int Intersection:: WMMAIntersection(){
	//
	size_t hash_bins = 256;
	// build matrix
	//compute A: from order 0
//	for(int i = 0; i< ;++i)
	//filtering
	//compute by binary search
}

int Intersection:: CPU_buildVectors(){
	ev = new EncodeVector [list_num];
	//initate inverted hash;
	inverted_map.reserve (128*list_num);
	for(int i = 0; i < 128*list_num ; ++ i ){
		auto pv = new std::vector <int>();
		inverted_map.push_back(pv);
	}
	//
	for (int i = 0; i < list_num; ++ i){
		for (int j = 0; j < list_size[i]; ++ j){
			int val = arrptr[i][j];
			unsigned int hashval = hash( val, 2*64);
//			std::cout << "val: " << val<< " ";
//			std::cout << "hashval: " << hashval;
//			std::cout << std::endl;
			if(hashval >= 64 ){
				//set l1 hashval 1
				setbit(ev[i].l1,hashval-64);
				//insert to inverted hash
				inverted_map[i*128 + hashval]->push_back(val);
			}else{
				//set l2 hashval 1
				setbit(ev[i].l2,hashval);
				//insert to inverted hash
				inverted_map[i*128 + hashval]->push_back(val);
			}
		}
	}


}

int Intersection:: binaryIntersection(){

}

int Intersection:: CPUBitwiseIntersection(){
	long t1 = clock();
	CPU_buildVectors();
	//bit-wise and for all arrays
	EncodeVector ev_result;
	for (int i = 0; i< list_num; ++i ){
//		std::cout << "i-th " << i <<" ";
//		std::cout << "high num of "<< ev[i].l1 << " ";
		auto m = getbit(ev[i].l2,1);
//		std::cout << "low num of "<< ev[i].l2<<" " << m<<std::endl;

		if(i==0){
			ev_result.l1 = ev[i].l1;
			ev_result.l2 = ev[i].l2;
		}else{
			ev_result.l1 &= ev[i].l1;
			ev_result.l2 &= ev[i].l2;
		}
//		std::cout << "result l1 " << ev_result.l1;
//		std::cout << "result l2 " << ev_result.l2<<std::endl;

	}
	// check how many 0 are left
//	std::cout << "num of bits: ";
//	std::cout << countSetBits(ev_result.l1) + countSetBits(ev_result.l2) <<std::endl;
	int count = countSetBits(ev_result.l1) + countSetBits(ev_result.l2);
	//for each pos has an 1 in it
	long t2 = clock();
	std::vector<int> res;
	for(size_t t = 0; t<128 ;++t){
		bool bit = 0;
		if(count == 0 ){
			break;
		}
		if(t >= 64 ){
			bit = getbit (ev_result.l1,t-64);
		}else{
			bit = getbit (ev_result.l2,t);
		}
//		std::cout <<"i-th bit: "<< t<<" val: "<< bit <<std::endl;
		if(bit == 1){
			count --;
			//compute a subset intersection of orignal one
			res = *(inverted_map[t]);
//			std::cout <<"list_num: "<< list_num <<std::endl;
			for(int k = 1; k < list_num; ++k){
				res = intersect(res,*(inverted_map[t+128*k]) );
			}
		}
//		std:: cout << "partial res_num: " << res.size() << std::endl;
//		for (auto i =0; i< res.size(); ++i){
//			std::cout<< "partial results: " << res[i] <<std::endl;
//		}
	}
	long t3 = clock();
	std:: cout <<"bitwise: " << t3 -t1 << " " << " preprocess: "<< t2-t1 << " " << std::endl;
}

int Intersection:: CPUSTLintersection(){
	long t1 = clock();

	std::vector<int> res(arrptr[0],arrptr[0]+ list_size[0]);
	for (int i = 1; i < list_num; ++ i){
		std::vector <int> append_list (arrptr[i],arrptr[i]+ list_size[i]);
		res = intersect(res, append_list);
	}
	std::cout <<"STL direct: " <<(clock() - t1)<< std::endl;
//	std:: cout << "res_num: " << res.size() << std::endl;
//	for (auto i =0; i< res.size(); ++i){
//		std::cout<< "results: " << res[i] <<std::endl;
//	}
}
