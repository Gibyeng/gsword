#pragma once

struct timer {
	double sampling_time;
	double enumerating_time;
	double cand_alloc_time;
	double reorder_time;
	unsigned long long est_path;
	unsigned long long est_workload;
	unsigned long long set_intersection_count;
	unsigned long long total_compare;
	unsigned long long real_workload;
	ui sample_time;
	ui total_sample_count;
	ui inter_count;
	double b;
	double alpha;
	ui fixednum;
	double total_path;
	unsigned int taskPerBlock;
	unsigned int taskPerWarp;
	unsigned int threadnum;
	unsigned int* arr_range_count ;
	bool successrun;
	double full_ratio;
	double adapt_ratio;
	double sample_ratio;
	double base_ratio;
	std::string msg;
	std::string msg_time;
	//var for cpu-gpu
	ui numberofhardcases;
	double old_est;
	double new_est;
	double gpu_sample_cost;
	double cpu_enumeration_cost;
	double pure_samping;
	double select_success_ratio_before;
	double select_success_ratio_after;
	double select_stdev_before;
	double select_stdev_after;
	double average_layer;
	ui batchnumber;
	// increase ratio of paths
	double SpeedupbyInheritance;
};

size_t hash (unsigned int k, size_t size ){
   k ^= k >> 16;
   k *= 0x85ebca6b;
   k ^= k >> 13;
   k *= 0xc2b2ae35;
   k ^= k >> 16;
   return k%size;
}

//128-bit vector
struct EncodeVector{
	unsigned long l1 = 0;
	unsigned long l2 = 0;
};

unsigned int countSetBits(unsigned int n)
{
    unsigned int count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}

void sortIndex (int array[], int n, int* index){
	if (array && n > 1){
		//初始化索引数组
		int i, j;
		for (i = 0; i < n; i++)
			index[i] = i;
		//类似于插入排序，在插入比较的过程中不断地修改index数组
		for (i = 0; i < n; i++)
		{
			j = i;
			while (j)
			{
				if (array[index[j]] < array[index[j - 1]])
					std::swap(index[j], index[j - 1]);
				else
					break;
				j--;
			}
		}
	}
}

void sortIndexDec (ui array[], ui n, ui* index){
	if (array && n > 1){
		//初始化索引数组
		ui i, j;
		for (i = 0; i < n; i++)
			index[i] = i;
		//类似于插入排序，在插入比较的过程中不断地修改index数组
		for (i = 0; i < n; i++)
		{
			j = i;
			while (j)
			{
				if (array[index[j]] > array[index[j - 1]])
					std::swap(index[j], index[j - 1]);
				else
					break;
				j--;
			}
		}
	}
}


std::vector<int> intersect(std::vector<int>& nums1, std::vector<int>& nums2) {
		std::vector<int> res;
        std::sort(nums1.begin(),nums1.end());
        std::sort(nums2.begin(),nums2.end());
        //require two arrays are sorted
        std::set_intersection(nums1.begin(),nums1.end(),nums2.begin(),nums2.end(),std::insert_iterator<std::vector<int>>(res,res.begin()));
        return res;    // res：2，2
}

ui divup(ui x, ui y)
{
    return (x + y - 1) / y;
}


