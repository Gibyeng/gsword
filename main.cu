#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <cuda_profiler_api.h>
#include <time.h>
#include <algorithm>
#include "Intersection.cu"
#include "until.cpp"
#include "graph/graph.cpp"
#include "matching/matching.cpp"
#include "matching/BuildTable.cpp"

#define maxListLength 1000

int main (int argc, char * argv[]){
	int opt = 0;
	std::string input_data_graph_file;
	std::string input_query_graph_file;
	int method = 2;
	ui step = 0;
	ui sample_time = 10;
	int inter_count = 5000;
	std::string output_file = "./output.txt";
	double b = 0.1;
	double alpha = 0.1;
	ui fixednum = 1;
	ui threadnum = sample_time;
	//编译时常量, cub需要
	constexpr ui BlockSize = 128;
	constexpr ui WarpSize = 32;
	std::cout << "block size is "<<  BlockSize <<std::endl;
	//random seed if necessory
//	srand((unsigned)time(NULL));
	cudaDeviceSynchronize();
	bool successrun = true;
	auto err = cudaGetLastError();
	if (err != cudaSuccess){
		successrun = false;
		std::cout <<" error, restart GPU! "<<std::endl;
	}else{
		std::cout <<" Pass GPU test "<<std::endl;
	}
	ui orderid = 1;
	ui batchsize = 2;
	const char *optstring = "d:q:m:s:o:t:i:b:a:n:c:e:h:";
	while ((opt = getopt (argc, argv, optstring))!= -1){
		switch(opt){
			case 'd':{
				input_data_graph_file = std::string (optarg);
				std::cout << "input_data_graph_file " << input_data_graph_file<<std::endl;
				break;
			}
			case 'q':{
				input_query_graph_file = std::string (optarg);
				std::cout << "input_query_graph_file " << input_query_graph_file<<std::endl;
				break;
			}
			/*m is the method to be run*/
			case 'm':{
				method = atoi(optarg);
				std::cout << "method " << method<<std::endl;
				break;
			}
			case 's':{
				step = atoi(optarg);
				std::cout << "step " << step<<std::endl;
				break;
			}
			case 't': {
				sample_time = atoi(optarg);
				std::cout << "sample_time: " << sample_time<<std::endl;
				break;
			}
			case 'i':{
				inter_count = atoi(optarg);
				std::cout << "inter_count:  " << inter_count<<std::endl;
				break;
			}
			case 'o':{
				output_file = std::string(optarg);
				std::cout << "output_file: " << output_file<<std::endl;
				break;
			}
			case 'b':{
				b = atof(optarg);
				std::cout << "branch var: " << b<<std::endl;
				break;
			}
			case 'a':{
				alpha = atof(optarg);
				std::cout << "alpha var: " <<alpha <<std::endl;
				break;
			}
			case 'n':{
				fixednum = atoi(optarg);
				std::cout << "fixnumber: " <<fixednum <<std::endl;
				break;
			}

			case 'c':{
				threadnum = atoi(optarg);
				std::cout << "threadnum: " <<threadnum <<std::endl;
				break;
			}
			case 'e':{
				orderid = atoi(optarg);
				std::cout << "using matching order id:" << orderid << std::endl;
				break;
			}
			case 'h':{
				batchsize = atoi(optarg);
				std::cout << "using batch size (only in CPU-GPU hybird methods ):" << batchsize << std::endl;
				break;
			}
		}
	}
	// argument check
	ui task_per_thread = sample_time/ threadnum;
	ui taskPerBlock = task_per_thread* BlockSize;
	ui taskPerWarp = task_per_thread* WarpSize;
	std::cout << "taskPerThread: " << task_per_thread <<" taskperBlock: " << taskPerBlock << " taskPerWarp: " << taskPerWarp <<std::endl;
	/* load graphs */
	Graph* query_graph = new Graph(true);
	Graph* data_graph = new Graph(true);
	data_graph->loadGraphFromFile(input_data_graph_file);
	query_graph->loadGraphFromFile(input_query_graph_file);
	//buildCoreTable
	std::cout << "buildCoreTable..." << std::endl;
	query_graph->buildCoreTable();
	std::cout << "-----" << std::endl;
	std::cout << "Query Graph Meta Information" << std::endl;
	query_graph->printGraphMetaData();
	std::cout << "-----" << std::endl;
	data_graph->printGraphMetaData();
	// ** reset step to max if undefine **/
	if(step == 0){
		step = query_graph->getVerticesCount();
	}

    /**
     * Start queries.
     */
	std::cout << "Start queries..." << std::endl;
	std::cout << "-----" << std::endl;
	std::cout << "Filter candidates..." << std::endl;
	ui** candidates = NULL;
	ui* candidates_count = NULL;
	ui* tso_order = NULL;
	TreeNode* tso_tree = NULL;
	ui* cfl_order = NULL;
	TreeNode* cfl_tree = NULL;
	ui* dpiso_order = NULL;
	TreeNode* dpiso_tree = NULL;
	TreeNode* ceci_tree = NULL;
	ui* ceci_order = NULL;

	GQLFilter(data_graph, query_graph, candidates, candidates_count);
	sortCandidates(candidates, candidates_count, query_graph->getVerticesCount());
//
	auto buildcand_start = std::chrono::high_resolution_clock::now();
	std::cout << "-----" << std::endl;
	std::cout << "Build indices..." << std::endl;
	Edges ***edge_matrix = NULL;
	edge_matrix = new Edges **[query_graph->getVerticesCount()];
	for (ui i = 0; i < query_graph->getVerticesCount(); ++i) {
		edge_matrix[i] = new Edges *[query_graph->getVerticesCount()];
	}
	/*build edge_matrix [node1][node2] -> candidateofnode1*/
	BuildTable::buildTables(data_graph, query_graph, candidates, candidates_count, edge_matrix);
	size_t memory_cost_in_bytes = 0;
	memory_cost_in_bytes = BuildTable::computeMemoryCostInBytes(query_graph, candidates_count, edge_matrix);
	BuildTable::printTableCardinality(query_graph, edge_matrix);

	std::cout << "-----" << std::endl;
	std::cout << "Generate a matching order..." << std::endl;

	ui* matching_order = NULL;
	ui* pivots = NULL;
	ui** weight_array = NULL;

	size_t order_num = 0;
//	sscanf(input_order_num.c_str(), "%zu", &order_num);
	//select matching order
	// the ordering is 0:QSI 1:GQL 2:TSO 3:CFL 4:DPiso 5:CECI 6:RI 7:VF2PP 8:Spectrum
	std::vector<std::vector<ui>> spectrum;
	if (orderid == 0) {
		std::cout << "use QSI query plan..." << std::endl;
		generateQSIQueryPlan(data_graph, query_graph, edge_matrix, matching_order, pivots);
	} else if (orderid == 1) {
		std::cout << "use GQL query plan..." << std::endl;
		generateGQLQueryPlan(data_graph, query_graph, candidates_count, matching_order, pivots);
	} else if (orderid == 2) {
		if (tso_tree == NULL) {
			generateTSOFilterPlan(data_graph, query_graph, tso_tree, tso_order);
		}
		std::cout << "use TSO query plan..." << std::endl;
		generateTSOQueryPlan(query_graph, edge_matrix, matching_order, pivots, tso_tree, tso_order);
	} else if (orderid == 3){
		if (cfl_tree == NULL) {
			int level_count;
			ui* level_offset;
			generateCFLFilterPlan(data_graph, query_graph, cfl_tree, cfl_order, level_count, level_offset);
			delete[] level_offset;
		}
		std::cout << "use CFL query plan..." << std::endl;
		generateCFLQueryPlan(data_graph, query_graph, edge_matrix, matching_order, pivots, cfl_tree, cfl_order, candidates_count);
	} else if (orderid == 4) {
		if (dpiso_tree == NULL) {
			generateDPisoFilterPlan(data_graph, query_graph, dpiso_tree, dpiso_order);
		}
		std::cout << "use DPiso query plan..." << std::endl;
		generateDSPisoQueryPlan(query_graph, edge_matrix, matching_order, pivots, dpiso_tree, dpiso_order,
													candidates_count, weight_array);
	}
	else if (orderid == 5) {
		std::cout << "use CECI query plan..." << std::endl;
		generateCECIQueryPlan(query_graph, ceci_tree, ceci_order, matching_order, pivots);
	}
	else if (orderid == 6) {
		std::cout << "use RI query plan..." << std::endl;
		generateRIQueryPlan(data_graph, query_graph, matching_order, pivots);
	}
	else if (orderid == 7) {
		std::cout << "use VF2 query plan..." << std::endl;
		generateVF2PPQueryPlan(data_graph, query_graph, matching_order, pivots);
	}
	else if (orderid == 8) {
		std::cout << "use Spectrum query plan..." << std::endl;
		generateOrderSpectrum(query_graph, spectrum, order_num);
	}
	else {
		std::cout << "The specified order id " << orderid << "' is not supported." << std::endl;
	}
	// ordering vertices
	generateGQLQueryPlan(data_graph, query_graph, candidates_count, matching_order, pivots);

	checkQueryPlanCorrectness(query_graph, matching_order, pivots);
	printSimplifiedQueryPlan(query_graph, matching_order);
	std::cout << "-----" << std::endl;
	auto buildcand_end = std::chrono::high_resolution_clock::now();
	std::cout<<"build candidates time: " << (double)std::chrono::duration_cast<std::chrono::nanoseconds>(buildcand_end - buildcand_start).count() /1000000000<< std::endl;
	std::cout << "Enumerate..." << std::endl;
	size_t output_limit = std::numeric_limits<size_t>::max();

	size_t embedding_count = 0;
	size_t call_count = 0;
	size_t time_limit = 0;
	timer record;
	record.sample_time = sample_time;
	record.inter_count = inter_count;
	record.b = b;
	record.alpha = alpha;
	record.fixednum = fixednum;
	record.taskPerBlock = taskPerBlock;
	record.taskPerWarp = taskPerWarp;
	record.threadnum = threadnum;
	record.successrun = successrun;
	record.batchnumber = batchsize;
	switch (method){

		case 0:{
			/* pure CPU enumeration*/
			std::cout << "CPU enmu: "<<std::endl;
			embedding_count = LFTJ(data_graph, query_graph, edge_matrix, candidates, candidates_count,
	                                              matching_order, output_limit, call_count, record);
			break;
		}


		case 2: {
			//GPU AL
			embedding_count = AL<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																		  matching_order, output_limit, call_count, step,record);
			break;
		}
	
		case 1:{
			//GPU WJ
			embedding_count = WJ<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																				  matching_order, output_limit, call_count, step,record);
			break;
		}

		
		case 3:{
			// GGE wj
			embedding_count = COWJ<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																																  matching_order, output_limit, call_count, step,record);
			break;
		}
		case 5:{
			// GGE balance workload within a warp
			embedding_count = RSAL<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																																  matching_order, output_limit, call_count, step,record);
			break;
		}

		case 4:{
			// alley 开合作 无warp内优化
			embedding_count = COAL<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																																			  matching_order, output_limit, call_count, step,record);
			break;
		}
	
		// cpu-gpu alley
		case 7:{
			embedding_count = HYBAL<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																								  matching_order, output_limit, call_count, step,record);
			break;
		}
	
		// cpu-gpu wanderjoin
		case 6:{
			embedding_count = HYBWJ<BlockSize>(data_graph, query_graph, edge_matrix, candidates, candidates_count,
																								  matching_order, output_limit, call_count, step,record);
			break;
		}
	}

	std::cout << "End... "<< std::endl;
	std::cout <<"Enumerate count: "<< embedding_count << std::endl;
	std::cout <<"Est path count: "<< record.est_path<< std::endl;
	std::cout << "Sampling_cost: " << record.sampling_time/1000000000 << std::endl;
	std::cout << "Enumerating_cost: " << record.enumerating_time/1000000000 << std::endl;
	std::cout << "candiate set cost: " << record.cand_alloc_time/1000000000 << std::endl;
	std::cout <<"call count: "<< call_count << std::endl;

	//write files
	ofstream out;
	if( method != 0 && method != 6 && method != 7){
	out.open(output_file, std::ios_base::app);
	}
	if (out.is_open())
	{
		//header add header manally
//		out << "data file\t" <<"query file\t"<< "enumerate count\t"<< "step\t"<< "recursion count\t" << "sampling cost\t" << "reorder cost\t"<< "enumerate cost" <<std::endl;
		// data_file
		std::size_t found = input_data_graph_file.find_last_of("/\\");
		out << input_data_graph_file.substr(found+1) << "\t";
		//query file
		found = input_query_graph_file.find_last_of("/\\");
		out << input_query_graph_file.substr(found+1) << "\t";
		// step
		out<< step << "\t";
//		//call count
//		out << call_count << "\t";
		// 2nd -  sampling -t
		out << record.sample_time <<"\t";
		out << taskPerBlock <<"\t";
		out << taskPerWarp <<"\t";
		// inter  -i
//		out << record.total_sample_count <<"\t";
		// b
//		out << record.b <<"\t";
		// a
//		out << record.alpha <<"\t";
		//enumerate count
		out << embedding_count<< "\t";
		// est_emb
		out << record.est_path<< "\t";
		//real_workload
//		out << record.real_workload<< "\t";
		// est_workload
//		out << record.est_workload<< "\t";
		// number of set intersections
//		out << record.set_intersection_count << "\t";
		// total_compare
//		out << record.total_compare << "\t";
		// total paths
//		out << record.total_path << "\t";
		// Q error ratio of emb
		size_t emb_c = embedding_count;
		size_t est_c = record.est_path;

		if(emb_c == 0){
			emb_c = 1;
		}
		if(est_c == 0){
			est_c = 1;
		}
		double qerr = (double) emb_c / est_c;
		if(qerr < 1){
			qerr = 1/qerr;
		}
		// overestimate or underestimate
		if(embedding_count > record.est_path){
			out << "-";
		}else{
			out << "+";
		}

//		out << (double)abs((long long)embedding_count -  (long long)record.est_path)/embedding_count<< "\t";

		out << qerr<< "\t";
		out << record.cand_alloc_time/1000000000 << "\t";
		// simpling cost_ by_GPU
		out << record.sampling_time/1000000000 << "\t";
//		// reorder cost_ by_GPU
//		out << record.reorder_time/1000000000 << "\t";
		// enumerating cost
		out << record.enumerating_time/1000000000 << "\t";
		// if gpu run successfully
		out << record.successrun << "\t";

		// 》 64
		if(method == 19){
			out << record.arr_range_count[0] << "\t";
			out << record.arr_range_count[1] << "\t";
			out << record.arr_range_count[2] << "\t";
			out << record.arr_range_count[3] << "\t";
			out << record.arr_range_count[4] << "\t";
		}

		out << std::endl;
		out.close();
	}
	
	if( method == 6 || method == 7){
		out.open(output_file, std::ios_base::app);
		if (out.is_open())
			{
				//header add header manally
		//		out << "data file\t" <<"query file\t"<< "enumerate count\t"<< "step\t"<< "recursion count\t" << "sampling cost\t" << "reorder cost\t"<< "enumerate cost" <<std::endl;
				// data_file
				std::size_t found = input_data_graph_file.find_last_of("/\\");
				out << input_data_graph_file.substr(found+1) << "\t";
				//query file
				found = input_query_graph_file.find_last_of("/\\");
				out << input_query_graph_file.substr(found+1) << "\t";
				// step
				out<< step << "\t";
		//		//call count
		//		out << call_count << "\t";
				// 2nd -  sampling -t
				out << record.sample_time <<"\t";
				out << taskPerBlock <<"\t";
				out << taskPerWarp <<"\t";
				// numberofhardcases;
				out << record.batchnumber<<"\t";
//				out << record.old_est<<"\t";
				out << record.new_est<<"\t";
				out << record.sampling_time/1000000000<<"\t";
//				out << record.gpu_sample_cost<<"\t";
//				out << record.cpu_enumeration_cost<<"\t";
//				out << record.select_success_ratio_before<<"\t";
				out << record.select_success_ratio_after<<"\t";
//				out << record.select_stdev_before<<"\t";
//				out << record.select_stdev_after<<"\t";
				out << record.average_layer<<"\t";

				out << std::endl;
				out.close();
			}
	}

	
	// reset GPU when exit
	cudaDeviceReset();
	return 0;
}
