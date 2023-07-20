#run alley in GPU
./build/GPUgmatch -d ./dataset/toy/data_graph/toy-data.graph -q ./dataset/toy/query_graph/toy-query.graph -m 2 -s 16 -t 102400 -i 1 -n 1 -c 5120 -e 0 -h 2
#run alley in GPU-CPU cooperate mode
./build/GPUgmatch -d ./dataset/toy/data_graph/toy-data.graph -q ./dataset/toy/query_graph/toy-query.graph -m 7 -s 16 -t 102400 -i 1 -n 1 -c 5120 -e 0 -h 2