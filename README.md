# GSWORD
GSWORD is a GPU framework for subgraph counting problem. It provides RW estimators to estimate the number of subgraphs for a data graph which are isomorphic with a given query graph.

Organization
--------
Code for "gSWORD: GPU-accelerated Sampling for Subgraph Counting"

Download
--------
There is a public tool to download the code from anonymous4open. (https://github.com/ShoufaChen/clone-anonymous4open)

Compilation
--------

Requirements

* CMake &gt;= 2.8
* CUDA environment

Compilation is done within the / (root) directory with CMake. 
Please make sure you have installed CMake software compilation tool.
Configure cmakelist.txt appropriately before you start compile. 
To compile, please create a build directory then simple run:

```
$ cd ./build
$ cmake ..
$ make
```

Running Code in GSWORD
--------
Running code is done within the build/ directory. 
Use "./build/gsword -d DataGraph -q QueryGraph -m method -s NumberOfQueryVetice" to estimate the count of QueryGraph in DataGraph.

| Method | code | Description                   |
| ------ | ---- | ----------------------------- |
| LFTJ   | 0    | Exact count by enumeration    |
| WJ     | 1    | GPU WJ  (O0)                  |
| AL     | 2    | GPU ALLEY  (O0)               |
| COWJ   | 3    | WJ with inheritance (O1)      |
| COAL   | 4    | AL with inheritance (O1)      |
| RSAL   | 5    | AL with Warp streaming (O2)   |
| HYBWJ  | 6    | WJ with CPU-GPU co-processing |
| HYBAL  | 7    | AL with CPU-GPU co-processing |

Method 0 is a graph enumeration algorithm that generate real count for the query. Method 1 and 2 are the methods with no optimizations. Method 3 and 4 are methods that deployed inheritances optimization (O1). Method 5 is deployed with warp streaming optimization (O2). Method 6 and 7 are our final solution with CPU-GPU co-processing optimations.

We also provide mpre advanced arguments for experienced users. 
-t NumberOfSamples,  -c NumberOfThreads, -e MatchOrder

| MatchOrder | code | Description                     |
| ---------- | ---- | ------------------------------- |
| QSI        | 0    | the ordering method of QuickSI  |
| GQL        | 1    | the ordering method of GraphQL  |
| TSO        | 2    | the ordering method of TurboIso |
| CFL        | 3    | the ordering method of CFL      |
| DPi        | 4    | the ordering method of DP-iso   |
| CECI       | 5    | the ordering method of CECI     |
| RI         | 6    | the ordering method of RI       |
| VF2        | 7    | the ordering method of VF2++    |

Examples
```
$ ./gsword -d datagraph.graph -q query.graph -m 1 -s 16
or
./gsword -d datagraph.graph -q query.graph -m 1 -s 16 -t 128000 -c 5120 -e 6
```

GPU-CPU cooperate in GSWORD
--------

![pipeline-crop](https://github.com/Gibyeng/gsword/assets/19706360/8a96fc3d-0301-476e-a231-0fe89663ea32)

We also support GPU-CPU cooperate executing mode for cases where existing RW estimators have severe underestimate issues.
When enable GPU-CPU cooperate methods, you can provide more arguments: -i "MaxNumberOfSamplesForEnumeration" -h "NumberOfBatches". 
We provide a toy datagraph with 3112 vertices and 12519 edges in the build/ directory. Please run the shell in example.sh and have a try.

Input Format for GSWORD
--------
 Graph starts with 't VertexNum EdgeNum' where VertexNum is the number of vertices and EdgeNum is the number of edges. Each vertex is represented as 'v VertexID LabelId Degree' and 'e VertexId VertexId' for edges. We give an input example in the following.

```
t 5 6
v 0 0 2
v 1 1 3
v 2 2 3
v 3 1 2
v 4 2 2
e 0 1
e 0 2
e 1 2
e 1 3
e 2 4
e 3 4
```

Output File Format for GSWORD
--------
The configuration information and results are showcased in the console during execution. Additionally, We also output the results into a file named "output.txt" by default. Each query corresponds to one line of the file. To get the Q-error please run enumeration (Method 0) to get the real subgraph count and compare the "estimatefromRW" with the real count. "GPUErrorDetection" flag indicates whether the GPU is functioning properly. If there is no GPU error, the flag is 1, otherwise 0. 


```
datagraph querygraph querysize numberofsamplesPerkernel numberofsamplesperblock numberofsamplesperwarp numberofBatches(only for co-processing) candidateBuildingTime samplingCost enumerationCount estimatefromRW GPUErrorDetection
```

| OutputTerm               | Description                                     |
| ------------------------ | ----------------------------------------------- |
| datagraph/querygraph     | The file name of data graph or query graph      |
| querysize                | The number of nodes of the query                |
| numberofsamplesPerkernel | The number of samples are assigns to one kernel |
| numberofsamplesperblock  | The number of samples are assigns to one block  |
| numberofsamplesPerwarp   | The number of samples are assigns to one warp   |
| numberofBatches          | Batch numbers used in co-processing approaches  |
| candidateBuildingTime    | The cost of building candidate graph            |
| samplingCost             | The cost of GPU sampling                        |
| enumerationCount         | The real count of CPU enumeration               |
| estimatefromRW           | The estimated count of RW estimator             |
| GPUErrorDetection        | The flag of GPU runtime error                   |



Datasets and Baselines
--------
We have updated four of the datasets (dblp, yeast, hprd, wordnet) and their corresponding queries utilized in the paper, and they can now be accessed in the "dataset/datasets.zip". Due to the large space of the datasets, we do not upload the rest datasets to the repo. However, we will provide Google Drive links for downloading these datasets upon publication. One can find the two baseline implementations, [gcare](https://github.com/yspark-dblab/gcare.git) and [nextDoor](https://github.com/plasma-umass/NextDoor) by clicking the provided links. 
