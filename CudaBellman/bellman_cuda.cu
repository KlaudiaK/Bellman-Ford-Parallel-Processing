
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <chrono>
using namespace std::chrono;
using namespace std;
using std::string;
using std::cout;
using std::endl;


struct Edge
{
	int Source;
	int Destination;
	int Weight;
};


struct Graph {
	int V, E;
	struct Edge* edge;
};

struct Graph* createGraph(int V, int E)
{
	struct Graph* graph = new Graph;
	graph->V = V;
	graph->E = E;
	graph->edge = new Edge[E];
	return graph;
}

#define cudaSafeCall(call) {  \
  cudaError err = call;       \
  if(cudaSuccess != err){     \
    fprintf(stderr, "cudaSafeCall: %s(%i) : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err));   \
    exit(EXIT_FAILURE);       \
}}

#define cudaCheckErr(errorMessage) {    \
  cudaError_t err = cudaGetLastError(); \
  if(cudaSuccess != err){               \
    fprintf(stderr, "cudaError: %s(%i) : %s : %s.\n", __FILE__, __LINE__, errorMessage, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                 \
}}


void printArr(int dist[], int n)
{
	printf("10 Vertex Distance from Source\n");
	for (int i = 0; i < 10 && i < n; ++i)
		printf("%d \t\t %d\n", i, dist[i]);
}


void Print(int distance[], int count)
{
	ofstream myfile;
	myfile.open("results_cuda.txt");

	for (int i = 0; i < count; ++i)
		myfile << distance[i] << "\n";


	myfile.close();
}

__global__ void relax(struct Edge* edges, int E, int* dist) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < E) {
		int u = edges[tid].Source;
		int v = edges[tid].Destination;
		int weight = edges[tid].Weight;
		if (dist[u] != INT_MAX && dist[u] + weight < dist[v])
			dist[v] = dist[u] + weight;
	}
}


__global__ void checkNegCycle(struct Edge* edges, int E, int* dist) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < E) {
		int u = edges[tid].Source;
		int v = edges[tid].Destination;
		int weight = edges[tid].Weight;
		if (dist[u] != INT_MAX
			&& dist[u] + weight < dist[v])
			printf("Graph contains negative weight cycle");
		return;
	}
}


void BellmanFordParallel(struct Graph* graph, int* dist, int src)
{
	int V = graph->V;
	int E = graph->E;

	for (int i = 0; i < V; i++)
		dist[i] = INT_MAX;
	dist[src] = 0.0;

	int* d_dist;
	cudaMalloc(&d_dist, V * sizeof(int));
	cudaMemcpy(d_dist, dist, V * sizeof(int), cudaMemcpyHostToDevice);


	struct Edge* edges = (struct Edge*)malloc(sizeof(struct Edge) * E);
	for (int i = 0; i < E; i++) {
		edges[i] = graph->edge[i];
	}
	struct Edge* d_edges;
	cudaMalloc((void**)&d_edges, sizeof(struct Edge) * E);
	cudaMemcpy(d_edges, edges, sizeof(struct Edge) * E, cudaMemcpyHostToDevice);

	for (int i = 1; i <= V - 1; i++) {

		relax << <1 + (E - 1) / 1024, 1024 >> > (d_edges, E, d_dist);
	}

	checkNegCycle << <1 + (E - 1) / 1024, 1024 >> > (d_edges, E, d_dist);

	cudaSafeCall(cudaDeviceSynchronize());
	cudaCheckErr("kernel error");

	cudaMemcpy(dist, d_dist, V * sizeof(int), cudaMemcpyDeviceToHost);

	Print(dist, V);

	free(edges);
	cudaFree(d_edges);
	cudaFree(d_dist);

	return;
}


void BellmanFordSerial(struct Graph* graph, int* dist, int src)
{
	int V = graph->V;
	int E = graph->E;

	for (int i = 0; i < V; i++) {
		dist[i] = INT_MAX;
	}

	dist[src] = 0;

	for (int i = 1; i <= V - 1; i++) {

		for (int j = 0; j < E; j++) {
			int u = graph->edge[j].Source;
			int v = graph->edge[j].Destination;
			int weight = graph->edge[j].Weight;
			if (dist[u] != INT_MAX
				&& dist[u] + weight < dist[v])
				dist[v] = dist[u] + weight;
		}

	}

	for (int i = 0; i < E; i++) {
		int u = graph->edge[i].Source;
		int v = graph->edge[i].Destination;
		int weight = graph->edge[i].Weight;
		if (dist[u] != INT_MAX
			&& dist[u] + weight < dist[v]) {
			printf("Graph contains negative weight cycle");
			return;
		}
	}

	Print(dist, V);

	return;
}

void checkSum(int distS[], int distP[], int size) {
	int count1 = 0;
	int count2 = 0;
	for (int i = 0; i < size; i++) {
		count1 += distS[i];
		count2 += distP[i];
	}
	printf("Sum of distances for Serial: %d.\nSum of distances for Parallel: %d.\n", count1, count2);
}

struct Graph* loadData(std::string filename) {

	int N;
	int edgesCount;
	std::ifstream inputf(filename, std::ifstream::in);
	int counter = 0;
	inputf >> N >> edgesCount;

	struct Graph* graph = createGraph(N, edgesCount);

	for (int i = 0; i < edgesCount; i++)
	{
		inputf >> graph->edge[i].Source >> graph->edge[i].Destination >> graph->edge[i].Weight;
	}

	return graph;
}


int main(int argc, char** argv)
{

	if (argc <= 1) {
		cout << "INPUT FILE WAS NOT FOUND!";
		return;
	}
	if (argc <= 2) {
		cout << "NUMBER OF THREADS WAS NOT FOUND!";
		return;
	}

	string filename = argv[1];
	struct Graph* graph = loadData(filename);
	cout << graph->V;


	int* distP = (int*)malloc(sizeof(int) * graph->V);
	int* distS = (int*)malloc(sizeof(int) * graph->V);
	auto start = high_resolution_clock::now();
	//BellmanFordSerial(graph, distS, 0);

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "Bellman serial took " << duration.count() << " ms\n";
	cout << "---------------------------------\n";


	start = high_resolution_clock::now();
	BellmanFordParallel(graph, distP, 0);
	stop = high_resolution_clock::now();
	 duration = duration_cast<microseconds>(stop - start);
	 cout << "Bellman parallel took "<< duration.count() << " ms\n";

	 cout << "---------------------------------\n";
	checkSum(distS, distP, graph->V);


	free(distP);
	free(distS);
	return 0;
}

