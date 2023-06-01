#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <chrono>
#include "omp.h"

using namespace std;

using namespace chrono;
#define INF 1000000

struct Edge
{
    int Source;
    int Destination;
    int Weight;
};

struct Graph
{
    int VerticesCount;
    int EdgesCount;
    struct Edge* edge;
};

static struct Graph* CreateGraph(int verticesCount, int edgesCount)
{
    struct Graph* graph = (struct Graph*)malloc(sizeof(struct Graph));
    graph->VerticesCount = verticesCount;
    graph->EdgesCount = edgesCount;
    graph->edge = (struct Edge*)malloc(graph->EdgesCount * sizeof(struct Edge));

    return graph;
}

static void Print(int distance[], int count, string output)
{
    ofstream myfile;
    myfile.open(output);

    printf("Vertex Distance from source\n");

    for (int i = 0; i < count; ++i)

        myfile << distance[i] << "\n";

    myfile.close();
}

static void BellmanFord(struct Graph* graph, int source, bool* has_negative_cycle, int p, string output)
{

    int verticesCount = graph->VerticesCount;
    int edgesCount = graph->EdgesCount;
    int* distance = (int*)malloc(sizeof(int) * verticesCount);

#pragma omp parralel for
    for (int i = 0; i < verticesCount; i++)
        distance[i] = INT_MAX;


    int q = verticesCount / p;

    distance[source] = 0;
    for (int i = 1; i < verticesCount; ++i)
        distance[i] = INF;

    *has_negative_cycle = false;

    int j, i;
#pragma omp parallel num_threads(p)
    {
        int my_rank = omp_get_thread_num();

#pragma omp barrier
        int start = my_rank * q + 1;
        int end = (my_rank + 1) * q;
#pragma omp parallel for
        for (i = start; i <= end; ++i)
        {
#pragma omp parallel for
            for (j = 0; j < edgesCount; ++j)
            {
                int u = graph->edge[j].Source;
                int v = graph->edge[j].Destination;
                int weight = graph->edge[j].Weight;

                if (distance[u] != INT_MAX && distance[u] + weight < distance[v])

                    distance[v] = distance[u] + weight;
            }
        }


    END: {}
    }

#pragma omp parallel for
    for (int i = 0; i < edgesCount; ++i)
    {
        int u = graph->edge[i].Source;
        int v = graph->edge[i].Destination;
        int weight = graph->edge[i].Weight;

        if (distance[u] != INT_MAX && distance[u] + weight < distance[v])
            cout << "Graph contains negative weight cycle." << endl;
    }

    Print(distance, verticesCount, output);

}

bool compareFiles(const string& file1, const string& file2) {
    ifstream f1(file1);
    ifstream f2(file2);
    string content1((istreambuf_iterator<char>(f1)), istreambuf_iterator<char>());
    string content2((istreambuf_iterator<char>(f2)), istreambuf_iterator<char>());

    return content1 == content2;
}


int main(int argc, char** argv) {

    if (argc <= 1) {
        cout << "INPUT FILE WAS NOT FOUND!";
        return 0;
    }
    if (argc <= 2) {
        cout << "OUTPUT FILE WAS NOT FOUND!";
        return 0;
    }
    if (argc <= 3) {
        cout << "NUMBER OF THREADS WAS NOT FOUND!";
        return 0;
    }

    string filename = argv[1];
    string output = argv[2];
    int p = atoi(argv[3]);
    int N;
    bool has_negative_cycle = false;
    int edgesCount;

    ifstream inputf(filename, ifstream::in);
    int counter = 0;
    inputf >> N >> edgesCount;
    assert(N < (1024 * 1024 * 20));
    struct Graph* graph = CreateGraph(N, edgesCount);
    for (int i = 0; i < edgesCount; i++)
    {
        inputf >> graph->edge[i].Source >> graph->edge[i].Destination >> graph->edge[i].Weight;
    }

    auto start = high_resolution_clock::now();
    BellmanFord(graph, 0, &has_negative_cycle, p, output);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    cout << "Time(s): " << duration.count() << endl;

    return 0;
}