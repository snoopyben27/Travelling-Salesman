
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <vector>
#include <cmath>
#include <iterator>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <random>
#include <limits>


int     EPOCHS = 100000;
int     POPULATION_SIZE = 750000;
float   ELITE_RATIO = 0.5f;
float   MUTATION_PROBABILITY = 0.85f;
char DATASET_FILEPATH[] = "C:\\Users\\ben\\Documents\\datasets\\a280_formatted.txt"; // "C:\\Users\\ben\\Documents\\datasets\\berlin52.txt";
int LOG_EVERY = 10;

float IDEAL_DISTANCE = 2579; // 7544.3659f;


struct City {
    size_t id;
    float x;
    float y;
};

City createNewCity(size_t id, float x, float y) {
    City city;
    city.id = id;
    city.x = x;
    city.y = y;
    return city;
}

void getCoordsFromArray(std::vector<City> const& arr, float* xCoordList, float* yCoordList) {
    for (int i = 0; i < arr.size(); i++) {
        xCoordList[i] = arr[i].x;
        yCoordList[i] = arr[i].y;
    }
}

std::vector<std::string> splitStringAtSpaces(const std::string& s) {
    std::istringstream iss(s);
    std::vector<std::string> words;
    std::string word;

    while (iss >> word) {
        words.push_back(word);
    }
    return words;
}

__device__ __host__ float distance(float x1, float y1, float x2, float y2) {
    return sqrtf((x2-x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

__global__ void init_rng(curandState* states, unsigned long long seed, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    // seed: same for all, sequence: unique per thread, offset: 0
    curand_init(seed, tid, 0, &states[tid]);
}

__device__ __forceinline__ int rand_int(curandState* state, int min, int max) {
    // inclusive range [min, max]
    unsigned int r = curand(state);
    unsigned int range = (unsigned int)(max - min + 1);
    // Better than r % range: reduce modulo bias with mul-hi (see below)
    unsigned int scaled = (unsigned int)(((unsigned long long)r * range) >> 32);
    return min + (int)scaled;
}

__device__ int generate_random_int(curandState* state, int min, int max) {
    unsigned int r = curand(state);
    int range = max - min + 1;
    return min + (r % range);
}

// This generates the same random number every fucking time
__device__ int generate_random_int(int min, int max) {
    curandState state;
    curand_init(1234, 10, 0, &state);

    unsigned int r = curand(&state);

    int range = max - min + 1;
    return min + (r % range);
}

/*
__global__ void shakeUpWithRandomMutations(int size, int populationSize, int* d_populations, curandState* states) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= populationSize) return;

    curandState local = states[threadId];

    int skip = rand_int(&local, 0, 10);
    if (skip < 5) return;

    int numberOfMutationToApply = rand_int(&local, 0, 10);
    for (int i = 0; i < numberOfMutationToApply; i++) {
        int startId = threadId * size;
        int gene1 = startId + rand_int(&local, 0, size - 1);
        int gene2 = startId + rand_int(&local, 0, size - 1);
        int tmp = d_populations[gene1];
        d_populations[gene1] = d_populations[gene2];
        d_populations[gene2] = tmp;
    }

    states[threadId] = local;
}
*/

__global__ void runEvolutionStep(int size, int populationSize, int nonElitePopulationSize, int* d_populations, float mutationProbability, curandState* states) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= nonElitePopulationSize) return;

    int replaceableItemStartIdx = (populationSize - nonElitePopulationSize) + threadId;
    curandState local = states[threadId];
    
    if (replaceableItemStartIdx >= populationSize) return;
    //if (threadId >= populationSize * size)  return;

    int crossoverType = rand_int(&local, 0, 1);
    if (crossoverType < 1) {
        // corssover 1-point

        int parentId1 = rand_int(&local, 0, (populationSize - nonElitePopulationSize) - 1);
        int parentId2 = rand_int(&local, 0, (populationSize - nonElitePopulationSize) - 1);
        int onePoint = rand_int(&local, 1, size - 2);

        for (int i = 0; i < onePoint; i++) {
            d_populations[replaceableItemStartIdx * size + i] = d_populations[parentId1 * size + i];
        }

        int tmpId = replaceableItemStartIdx * size + onePoint;
        bool found = false;

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < onePoint; j++) {
                if (d_populations[parentId2 * size + i] == d_populations[replaceableItemStartIdx * size + j]) {
                    found = true;
                    break;
                }

            }
            if (!found) {
                d_populations[tmpId] = d_populations[parentId2 * size + i];
                tmpId++;
            }
            found = false;
        }
    }
    else {
        // crossover 2-point

        int parentId1 = rand_int(&local, 0, (populationSize - nonElitePopulationSize) - 1);
        int parentId2 = rand_int(&local, 0, (populationSize - nonElitePopulationSize) - 1);
        int onePoint = rand_int(&local, 0, size - 2);
        int twoPoint = rand_int(&local, onePoint + 1, size - 1);

        for (int i = 0; i < onePoint; i++) {
            d_populations[replaceableItemStartIdx * size + i] = d_populations[parentId1 * size + i];
        }

        for (int i = twoPoint; i < size; i++) {
            d_populations[replaceableItemStartIdx * size + i] = d_populations[parentId1 * size + i];
        }

        int tmpId = replaceableItemStartIdx * size + onePoint;
        bool found = false;

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < onePoint; j++) {
                if (d_populations[parentId2 * size + i] == d_populations[replaceableItemStartIdx * size + j]) {
                    found = true;
                    break;
                }

            }

            for (int j = twoPoint; j < size; j++) {
                if (d_populations[parentId2 * size + i] == d_populations[replaceableItemStartIdx * size + j]) {
                    found = true;
                    break;
                }

            }

            if (!found) {
                d_populations[tmpId] = d_populations[parentId2 * size + i];
                tmpId++;
            }
            found = false;
        }
    }

    // mutation
    
    int doMutation = rand_int(&local, 0, 100);
    if (doMutation <= int(mutationProbability * 100.0f)) {
        int numberOfMutations = rand_int(&local, 1, 10);

        for (int m = 0; m < numberOfMutations; m++) {
            int startId = replaceableItemStartIdx * size;
            int gene1 = startId + rand_int(&local, 0, size - 1);
            int gene2 = startId + rand_int(&local, 0, size - 1);
            int tmp = d_populations[gene1];
            d_populations[gene1] = d_populations[gene2];
            d_populations[gene2] = tmp;
        }
    }

    states[threadId] = local;
}

__global__ void calculatePopulationsFitness(int size, int populationSize, int* populations, float* fitnessScores, float* d_xCoordList, float* d_yCoordList) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int startIdx = threadId * size;
    int stopIdx = threadId * size + size;

    if (threadId >= populationSize) return;
    
    float tmpDistance = 0.0f;

    // for (int i = startIdx+1; i < stopIdx; i++) {
    
    for (int i = startIdx+1; i < stopIdx; i++) {
        tmpDistance += distance(
            d_xCoordList[populations[i - 1]],
            d_yCoordList[populations[i - 1]],
            d_xCoordList[populations[i]],
            d_yCoordList[populations[i]]
        );
    }
    
    tmpDistance += distance(
        d_xCoordList[populations[startIdx]],
        d_yCoordList[populations[startIdx]],
        d_xCoordList[populations[stopIdx-1]],
        d_yCoordList[populations[stopIdx-1]]
    );
    
    
    fitnessScores[threadId] = tmpDistance;
}

__host__ void reorderPopulations(int size, int populationSize, int* populations, float* fitnessScores) {
    std::vector<int> order(populationSize);
    std::iota(order.begin(), order.end(), 0);

    std::sort(order.begin(), order.end(),[&](int a, int b) { return fitnessScores[a] < fitnessScores[b]; });

    std::vector<int> popTmp(populationSize * size);
    std::vector<float> fitTmp(populationSize);

    for (int i = 0; i < populationSize; i++) {
        memcpy(&popTmp[i * size], &populations[order[i] * size], size * sizeof(int));
        fitTmp[i] = fitnessScores[order[i]];
    }

    memcpy(populations, popTmp.data(), popTmp.size() * sizeof(int));
    memcpy(fitnessScores, fitTmp.data(), fitTmp.size() * sizeof(float));
}

__host__ void executeEpochs(int blocks, int threadsPerBlock, int size, int populationSize, int* populations, int nonElitePopulationSize, int* d_populations, float* d_xCoordList, float* d_yCoordList) {
    
    float* d_fitnessScores;
    float* localFitnessScores;

    cudaError_t err = cudaMalloc((void**)&d_fitnessScores, populationSize * sizeof(float));
    localFitnessScores = reinterpret_cast<float*>(malloc(populationSize * sizeof(float)));

    curandState* d_states;
    cudaMalloc(&d_states, nonElitePopulationSize * sizeof(curandState));
    int TPB_RNG = 256;
    int rngBlocks = (nonElitePopulationSize + TPB_RNG - 1) / TPB_RNG;

    init_rng << <rngBlocks, TPB_RNG >> > (d_states, 1234ULL, nonElitePopulationSize);

    int TPB = 1024;                       // threads per block
    int testBlocks = (populationSize + TPB - 1) / TPB;
    int testBlocksE = (nonElitePopulationSize + TPB - 1) / TPB;

    bool shakeUpDone = false;

    // main optimization loop
    for (int e = 0; e < EPOCHS; e++) {
        // we need to reorder the list to have it in increasing fitness score based
        calculatePopulationsFitness<<<testBlocks, TPB>>>(size, populationSize, d_populations, d_fitnessScores, d_xCoordList, d_yCoordList);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { printf("Fitness calc kernel runtime error: %s\n", cudaGetErrorString(err)); }

        // copy the fitness scores back...
        cudaMemcpy(localFitnessScores, d_fitnessScores, populationSize * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(populations, d_populations, size * populationSize * sizeof(int), cudaMemcpyDeviceToHost);
        
        // reorder population to fitness ascending order
        reorderPopulations(size, populationSize, populations, localFitnessScores);
        if (e % LOG_EVERY == 0) {
            std::cout << "EPOCH: " << e << "\t";
            std::cout << "Best fitness score: " << localFitnessScores[0] << " | distance from optimal: " << localFitnessScores[0] - IDEAL_DISTANCE << "\n";
            if (std::abs(localFitnessScores[0] - IDEAL_DISTANCE) < 1.0f) {
                std::cout << "IDEAL SOLUTION FOUND, EXITING MAIN LOOP...\n";
                break;
            }
        }
        
        cudaMemcpy(d_populations, populations, sizeof(int) * size * populationSize, cudaMemcpyHostToDevice);

        // this generates the new populations by replacing the non elite ones
        runEvolutionStep<<<testBlocksE, TPB>>>(size, populationSize, nonElitePopulationSize, d_populations, MUTATION_PROBABILITY, d_states);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { printf("Evolution step kernel runtime error: %s\n", cudaGetErrorString(err)); }

        /*
        if (!shakeUpDone && e > (int)(EPOCHS / 2)) {
            shakeUpDone = true;
            shakeUpWithRandomMutations <<<testBlocks, TPB >>> (size, populationSize, d_populations, d_states);
        }
        */
    }

    std::cout << "\nOverall best fitness score: " << localFitnessScores[0] << " | distance from optimal: " << localFitnessScores[0] - IDEAL_DISTANCE << "\n";

    cudaFree(d_states);
}

void readInitDataFromFile(std::vector<City>& cities) {
    std::ifstream infile(DATASET_FILEPATH);
    std::string line;
    std::vector<std::string> splitLine;
    size_t id = 0;

    while (std::getline(infile, line)) {
        splitLine = splitStringAtSpaces(line);
        cities.emplace_back(createNewCity(id, std::stof(splitLine[1]), std::stof(splitLine[2])));
        id++;
    }
}

void generateIdList(int size, int* idList) {
    for (int i = 0; i < size; i++) {
        idList[i] = i;
    }
}

void generateSamePopulations(int size, int* populations) {
    for (int p = 0; p < POPULATION_SIZE; p++) {
        for (int i = 0; i < size; i++) {
            populations[p * size + i] = i; // same incremental ids for all populations
        }
    }
}

void generateRandomlyShuffledInitPopulation(int size, int* populations) {
    std::random_device dev;
    std::mt19937 rng(dev());

    for (int p = 0; p < POPULATION_SIZE; p++) {
        for (int i = 0; i < size; i++) {
            std::uniform_int_distribution<std::mt19937::result_type> dist(0, size-1);
            auto const n = dist(rng);
            auto const tmp = populations[p * size + n];
            populations[p * size + n] = populations[p * size + i];
            populations[p * size + i] = tmp;
        }
    }
}

void displayPopulation(int size, int* populations) {
    for (int i = 0; i < size; i++) {
        std::cout << populations[i] << " ";
    }
    std::cout << std::endl;
}

void getPopulationsStat(int size, int* populations, float* xCoords, float* yCoords) {
    float bestDistance = std::numeric_limits<float>::max();
    float avgDistance = 0.0f;
    float tmpDistance = 0.0f;
    int idx = 0;

    for (int p = 0; p < POPULATION_SIZE; p++) {
        tmpDistance = 0.0f;
        for (int i = 1; i < size; i++) {
            idx = p * size + i;
            tmpDistance += distance(
                xCoords[populations[idx - 1]],
                yCoords[populations[idx - 1]],
                xCoords[populations[idx]],
                yCoords[populations[idx]]
            );
        }

        tmpDistance += distance(
            xCoords[0],
            yCoords[0],
            xCoords[p * size + size-1],
            yCoords[p * size + size-1]
        );

        if (tmpDistance < bestDistance)
            bestDistance = tmpDistance;

        avgDistance += tmpDistance;
    }

    avgDistance /= size;

    std::cout << "avg distance: " << avgDistance << " --- best distance: " << bestDistance << "\n";
}

bool hasDuplicate(int arr[], int n) {
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (arr[i] == arr[j]) {
                std::cout << arr[i] << " " << arr[j] << "\n";
            }
        }
    }
    return false;
}

int main(int argc, char** argv) {
    if (argc == 0) {

    }

    // step 1, basics like reading the dataset, parsing the datapoints, etc.
    std::vector<City> cities;
    readInitDataFromFile(cities);
    std::cout << "[i] lines parsed: " << cities.size() << "\n";

    // id list is to store correlation between cities and coords, I'm not sure if I even need this...
    // TODO: check if I need this
    int* idList;
    idList = (int*)malloc(cities.size() * sizeof(int));
    generateIdList(cities.size(), idList);
    std::cout << "[i] id list generated\n";

    // so populations is a flat array, indexed sequentially for each population like pId * size + currentPopId
    int* populations;
    populations = static_cast<int*>(malloc(cities.size() * sizeof(int) * POPULATION_SIZE));
    generateSamePopulations(cities.size(), populations);
    // TODO: later should write some tests and validation maybe...
    generateRandomlyShuffledInitPopulation(cities.size(), populations);
    
    // displayPopulation(cities.size(), populations);

    cudaError_t err;

    // TODO: check back if i even need this array
    int* d_idList;
    err = cudaMalloc((void**)&d_idList, cities.size() * sizeof(int));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed for ids: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // step 2, main arrays for calculations
    float* d_xCoordList;
    float* d_yCoordList;
    int* d_populations;

    err = cudaMalloc((void**)&d_xCoordList, cities.size() * sizeof(float));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed for x cords: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc((void**)&d_yCoordList, cities.size() * sizeof(float));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed for y cords: %s\n", cudaGetErrorString(err));
        return 1;
    }
    std::cout << "malloc and cudaMalloc done\n";

    err = cudaMalloc((void**)&d_populations, cities.size() * sizeof(int) * POPULATION_SIZE);
    if (err != cudaSuccess) {
        printf("cudaMalloc failed for populations: %s\n", cudaGetErrorString(err));
        return 1;
    }
    std::cout << "malloc and cudaMalloc done\n";

    // step 3, move data to device
    float* xCoordList = static_cast<float*>(malloc(cities.size() * sizeof(float)));
    float* yCoordList = static_cast<float*>(malloc(cities.size() * sizeof(float)));

    getCoordsFromArray(cities, xCoordList, yCoordList);
    // getPopulationsStat(cities.size(), populations, xCoordList, yCoordList);

    cudaMemcpy(d_xCoordList, xCoordList, cities.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_yCoordList, yCoordList, cities.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_populations, populations, cities.size() * sizeof(int) * POPULATION_SIZE, cudaMemcpyHostToDevice);

    // TODO: handle multithreading better, dont use only one block
    int nonEliteArraySize = (int) (float(POPULATION_SIZE) * (1.0f - ELITE_RATIO));
    std::cout << "Non-elite population size: " << nonEliteArraySize << "\n";
    executeEpochs(1, POPULATION_SIZE, cities.size(), POPULATION_SIZE, populations, nonEliteArraySize, d_populations, d_xCoordList, d_yCoordList);

    cudaMemcpy(populations, d_populations, cities.size() * sizeof(int) * POPULATION_SIZE, cudaMemcpyDeviceToHost);

    std::vector<int> checkArr;
    for (int i = 0; i < cities.size(); i++) {
        std::cout << populations[i] << " ";
        checkArr.push_back(populations[i]);
    }
    auto duplicateTest = hasDuplicate(checkArr.data(), cities.size());
    std::cout << "\nDuplicate found (bool): " << duplicateTest << "\n";

    free(idList);
    free(populations);
    free(xCoordList);
    free(yCoordList);
    cudaFree(d_idList);
    cudaFree(d_xCoordList);
    cudaFree(d_yCoordList);


    return 0;
}