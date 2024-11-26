
#ifndef RIPPLES_MPI_IMM_COMM_ENGINE_H
#define RIPPLES_MPI_IMM_COMM_ENGINE_H

#include <vector>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <map>
#include <mutex>
#include <iostream> 
#include <utility>
#include <chrono>
#include <random>
#include <cmath>

namespace ripples {
    namespace mpi {

        typedef struct linearizedSetsSize {
            int count;
            std::vector<int> countPerProcess;
        } LinearizedSetsSize;

        template <typename GraphTy>
        class CommunicationEngine
        {
            private: 
            const bool streaming;
            const int block_size = (1024 / sizeof(unsigned int)); // 256 ints per block

            const unsigned int world_size;
            const unsigned int world_rank;

            static const unsigned int RRR_SET_STOP = -1;
            static const unsigned int CANDIDATE_SET_STOP = -2;
            static const unsigned int BLOCK_STOP = -3;

            public:
            CommunicationEngine
            (
                const unsigned int world_size, 
                const unsigned int world_rank,
                const bool streaming
            ) : world_size(world_size), world_rank(world_rank), streaming(streaming)
            {
                // TODO: Extract all MPI_Datatype operations into constructor and deconstructor. Last time you tried
                //  this you got MPI_DATATYPE_NULL errors from mpich. Figure out how to prevent this.
            }

            ~CommunicationEngine() 
            {
            }

            size_t GetSendReceiveBufferSize(const size_t data_size) const
            {
                return (size_t)(data_size + (size_t)(data_size % this->block_size == 0 ? 0 : (this->block_size - (data_size % this->block_size))));
            }

            unsigned int GetRank() const
            {
                return this->world_rank;
            }

            unsigned int GetSize() const
            {
                return this->world_size;
            }

            int TestSend(MPI_Request &request) const
            {
                MPI_Status status;
                int flag = 0;
                MPI_Test(&request, &flag, &status);
                return flag;
            }

            void SendNextSeed(
                const unsigned int* data, 
                const size_t data_size, 
                const unsigned int tag
            ) const
            {
                MPI_Datatype batch_int;
                MPI_Type_contiguous( this->block_size, MPI_INT, &(batch_int) );
                MPI_Type_commit(&(batch_int));

                MPI_Send (
                    data,
                    data_size / this->block_size,
                    batch_int, 0,
                    tag,
                    MPI_COMM_WORLD
                );
                MPI_Type_free(&(batch_int));
            }

            void QueueNextSeed(
                const unsigned int* data, 
                const size_t data_size, 
                const unsigned int tag, 
                MPI_Request &request
            ) const
            {
                MPI_Datatype batch_int;
                MPI_Type_contiguous( this->block_size, MPI_INT, &(batch_int) );
                MPI_Type_commit(&(batch_int));
                MPI_Isend (
                    data,
                    data_size / this->block_size,
                    batch_int, 
                    0,
                    tag,
                    MPI_COMM_WORLD,
                    &request
                );
                MPI_Type_free(&(batch_int));
            }

            size_t GetBlockSize() const
            {
                return this->block_size;
            }

            std::vector<int> DistributeVertices (
                const GraphTy &G
            ) const {
                size_t nodes = G.num_nodes();

                std::vector<int> vertex_mapping(nodes, -1);
                unsigned int seed = (unsigned int)time(0);
                MPI_Bcast(&seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

                std::uniform_int_distribution<int> uniform_distribution(this->streaming ? 1 : 0, this->world_size - 1);
                std::default_random_engine number_selecter(seed);

                for (int i = 0; i < nodes; i++) {
                    vertex_mapping[i] = uniform_distribution(number_selecter);
                }

                return vertex_mapping;
            }

            void AggregateThroughAllToAll(
                const TransposeRRRSets<GraphTy> &tRRRSets,
                const std::vector<int> &vertexToProcess,
                std::vector<size_t> &old_sampling_sizes,
                const size_t localThetaPrime,
                std::map<int, std::vector<int>> &aggregateSets
            ) const
            {
                std::vector<unsigned int> countPerProcess;
                size_t totalCount = this->countPerProcessForBatchSend(countPerProcess, tRRRSets, vertexToProcess, old_sampling_sizes);

                std::vector<unsigned int> psum;
                this->buildPrefixSum(psum, countPerProcess.data(), this->world_size);

                std::vector<unsigned int> linear_sets = this->LinearizeColumnsInRRRSetMatrix(
                    tRRRSets, 
                    vertexToProcess, 
                    psum, 
                    totalCount, 
                    old_sampling_sizes
                );

                for (size_t i = 0; i < old_sampling_sizes.size(); i++)
                {
                    old_sampling_sizes[i] = tRRRSets.sets[i].second.size();
                }

                // std::cout << "linearizing data, rank = " << world_rank << std::endl;

                for (auto & processCount : countPerProcess) {
                    processCount = processCount / block_size;
                }
                
                this->GetProcessSpecificVertexRRRSets(aggregateSets, linear_sets.data(), countPerProcess.data(), localThetaPrime);
            }

            // void BuildBufferForStreamingSend(std::vector<unsigned int>& buffer) const
            // {   
            //     size_t total_data = this->GetSendReceiveBufferSize((size_t)sendData.first);

            //     buffer.resize(total_data);

            //     size_t i = 0;

            //     for (; i < sendData.first; i++)
            //     {
            //         buffer[i] = sendData.second[i];
            //     }

            //     for (; i < total_data; i++)
            //     {
            //         buffer[i] = -1; //TODO: Verify that this end of message flag is correct
            //     }

            //     delete sendData.second;
            // }

            void QueueReceive(unsigned int* buffer, size_t max_receive_size, MPI_Request &request) const 
            {
                MPI_Datatype batch_int;
                MPI_Type_contiguous( this->block_size, MPI_INT, &(batch_int) );
                MPI_Type_commit(&(batch_int));
                MPI_Irecv(
                    buffer,
                    max_receive_size / this->block_size,
                    batch_int,
                    MPI_ANY_SOURCE,
                    MPI_ANY_TAG,
                    MPI_COMM_WORLD,
                    &request
                );
                MPI_Type_free(&(batch_int));
            }

            void WaitForSend(MPI_Request &request) const 
            {
                MPI_Status status;
                MPI_Wait(&request, &status);
            }

            size_t GatherPartialSolutions(
                std::vector<unsigned int> &receiveBuffer, // modified
                const std::pair<std::vector<unsigned int>, ssize_t> &localSeeds,
                const std::map<int, std::vector<int>> &solutionSpace,
                const MPI_Comm &commWorld
            ) const
            {
                std::vector<unsigned int> sendBuffer;
                size_t totalSendData = this->linearizeLocalSeeds(sendBuffer, solutionSpace, localSeeds.first, localSeeds.second);
                return this->aggregateAggregateSets(receiveBuffer, totalSendData, sendBuffer.data(), commWorld);
            }

            // Returns total count, modifies the countPerProcess vector;
            size_t count(std::vector<unsigned int>& countPerProcess, TransposeRRRSets<GraphTy> &tRRRSets, std::vector<int> vertexToProcessor) const
            {
                std::vector<std::mutex> locksPerProcess(this->world_size);
                countPerProcess.resize(this->world_size, 0);

                size_t count = 0;
                #pragma omp parallel for reduction(+:count)
                for (int i = 0; i < vertexToProcessor.size(); i++) 
                {
                    count += tRRRSets.sets[i].second.size() + 2;

                    locksPerProcess[vertexToProcessor[i]].lock();
                    countPerProcess[vertexToProcessor[i]] += tRRRSets.sets[i].second.size() + 2;
                    locksPerProcess[vertexToProcessor[i]].unlock();
                }   

                return count;
            }

            size_t countPerProcessForBatchSend(
                std::vector<unsigned int>& countPerProcess, 
                const TransposeRRRSets<GraphTy> &tRRRSets, 
                const std::vector<int> vertexToProcessor, 
                const std::vector<size_t>& old_sizes
            ) const 
            {
                std::vector<std::mutex> locksPerProcess(this->world_size);
                countPerProcess.resize(this->world_size, 0);

                size_t count = 0;

                #pragma omp parallel for reduction(+:count)
                for (size_t i = 0; i < vertexToProcessor.size(); i++) 
                {
                    size_t old_count = old_sizes[i];
                    size_t new_size = tRRRSets.sets[i].second.size();
                    count += new_size - old_count + 2;
                    // count += tRRRSets.sets[i].second.size() + 2;

                    locksPerProcess[vertexToProcessor[i]].lock();
                    countPerProcess[vertexToProcessor[i]] += new_size - old_count + 2;
                    locksPerProcess[vertexToProcessor[i]].unlock();
                }   

                size_t block_tax = 0;
                for (auto & processCount : countPerProcess)
                {
                    size_t blocks_to_add = this->block_size - (processCount % this->block_size);
                    // size_t blocks_to_add = processCount % block_size;
                    block_tax += blocks_to_add;
                    processCount += blocks_to_add;
                }

                return count + block_tax;
            }

            void buildPrefixSum(std::vector<unsigned int>& prefixSum, unsigned int* v, unsigned int total_count) const
            {
                prefixSum.resize(total_count, 0);

                for (int i = 1; i < total_count; i++) {
                    prefixSum[i] = prefixSum[i - 1] + v[i - 1];
                }
            }

            void SendToGlobal(int* data, int size, int tag) const
            {
                MPI_Send(data, size, MPI_INT, 0, tag,
                    MPI_COMM_WORLD);
            }

            
            size_t linearizeLocalSeeds(
                std::vector<unsigned int>& sendBuffer, // modified
                const std::map<int, std::vector<int>> &solutionSpace, 
                const std::vector<unsigned int>& localSeeds, 
                const size_t total_utility
            ) const
            {
                std::vector<std::pair<int, int>> setsPrefixSum;

                size_t runningSum = 0;
                for (const auto & seed : localSeeds) {
                    auto seed_set = solutionSpace.at(seed);
                    setsPrefixSum.push_back(std::make_pair(seed, runningSum));
                    runningSum += seed_set.size() + 2;
                }

                size_t totalData = runningSum + 3;
                size_t total_block_data = totalData + (this->block_size - (totalData % this->block_size));

                sendBuffer.resize(total_block_data, -1);

                #pragma omp parallel for
                for (int setIndex = 0; setIndex < setsPrefixSum.size(); setIndex++) {
                    sendBuffer[setsPrefixSum[setIndex].second] = setsPrefixSum[setIndex].first;
                    int offset = setsPrefixSum[setIndex].second + 1;
                    for (const auto & RRRSetID : solutionSpace.at(setsPrefixSum[setIndex].first)) {
                        sendBuffer[offset++] = RRRSetID;
                    }
                    sendBuffer[setsPrefixSum[setIndex].second + solutionSpace.at(setsPrefixSum[setIndex].first).size() + 1] = this->RRR_SET_STOP;
                }

                // mark end of process seeds
                sendBuffer[totalData - 2] = this->CANDIDATE_SET_STOP;

                // mark total utility of local process
                sendBuffer[totalData - 1] = total_utility;

                for (int i = totalData; i < total_block_data; i++)
                {
                    sendBuffer[i] = this->BLOCK_STOP;
                }

                return total_block_data;
            }

            std::pair<size_t, int*> LinearizeSingleSeed(const int vertexID, const std::vector<int>& set) const
            {
                int* linearAggregateSets = new int[set.size()+2];
                size_t offset = 1;

                *linearAggregateSets = vertexID;

                for (const auto & RRRSetID : set) {
                    *(linearAggregateSets + offset++) = RRRSetID;
                }

                *(linearAggregateSets + offset) = -1;

                return std::make_pair(set.size()+2, linearAggregateSets);
            }

            std::vector<unsigned int> LinearizeColumnsInRRRSetMatrix(
                const TransposeRRRSets<GraphTy> &tRRRSets, 
                const std::vector<int> &vertexToProcessor, 
                const std::vector<unsigned int> &dataStartPartialSum, 
                const size_t total_data, 
                const std::vector<size_t>& old_sizes
            ) const
            {
                std::vector<unsigned int> linear_sets(total_data);

                #pragma omp parallel for
                for (int rank = 0; rank < this->world_size; rank++) {
                    this->LinearizeRank(tRRRSets, linear_sets, vertexToProcessor, dataStartPartialSum, rank, old_sizes);
                }

                return linear_sets;
            }

            void LinearizeRank(
                const TransposeRRRSets<GraphTy> &tRRRSets, 
                std::vector<unsigned int>& linearTRRRSets, 
                const std::vector<int> &vertexToProcessor, 
                const std::vector<unsigned int> &dataStartPartialSum, 
                const int rank, 
                const std::vector<size_t>& old_sizes
            ) const
            {
                size_t index = dataStartPartialSum[rank];
                for (size_t i = 0; i < tRRRSets.sets.size(); i++) {
                    if (vertexToProcessor[i] == rank) {
                        linearTRRRSets[index++] = i;
                        for (auto itr = tRRRSets.sets[i].second.begin() + old_sizes[i]; itr != tRRRSets.sets[i].second.end(); itr++) {
                            const auto & RRRid = *itr;
                            linearTRRRSets[index++] = RRRid;
                        }
                        linearTRRRSets[index++] = -1;
                    }
                }

                size_t stop = rank < (dataStartPartialSum.size() - 1) ? dataStartPartialSum[rank+1] : linearTRRRSets.size() ;
                for (; index < stop; index++) {
                    linearTRRRSets[index] = -1;
                }
            }

            void DEBUG_printLinearizedSets(int* linearizedData, int totalData) const
            {
                for (int i = 0; i < totalData; i++) {
                    std::cout << linearizedData[i] << " ";
                    if (linearizedData[i] == -1) {
                    std::cout << std::endl;
                    }
                }
                std::cout << std::endl;
            }

            /// @brief every process calls this function after they are sent the bulk data from MPI_alltoallV. Each
            ///     process only gets a certain set of values. 
            /// @param tRRRSets is the local process's tRRRSets. This should be a new object (i.e, empty).
            /// @param data is the data collected from MPI_alltoallV
            /// @param receivedDataSizes is the data collected from MPI_alltoall
            /// @param p number of processes
            /// @param RRRIDsPerProcess the upper bound of the maximum number of RRRIDs that each process is responsible for generating
            void AggregateTRRRSets(
                std::map<int, std::vector<int>> &aggregateSets, 
                unsigned int* data, 
                unsigned int* receivedDataSizes, 
                size_t RRRIDsPerProcess
            ) const
            {
                size_t totalData = 0;
                for (int i = 0; i < this->world_size; i++) {
                    totalData += *(receivedDataSizes + i);
                }            

                int receivedDataRank = 0;

                // cycle over data
                int vertexID = *data; // TODO: Fix all these datatypes, need to be vector_type, will not scale with int as number of nodes increases
                // aggregateSets.insert({ vertexID, std::vector<int>() });
                for (size_t rankDataProcessed = 1, i = 1; i < totalData - 1; i++, rankDataProcessed++) {
                    if (*(data + i) == -1 && *(data + i + 1) != -1) {
                        vertexID = *(data + ++i);
                        rankDataProcessed++;
                        // aggregateSets.insert({ vertexID, std::vector<int>() });
                    }

                    else if (*(data + i) != -1) {
                        // add each RRRSetID to the map indexed by the target vertex id. 
                        // The RRRSetIDs need to be modified such that they are unique from the sent process. 
                        // This is possible because the order is known and the expected sizes of data that should be sent. 
                        aggregateSets[vertexID].push_back(*(data + i) + (RRRIDsPerProcess * receivedDataRank));
                    }

                    // track the current receiving process rank
                    if (rankDataProcessed > *(receivedDataSizes + receivedDataRank)) {
                        receivedDataRank += 1;
                        rankDataProcessed = 1;
                    }
                }
            }

            void deserializeGatherData (
                std::map<int, std::vector<int>> &newSolutionSpace, // modified
                std::vector<std::pair<std::vector<unsigned int>, unsigned int>> &candidateSets, // modified
                unsigned int* data, 
                size_t totalData
            ) const {
                size_t position = 0;

                while (position < totalData) {
                    std::vector<unsigned int> currentCandidateSet;

                    while (data[position] != CANDIDATE_SET_STOP) {
                        unsigned int currentVertex = data[position++];
                        currentCandidateSet.push_back(currentVertex);
                        std::vector<int> &currentRRRSet = newSolutionSpace[currentVertex];

                        while (data[position] != RRR_SET_STOP) {
                            currentRRRSet.push_back(data[position]);
                            position++;
                        }
                    
                        // data[position] == -1, this is the end of current vertex, next element is either -2 or the next vertex
                    
                        while (data[position] == RRR_SET_STOP) {
                            position++;
                        }
                    }

                    // data[position] == -2, this tells us that we have reached the end of currentVertex's RRR sets, 
                    //  and that the next element is the utility

                    position++;
                    candidateSets.push_back(std::make_pair(currentCandidateSet, data[position]));
                    position++;

                    while (data[position] == BLOCK_STOP && position < totalData) {
                        position++;
                    }
                }
            }

            void GetProcessSpecificVertexRRRSets(
                std::map<int, std::vector<int>> &aggregateSets, 
                unsigned int* linearizedData,
                unsigned int* countPerProcess,
                size_t RRRIDsPerProcess
            ) const {
                unsigned int* receiveSizes = new unsigned int[this->world_size];

                MPI_Alltoall(countPerProcess, 1, MPI_UNSIGNED, receiveSizes, 1, MPI_UNSIGNED, MPI_COMM_WORLD);

                size_t totalReceivingBlocks = 0;
                for (int i = 0; i < this->world_size; i++) {
                    totalReceivingBlocks += (unsigned int)*(receiveSizes + i);
                // std::cout << "rank " << GetRank() << " receiving " << (unsigned int)*(receiveSizes + i) << " blocks from " << i << std::endl;
                }
                
                unsigned int* linearizedLocalData = new unsigned int[totalReceivingBlocks * this->block_size];

                std::vector<unsigned int> sendPrefixSum;
                buildPrefixSum(sendPrefixSum, countPerProcess, this->world_size);
                std::vector<unsigned int> receivePrefixSum; 
                buildPrefixSum(receivePrefixSum, receiveSizes, this->world_size);

                // call alltoall_v
                MPI_Datatype batch_int;
                MPI_Type_contiguous( this->block_size, MPI_INT, &(batch_int) );
                MPI_Type_commit(&(batch_int));
                MPI_Alltoallv(
                    linearizedData, 
                    (int*)countPerProcess, 
                    (int*)sendPrefixSum.data(), 
                    batch_int, 
                    linearizedLocalData, 
                    (int*)receiveSizes, 
                    (int*)receivePrefixSum.data(),
                    batch_int,
                    MPI_COMM_WORLD
                );
                MPI_Type_free(&(batch_int));

                for (int i = 0; i < this->world_size; i++) {
                    receiveSizes[i] *= this->block_size;
                }

                // TODO: make this not terrible
            //  for streaming this might crash so you may need to add a dummy value to aggregate sets.
                if (this->world_rank == 0 && this->streaming == true) {
                aggregateSets.insert({0, std::vector<int>()});
                } else {
                    this->AggregateTRRRSets(aggregateSets, linearizedLocalData, receiveSizes, RRRIDsPerProcess);
                }
                
                delete receiveSizes;
                delete linearizedLocalData;
            }

            std::vector<unsigned int> distributeSeedSet(const std::vector<unsigned int> &seed_set, const size_t k) const {
                std::vector<unsigned int> res = std::vector<unsigned int>(k, UINT_MAX);
                for (int i = 0; i < seed_set.size(); i++) {
                    res[i] = seed_set[i];
                }

                MPI_Bcast(res.data(), k, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

                for (size_t i = res.size() - 1; i >= 0 && res[i] == UINT_MAX; i--) {
                    res.pop_back();
                }

                return res;
            }

            unsigned int sumCoverage(unsigned int coverage) const {
                unsigned int res = 0;
                MPI_Reduce(&coverage, &res, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
                return res;
            }

            void reduceLocalNonCovered(const std::vector<unsigned int> &local_covered, std::vector<unsigned int> &global_covered) const {
                MPI_Reduce(
                    local_covered.data(),
                    global_covered.data(),
                    local_covered.size(),
                    MPI_INT,
                    MPI_SUM,
                    0,
                    MPI_COMM_WORLD);
            }

            size_t aggregateAggregateSets(
                std::vector<unsigned int>& receiveBuffer, // modified
                const size_t totalGatherData, 
                const unsigned int* sendData,
                const MPI_Comm &commWorld
            ) const
            {
                std::vector<unsigned int> gatherSizes(this->world_size);
                gatherSizes[0] = -1; // TODO: Find out why this is here

                size_t total_block_send = totalGatherData / this->block_size;
                // std::cout << "totalGatherData " << totalGatherData << std::endl; 

                MPI_Gather(
                    &total_block_send, 1, MPI_INT,
                    gatherSizes.data(), 1, MPI_INT, 0, commWorld
                );

                std::vector<unsigned int> displacements;
                
                // mpi_gatherv for all buffers of linearized aggregateSets. 
                // TODO: Decipher this moon rune code
                size_t totalData = 0;
                if (gatherSizes[0] == -1) {
                    totalData = 1;
                }
                else{
                    for (int i = 0; i < this->world_size; i++) {
                        totalData += (unsigned int)gatherSizes[i];
                    }  

                    buildPrefixSum(displacements, gatherSizes.data(), this->world_size);
                }

                receiveBuffer.resize(totalData * this->block_size);

                MPI_Datatype batch_int;
                MPI_Type_contiguous( this->block_size, MPI_INT, &(batch_int) );
                MPI_Type_commit(&(batch_int));
                MPI_Gatherv(
                    sendData,
                    total_block_send,
                    batch_int,
                    (int*)receiveBuffer.data(),
                    (int*)gatherSizes.data(),
                    (int*)displacements.data(),
                    batch_int,
                    0,
                    commWorld
                );
                MPI_Type_free(&(batch_int));

                return totalData * this->block_size;
            }

            void distributeF(double* f) const
            {
                MPI_Bcast(f, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }
        };

        template <typename GraphTy>
        class CommunicationEngineBuilder
        {
            public:
            static CommunicationEngine<GraphTy> BuildCommunicationEngine(const bool streaming)
            {
                int world_size, world_rank;
                MPI_Comm_size(MPI_COMM_WORLD, &world_size);
                MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

                return CommunicationEngine<GraphTy>(world_size, world_rank, streaming);
            }
        };
    }
}

#endif  