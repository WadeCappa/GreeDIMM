#ifndef RIPPLES_MPI_STREAMING_ENGINE_H
#define RIPPLES_MPI_STREAMING_ENGINE_H

#include <vector>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <map>
#include <mutex>
#include <iostream> 
#include <utility>
#include <chrono>
#include <math.h>
#include <algorithm>
#include <deque>

#include <thread>
#include <future>

#include <cmath>

namespace ripples {
namespace mpi {

typedef struct origin {
    int source;
    int seed;
} Origin;

typedef struct candidateSet {
    unsigned int vertex;
    std::vector<int> covered;
} CandidateSet;

class ThresholdBucket 
{
    private:
    ripples::Bitmask<int> localCovered;
    std::vector<std::pair<unsigned int, size_t>> seeds;
    double marginalGainThreshold;
    int k;
    size_t theta;

    public:
    ThresholdBucket(size_t theta, size_t deltaZero, int k, double epsilon, size_t numBucket) 
        : localCovered(theta)
    {
        this->theta = theta;
        this->marginalGainThreshold = ( (double)deltaZero / (double)( 2 * k )) * (double)std::pow(1 + epsilon, numBucket);
        this->k = k;
    }

    size_t getUtility() 
    {
        return localCovered.popcount();
    }

    std::vector<std::pair<unsigned int, size_t>> getSeeds()
    {
        return this->seeds;
    }

    size_t getTotalCovered()
    {
        return this->seeds.size();
    }

    bool attemptInsert(const CandidateSet& element) 
    {
        if (this->seeds.size() >= this->k)
        {
            return false;
        }
       
        std::vector<size_t> temp;
        for (const auto e : element.covered) {
            if (!localCovered.get(e))
                temp.push_back(e);
        }

        if (temp.size() >= this->marginalGainThreshold) {
            for (const auto e : temp) {
                localCovered.set(e);
            }

            this->seeds.push_back(std::make_pair(element.vertex, temp.size()));             
            return true;
        }

        return false;
    }
};

class BucketController 
{
    private:
    std::vector<ThresholdBucket*> buckets;
    std::vector<std::vector<ThresholdBucket*>> threadMap;

    int k;
    size_t theta;
    double epsilon;
    
    size_t deltaZero;    
    int fullBuckets = 0;

    public:
    void CreateBuckets(size_t deltaZero)
    {
        this->deltaZero = deltaZero;

        int num_buckets = (int)(0.5 + [](double val, double base) {
            return log2(val) / log2(base);
        }((double)k, (1+this->epsilon)));

        // std::cout << "deltazero: " << this->deltaZero << ", number of buckets: " << num_buckets << std::endl;

        for (int i = 0; i < num_buckets + 1; i++)
        {
            this->buckets.push_back(new ThresholdBucket(this->theta, this->deltaZero, this->k, this->epsilon, i));
        }
    }

    void MapBucketsToThreads(const unsigned int threads)
    {
        // build mapping of each thread to number of buckets
        int buckets_per_thread = std::ceil((double)this->buckets.size() / ((double)threads));
        // std::cout << "number of buckets; " << this->buckets.size() << ", threads; " << threads << ", buckets per thread; " << buckets_per_thread << std::endl;
        this->threadMap.resize(threads);

        int bucket_index = 0;
        for (int i = 0; i < this->threadMap.size(); i++ )
        {
            int buckets_added = 0;
            while (bucket_index < this->buckets.size() && buckets_added < buckets_per_thread)
            {
                this->threadMap[i].push_back(this->buckets.at(bucket_index));
                bucket_index++;
                buckets_added++;
            }
        }
    }

    void ProcessElements(
        const unsigned int threads, 
        const std::vector<std::pair<int, Origin>> &availability_index,
        const std::vector<std::vector<CandidateSet>> &element_matrix
    )
    {
        #pragma omp parallel for num_threads(threads)
        for (int i = 0; i < this->threadMap.size(); i++)
        {
            const std::vector<ThresholdBucket*>& thread_buckets = this->threadMap[i];
            int local_dummy_value = 0;

            for (int local_received_index = 0; local_received_index < availability_index.size(); local_received_index++)
            {
                while (true) 
                {
                    if (availability_index[local_received_index].first == 1)
                    {
                        break;
                    }

                    # pragma omp atomic
                    local_dummy_value++;
                }

                // std::cout << "inserting seed " << local_received_index << " " << availability_index[local_received_index].first << std::endl;

                for (const auto & b : thread_buckets)
                {
                    const auto & p = availability_index[local_received_index].second;
                    b->attemptInsert(element_matrix[p.source][p.seed]);
                }
            }
        }
    }

    size_t GetNumberOfBuckets()
    {
        return this->buckets.size();
    }

    std::vector<ThresholdBucket*> GetBuckets()
    {
        return this->buckets;
    }

    bool Initialized()
    {
        return this->buckets.size() > 0;
    }

    bool AllBucketsFull()
    {
        // return this->buckets.size() == 0 ? false : this->fullBuckets == this->buckets.size();
        return false;
    }

    std::pair<std::vector<unsigned int>, size_t> GetBestSeeds()
    {
        size_t max_covered = 0;
        int max_covered_index = 0;

        for (int i = 0; i < this->buckets.size(); i++)
        {
            size_t bucket_utility = this->buckets.at(i)->getUtility();
            if (bucket_utility > max_covered) {
                max_covered = bucket_utility;
                max_covered_index = i;
            }
        }
        
        std::vector<unsigned int> seeds;
        for (const auto p : this->buckets.at(max_covered_index)->getSeeds()) {
            seeds.push_back(p.first);
        }

        // std::cout << "selected bucket " << max_covered_index << " with size of " << this->buckets.at(max_covered_index)->getSeeds().size() << std::endl;

        return std::make_pair(seeds, max_covered);
    }

    BucketController(int k, size_t theta, double epsilon)
    {
        this->theta = theta;
        this->k = k;
        this->epsilon = epsilon;
    }

    ~BucketController() 
    {
        for (auto b : this->buckets)
        {
            delete b;
        }
    }
};

template <typename GraphTy>
class StreamingRandGreedIEngine 
{
    private: 
    std::vector<unsigned int> buffer;
    MPI_Request request;
    BucketController buckets;
    const CommunicationEngine<GraphTy> &cEngine;

    int k;
    int kprime;
    int active_senders;
    int first_values_from_senders;
    int number_of_senders;
    size_t theta;
    double epsilon;
    TimerAggregator& timer;

    std::vector<std::vector<CandidateSet>> element_matrix;
    std::vector<std::vector<unsigned int>> local_seed_sets;
    std::vector<std::pair<int, Origin>> availability_index;
    std::vector<unsigned int> local_utilities;

    static void GetSeedSet(std::vector<unsigned int>& receive_buffer, std::vector<unsigned int>& seed_set)
    {
        int negatives = 0;
        unsigned int* e = receive_buffer.data();
        for (; negatives != 2; e++)
        {
            if (negatives == 1 && *e != -1)
            {
                seed_set.push_back(*e);
            }

            if (*e == -1)
            {
                negatives++;
            }
        }
    }

    CandidateSet ExtractElement(std::vector<unsigned int>& receive_buffer, std::map<int, std::vector<int>> &newSolutionSpace)
    {
        std::vector<int> &seed_data = newSolutionSpace[receive_buffer[0]];
        for (unsigned int* e = receive_buffer.data() + 1; *(e) != -1; e++) 
        {
            seed_data.push_back(*e);
        }

        return (CandidateSet){receive_buffer[0], seed_data};
    }

    void ResetBuffer(std::vector<unsigned int>& receive_buffer)
    {
        this->cEngine.QueueReceive(receive_buffer.data(), receive_buffer.size(), this->request);
    }

    void HandleStatus(const MPI_Status& status)
    {
        if (status.MPI_TAG == 0)
        {
            this->first_values_from_senders++;
        }

        if (status.MPI_TAG > this->kprime - 1)
        {
            this->active_senders--;
        }
    }

    void ReceiveNextSend(MPI_Status& status)
    {
        this->timer.receiveTimer.startTimer();
        MPI_Wait(&(this->request), &status);
        this->timer.receiveTimer.endTimer();
    }

    static void WaitToProcess(int &dummy_value, int &buckets_initialized)
    {
        while (true) 
        {
            # pragma omp atomic
            dummy_value++;
            
            if (buckets_initialized == 1)
            {
                break;
            }
        }
    }

    public:
    StreamingRandGreedIEngine(
        int k, int kprime, size_t theta,
        double epsilon, int number_of_senders, 
        const CommunicationEngine<GraphTy> &cEngine,
        TimerAggregator& timer
    ) 
        : buckets(k, theta, epsilon), cEngine(cEngine), timer(timer), buffer(cEngine.GetSendReceiveBufferSize(theta))
    {
        this->active_senders = number_of_senders;
        this->k = k;
        this->kprime = kprime;
        this->epsilon = epsilon;
        this->number_of_senders = number_of_senders;
        this->first_values_from_senders = 0;
        this->theta = theta;
    
        this->element_matrix = std::vector<std::vector<CandidateSet>>(
            this->number_of_senders, 
            std::vector<CandidateSet>(
                this->kprime,
                (CandidateSet){0, std::vector<int>()}
            )
        );

        this->local_seed_sets = std::vector<std::vector<unsigned int>>(this->number_of_senders);

        this->availability_index = std::vector<std::pair<int, Origin>>(
            this->number_of_senders * this->kprime, 
            std::make_pair(0, (Origin){-1,-1})
        );

        this->local_utilities = std::vector<unsigned int>(this->number_of_senders);
    }

    ~StreamingRandGreedIEngine()
    {

    }

    std::pair<std::vector<unsigned int>, int> Stream(std::map<int, std::vector<int>> &newSolutionSpace)
    {
        int buckets_initialized = 0;
        int dummy_value = 0;
        int kill_processors = 0;
        size_t maxVal = 0;

        int threads = omp_get_max_threads();
        omp_set_nested(2);

        # pragma omp parallel num_threads(2)
        {
            if (omp_get_thread_num() == 0) // receiver
            {
                // std::cout << "RECEIVING WITH THREAD ID " << omp_get_thread_num() << std::endl;

                for (int i = 0; i < (this->number_of_senders * this->kprime); i++)
                {                   
                    MPI_Status status;
                    this->ResetBuffer(this->buffer);
                    this->ReceiveNextSend(status);

                    this->timer.processingReceiveTimer.startTimer();

                    int tag = status.MPI_TAG > this->kprime - 1 ? this->kprime - 1 : status.MPI_TAG;
                    int source = status.MPI_SOURCE - 1;

                    // std::cout << "tag: " << tag << ", source: " << source << std::endl;

                    if (status.MPI_TAG > this->kprime - 1)
                    {
                        // std::cout << "got last seed" << std::endl;

                        local_utilities[source] = status.MPI_TAG;
                        this->GetSeedSet(this->buffer, local_seed_sets[source]);
                    }

                    element_matrix[source][tag] = this->ExtractElement(this->buffer, newSolutionSpace);

                    this->timer.processingReceiveTimer.endTimer();

                    // std::cout << "extracted elements" << std::endl;

                    if (buckets_initialized == 0)
                    {
                        maxVal = std::max(maxVal, element_matrix[source][tag].covered.size());
                    }

                    availability_index[i].second.source = source;
                    availability_index[i].second.seed = tag;

                    this->timer.atomicUpdateTimer.startTimer();

                    #pragma omp atomic 
                    availability_index[i].first++;
                    
                    this->timer.atomicUpdateTimer.endTimer();

                    this->HandleStatus(status);

                    if (this->first_values_from_senders == this->number_of_senders && buckets_initialized == 0)
                    {
                        #pragma omp atomic 
                        buckets_initialized++;
                    }

                    // std::cout << "handled status" << std::endl;
                }

                // std::cout << "killing processors, waiting for them to exit..." << std::endl;
            }
            else // processor
            {   
                this->WaitToProcess(dummy_value, buckets_initialized);

                this->timer.initBucketTimer.startTimer();
                this->buckets.CreateBuckets(maxVal);
                this->buckets.MapBucketsToThreads(threads - 1);
                this->timer.initBucketTimer.endTimer();

                // std::cout << "starting to process elements..." << std::endl;

                this->timer.max_k_globalTimer.startTimer();
                this->buckets.ProcessElements(threads - 1, availability_index, element_matrix);                
                this->timer.max_k_globalTimer.endTimer();
            }
        }

        auto bestSeeds = this->buckets.GetBestSeeds();

        for (int i = 0; i < local_utilities.size(); i++)
        {
            // std::cout << "streaming utility is " << bestSeeds.second << std::endl;
            // std::cout << "looking at " << local_utilities[i] << " from local process " << i+1 << std::endl;
            if (local_utilities[i] > bestSeeds.second)
            {
                bestSeeds.second = local_utilities[i];
                bestSeeds.first = local_seed_sets[i];
            }
        }

        // std::cout << "number of seeds: " << bestSeeds.first.size() << std::endl;

        return bestSeeds;
    }
};

}

}
#endif  