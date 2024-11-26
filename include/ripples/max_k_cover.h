#include <vector>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <map>
#include <set>
#include <mutex>
#include <iostream>
#include "bitmask.h"
#include <utility>
#include <queue>
#include <cstdlib>
#include <bits/stdc++.h>
#include <cmath>
// #include "mpi/communication_engine.h"

template <typename GraphTy>
class MaxKCoverBase 
{
private:
    class NextMostInfluentialFinder
    { 
    protected:
        std::vector<unsigned int> vertex_subset;
        std::map<int, std::vector<int>*> replicated_data;
        ripples::Bitmask<int> covered;
        size_t subset_size;

    public:
        NextMostInfluentialFinder(
            size_t theta
        ) : covered(theta)
        {}

        virtual void Setup(const std::map<int, std::vector<int>>& original_data)
        {
            for (const auto & l : original_data)
            {
                this->replicated_data.insert({ l.first, new std::vector<int>(l.second.begin(), l.second.end()) });
            }
        }

        virtual ~NextMostInfluentialFinder()
        {
            // std::cout << "deallocating base class in max_k_cover..." << std::endl;
            for (const auto & l : this->replicated_data)
            {
                delete l.second;
            }
        }

        unsigned int GetUtility()
        {
            return covered.popcount();
        }

        virtual size_t findNextInfluential () = 0;

        virtual NextMostInfluentialFinder* setSubset (
            const std::vector<unsigned int>& subset_of_selection_sets,
            size_t subset_size
        ) = 0;

        virtual NextMostInfluentialFinder* reloadSubset () = 0;
    };

    class LazyGreedy : public NextMostInfluentialFinder
    {
    private:
        template <typename idTy>
        struct CompareMaxHeap {

            bool operator()(const std::pair<idTy, std::vector<idTy>*> a,
                            const std::pair<idTy, std::vector<idTy>*> b) {
                return a.second->size() < b.second->size();
            }
        };

        CompareMaxHeap<int> cmp;            
        std::vector<std::pair<int, std::vector<int>*>>* heap;

        void generateQueue(const std::vector<unsigned int>& subset_of_selection_sets, size_t subset_size)
        {
            for (int i = 0; i < subset_size; i++)
            {
                this->heap->at(i) = std::make_pair(subset_of_selection_sets[i], this->replicated_data.at(subset_of_selection_sets[i]));
            }

            std::make_heap(this->heap->begin(), this->heap->end(), this->cmp);
        }

    public:
        LazyGreedy(size_t theta) : NextMostInfluentialFinder(theta) {}

        ~LazyGreedy()
        {
            // std::cout << "deallocating Lazy-Greedy finder ..." << std::endl;
            delete heap;
        }

        NextMostInfluentialFinder* setSubset(const std::vector<unsigned int>& subset_of_selection_sets, size_t subset_size) override
        {
            this->subset_size = subset_size;
            this->heap = new std::vector<std::pair<int, std::vector<int>*>>(subset_size);

            generateQueue(subset_of_selection_sets, subset_size);
            this->vertex_subset = subset_of_selection_sets;
            return this;
        }

        NextMostInfluentialFinder* reloadSubset () override 
        {
            generateQueue(this->vertex_subset, this->subset_size);
            return this;
        }

        // TODO: this is broken, pull this code offline and figure out how to fix it. It never makes recusive calls, always accepts first solution.
        size_t findNextInfluential() override
        {
            // std::cout << "entered into finder" << std::endl;
            std::pair<int, std::vector<int>*> l = this->heap->front();
            std::pop_heap(this->heap->begin(), this->heap->end(), this->cmp);
            this->heap->pop_back();

            // std::cout << "found " << l.first << " from heap " << std::endl;

            std::vector<int> new_set;

            // remove RR IDs from l that are already covered. 
            for (unsigned int e: *(l.second)) {
                if (!(this->covered.get(e))) {
                    new_set.push_back(e);
                }
            }      

            if (this->replicated_data.find(l.first) == this->replicated_data.end())
            {
                // std::cout << "could not find " << l.first << " in sets" << std::endl;
                exit(1);
            }
            
            // delete l.second;
            l.second->empty();

            for (const auto & e : new_set)
            {
                l.second->push_back(e);
            }
            
            // calculate marginal gain
            auto marginal_gain = l.second->size();

            // calculate utiluty of next best
            auto r = this->heap->front();
            
            // if marginal of l is better than r's utility, l is the current best     
            if (marginal_gain >= r.second->size()) 
            {
                for (unsigned int e: *(l.second)) {
                    if (!(this->covered.get(e))) {
                        this->covered.set(e);
                    }
                }

                // std::cout << "returning from lazy greedy " << l.first << " of size " << l.second->size() << std::endl;

                return l.first;
            }
            // else push l's marginal into the heap 
            else {
                // std::cout << "recursive call, l.first: " << l.first << ", l.second.size(): " << l.second->size() << ", r.second.size(): " << r.second->size() << std::endl;
                // TODO: Verify that this branch works.

                this->heap->push_back(l);
                std::push_heap(this->heap->begin(), this->heap->end(), this->cmp);

                return findNextInfluential();
            }
        }
    };

    class NaiveGreedy : public NextMostInfluentialFinder
    {
    public:
        NaiveGreedy(
            size_t theta
        ) : NextMostInfluentialFinder(theta)
        {}

        ~NaiveGreedy(){}

        NextMostInfluentialFinder* setSubset(const std::vector<unsigned int>& subset_of_selection_sets, size_t subset_size) override
        {
            this->vertex_subset = subset_of_selection_sets;
            this->subset_size = subset_size;
            return this;
        } 

        NextMostInfluentialFinder* reloadSubset () override 
        {
            return this;
        }

        size_t findNextInfluential() override
        {
            int max = 0;
            int max_key = -1;

            for ( int i = 0; i < this->subset_size; i++ )
            {
                int vertex = this->vertex_subset.at(i);
                if (this->replicated_data.find(vertex) != this->replicated_data.end() && this->replicated_data.at(vertex)->size() > max)
                {
                    max = this->replicated_data.at(vertex)->size();
                    max_key = vertex;
                }
            }

            for (int e: *(this->replicated_data.at(max_key))) {
                if (!(this->covered.get(e))) {
                    this->covered.set(e);
                }
            }

            #pragma omp parallel 
            {
                # pragma omp for schedule(static)
                for( int i = 0; i < this->subset_size; i++ ) {
                    if (this->replicated_data.find(this->vertex_subset.at(i)) != this->replicated_data.end()) 
                    {
                        auto RRRSets = this->replicated_data.at(this->vertex_subset.at(i));

                        std::vector<int> temp;
                        if (this->vertex_subset.at(i) != max_key) {
                            for (int e: *RRRSets) {
                                if (this->covered.get(e)) {
                                    temp.push_back(e);
                                }
                            }
                            for (int e: temp) {
                                this->replicated_data.at(this->vertex_subset.at(i))->erase(e); 
                            }

                        }
                    }
                }
            }

            this->replicated_data.erase(max_key);
            return max_key;
        }
    };

protected: 
    int k;
    int kprime;
    double epsilon;
    const size_t theta;
    bool usingStochastic = false;
    NextMostInfluentialFinder* finder = 0;
    TimerAggregator &timer;

    void reorganizeVertexSet(std::vector<unsigned int>& vertices, size_t size, std::vector<unsigned int>& seedSet)
    {
        std::set<int> seeds(seedSet.begin(), seedSet.end());

        for (int i = 0; i < size; i++) 
        {
            int j = getRandom(i, vertices.size()-1);
            while (seeds.find(vertices.at(j)) != seeds.end())
            {
                j = getRandom(i, vertices.size()-1);
            }
            std::swap(vertices.at(i), vertices.at(j));
        }
    }

    int getRandom(int min, int max)
    {
        return min + (rand() % static_cast<int>(max - min + 1));
    }

    size_t getSubsetSize(size_t n, int k, double epsilon)
    {
        // new set R = (n/k)*log(1/epsilon),
        return ((size_t)std::round((double)n/(double)k) * std::log10(1/epsilon)) + 1;
    }

public:
    MaxKCoverBase(int k, int kprime, size_t theta, TimerAggregator &timer) 
        : timer(timer), theta(theta)
    {
        this->k = k;
        this->kprime = kprime;
    };

    ~MaxKCoverBase() {
        delete this->finder;
    }

    MaxKCoverBase& useStochasticGreedy(double e)
    {
        this->epsilon = e;
        this->usingStochastic = true;
        return *this;
    }

    MaxKCoverBase& useLazyGreedy()
    {
        this->finder = new LazyGreedy(this->theta);

        return *this;
    }

    MaxKCoverBase& useNaiveGreedy()
    {
        this->finder = new NaiveGreedy(this->theta);

        return *this;
    }

    virtual std::pair<std::vector<unsigned int>, unsigned int> run_max_k_cover (const std::map<int, std::vector<int>>& data) = 0;
};

template <typename GraphTy>
class MaxKCover : public MaxKCoverBase<GraphTy>
{
    public:
    MaxKCover(int k, int kprime, size_t theta, TimerAggregator &timer) 
        : MaxKCoverBase<GraphTy>(k, kprime, theta, timer)
    {
    }

    std::pair<std::vector<unsigned int>, unsigned int> run_max_k_cover (const std::map<int, std::vector<int>>& data) override
    {
        this->finder->Setup(data);
        std::vector<unsigned int> res(this->k, -1);

        size_t subset_size = (this->usingStochastic) ? this->getSubsetSize(data.size(), this->k, this->epsilon) : data.size() ;

        std::vector<unsigned int> all_vertices;  
        for (const auto & l : data) { all_vertices.push_back(l.first); }
        this->finder->setSubset(all_vertices, subset_size);

        for (unsigned int currentSeed = 0; currentSeed < this->k; currentSeed++)
        {
            // std::cout << "working on seed " <<  currentSeed  << "..." << std::endl;
            if (this->usingStochastic)
            {
                this->reorganizeVertexSet(all_vertices, subset_size, res);
                this->finder->reloadSubset();
            }

            res[currentSeed] = this->finder->findNextInfluential();
        }

        auto resUtility = this->finder->GetUtility();
        return std::make_pair(res, resUtility);
    }
};

template <typename GraphTy>
class StreamingMaxKCover : public MaxKCoverBase<GraphTy>
{
private:
    const ripples::mpi::CommunicationEngine<GraphTy> &cEngine;
    MPI_Request request;
    std::vector<std::vector<unsigned int>> send_buffers;

    void QueueNextSeed(
        const unsigned int current_send_index,
        const std::vector<unsigned int>& res,
        const std::map<int, std::vector<int>>& data,
        const unsigned int kprime
    )
    {
        current_send_index == this->kprime - 1 ? 
            this->InsertLastSeed(current_send_index, res, data.at(res[current_send_index]), kprime) :
            this->InsertNextSeedIntoSendBuffer(current_send_index, res[current_send_index], data.at(res[current_send_index]))
        ;

        this->cEngine.QueueNextSeed(
            this->send_buffers[current_send_index].data(),
            this->send_buffers[current_send_index].size(), 
            current_send_index,
            this->request
        );        
    }

    void InsertNextSeedIntoSendBuffer (
        const unsigned int current_seed, 
        const unsigned int vertex_id,
        const std::vector<int>& RRRSetIDs
    )
    {
        // TODO: this code is slow, can potentially be sped up (marginally) by skipping the delete step, and giving
        //  the communciation engine the vector buffer itself instead of making it create it's own temp buffer.
        std::pair<size_t, int*> sendData = this->cEngine.LinearizeSingleSeed(vertex_id, RRRSetIDs);

        // this->cEngine.BuildBufferForStreamingSend(send_buffers[current_seed]);

        size_t total_data = this->cEngine.GetSendReceiveBufferSize(sendData.first);

        this->send_buffers[current_seed].resize(total_data);
        
        size_t i = 0;

        for (; i < sendData.first; i++)
        {
            this->send_buffers[current_seed][i] = sendData.second[i];
        }

        for (; i < total_data; i++)
        {
            this->send_buffers[current_seed][i] = -1; //TODO: Verify that this end of message flag is correct
        }

        // this->send_buffers[current_seed][0]

        delete sendData.second;

        // this->send_buffers[current_seed].second = sendData.second;
        // this->send_buffers[current_seed].first = sendData.first;
    }

    void InsertLastSeed(
        const unsigned int current_seed, 
        const std::vector<unsigned int>& local_seed_set, 
        const std::vector<int>& RRRSetIDs,
        const unsigned int kprime
    )
    {
        this->InsertNextSeedIntoSendBuffer(
            current_seed, 
            local_seed_set[current_seed], 
            RRRSetIDs
        );

        const unsigned int number_of_blocks = std::ceil((double)kprime / (double)this->cEngine.GetBlockSize());
        std::vector<unsigned int> newBuffer = std::vector<unsigned int>(this->send_buffers[current_seed].size() + number_of_blocks * this->cEngine.GetBlockSize());

        int start = 0;
        for (; this->send_buffers[current_seed][start] != -1; start++)
        {
            newBuffer[start] = this->send_buffers[current_seed][start];
        }

        newBuffer[start] = -1;

        start++;

        for (const auto & seed : local_seed_set)
        {
            newBuffer[start++] = seed;
        }

        newBuffer[start] = -1;

        // delete this->send_buffers[current_seed].second;
        this->send_buffers[current_seed] = newBuffer;
        // this->send_buffers[current_seed].second = newBuffer;

        // this->send_buffers[current_seed].first = this->send_buffers[current_seed].size() + local_seed_set.size() + 1;
    }

    public:
    StreamingMaxKCover(
        int k, 
        int kprime, 
        size_t theta,
        TimerAggregator &timer,
        const ripples::mpi::CommunicationEngine<GraphTy> &cEngine
    ) : MaxKCoverBase<GraphTy>(k, kprime, theta, timer), cEngine(cEngine), send_buffers(kprime)
    {}

    std::pair<std::vector<unsigned int>, unsigned int> run_max_k_cover(const std::map<int, std::vector<int>>& data) override
    {
        this->finder->Setup(data);

        std::vector<unsigned int> res(this->k, -1);

        size_t subset_size = (this->usingStochastic) ? this->getSubsetSize(data.size(), this->k, this->epsilon) : data.size() ;

        std::vector<unsigned int> all_vertices;  
        for (const auto & l : data) { all_vertices.push_back(l.first); }
        this->finder->setSubset(all_vertices, subset_size);

        int current_send_index = 0;

        std::pair<int, int*> sendData;

        for (unsigned int currentSeed = 0; currentSeed < this->k; currentSeed++)
        {
            // std::cout << "finding seed " << currentSeed << " while trying to send " << current_send_index << std::endl;

            if (this->usingStochastic)
            {
                this->reorganizeVertexSet(all_vertices, subset_size, res);
                this->finder->reloadSubset();
            }

            this->timer.max_k_localTimer.startTimer();

            res[currentSeed] = this->finder->findNextInfluential();

            // std::cout << "found seed " << currentSeed << " of id " << res[currentSeed] << std::endl;

            this->timer.max_k_localTimer.endTimer();

            // This code block sends data to the global protion of the streaming solution
            this->timer.sendTimer.startTimer();

            int flag = 0;
            if (currentSeed > 0 && current_send_index < this->kprime - 1)
            {
                flag = this->cEngine.TestSend(this->request);
                if (flag == 1)
                {
                    // std::cout << "sent seed " << current_send_index << std::endl;
                    current_send_index++;
                }
            }

            if (flag == 1 || currentSeed == 0)
            {
                // std::cout << "queuing seed " << current_send_index << std::endl;
                this->QueueNextSeed(current_send_index, res, data, this->kprime);
                // std::cout << "finished queuing " << current_send_index << std::endl;
            }

            this->timer.sendTimer.endTimer();
        }

        this->timer.sendTimer.startTimer();
        
        this->cEngine.WaitForSend(this->request);
        current_send_index++;

        // std::cout << "starting to batch send" << std::endl;
        for (; current_send_index < this->kprime; current_send_index++)
        {

            current_send_index == this->kprime - 1 ? 
                this->InsertLastSeed(current_send_index, res, data.at(res[current_send_index]), this->kprime) :
                this->InsertNextSeedIntoSendBuffer(current_send_index, res[current_send_index], data.at(res[current_send_index]))
            ;

            this->cEngine.SendNextSeed(                    
                this->send_buffers[current_send_index].data(),
                this->send_buffers[current_send_index].size(),
                current_send_index == this->kprime - 1 ? this->finder->GetUtility() : current_send_index
            );
        }

        // std::cout << "Sent all seeds" << std::endl;

        this->timer.sendTimer.endTimer();

        return std::make_pair(res, this->finder->GetUtility());
    }
};
