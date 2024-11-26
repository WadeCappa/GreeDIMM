#ifndef RIPPLES_MPI_RRR_SETS_H
#define RIPPLES_MPI_RRR_SETS_H

#include <vector>
#include <set>
#include <unordered_set>
#include <mutex>
#include <iostream>
#include "ripples/bitmask.h"

namespace ripples {

    template <typename GraphTy>
    class TransposeRRRSets
    {
        public:
        std::vector<std::pair<std::mutex, std::vector<size_t>>> sets;

        public:
        TransposeRRRSets(size_t numberOfVertices) 
            : sets(numberOfVertices)
        {
        }

        ~TransposeRRRSets() 
        {
        }

        void addToSet(int vertex, size_t RRRId) 
        {
            sets[vertex].first.lock(); 
            sets[vertex].second.push_back(RRRId);
            sets[vertex].first.unlock();
        }

        void GetSetSizes(std::vector<unsigned int>& setSizes)
        {
            for (int i = 0; i < setSizes.size() && i < this->sets.size(); i++)
            {
                setSizes[i] = (unsigned(this->sets[i].second.size()));
            }
        }

        void RemoveDuplicates()
        {
            #pragma omp parallel for
            for (int i = 0; i < this->sets.size(); i++)
            {
                int sizeBefore = this->sets[i].second.size();

                std::unordered_set<size_t> seen;
                for (const auto & r : this->sets[i].second)
                {
                    seen.insert(r);
                }

                this->sets[i].second.assign(seen.begin(), seen.end());

                if (sizeBefore != this->sets[i].second.size())
                {
                    std::cout << "size difference; " << sizeBefore -  this->sets[i].second.size() << std::endl;
                }
            }
        }

        void getLocalNonCovered(
            std::vector<unsigned int> &non_covered,
            const GraphTy &G, 
            const std::vector<unsigned int> &seeds, 
            const size_t theta
        ) const {
            ripples::Bitmask<int> covered(theta);

            #pragma omp parallel for
            for (size_t i = 0; i < seeds.size(); i++) {
                const auto &vertex = seeds[i];
                for (const auto & rrr_set : this->sets[vertex].second) {
                    covered.set(rrr_set);
                }
            }

            #pragma omp parallel for
            for (size_t vertex = 0; vertex < this->sets.size(); vertex++) {
                const auto & rrr_sets = this->sets[vertex].second;
                for (const auto & rrr_set : rrr_sets) {
                    if (!covered.get(rrr_set)) {
                        non_covered[vertex]++;
                    }
                }
            }
        }

        unsigned int calculateInfluence(
            const std::vector<unsigned int> &seeds, 
            const size_t theta
        ) const {
            ripples::Bitmask<int> covered(theta);

            #pragma omp parallel for
            for (size_t i = 0; i < seeds.size(); i++) {
                const auto &vertex = seeds[i];
                for (const auto & rrr_set : this->sets[vertex].second) {
                    covered.set(rrr_set);
                }
            }

            return covered.popcount();
        }
    };
}

#endif  
