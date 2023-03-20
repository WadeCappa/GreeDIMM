#include <vector>
#include <set>
#include <unordered_set>
#include <mutex>
#include <iostream>

template <typename GraphTy>
class TransposeRRRSets
{
    public:
    std::vector<std::pair<std::mutex, std::vector<typename GraphTy::vertex_type>>> sets;

    public:
    TransposeRRRSets(int numberOfVertices) 
        : sets(numberOfVertices)
    {
    }

    ~TransposeRRRSets() 
    {
    }

    void addToSet(int index, typename GraphTy::vertex_type vertex) 
    {
        sets[index].first.lock(); 
        sets[index].second.push_back(vertex);
        sets[index].first.unlock();
    }

    auto getBegin() 
    {
        return sets.begin();
    }

    auto getEnd() 
    {
        return sets.end();
    }

    void removeDuplicates()
    {
        #pragma omp parallel for
        for (int i = 0; i < this->sets.size(); i++)
        {
            std::unordered_set<typename GraphTy::vertex_type> seen;

            for (const auto & rrr_id : this->sets[i].second)
            {
                seen.insert(rrr_id);
            }

            this->sets[i].second.assign(seen.begin(), seen.end());
        }
    }
};

