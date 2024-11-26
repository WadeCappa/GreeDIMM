#ifndef RIPPLES_MPI_IMM_OWNERSHIP_H
#define RIPPLES_MPI_IMM_OWNERSHIP_H

namespace ripples {
    template <typename GraphTy>
    class OwnershipManager {
        private: 
            const ripples::mpi::CommunicationEngine<GraphTy> &cEngine;
            const std::vector<int> &vertexToProcess;
            std::vector<size_t> oldSamplingSizes;

        public:

        OwnershipManager(
            size_t num_nodes,
            const ripples::mpi::CommunicationEngine<GraphTy> &cEngine,
            const std::vector<int> &vertexToProcess
        ) : cEngine(cEngine), vertexToProcess(vertexToProcess), oldSamplingSizes(num_nodes, 0) {}
        
        void redistributeSeedSets(
            const TransposeRRRSets<GraphTy> &tRRRSets, 
            std::map<int, std::vector<int>> &aggregateSets,
            const size_t delta
        ) {
            this->cEngine.AggregateThroughAllToAll(
                tRRRSets,
                this->vertexToProcess,
                this->oldSamplingSizes,
                delta,
                aggregateSets
            );
        }
    };
}


#endif  