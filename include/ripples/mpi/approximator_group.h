#ifndef RIPPLES_MPI_IMM_APPROX_GROUP_H
#define RIPPLES_MPI_IMM_APPROX_GROUP_H

namespace ripples {
    namespace mpi {

        typedef struct solutionState {
            const std::map<int, std::vector<int>> &solutionSpace;
            const std::pair<std::vector<unsigned int>, unsigned int> &bestSolution;
        } SolutionState ;

        class ApproximatorGroup {
            protected: 
            const MPI_Comm groupWorld;
            const std::vector<int> &vertexToProcess;

            static std::pair<std::vector<unsigned int>, unsigned int> getBestCandidate(
                const std::vector<std::pair<std::vector<unsigned int>, unsigned int>> &allCandidateSets,
                unsigned int kprime
            ) {
                unsigned int bestUtility = 0;
                std::vector<unsigned int> bestCandidate;
                
                for (const auto & e : allCandidateSets) {
                    if (e.second > bestUtility) {
                        bestCandidate = e.first;
                        bestUtility = e.second;

                        if (e.first.size() > kprime) {
                            std::cout << "found candidate that was greater than allowed size, size of " << e.first.size() << std::endl;
                        }
                    }
                }

                return std::make_pair(bestCandidate, bestUtility);
            }

            public:
            ApproximatorGroup(
                MPI_Comm groupWorld,
                const std::vector<int> &vertexToProcess
            ) : vertexToProcess(vertexToProcess), groupWorld(groupWorld) {}

            virtual void approximate(
                const ripples::mpi::SolutionState &currentState,
                std::map<int, std::vector<int>> &nextSolutionSpace, // modified
                std::pair<std::vector<unsigned int>, unsigned int> &nextBestSolution, // modified
                const int kprime, 
                const size_t theta
            ) = 0;
        };

        template <typename GraphTy, typename ConfTy>
        class StreamingApproximatorGroup : public ApproximatorGroup { 
            private:
            TimerAggregator &timeAggregator;
            const ConfTy &CFG;
            const CommunicationEngine<GraphTy> &cEngine;

            public:
            StreamingApproximatorGroup(
                MPI_Comm groupWorld,
                const std::vector<int> &vertexToProcess,
                TimerAggregator &timeAggregator,
                const ConfTy &input_CFG,
                const CommunicationEngine<GraphTy> &cEngine
            ) : ApproximatorGroup(groupWorld, vertexToProcess), timeAggregator(timeAggregator), CFG(input_CFG), cEngine(cEngine) {}

            void approximate(
                const ripples::mpi::SolutionState &currentState,
                std::map<int, std::vector<int>> &nextSolutionSpace, // modified
                std::pair<std::vector<unsigned int>, unsigned int> &nextBestSolution, // modified
                const int kprime, 
                const size_t theta
            ) override {
                int worldSize;
                int groupRank;
                MPI_Comm_rank(this->groupWorld, &groupRank);
                MPI_Comm_size(this->groupWorld, &worldSize);
                if (groupRank == 0) {
                    ripples::mpi::StreamingRandGreedIEngine<GraphTy> streamingEngine(this->CFG.k, kprime, theta, (double)this->CFG.epsilon_2, worldSize - 1, this->cEngine, this->timeAggregator);
                    std::cout << "starting to receive..." << std::endl;
                    nextBestSolution = streamingEngine.Stream(nextSolutionSpace);
                } else {
                    StreamingMaxKCover<GraphTy> localKCoverEngine(this->CFG.k, kprime, theta, this->timeAggregator, this->cEngine);
                    localKCoverEngine.useLazyGreedy();
                    localKCoverEngine.run_max_k_cover(currentState.solutionSpace);
                }
            }
        };

        template <typename GraphTy, typename ConfTy>
        class LazyLazyApproximatorGroup : public ApproximatorGroup {
            private:
            TimerAggregator &timeAggregator;
            const ConfTy &CFG;
            const CommunicationEngine<GraphTy> &cEngine;

            static std::pair<std::vector<unsigned int>, unsigned int> SolveKCover(
                const int k,
                const int kprime,
                const size_t max_element, 
                TimerAggregator &timeAggregator,
                const std::map<int, std::vector<int>>& elements
            ) {
                MaxKCover<GraphTy> localKCoverEngine(k, kprime, max_element, timeAggregator);
                localKCoverEngine.useLazyGreedy();
                return localKCoverEngine.run_max_k_cover(elements);
            }

            public:
            LazyLazyApproximatorGroup(
            MPI_Comm groupWorld,
            const std::vector<int> &vertexToProcess,
            TimerAggregator &timeAggregator,
            const ConfTy &input_CFG,
            const CommunicationEngine<GraphTy> &cEngine
        ) : ApproximatorGroup(groupWorld, vertexToProcess), timeAggregator(timeAggregator), CFG(input_CFG), cEngine(cEngine) {}

        void approximate(
            const ripples::mpi::SolutionState &currentState,
            std::map<int, std::vector<int>> &nextSolutionSpace, // modified
            std::pair<std::vector<unsigned int>, unsigned int> &nextBestSolution, // modified
            const int kprime, 
            const size_t theta
        ) override {
            nextBestSolution.first = currentState.bestSolution.first;
            nextBestSolution.second = currentState.bestSolution.second;

            if (nextBestSolution.first.size() == 0) {
                // std::cout << "init of local solution using world size of " << currentState.solutionSpace.size() << std::endl;;
                this->timeAggregator.max_k_localTimer.startTimer();

                nextBestSolution = this->SolveKCover(
                    this->CFG.k, kprime, theta, this->timeAggregator, currentState.solutionSpace
                );

                this->timeAggregator.max_k_localTimer.endTimer();
            }

            this->timeAggregator.allGatherTimer.startTimer();

            std::vector<unsigned int> receiveBuffer;
            // std::cout << "before gather" << std::endl;
            size_t totalData = this->cEngine.GatherPartialSolutions(
                receiveBuffer, 
                nextBestSolution,
                currentState.solutionSpace, 
                this->groupWorld
            );

            this->timeAggregator.allGatherTimer.endTimer();

            int globalRank; 
            MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);

            int groupRank;
            MPI_Comm_rank(this->groupWorld, &groupRank);
            if (groupRank == 0) {
                // std::cout << "global rank " << globalRank << " is group leader" << std::endl;

                this->timeAggregator.allGatherTimer.startTimer();

                std::vector<std::pair<std::vector<unsigned int>, unsigned int>> candidateSets;
                // std::cout << "before deserialization" << std::endl;
                this->cEngine.deserializeGatherData(nextSolutionSpace, candidateSets, receiveBuffer.data(), totalData);

                this->timeAggregator.allGatherTimer.endTimer();

                // spdlog::get("console")->info("global max_k_cover...");
                this->timeAggregator.max_k_globalTimer.startTimer();

                // std::cout << "before global k cover" << std::endl;
                auto globalSeeds = this->SolveKCover(
                    this->CFG.k, kprime, theta, this->timeAggregator, nextSolutionSpace
                );

                candidateSets.push_back(globalSeeds);
                nextBestSolution = this->getBestCandidate(candidateSets, kprime);

                // nextBestSolution = globalSeeds;

                this->timeAggregator.max_k_globalTimer.endTimer();
            } else {
                nextBestSolution = std::make_pair(std::vector<unsigned int>(), 0);
            }
        }
    };
    }
}
#endif  