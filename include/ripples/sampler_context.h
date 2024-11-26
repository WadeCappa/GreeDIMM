
#ifndef RIPPLES_MPI_SAMPLER_H
#define RIPPLES_MPI_SAMPLER_H

namespace ripples {
    template <typename GraphTy, typename PRNGeneratorTy,
            typename ItrTy, typename ExecRecordTy,
            typename diff_model_tag>
    void GenerateTransposeRRRSets(
                        TransposeRRRSets<GraphTy> &transposeRRRSets,
                        size_t current_sets,
                        size_t delta,
                        const GraphTy &G,
                        StreamingRRRGenerator<GraphTy, PRNGeneratorTy, ItrTy, diff_model_tag> &se,
                        ExecRecordTy &,
                        diff_model_tag &&,
                        omp_parallel_tag &&) {
        se.transposeGenerate(transposeRRRSets, current_sets, delta);
    }

    template <
        typename GraphTy,
        typename diff_model_tag,
        typename RRRGeneratorTy,
        typename execution_tag
    >
    class DefaultSampler {
        private:
        ripples::omp_parallel_tag ex_tag;
        diff_model_tag &model_tag;

        const unsigned int world_size;
        const GraphTy &graph;
        RRRGeneratorTy &gen;
        ripples::IMMExecutionRecord &record;

        public:
        DefaultSampler(
            unsigned int world_size,
            const GraphTy &input_G, 
            RRRGeneratorTy &gen,
            ripples::IMMExecutionRecord &record,
            diff_model_tag &model_tag
        ) : world_size(world_size), graph(input_G), gen(gen), record(record), model_tag(model_tag) {
        }

        // Non-const, but because we want random generation this is fine.
        void addNewSamples(
            TransposeRRRSets<GraphTy> &tRRRSets, 
            const size_t counterStart, 
            const size_t delta
        ) {
            GenerateTransposeRRRSets(
                tRRRSets, counterStart, delta, 
                this->graph, this->gen, this->record,
                std::forward<diff_model_tag>(this->model_tag),
                std::forward<execution_tag>(this->ex_tag)
            );
        }

    };
}

#endif 
