//===------------------------------------------------------------*- C++ -*-===//
//
//             Ripples: A C++ Library for Influence Maximization
//                  Marco Minutoli <marco.minutoli@pnnl.gov>
//                   Pacific Northwest National Laboratory
//
//===----------------------------------------------------------------------===//
//
// Copyright (c) 2019, Battelle Memorial Institute
//
// Battelle Memorial Institute (hereinafter Battelle) hereby grants permission
// to any person or entity lawfully obtaining a copy of this software and
// associated documentation files (hereinafter “the Software”) to redistribute
// and use the Software in source and binary forms, with or without
// modification.  Such person or entity may use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and may permit
// others to do so, subject to the following conditions:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Other than as used herein, neither the name Battelle Memorial Institute or
//    Battelle may be used in any form whatsoever without the express written
//    consent of Battelle.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL BATTELLE OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//

#ifndef RIPPLES_MPI_IMM_GREED_H
#define RIPPLES_MPI_IMM_GREED_H

#include "communication_engine.h"
#include "ripples/sampler_context.h"
#include "ripples/mpi/ownership_manager.h"
#include "ripples/max_k_cover.h"
#include "ripples/mpi/streaming_engine.h"
#include "ripples/mpi/approximator_context.h"
#include "ripples/mpi/approximator_group.h"
#include "ripples/mpi/martingale_context.h"
#include "ripples/mpi/martingale_builder.h"

namespace ripples {
namespace mpi {

template <typename ex_tag>
struct MPI_Plus_X {
  // using generate_ex_tag
  // using seed_selection_ex_tag
};

template <>
struct MPI_Plus_X<mpi_omp_parallel_tag> {
  using generate_ex_tag = omp_parallel_tag;
  using seed_selection_ex_tag = mpi_omp_parallel_tag;
};

//! Split a random number generator into one sequence per MPI rank.
//!
//! \tparam PRNG The type of the random number generator.
//!
//! \param gen The parallel random number generator to split.
template <typename PRNG>
void split_generator(PRNG &gen) {
  // Find out rank, size
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  gen.split(world_size, world_rank);
}

template <typename PRNG>
std::vector<PRNG> rank_split_generator(const PRNG &gen) {
  size_t max_num_threads(1);

#pragma omp single
  max_num_threads = omp_get_max_threads();

  std::vector<trng::lcg64> generator(max_num_threads, gen);

#pragma omp parallel
  {
    generator[omp_get_thread_num()].split(omp_get_num_threads(),
                                          omp_get_thread_num());
  }
  return generator;
}

double getPessimisticApproximation() {
  return 1.0 - (1.0 / (double)std::exp(1.0));
}

template <typename GraphTy, typename ConfTy, typename diff_model_tag,
          typename RRRGeneratorTy, typename ExTagTrait>
auto run_greedimm(
  const GraphTy &G, 
  const ConfTy &CFG, 
  TimerAggregator &timeAggregator,
  double l_value, 
  RRRGeneratorTy &gen,
  IMMExecutionRecord &record, 
  diff_model_tag &&model_tag, 
  ExTagTrait &&
) 
{
  // no sequential version available
  using execution_tag = ripples::omp_parallel_tag;
  using vertex_type = typename GraphTy::vertex_type;

  const bool use_streaming = true;
  CommunicationEngine<GraphTy> cEngine = CommunicationEngineBuilder<GraphTy>::BuildCommunicationEngine(use_streaming);
  TransposeRRRSets<GraphTy> tRRRSets(G.num_nodes());
  std::vector<int> vertexToProcess(cEngine.DistributeVertices(G));

  DefaultSampler<GraphTy, diff_model_tag, RRRGeneratorTy, execution_tag> sampler(
      cEngine.GetSize(), G, gen, record, model_tag
  );

  OwnershipManager<GraphTy> ownershipManager(G.num_nodes(), cEngine, vertexToProcess);

  std::vector<ApproximatorGroup*> approximatorGroups;
  approximatorGroups.push_back(new StreamingApproximatorGroup<GraphTy, ConfTy>(MPI_COMM_WORLD, vertexToProcess, timeAggregator, CFG, cEngine));
  std::vector<ApproximatorContext> approximators;
  approximators.push_back(ApproximatorContext (approximatorGroups));

  MartingaleContext<GraphTy, ConfTy, RRRGeneratorTy, diff_model_tag, execution_tag> martingaleContext(
    sampler, ownershipManager, approximators, G, CFG, l_value, record, cEngine, timeAggregator
  );

  const double a = 1.0 - std::pow((double)std::exp(1.0), 0 - CFG.alpha);
  const double b = 0.5 - CFG.epsilon_2;
  double approx = (a * b) / (a + b);
  approx = CFG.use_pessimistic_approximation ? getPessimisticApproximation() : approx;
  if (CFG.use_opimc >= 0) {
    auto res = martingaleContext.useOpimc(approx);
    return res;
  } else {
    auto res = martingaleContext.useImm(approx);
    return res;
  }
}

template <typename GraphTy, typename ConfTy, typename diff_model_tag,
          typename RRRGeneratorTy, typename ExTagTrait>
auto run_randgreedi(
  const GraphTy &G, 
  const ConfTy &CFG, 
  TimerAggregator &timeAggregator,
  double l_value, 
  RRRGeneratorTy &gen,
  IMMExecutionRecord &record, 
  diff_model_tag &&model_tag, 
  ExTagTrait &&
) 
{
  // no sequential version available
  using execution_tag = ripples::omp_parallel_tag;
  using vertex_type = typename GraphTy::vertex_type;

  const bool use_streaming = false;
  CommunicationEngine<GraphTy> cEngine = CommunicationEngineBuilder<GraphTy>::BuildCommunicationEngine(use_streaming);
  TransposeRRRSets<GraphTy> tRRRSets(G.num_nodes());
  std::vector<int> vertexToProcess(cEngine.DistributeVertices(G));

  DefaultSampler<GraphTy, diff_model_tag, RRRGeneratorTy, execution_tag> sampler(
      cEngine.GetSize(), G, gen, record, model_tag
  );

  OwnershipManager<GraphTy> ownershipManager(G.num_nodes(), cEngine, vertexToProcess);

  std::vector<unsigned int> branchingFactors = MartingleBuilder::getBranchingFactors(CFG.branching_factors);

  std::vector<ApproximatorContext> approximators = MartingleBuilder::buildApproximatorContexts<GraphTy, ConfTy>(branchingFactors, cEngine.GetSize(), CFG, timeAggregator, vertexToProcess, cEngine);

  MartingaleContext<GraphTy, ConfTy, RRRGeneratorTy, diff_model_tag, execution_tag> martingaleContext(
    sampler, ownershipManager, approximators, G, CFG, l_value, record, cEngine, timeAggregator
  );

  double approx = (1.0 - std::pow((double)std::exp(1.0), 0 - CFG.alpha)) / 2.0; // last term should be number of levels.
  approx = CFG.use_pessimistic_approximation ? getPessimisticApproximation() : approx;
  if (CFG.use_opimc >= 0) {
    auto res = martingaleContext.useOpimc(approx);
    return res;
  } else {
    auto res = martingaleContext.useImm(approx);
    return res;
  }
}



}  // namespace mpi
}  // namespace ripples

#endif  // RIPPLES_MPI_IMM_H
