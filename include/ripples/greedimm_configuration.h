

#ifndef RIPPLES_IMM_GREEDIMM_CONFIGURATION_H
#define RIPPLES_IMM_GREEDIMM_CONFIGURATION_H

#include "ripples/imm_configuration.h"

namespace ripples {
  struct GreedimmConfiguration : public IMMConfiguration {
    double epsilon_2 = 0.13; 
    double alpha = 1;
    bool use_diimm = false;
    bool verbose = false;
    std::string branching_factors = "";
    int use_opimc = -1;
    bool use_pessimistic_approximation = false;

    void addCmdOptions(CLI::App &app) {
      IMMConfiguration::addCmdOptions(app);
      app.add_option("--epsilon-2", epsilon_2,
                    "Set the error parameter for the streaming step. Default of 0.13 to acheive approximation garuntee of 21%")
          ->group("Streaming-Engine Options");
      app.add_option("--alpha", alpha,
                  "Set the fraction of local seeds to send to the final selection step, defaults to 1")
          ->group("Streaming-Engine Options");
      app.add_option("--DIiMM", use_diimm,
                  "Use the 2022 DIiMM algorithm to approximate influence maximization")
          ->group("Streaming-Engine Options");
      app.add_option("--verbose", verbose,
                  "Output more granular runtime data")
          ->group("Streaming-Engine Options");
      app.add_option("--branching-factors", branching_factors,
                  "Provide a set of branching factors. The most optimal will be determined at runtime. Format as '.' delimited string, such as \"2.4.8.16\". Defaults empty.")
          ->group("Streaming-Engine Options"); // todo: figure out if you can change the group without breaking anything
      app.add_option("--opimc", use_opimc,
                  "Use OPIM-C instead of IMM. Set to -1 by default which means it will not be used. Mode 0 uses a pessimistic but easy to calculate upper bound. Mode 1 uses a more optimistic upper bound but is harder to calculate.")
          ->group("Streaming-Engine Options"); // todo: figure out if you can change the group without breaking anything
      app.add_option("--pessimistic_approximation", use_pessimistic_approximation,
                  "Sets the exit condition value to the highest possible approximation.")
            ->group("Streaming-Engine Options"); // todo: figure out if you can change the group without breaking anything
    }
  };
}

#endif
