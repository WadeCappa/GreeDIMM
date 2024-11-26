GreeDIMM
*******

This repository contains the software application described in *Scalable Influence Maximization using Distributed Streaming Maximum Cover*.

Quickstart with Conan
=====================

First of all we need to set up the Python environment needed.

.. code-block:: shell

   $ python -m venv --prompt ripples-dev .venv
   $ source .venv/bin/activate
   $ pip install conan

Then, we set up the conan profile:

.. code-block:: shell

   $ conan profile detect

You can check that the conan has detected the correct compiler by:

.. code-block:: shell

   $ conan profile show

In many cases this will show the correct configuration. Notable exceptions are
systems where you want to use a provided compiler wrapper (e.g., many HPE
machines ship compiler wrappers) or you want to use hipcc to compile the
framework. In that case you want to edit your conan profile file with:

.. code-block:: shell

   $ vim $(conan profile path default)

Here as a reference you can find how to change the profile to use hipcc:

.. code-block:: conf

    [settings]
    arch=x86_64
    build_type=Release
    compiler=clang
    compiler.cppstd=gnu14
    compiler.libcxx=libstdc++11
    compiler.version=15
    os=Linux
    [buildenv]
    *:CC=hipcc
    *:CXX=hipcc

The next step is to install dependencies:

.. code-block:: shell

    $ conan create conan/trng
    $ conan create conan/rocThrust # if compiling with AMD GPU support.
    $ conan create conan/metall    # if compiling with Metall support.
    $ conan install . --build missing
    $ conan install . --build missing -o gpu=amd # for AMD GPU support.

We can now compile ripples:

.. code-block:: shell

    $ conan build .               # CPU only version
    $ conan build . -o gpu=amd    # with AMD GPU support.

To enable Memkind or Metall please replace the conan install command with one of:

Allocate RRRSets Using Metall
=============================

Ripples + Metall has another mode that allocates intermediate data (called RRRSets) using Metall.

To enable the mode, define ENABLE_METALL_RRRSETS macro (e.g., insert ``#define ENABLE_METALL_RRRSETS`` at the beginning of tools/imm.cc).

The storage directory can be specified with ``--rr-store-dir=<PATH>`` argument when executing imm.

GreeDIMM can be run with ``build/release/tools/mpi-greedi-im``. Running ``build/release/tools/mpi-greedi-im -h`` will provide the following information; 

.. code-block::
   
   Usage: ./build/release/tools/mpi-greedi-im [OPTIONS]

   Options:
      -h,--help                   Print this help message and exit
      

   Input Options:
      -i,--input-graph TEXT REQUIRED
                                    The input file with the edge-list.
      --reload-binary             Reload a graph from binary input
      -u,--undirected             The input graph is undirected
      -w,--weighted               The input graph is weighted
      --distribution TEXT         The distribution to be used (uniform|normal) to generate weights
      --mean FLOAT                The mean for the normal distribution
      --variance FLOAT            The variance for the normal distribution
      --scale-factor FLOAT        Scaling Factor for the generated weights
      --disable-renumbering       Load the graph as is from the input.


   Algorithm Options:
      -k,--seed-set-size UINT REQUIRED
                                    The size of the seed set.
      -p,--parallel               Trigger the parallel implementation
      -d,--diffusion-model TEXT REQUIRED
                                    The diffusion model to use (LT|IC)
      -e,--epsilon FLOAT REQUIRED The size of the seed set.


   Streaming-Engine Options:
      --dump-sampling-data BOOLEAN
                                    Output all sampling data to your output file
      --run-streaming BOOLEAN     Run max-k-cover within a streaming algorithm. False by default.
      --epsilon-2 FLOAT           Set the error parameter for the streaming step. Default of 0.13 to acheive approximation garuntee of 21%
      --alpha FLOAT               Set the fraction of local seeds to send to the final selection step, defaults to 1


   Output Options:
      -o,--output TEXT            The file name of the log.


GreeDIMM Team
============

- `Reet Barik | WSU <reet.barik@wsu.edu>`_
- `Wade Cappa | WSU <wade.cappa@wsu.edu>`_
- `S M Ferdous | PNNL <sm.ferdous@pnnl.gov>`_
- `Marco Mintutoli | PNNL <marco.minutoli@pnnl.gov>`_
- `Mahantesh Halappanavar | PNNL, WSU <mahantesh.halappanavar@pnnl.gov>`_
- `Ananth Kalyanaraman | WSU, PNNL <ananth@wsu.edu>`_

This software was produced in collaboration between authors from Washington State University Pullman, and Pacific Northwest National Laboratory Richland. 

Disclamer Notice
================

This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or any
information, apparatus, product, software, or process disclosed, or represents
that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.

.. raw:: html

   <div align=center>
   <pre style="align-text:center">
   PACIFIC NORTHWEST NATIONAL LABORATORY
   operated by
   BATTELLE
   for the
   UNITED STATES DEPARTMENT OF ENERGY
   under Contract DE-AC05-76RL01830
   </pre>
   </div>
