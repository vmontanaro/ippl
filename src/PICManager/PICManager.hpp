// PCONTAINER header file
//   Defines a particle attribute for charged particles to be used in
//   applications
//
// Copyright (c) 2021 Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//
#ifndef PCONTAINER_H
#define PCONTAINER_H

#include "Ippl.h"

#include <csignal>
#include <thread>

#include "Utility/TypeUtils.h"

/*
  FixMe:

  If we need the solver here then:

  "Solver/Solver.h"
*/

#include "Solver/ElectrostaticsCG.h"
#include "Solver/FFTPeriodicPoissonSolver.h"
#include "Solver/FFTPoissonSolver.h"
#include "Solver/P3MSolver.h"

unsigned LoggingPeriod = 1;

// some typedefs
template <unsigned Dim = 3>
using Mesh_t = ippl::UniformCartesian<double, Dim>;

template <typename T, unsigned Dim = 3>
using PLayout_t = typename ippl::ParticleSpatialLayout<T, Dim, Mesh_t<Dim>>;

template <unsigned Dim = 3>
using Centering_t = typename Mesh_t<Dim>::DefaultCentering;

template <unsigned Dim = 3>
using FieldLayout_t = ippl::FieldLayout<Dim>;

template <typename T = double, unsigned Dim = 3>
using ORB = ippl::OrthogonalRecursiveBisection<double, Dim, Mesh_t<Dim>, Centering_t<Dim>, T>;

using size_type = ippl::detail::size_type;

template <typename T, unsigned Dim = 3>
using Vector = ippl::Vector<T, Dim>;

template <typename T, unsigned Dim = 3, class... ViewArgs>
using Field = ippl::Field<T, Dim, Mesh_t<Dim>, Centering_t<Dim>, ViewArgs...>;

template <typename T>
using ParticleAttrib = ippl::ParticleAttrib<T>;

template <typename T, unsigned Dim = 3>
using Vector_t = ippl::Vector<T, Dim>;

template <unsigned Dim = 3, class... ViewArgs>
using Field_t = Field<double, Dim, ViewArgs...>;

template <typename T = double, unsigned Dim = 3, class... ViewArgs>
using VField_t = Field<Vector_t<T, Dim>, Dim, ViewArgs...>;

// heFFTe does not support 1D FFTs, so we switch to CG in the 1D case
template <typename T = double, unsigned Dim = 3>
using CGSolver_t = ippl::ElectrostaticsCG<Field<T, Dim>, Field_t<Dim>>;

using ippl::detail::ConditionalType, ippl::detail::VariantFromConditionalTypes;

template <typename T = double, unsigned Dim = 3>
using FFTSolver_t = ConditionalType<Dim == 2 || Dim == 3,
                                    ippl::FFTPeriodicPoissonSolver<VField_t<T, Dim>, Field_t<Dim>>>;

template <typename T = double, unsigned Dim = 3>
using P3MSolver_t = ConditionalType<Dim == 3, ippl::P3MSolver<VField_t<T, Dim>, Field_t<Dim>>>;

template <typename T = double, unsigned Dim = 3>
using OpenSolver_t =
    ConditionalType<Dim == 3, ippl::FFTPoissonSolver<VField_t<T, Dim>, Field_t<Dim>>>;

template <typename T = double, unsigned Dim = 3>
using Solver_t = VariantFromConditionalTypes<CGSolver_t<T, Dim>, FFTSolver_t<T, Dim>,
                                             P3MSolver_t<T, Dim>, OpenSolver_t<T, Dim>>;

const double pi = Kokkos::numbers::pi_v<double>;

/*
  FixMe: the include needs to go up, but we need this for Connector.hpp

*/

#include "Connector/Connector.hpp"

// Signal handling
int interruptSignalReceived = 0;

/*!
 * Signal handler records the received signal
 * @param signal received signal
 */
void interruptHandler(int signal) {
    interruptSignalReceived = signal;
}

/*!
 * Checks whether a signal was received
 * @return Signal handler was called
 */
bool checkSignalHandler() {
    ippl::Comm->barrier();
    return interruptSignalReceived != 0;
}

/*!
 * Sets up the signal handler, for SIGTERM and SIGINT
 */
void setSignalHandler() {
    struct sigaction sa;
    sa.sa_handler = interruptHandler;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGTERM, &sa, NULL) == -1) {
        std::cerr << ippl::Comm->rank() << ": failed to set up signal handler for SIGTERM ("
                  << SIGTERM << ")" << std::endl;
    }
    if (sigaction(SIGINT, &sa, NULL) == -1) {
        std::cerr << ippl::Comm->rank() << ": failed to set up signal handler for SIGINT ("
                  << SIGINT << ")" << std::endl;
    }
}

/**
 * @class PICManager
 * @brief Class for managing particles in a Particle-In-Cell (PIC) context
 * @tparam PLayout Particle layout type
 * @tparam T Data type for particle attributes
 * @tparam Dim Dimension of the simulation
 */

template <class PLayout, typename T, unsigned Dim = 3>
class PICManager : public ippl::ParticleBase<PLayout> {
public:
    using Base = ippl::ParticleBase<PLayout>;

    VField_t<T, Dim> F_m;  /// force field

    Field_t<Dim> rhs_m;   /// the right hand side
    Field<T, Dim> sol_m;  /// the solution

    typedef ippl::BConds<Field<T, Dim>, Dim> bc_type;

    bc_type bc_m;

    // ORB
    ORB<T, Dim> orb;

    Vector_t<T, Dim> nr_m;

    ippl::e_dim_tag decomp_m[Dim];

    Vector_t<double, Dim> hr_m;
    Vector_t<double, Dim> rmin_m;
    Vector_t<double, Dim> rmax_m;

    std::string stype_m;

private:
    Solver_t<T, Dim> solver_m;

public:
    double time_m;

    unsigned int loadbalancefreq_m;

    double loadbalancethreshold_m;

public:
    /**
     * @brief Default constructor is mandatory for all derived classes from ParticleBaseTest
     * @param pl Particle layout
     */
    PICManager(PLayout& pl)
        : Base(pl) {}
    /**
     * @brief Constructor with parameters
     * @param pl Particle layout
     * @param hr Vector of dimensions
     * @param rmin Minimum vector
     * @param rmax Maximum vector
     * @param decomp Decomposition tag
     * @param solver Solver type
     */
    PICManager(PLayout& pl, Vector_t<double, Dim> hr, Vector_t<double, Dim> rmin,
               Vector_t<double, Dim> rmax, ippl::e_dim_tag decomp[Dim], std::string solver)
        : Base(pl)
        , hr_m(hr)
        , rmin_m(rmin)
        , rmax_m(rmax)
        , stype_m(solver) {
        for (unsigned int i = 0; i < Dim; i++) {
            decomp_m[i] = decomp[i];
        }
    }

    /**
     * @brief Destructor
     */
    ~PICManager() {}

    /**
     * @brief Update the layout
     * @param fl Field layout
     * @param mesh Mesh
     * @param buffer Buffer
     * @param isFirstRepartition Flag for first repartition
     */
    void updateLayout(FieldLayout_t<Dim>& fl, Mesh_t<Dim>& mesh,
                      PICManager<PLayout, T, Dim>& buffer, bool& isFirstRepartition) {
        // Update local fields
        static IpplTimings::TimerRef tupdateLayout = IpplTimings::getTimer("updateLayout");
        IpplTimings::startTimer(tupdateLayout);
        F_m.updateLayout(fl);
        rhs_m.updateLayout(fl);
        if (stype_m == "CG") {
            this->sol_m.updateLayout(fl);
            sol_m.setFieldBC(bc_m);
        }

        // Update layout with new FieldLayout
        PLayout& layout = this->getLayout();
        layout.updateLayout(fl, mesh);
        IpplTimings::stopTimer(tupdateLayout);
        static IpplTimings::TimerRef tupdatePLayout = IpplTimings::getTimer("updatePB");
        IpplTimings::startTimer(tupdatePLayout);
        if (!isFirstRepartition) {
            layout.update(*this, buffer);
        }
        IpplTimings::stopTimer(tupdatePLayout);
    }

    /**
     * @brief Initialize fields
     * @param mesh Mesh
     * @param fl Field layout
     */

    void initializeFields(Mesh_t<Dim>& mesh, FieldLayout_t<Dim>& fl) {
        F_m.initialize(mesh, fl);
        rhs_m.initialize(mesh, fl);
        if (stype_m == "CG") {
            sol_m.initialize(mesh, fl);
            sol_m.setFieldBC(bc_m);
        }
    }

    /**
     * @brief Set all boundary conditions to periodic
     */

    void setBCAllPeriodic() { this->setParticleBC(ippl::BC::PERIODIC); }

    /**
     * @brief Initialize ORB
     * @param fl Field layout
     * @param mesh Mesh
     */

    void initializeORB(FieldLayout_t<Dim>& fl, Mesh_t<Dim>& mesh) {
        orb.initialize(fl, mesh, rhs_m);
    }

    /**
     * @brief Repartition
     * @param fl Field layout
     * @param mesh Mesh
     * @param buffer Buffer
     * @param isFirstRepartition Flag for first repartition
     */

    void repartition(FieldLayout_t<Dim>& fl, Mesh_t<Dim>& mesh, PICManager<PLayout, T, Dim>& buffer,
                     bool& isFirstRepartition) {
        // Repartition the domains
        bool res = orb.binaryRepartition(this->R, fl, isFirstRepartition);

        if (res != true) {
            std::cout << "Could not repartition!" << std::endl;
            return;
        }
        // Update
        this->updateLayout(fl, mesh, buffer, isFirstRepartition);
        if constexpr (Dim == 2 || Dim == 3) {
            if (stype_m == "FFT") {
                std::get<FFTSolver_t<T, Dim>>(solver_m).setRhs(rhs_m);
            }
            if constexpr (Dim == 3) {
                if (stype_m == "P3M") {
                    std::get<P3MSolver_t<T, Dim>>(solver_m).setRhs(rhs_m);
                } else if (stype_m == "OPEN") {
                    std::get<OpenSolver_t<T, Dim>>(solver_m).setRhs(rhs_m);
                }
            }
        }
    }

    /**
     * @brief Balance
     * @param totalP Total particles
     * @param nstep Number of steps
     * @param TestName Test name
     * @return True if balanced, false otherwise
     */

    bool balance(size_type totalP, const unsigned int nstep, const char* TestName) {
        if (ippl::Comm->size() < 2) {
            return false;
        }
        if (std::strcmp(TestName, "UniformPlasmaTest") == 0) {
            return (nstep % loadbalancefreq_m == 0);
        } else {
            int local = 0;
            std::vector<int> res(ippl::Comm->size());
            double equalPart = (double)totalP / ippl::Comm->size();
            double dev       = std::abs((double)this->getLocalNum() - equalPart) / totalP;
            if (dev > loadbalancethreshold_m) {
                local = 1;
            }
            MPI_Allgather(&local, 1, MPI_INT, res.data(), 1, MPI_INT,
                          ippl::Comm->getCommunicator());

            for (unsigned int i = 0; i < res.size(); i++) {
                if (res[i] == 1) {
                    return true;
                }
            }
            return false;
        }
    }

    /*
      All about the solver(s) in a PIC context

     */

    /**
     * @brief Initialize solver
     */

    void initSolver() {
        Inform m("solver ");
        if (stype_m == "FFT") {
            initFFTSolver();
        } else if (stype_m == "CG") {
            initCGSolver();
        } else if (stype_m == "P3M") {
            initP3MSolver();
        } else if (stype_m == "OPEN") {
            initOpenSolver();
        } else {
            m << "No solver matches the argument" << endl;
        }
    }

    /**
     * @brief run selected solver
     */
    void runSolver() {
        if (stype_m == "CG") {
            CGSolver_t<T, Dim>& solver = std::get<CGSolver_t<T, Dim>>(solver_m);
            solver.solve();

            if (ippl::Comm->rank() == 0) {
                std::stringstream fname;
                fname << "data/CG_";
                fname << ippl::Comm->size();
                fname << ".csv";

                Inform log(NULL, fname.str().c_str(), Inform::APPEND);
                int iterations = solver.getIterationCount();
                // Assume the dummy solve is the first call
                if (time_m == 0 && iterations == 0) {
                    log << "time,residue,iterations" << endl;
                }
                // Don't print the dummy solve
                if (time_m > 0 || iterations > 0) {
                    log << time_m << "," << solver.getResidue() << "," << iterations << endl;
                }
            }
            ippl::Comm->barrier();
        } else if (stype_m == "FFT") {
            if constexpr (Dim == 2 || Dim == 3) {
                std::get<FFTSolver_t<T, Dim>>(solver_m).solve();
            }
        } else if (stype_m == "P3M") {
            if constexpr (Dim == 3) {
                std::get<P3MSolver_t<T, Dim>>(solver_m).solve();
            }
        } else if (stype_m == "OPEN") {
            if constexpr (Dim == 3) {
                std::get<OpenSolver_t<T, Dim>>(solver_m).solve();
            }
        } else {
            throw std::runtime_error("Unknown solver type");
        }
    }

    /**
     * @brief Configure solver based on ippl::ParameterList
     */

    template <typename Solver>
    void initSolverWithParams(const ippl::ParameterList& sp) {
        solver_m.template emplace<Solver>();
        Solver& solver = std::get<Solver>(solver_m);

        solver.mergeParameters(sp);

        solver.setRhs(rhs_m);

        if constexpr (std::is_same_v<Solver, CGSolver_t<T, Dim>>) {
            // The CG solver computes the potential directly and
            // uses this to get the electric field
            solver.setLhs(sol_m);
            solver.setGradient(F_m);
        } else {
            // The periodic Poisson solver, Open boundaries solver,
            // and the P3M solver compute the electric field directly
            solver.setLhs(F_m);
        }
    }

    /**
     * @brief Configure CG solver
     */

    void initCGSolver() {
        ippl::ParameterList sp;
        sp.add("output_type", CGSolver_t<T, Dim>::GRAD);
        // Increase tolerance in the 1D case
        sp.add("tolerance", 1e-10);

        initSolverWithParams<CGSolver_t<T, Dim>>(sp);
    }

    /**
     * @brief Configure FFT solver
     */

    void initFFTSolver() {
        if constexpr (Dim == 2 || Dim == 3) {
            ippl::ParameterList sp;
            sp.add("output_type", FFTSolver_t<T, Dim>::GRAD);
            sp.add("use_heffte_defaults", false);
            sp.add("use_pencils", true);
            sp.add("use_reorder", false);
            sp.add("use_gpu_aware", true);
            sp.add("comm", ippl::p2p_pl);
            sp.add("r2c_direction", 0);

            initSolverWithParams<FFTSolver_t<T, Dim>>(sp);
        } else {
            throw std::runtime_error("Unsupported dimensionality for FFT solver");
        }
    }

    /**
     * @brief Configure P3M solver
     */

    void initP3MSolver() {
        if constexpr (Dim == 3) {
            ippl::ParameterList sp;
            sp.add("output_type", P3MSolver_t<T, Dim>::GRAD);
            sp.add("use_heffte_defaults", false);
            sp.add("use_pencils", true);
            sp.add("use_reorder", false);
            sp.add("use_gpu_aware", true);
            sp.add("comm", ippl::p2p_pl);
            sp.add("r2c_direction", 0);

            initSolverWithParams<P3MSolver_t<T, Dim>>(sp);
        } else {
            throw std::runtime_error("Unsupported dimensionality for P3M solver");
        }
    }

    /**
     * @brief Configure Open solver
     * @FixMe what is the difference to the FFT solver?
     */

    void initOpenSolver() {
        if constexpr (Dim == 3) {
            ippl::ParameterList sp;
            sp.add("output_type", OpenSolver_t<T, Dim>::GRAD);
            sp.add("use_heffte_defaults", false);
            sp.add("use_pencils", true);
            sp.add("use_reorder", false);
            sp.add("use_gpu_aware", true);
            sp.add("comm", ippl::p2p_pl);
            sp.add("r2c_direction", 0);
            sp.add("algorithm", OpenSolver_t<T, Dim>::HOCKNEY);

            initSolverWithParams<OpenSolver_t<T, Dim>>(sp);
        } else {
            throw std::runtime_error("Unsupported dimensionality for OPEN solver");
        }
    }

    /**
     * @brief
     *
     */
    typename VField_t<T, Dim>::HostMirror getForceFieldMirror() const {
        auto Eview = F_m.getHostMirror();
        updateEMirror(Eview);
        return Eview;
    }

    /**
     * @brief
     *
     */
    void updateForceFieldMirror(typename VField_t<T, Dim>::HostMirror& mirror) const {
        Kokkos::deep_copy(mirror, F_m.getView());
    }
};
#endif