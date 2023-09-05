// Landau Damping Test
//   Usage:
//     srun ./LandauDamping
//                  <nx> [<ny>...] <Np> <Nt> <stype>
//                  <lbthres> --overallocate <ovfactor> --info 10
//     nx       = No. cell-centered points in the x-direction
//     ny...    = No. cell-centered points in the y-, z-, ...-direction
//     Np       = Total no. of macro-particles in the simulation
//     Nt       = Number of time steps
//     stype    = Field solver type (FFT and CG supported)
//     lbthres  = Load balancing threshold i.e., lbthres*100 is the maximum load imbalance
//                percentage which can be tolerated and beyond which
//                particle load balancing occurs. A value of 0.01 is good for many typical
//                simulations.
//     ovfactor = Over-allocation factor for the buffers used in the communication. Typical
//                values are 1.0, 2.0. Value 1.0 means no over-allocation.
//     Example:
//     srun ./LandauDamping 128 128 128 10000 10 FFT 0.01 --overallocate 2.0 --info 10
//
// Copyright (c) 2021, Sriramkrishnan Muralikrishnan,
// Paul Scherrer Institut, Villigen PSI, Switzerland
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

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "Utility/IpplTimings.h"

//#include "ChargedParticles.hpp"
#include "Manager/PicManager.h"
#include "datatypes.h"
#include "ParticleContainer.hpp"
#include "FieldContainer.hpp"
#include "FieldSolver.hpp"
 
 constexpr unsigned Dim = 3;
using T = double;

template <typename T>
struct Newton1D {
    double tol   = 1e-12;
    int max_iter = 20;
    double pi    = Kokkos::numbers::pi_v<double>;

    T k, alpha, u;

    KOKKOS_INLINE_FUNCTION Newton1D() {}

    KOKKOS_INLINE_FUNCTION Newton1D(const T& k_, const T& alpha_, const T& u_)
        : k(k_)
        , alpha(alpha_)
        , u(u_) {}

    KOKKOS_INLINE_FUNCTION ~Newton1D() {}

    KOKKOS_INLINE_FUNCTION T f(T& x) {
        T F;
        F = x + (alpha * (Kokkos::sin(k * x) / k)) - u;
        return F;
    }

    KOKKOS_INLINE_FUNCTION T fprime(T& x) {
        T Fprime;
        Fprime = 1 + (alpha * Kokkos::cos(k * x));
        return Fprime;
    }

    KOKKOS_FUNCTION
    void solve(T& x) {
        int iterations = 0;
        while (iterations < max_iter && Kokkos::fabs(f(x)) > tol) {
            x = x - (f(x) / fprime(x));
            iterations += 1;
        }
    }
};

template <typename T, class GeneratorPool, unsigned Dim>
struct generate_random {
    using view_type  = typename ippl::detail::ViewType<T, 1>::view_type;
    using value_type = typename T::value_type;
    // Output View for the random numbers
    view_type x, v;

    // The GeneratorPool
    GeneratorPool rand_pool;

    value_type alpha;

    T k, minU, maxU;

    // Initialize all members
    generate_random(view_type x_, view_type v_, GeneratorPool rand_pool_, value_type& alpha_, T& k_,
                    T& minU_, T& maxU_)
        : x(x_)
        , v(v_)
        , rand_pool(rand_pool_)
        , alpha(alpha_)
        , k(k_)
        , minU(minU_)
        , maxU(maxU_) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        // Get a random number state from the pool for the active thread
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

        value_type u;
        for (unsigned d = 0; d < Dim; ++d) {
            u       = rand_gen.drand(minU[d], maxU[d]);
            x(i)[d] = u / (1 + alpha);
            Newton1D<value_type> solver(k[d], alpha, u);
            solver.solve(x(i)[d]);
            v(i)[d] = rand_gen.normal(0.0, 1.0);
        }

        // Give the state back, which will allow another thread to acquire it
        rand_pool.free_state(rand_gen);
    }
};

double CDF(const double& x, const double& alpha, const double& k) {
    double cdf = x + (alpha / k) * std::sin(k * x);
    return cdf;
}

KOKKOS_FUNCTION
double PDF(const Vector_t<double, Dim>& xvec, const double& alpha, const Vector_t<double, Dim>& kw,
           const unsigned Dim) {
    double pdf = 1.0;

    for (unsigned d = 0; d < Dim; ++d) {
        pdf *= (1.0 + alpha * Kokkos::cos(kw[d] * xvec[d]));
    }
    return pdf;
}

const char* TestName = "LandauDamping";

//template <typename T, unsigned Dim>
class MyPicManager : public ippl::PicManager<ParticleContainer<double, 3>, FieldContainer<double, 3>, FieldSolver<double, 3>> {
public:
    double Q_m;
    MyPicManager(double Q)
        : ippl::PicManager<ParticleContainer<double, 3>, FieldContainer<double, 3>, FieldSolver<double, 3>>(), Q_m(Q){
    }

    void par2grid() override {
        scatterCIC();
    }

    void grid2par() override {
        gatherCIC();
    }
    
    void gatherCIC() { gather(pcontainer_m->E, fcontainer_m->E_m, pcontainer_m->R); }
    
    void scatterCIC() {
        Inform m("scatter ");

        fcontainer_m->rho_m = 0.0;
        scatter(pcontainer_m->q, fcontainer_m->rho_m, pcontainer_m->R);

        std::cout << std::fabs((Q_m - fcontainer_m->rho_m.sum()) / Q_m)  << std::endl;

        size_type Total_particles = 0;
        size_type local_particles = pcontainer_m->getLocalNum();

        MPI_Reduce(&local_particles, &Total_particles, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
                   ippl::Comm->getCommunicator());

        double cellVolume =
            std::reduce(fcontainer_m->hr_m.begin(), fcontainer_m->hr_m.end(), 1., std::multiplies<double>());
        fcontainer_m->rho_m = fcontainer_m->rho_m / cellVolume;

        // rho = rho_e - rho_i (only if periodic BCs)
        if (fsolver_m->stype_m != "OPEN") {
            double size = 1;
            for (unsigned d = 0; d < Dim; d++) {
                size *= fcontainer_m->rmax_m[d] - fcontainer_m->rmin_m[d];
            }
            fcontainer_m->rho_m = fcontainer_m->rho_m - (Q_m / size);
        }
    }
};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg("LandauDamping");
        Inform msg2all("LandauDamping", INFORM_ALL_NODES);

        int arg = 1;

        Vector_t<int, Dim> nr;
        for (unsigned d = 0; d < Dim; d++) {
            nr[d] = std::atoi(argv[arg++]);
        }

        const size_type totalP = std::atoll(argv[arg++]);
        const unsigned int nt  = std::atoi(argv[arg++]);

        msg << "Landau damping" << endl
            << "nt " << nt << " Np= " << totalP << " grid = " << nr << endl;

       ippl::NDIndex<Dim> domain;
        for (unsigned i = 0; i < Dim; i++) {
            domain[i] = ippl::Index(nr[i]);
        }

        ippl::e_dim_tag decomp[Dim];
        for (unsigned d = 0; d < Dim; ++d) {
            decomp[d] = ippl::PARALLEL;
        }
        
        // create mesh and layout objects for this problem domain
        Vector_t<double, Dim> kw = 0.5;
        double alpha             = 0.05;
        Vector_t<double, Dim> rmin(0.0);
        Vector_t<double, Dim> rmax = 2 * pi / kw;

        Vector_t<double, Dim> hr = rmax / nr;
        // Q = -\int\int f dx dv
        double Q = std::reduce(rmax.begin(), rmax.end(), -1., std::multiplies<double>());
        Vector_t<double, Dim> origin = rmin;
        const double dt              = std::min(.05, 0.5 * *std::min_element(hr.begin(), hr.end()));
        
        const bool isAllPeriodic = true;
        Mesh_t<Dim> mesh(domain, hr, origin);
        FieldLayout_t<Dim> FL(domain, decomp, isAllPeriodic);
        PLayout_t<double, Dim> PL(FL, mesh);

        std::string solver = argv[arg++];

        if (solver == "OPEN") {
            throw IpplException("LandauDamping",
                                "Open boundaries solver incompatible with this simulation!");
        }
        
        using ParticleContainer_t = ParticleContainer<T, Dim>;
        std::shared_ptr<ParticleContainer_t> pc = std::make_shared<ParticleContainer_t>(PL);
        
        using FieldContainerType = FieldContainer<T, Dim>;
        std::shared_ptr<FieldContainerType> fc = std::make_shared<FieldContainerType>(hr, rmin, rmax, decomp);
        
        printf("initializeFields\n");
        fc->initializeFields(mesh, FL);
        
        using FieldSolverType = FieldSolver<T, Dim>;
        std::shared_ptr<FieldSolverType> fs = std::make_shared<FieldSolverType>(solver, fc->rho_m, fc->E_m);
        
        MyPicManager manager(Q);
        
        manager.setParticleContainer(pc);
        manager.setFieldContainer(fc);
        manager.setFieldSolver(fs);
        fs->initSolver();
        
        //bool isFirstRepartition;

        typedef ippl::detail::RegionLayout<double, Dim, Mesh_t<Dim>>::uniform_type RegionLayout_t;
        const RegionLayout_t& RLayout                           = PL.getRegionLayout();
        const typename RegionLayout_t::host_mirror_type Regions = RLayout.gethLocalRegions();
        Vector_t<double, Dim> Nr, Dr, minU, maxU;
        int myRank    = ippl::Comm->rank();
        double factor = 1;
        for (unsigned d = 0; d < Dim; ++d) {
            Nr[d] = CDF(Regions(myRank)[d].max(), alpha, kw[d])
                    - CDF(Regions(myRank)[d].min(), alpha, kw[d]);
            Dr[d]   = CDF(rmax[d], alpha, kw[d]) - CDF(rmin[d], alpha, kw[d]);
            minU[d] = CDF(Regions(myRank)[d].min(), alpha, kw[d]);
            maxU[d] = CDF(Regions(myRank)[d].max(), alpha, kw[d]);
            factor *= Nr[d] / Dr[d];
        }

        size_type nloc            = (size_type)(factor * totalP);
        size_type Total_particles = 0;

        MPI_Allreduce(&nloc, &Total_particles, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                      ippl::Comm->getCommunicator());

        int rest = (int)(totalP - Total_particles);

        if (ippl::Comm->rank() < rest) {
            ++nloc;
        }
        
        pc->create(nloc);
        
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42 + 100 * ippl::Comm->rank()));
        Kokkos::parallel_for(
            nloc, generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                      pc->R.getView(), pc->P.getView(), rand_pool64, alpha, kw, minU, maxU));

        Kokkos::fence();
        ippl::Comm->barrier();

        pc->q = Q / totalP;
        msg << "particles created and initial conditions assigned " << endl;
        
        fc->rho_m = 0.0;
        fs->runSolver();
        manager.par2grid();
        fs->runSolver();
        manager.grid2par();

        
        // begin main timestep loop
        msg << "Starting iterations ..." << endl;
        for (unsigned int it = 0; it < nt; it++) {
            // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
            // Here, we assume a constant charge-to-mass ratio of -1 for
            // all the particles hence eliminating the need to store mass as
            // an attribute
            // kick

            //IpplTimings::startTimer(PTimer);
            pc->P = pc->P - 0.5 * dt * pc->E;
            //IpplTimings::stopTimer(PTimer);

            // drift
            //IpplTimings::startTimer(RTimer);
            pc->R = pc->R + dt * pc->P;
            //IpplTimings::stopTimer(RTimer);
            // P->R.print();

            // Since the particles have moved spatially update them to correct processors
            //IpplTimings::startTimer(updateTimer);
            //PL.update(*P, bunchBuffer);
            //IpplTimings::stopTimer(updateTimer);

            // Domain Decomposition
            //if (P->balance(totalP, it + 1)) {
            //    msg << "Starting repartition" << endl;
            //    IpplTimings::startTimer(domainDecomposition);
            //    P->repartition(FL, mesh, bunchBuffer, isFirstRepartition);
            //    IpplTimings::stopTimer(domainDecomposition);
            //    // IpplTimings::startTimer(dumpDataTimer);
            //    // P->dumpLocalDomains(FL, it+1);
            //    // IpplTimings::stopTimer(dumpDataTimer);
            //}

            // scatter the charge onto the underlying grid
            manager.par2grid();
            
            // Field solve
            //IpplTimings::startTimer(SolveTimer);
            fs->runSolver();
            //IpplTimings::stopTimer(SolveTimer);

            // gather E field
            manager.grid2par();

            // kick
            //IpplTimings::startTimer(PTimer);
            pc->P = pc->P - 0.5 * dt * pc->E;
            //IpplTimings::stopTimer(PTimer);

            //P->time_m += dt;
            //IpplTimings::startTimer(dumpDataTimer);
            //P->dumpLandau();
            //P->gatherStatistics(totalP);
            //IpplTimings::stopTimer(dumpDataTimer);
            //msg << "Finished time step: " << it + 1 << " time: " << P->time_m << endl;

            //if (checkSignalHandler()) {
            //    msg << "Aborting timestepping loop due to signal " << interruptSignalReceived
            //        << endl;
            //    break;
            //}
        }
        msg << "LandauDamping: End." << endl;
        //IpplTimings::stopTimer(mainTimer);
        //IpplTimings::print();
        //IpplTimings::print(std::string("timing.dat"));
        //auto end = std::chrono::high_resolution_clock::now();

        //std::chrono::duration<double> time_chrono =
        //std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        //std::cout << "Elapsed time: " << time_chrono.count() << std::endl;
        
    }
    ippl::finalize();

    return 0;
}