#ifndef IPPL_FFT_NFFT_H
#define IPPL_FFT_NFFT_H

#include "FFT.h"

/*
template <typename T, class... Properties>
    template <typename Field, class PT>
    void ParticleAttrib<T, Properties...>::spread(
        Field& f, const ParticleAttrib<Vector<PT, Field::dim>, Properties...>& pp) const {
        constexpr unsigned Dim = Field::dim;
        using PositionType     = typename Field::Mesh_t::value_type;

        static IpplTimings::TimerRef scatterTimer = IpplTimings::getTimer("scatter");
        IpplTimings::startTimer(scatterTimer);
        using view_type = typename Field::view_type;
        view_type view  = f.getView();

        using mesh_type       = typename Field::Mesh_t;
        const mesh_type& mesh = f.get_mesh();

        using vector_type = typename mesh_type::vector_type;
        using value_type  = typename ParticleAttrib<T, Properties...>::value_type;

        const vector_type& dx     = mesh.getMeshSpacing();
        const vector_type& origin = mesh.getOrigin();
        const vector_type invdx   = 1.0 / dx;

        const FieldLayout<Dim>& layout = f.getLayout();
        const NDIndex<Dim>& lDom       = layout.getLocalNDIndex();
        const int nghost               = f.getNghost();

        using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::scatter", policy_type(0, *(this->localNum_mp)),
            KOKKOS_CLASS_LAMBDA(const size_t idx) {
                // find nearest grid point
                vector_type l                        = (pp(idx) - origin) * invdx + 0.5;
                Vector<int, Field::dim> index        = l;
                Vector<PositionType, Field::dim> whi = l - index;
                Vector<PositionType, Field::dim> wlo = 1.0 - whi;

                Vector<size_t, Field::dim> args = index - lDom.first() + nghost;

                // scatter
                const value_type& val = dview_m(idx);
                detail::scatterToField(std::make_index_sequence<1 << Field::dim>{}, view, wlo, whi,
                                       args, val);
            });
        IpplTimings::stopTimer(scatterTimer);

        static IpplTimings::TimerRef accumulateHaloTimer = IpplTimings::getTimer("accumulateHalo");
        IpplTimings::startTimer(accumulateHaloTimer);
        f.accumulateHalo();
        IpplTimings::stopTimer(accumulateHaloTimer);
    }*/

int main(int argc, char **argv){
    int M = 1e7;                                   // number of nonuniform points
    View<double, M> x;              // Non uniform points 
    View<Kokkos::complex<double>, M > c;        // complex strenghts
    Kokkos::complex<double> I = Kokkos::complex<double>(0.0,1.0);  // the imaginary unit

    Kokkos::Random_XorShift64_Pool<> random_pool(12345);

     using policy_type = Kokkos::RangePolicy<execution_space>;
        Kokkos::parallel_for(
            "ParticleAttrib::scatter", policy_type(0, M),
            KOKKOS_CLASS_LAMBDA(const size_t idx) {
                auto generator = random_pool.get_state();

                x(j) = Kokkos::numbers::pi_v<double>*(2*(generator.drand(0., 1.)/RAND_MAX)-1); // uniform random in [-pi,pi)
                c(j) = 2*((double)rand()/RAND_MAX)-1 + I*(2*((generator.drand(0., 1.)/RAND_MAX)-1);

                random_pool.free_state(generator);
            });

// 1. Compute kernels & rescale them
//      -Find nearest gridpoint & scatter to field, evaluate phi on field gridpoint   


// 2. Convolution FFT{FFT^(-1){cj * phi}}

// 3. Truncate to the central N frequencies and compute f_k = p_k b_k, p_k = 2/(wφ̂(αk))




    return 0;
}