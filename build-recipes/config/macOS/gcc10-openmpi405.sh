export OTB_TOOLSET=gcc
export OTB_COMPILER=gcc
export OTB_COMPILER_VERSION=10.3.0
export OTB_MPI=openmpi
export OTB_MPI_VERSION=4.0.5

declare -a OTB_RECIPES=(
	010-build-gmp
	020-build-mpfr
	030-build-mpc
	040-build-gcc
	050-build-cmake
	060-build-openmpi
	070-build-zlib
	080-build-hdf5
	090-build-gsl
	100-build-h5hut
	105-build-icu4c
	110-build-boost
	200-build-parmetis
	210-build-openblas
	220-build-trilinos
	230-build-amrex
	240-build-MITHRA
	300-build-gtest)

declare -A OTB_SYMLINKS
