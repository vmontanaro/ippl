 #include <iostream>
 #include <mpi.h>
 #include <Kokkos_Core.hpp> 

int main(int argc, char *argv[]) {

     MPI_Init(&argc, &argv);

     Kokkos::initialize(argc,argv);
     {
         int rank = 0;
         MPI_Comm_rank(MPI_COMM_WORLD, &rank);

         int size = 0;
         MPI_Comm_size(MPI_COMM_WORLD, &size);

         typedef Kokkos::View<int*> buffer_type;
	 buffer_type buffer;
	 size_t totalRequests = 0;
	 if(rank == 0)
            totalRequests = size-1;
	  
         std::vector<MPI_Request> requests(totalRequests);
         for (int niter = 0; niter < 10; ++niter) {
             Kokkos::realloc(buffer, (niter*10000)+100);
             //buffer_type buffer("buffer", (niter*10000)+100);
             if(rank == 0) {
                 Kokkos::deep_copy(buffer, 1);
		 size_t rIndex=0;
                 for (int nr = 1; nr < size; ++nr) {
                     MPI_Isend(buffer.data(), buffer.size(),
                              MPI_INT, nr, 42, MPI_COMM_WORLD, &requests[rIndex++]);
                 }
             }
             else {
                 MPI_Status status;
                 MPI_Recv(buffer.data(), buffer.size(),
                          MPI_INT, 0, 42, MPI_COMM_WORLD, &status);
                 buffer_type::HostMirror host_buffer = Kokkos::create_mirror_view(buffer);
                 Kokkos::deep_copy(host_buffer, buffer);
                 std::cout << "Rank: " << rank << " -niter: " << niter << "-----------" << std::endl;
                 //for (size_t i = 0; i < host_buffer.size(); ++i) {
                 //    std::cout << host_buffer(i) << std::endl;
                 //}
             }
             if (totalRequests > 0) {
                MPI_Waitall(totalRequests, requests.data(), MPI_STATUSES_IGNORE);
             }
         }

     }
     Kokkos::finalize();
     MPI_Finalize();

     return 0;
 }
