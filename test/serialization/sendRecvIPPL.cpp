#include <iostream>
#include "Ippl.h"

int main(int argc, char *argv[]) {

	Ippl ippl(argc, argv);
     	int rank = Ippl::Comm->rank();
     	int size = Ippl::Comm->size();

    	Ippl::Comm->setDefaultOverallocation(std::atof(argv[1]));
	using buffer_type = ippl::Communicate::buffer_type;
      	size_t totalRequests = 0;
      	if(rank == 0)
       	   totalRequests = size-1;
	  
      	std::vector<MPI_Request> requests(totalRequests);
	ippl::detail::FieldBufferData<double> fd;

      	for (int niter = 0; niter < 10; ++niter) {
		int nsends = (niter+1)*100;
		int nrecvs = nsends;
		auto& fbuffer = fd.buffer;
                Kokkos::realloc(fbuffer, nsends);
             	if(rank == 0) {
                	Kokkos::deep_copy(fbuffer, 1.0);
		 	size_t rIndex=0;
                 	for (int nr = 1; nr < size; ++nr) {
         			buffer_type buf = Ippl::Comm->getBuffer<double>(1000+nr,nsends);
                 		Ippl::Comm->isend(nr, 42, fd, *buf, requests[rIndex++], nsends);
                    		buf->resetWritePos();
			}
             	}
             	else {
                	buffer_type buf = Ippl::Comm->getBuffer<double>(2000, nrecvs);
                    	Ippl::Comm->recv(0, 42, fd, *buf, nrecvs * sizeof(double), nrecvs);
                    	buf->resetReadPos();
             	}
             	if (totalRequests > 0) {
                	MPI_Waitall(totalRequests, requests.data(), MPI_STATUSES_IGNORE);
             	}
		if(rank == 0) {
			std::cout << "Iter: " << niter+1 << " completed " << std::endl;
		}

        }


     	return 0;
}
