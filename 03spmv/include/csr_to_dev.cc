/**
   @file csr_to_dev.cc
   @brief make a deivce copy of a sparse matrix in csr format
*/

/** 
    @brief make a deivce copy of a sparse matrix in csr format.
    @param (A) the reference to a matrix whose elems_dev has not 
    been set (i.e., = NULL)
    @returns 1 if succeed. 0 if failed.
    @details this function allocates memory blocks on the device and
    transfers A's row_start array and non-zero elements in 
    the allocated blocks.
    it also should set A's elems_dev and row_start_dev 
    to the addresses of the allocated 
    blocks, so that if you pass A as an argument of a kernel launch,
    the device code can obtain all necessary information of A from
    the parameter.
    @sa sparse_to_dev
*/

static int csr_to_dev(sparse_t& A) {
  fprintf(stderr,
          "*************************************************************\n"
          "%s:%d:csr_to_dev:\n"
          "write a code that copies the elements of A to the device.\n"
          "use dev_malloc and to_dev utility functions in cuda_util.h\n"
          "*************************************************************\n",
          __FILE__, __LINE__);
  //exit(1);
  /*runqiu:copying csr to device*/
  A.csr.elems_dev=(csr_elem_t*)dev_malloc(sizeof(csr_elem_t)*A.nnz);
  to_dev(A.csr.elems_dev, A.csr.elems, sizeof(csr_elem_t)*A.nnz);
  //check_api_error((cudaMalloc((void **)&A.csr.elems_dev, sizeof(*(A.csr.elems))*A.nnz)));
  //check_api_error((cudaMemcpy(A.csr.elems_dev, A.csr.elems, sizeof(*(A.csr.elems))*A.nnz, cudaMemcpyHostToDevice)));
  A.csr.row_start_dev=(idx_t*)dev_malloc(sizeof(idx_t)*A.M);
  to_dev(A.csr.row_start_dev, A.csr.row_start, sizeof(idx_t)*A.M);
  //check_api_error((cudaMalloc((void **)&A.csr.row_start_dev, sizeof(int)*A.M)));
  //check_api_error((cudaMemcpy(A.csr.row_start_dev, A.csr.row_start, sizeof(int)*A.M, cudaMemcpyHostToDevice)));


  return 1;
}


