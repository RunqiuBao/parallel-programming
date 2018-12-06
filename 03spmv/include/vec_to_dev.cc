/**
   @file vec_to_dev.cc
   @brief make a deivce copy of a vector.
*/

/** 
    @brief make a deivce copy of a vector.
    @param (v) the reference to a matrix whose elems_dev has not 
    been set (i.e., = NULL)
    @returns 1 if succeed. 0 if failed.
    @sa sparse_to_dev
*/
static int vec_to_dev(vec_t& v) {
  fprintf(stderr,
          "*************************************************************\n"
          "%s:%d:vec_to_dev:\n"
          "write a code that copies the elements of v to the device.\n"
          "use dev_malloc and to_dev utility functions in cuda_util.h\n"
          "*************************************************************\n",
          __FILE__, __LINE__);

  /*runqiu:copying vec to device*/
  check_api_error((cudaMalloc((void **)&v.elems_dev, sizeof(real)*v.n)));
  check_api_error((cudaMemcpy(v.elems_dev, v.elems, sizeof(real)*v.n, cudaMemcpyHostToDevice)));
  
  return 1;
}
