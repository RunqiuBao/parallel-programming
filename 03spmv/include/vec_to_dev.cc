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
  /*fprintf(stderr,
          "*************************************************************\n"
          "%s:%d:vec_to_dev:\n"
          "write a code that copies the elements of v to the device.\n"
          "use dev_malloc and to_dev utility functions in cuda_util.h\n"
          "*************************************************************\n",
          __FILE__, __LINE__);*/

    //sparse_t dst; //address on device
      fprintf(stderr,
              "*************************************************************\n"
              "v allocating\n"
              "*************************************************************\n",
              __FILE__, __LINE__);

    //first get the size of A
    size_t sz = sizeof(*(v.elems));
    fprintf(stderr,
              "*************************************************************\n"
              "v got the size\n"
              "*************************************************************\n",
              __FILE__, __LINE__);

    //allocate the dev memory
    void * dst = dev_malloc(sz);
    v.elems_dev =(real *)dst;
    fprintf(stderr,
              "*************************************************************\n"
              "v assigned the size\n"
              "*************************************************************\n",
              __FILE__, __LINE__);

    //copy the data to device
    to_dev( (void *)v.elems_dev, (void *)v.elems, sz);

    fprintf(stderr,
            "*************************************************************\n"
            "v allocated\n"
            "*************************************************************\n",
            __FILE__, __LINE__);
  return 1;
}
