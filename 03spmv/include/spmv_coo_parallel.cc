/**
  @file spmv_coo_parallel.cc
  @brief y = A * x for coo with parallel for
 */

/** 
    @brief y = A * x for coo with parallel for
    @param (A) a sparse matrix
    @param (vx) a vector
    @param (vy) a vector
    @returns 1 if succeed, 0 if failed
*/
static int spmv_coo_parallel(sparse_t A, vec_t vx, vec_t vy) {
  /*fprintf(stderr,
          "*************************************************************\n"
          "%s:%d:spmv_coo_parallel:\n"
          "write a code that performs SPMV for COO format in parallel\n"
          "using parallel for + atomic directives\n"
          "*************************************************************\n",
          __FILE__, __LINE__);*/
  #pragma omp parallel
  //int t = omp_get_thread_num();
  //int nt = opm_num_threads();

  idx_t M = A.M;
  idx_t nnz = A.nnz;
  coo_elem_t * elems = A.coo.elems;
  real * x = vx.elems;
  real * y = vy.elems;
  for (idx_t i = 0; i < M; i++) {
    y[i] = 0.0;
  }
  #pragma omp for
  for (idx_t k = 0; k < nnz; k++) {
      //i,j, Aij = A.elems[k];
      //#pragma omp atomic
      //y[i] += Aij * x[j];
      coo_elem_t * e = elems + k;
      idx_t i = e->i;
      idx_t j = e->j;
      real  a = e->a;
      real ax = a * x[j];
      #pragma omp atomic
      y[i] += ax;
  }
  return 1;


  //exit(1);

  /* this is a serial code for your reference */
  /*
  idx_t M = A.M;
  idx_t nnz = A.nnz;
  coo_elem_t * elems = A.coo.elems;
  real * x = vx.elems;
  real * y = vy.elems;
  for (idx_t i = 0; i < M; i++) {
    y[i] = 0.0;
  }
  for (idx_t k = 0; k < nnz; k++) {
    coo_elem_t * e = elems + k;
    idx_t i = e->i;
    idx_t j = e->j;
    real  a = e->a;
    real ax = a * x[j];
    y[i] += ax;
  }
  return 1;
  */
}
