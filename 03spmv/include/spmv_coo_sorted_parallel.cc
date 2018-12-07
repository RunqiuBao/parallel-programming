/**
  @file spmv_coo_sorted_parallel.cc
  @brief y = A * x for coo_sorted with parallel for
 */

/** 
    @brief y = A * x for coo_sorted with parallel for
    @param (A) a sparse matrix
    @param (vx) a vector
    @param (vy) a vector
    @returns 1 if succeed, 0 if failed
*/
static int spmv_coo_sorted_parallel(sparse_t A, vec_t vx, vec_t vy) {
  /* the same no matter whether elements are sorted or not.
     call spmv_coo_cuda and we are done */
  idx_t M = A.M;
  idx_t nnz = A.nnz;
  coo_elem_t * elems = A.coo.elems;
  real * x = vx.elems;
  real * y = vy.elems;
  for (idx_t i = 0; i < M; i++) {
    y[i] = 0.0;
  }
  #pragma omp parallel for
  for (idx_t k = 0; k < nnz; k++) {
      coo_elem_t * e = elems + k;
      idx_t i = e->i;
      idx_t j = e->j;
      real  a = e->a;
      real ax = a * x[j];
      #pragma omp atomic
      y[i] += ax;
  }
  return 1;
}

