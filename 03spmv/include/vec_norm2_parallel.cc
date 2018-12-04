/** 
    @file vec_norm2_parallel.cc
    @brief square norm of a vector in serial
*/

/** 
    @brief square norm of a vector in serial
    @param (v) a vector
    @returns the square norm of v (v[0]^2 + ... + v[n-1]^2)
*/
static real vec_norm2_parallel(vec_t v) {
  /*fprintf(stderr,
          "*************************************************************\n"
          "%s:%d:vec_norm2_parallel:\n"
          "write a code that computes square norm of a vector v\n"
          "using parallel for + reduction.\n"
          "*************************************************************\n",
          __FILE__, __LINE__);
  exit(1);*/

  real s = 0.0;
  real * x = v.elems;
  idx_t n = v.n;
  #pragma omp declare reduction (sp: real: s_add (&omp_out, &opm_in)) initializer(s_init(&omp_priv))
  #pragma omp for reduction (sp : s)
  for (idx_t i = 0; i < n; i++) {
    #pragma omp parallel reduction()
    s += x[i] * x[i];
  }
  return s;
}

