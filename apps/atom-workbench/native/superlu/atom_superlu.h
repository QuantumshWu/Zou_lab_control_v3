#ifndef ATOM_SUPERLU_H
#define ATOM_SUPERLU_H
#include <stddef.h>
#include <stdint.h>
#ifdef _WIN32
#define ATOM_API __declspec(dllexport)
#else
#define ATOM_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
/* ABI 1. LP64 SuperLU: int32 CSC, double values. The original CSC is copied.
 * Result is an ILUTP PRECONDITIONER, never a replacement for the original A.
 * 0=success, negative=bridge error, positive=SuperLU info. On failure *out=NULL.
 * Factor creation is serialized (upstream qsort context is process-global).
 * Independent solves may run concurrently; do not concurrently use one handle.
 * `error` may be NULL when error_capacity=0. All functions are Python-free. */
ATOM_API int atom_superlu_factor(int32_t n, int32_t nnz,
    const int32_t *colptr, const int32_t *rowind, const double *values,
    double drop_tol, double fill_factor, double pivot_threshold,
    void **out, char *error, size_t error_capacity);
/* In-place solve of L/U factors, NOT iterative refinement. Column-major RHS.
 * transpose=0 for A, transpose=1 for transpose(A).
 * For one RHS ld>=n. B is unchanged on argument validation failure. */
ATOM_API int atom_superlu_solve(void *handle, int32_t transpose, int32_t nrhs, double *b,
    int32_t ld, char *error, size_t error_capacity);
ATOM_API uint64_t atom_superlu_factor_nnz(const void *handle);
ATOM_API void atom_superlu_destroy(void *handle);
ATOM_API int atom_superlu_abi_version(void);
#ifdef __cplusplus
}
#endif
#endif
