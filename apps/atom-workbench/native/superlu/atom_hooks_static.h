#ifndef ATOM_SUPERLU_HOOKS_H
#define ATOM_SUPERLU_HOOKS_H
#include <stddef.h>
#include <stdlib.h>
void *atom_slu_malloc(size_t size);
void atom_slu_free(void *ptr);
_Noreturn void atom_slu_abort(const char *message);
_Noreturn void atom_slu_exit(int code);
#define USER_MALLOC(size) atom_slu_malloc(size)
#define USER_FREE(ptr) atom_slu_free(ptr)
#define USER_ABORT(message) atom_slu_abort(message)
/* Upstream ilu_dpivotL also calls exit directly on structural singularity. */
#define exit(code) atom_slu_exit(code)
/* Same LP64 BLAS as the production Rust solver, without Python hooks. */
#endif

/* SuperLU int32 API aliases; official COLAMD source remains unchanged. */
#define COLAMD_MAIN colamd
#define COLAMD_recommended colamd_recommended
#define COLAMD_set_defaults colamd_set_defaults
