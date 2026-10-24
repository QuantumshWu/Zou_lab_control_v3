#include "atom_superlu.h"
#include "slu_ddefs.h"
#include <setjmp.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

typedef struct AtomFactor AtomFactor;
typedef union Allocation Allocation;
union Allocation {
    struct { Allocation *prev, *next; AtomFactor *owner; } link;
    max_align_t alignment;
};
struct AtomFactor {
    int n;
    int *perm_c, *perm_r;
    SuperMatrix L, U;
    Allocation *allocations;
    uint64_t nnz;
    int valid;
};
typedef struct { AtomFactor *factor; jmp_buf jump; char message[512]; } AtomCall;
static __declspec(thread) AtomCall *active_call;
/* ilu_dcopy_to_ucol/ilu_ddrop_row use a static qsort comparison context. */
static SRWLOCK factor_lock = SRWLOCK_INIT;

static void write_error(char *error, size_t cap, const char *message) {
    if (error && cap) {
        size_t len = strlen(message);
        if (len >= cap) len = cap - 1;
        memcpy(error, message, len); error[len] = 0;
    }
}
_Noreturn void atom_slu_abort(const char *message) {
    if (active_call) {
        write_error(active_call->message, sizeof(active_call->message), message);
        longjmp(active_call->jump, 1);
    }
    /* Cannot occur through the exported API; protects internal misuse. */
    abort();
}
_Noreturn void atom_slu_exit(int code) {
    char message[96];
    snprintf(message,sizeof(message),"SuperLU stopped with exit status %d",code);
    atom_slu_abort(message);
}
void *atom_slu_malloc(size_t size) {
    if (!active_call || size > SIZE_MAX - sizeof(Allocation))
        atom_slu_abort("SuperLU allocation outside active call or size overflow");
    Allocation *a = malloc(sizeof(Allocation) + (size ? size : 1));
    if (!a) atom_slu_abort("SuperLU allocation failed");
    AtomFactor *f = active_call->factor;
    a->link.owner = f; a->link.prev = NULL; a->link.next = f->allocations;
    if (a->link.next) a->link.next->link.prev = a;
    f->allocations = a;
    return a + 1;
}
void atom_slu_free(void *ptr) {
    if (!ptr) return;
    Allocation *a = (Allocation *)ptr - 1;
    if (a->link.prev) a->link.prev->link.next = a->link.next;
    else a->link.owner->allocations = a->link.next;
    if (a->link.next) a->link.next->link.prev = a->link.prev;
    free(a);
}
ATOM_API void atom_superlu_destroy(void *handle) {
    AtomFactor *f = handle;
    if (!f) return;
    /* Includes allocations left behind by an interrupted factorization. */
    Allocation *a = f->allocations;
    while (a) { Allocation *next = a->link.next; free(a); a = next; }
    free(f);
}
ATOM_API int atom_superlu_abi_version(void) { return 1; }
ATOM_API uint64_t atom_superlu_factor_nnz(const void *handle) {
    const AtomFactor *f = handle; return f && f->valid ? f->nnz : 0;
}
ATOM_API int atom_superlu_factor(int32_t n, int32_t nnz,
    const int32_t *colptr, const int32_t *rowind, const double *values,
    double drop_tol, double fill_factor, double pivot_threshold,
    void **out, char *error, size_t error_capacity) {
    if (out) *out = NULL;
    write_error(error, error_capacity, "");
    if (!out || n <= 0 || nnz < 0 || !colptr || !rowind || !values ||
        !isfinite(drop_tol) || drop_tol < 0 || !isfinite(fill_factor) || fill_factor < 1 ||
        !isfinite(pivot_threshold) || pivot_threshold < 0 || pivot_threshold > 1 || active_call) {
        write_error(error, error_capacity, "Invalid SuperLU factor arguments"); return -1;
    }
    if (colptr[0] != 0 || colptr[n] != nnz) {
        write_error(error, error_capacity, "Invalid CSC column pointers"); return -1;
    }
    for (int col = 0; col < n; ++col) {
        if (colptr[col] > colptr[col+1] || colptr[col] < 0 || colptr[col+1] > nnz) {
            write_error(error, error_capacity, "Nonmonotone CSC column pointers"); return -1;
        }
        int previous = -1;
        for (int k = colptr[col]; k < colptr[col+1]; ++k) {
            if (rowind[k] <= previous || rowind[k] >= n || !isfinite(values[k])) {
                write_error(error, error_capacity, "CSC needs sorted unique row indices and finite values"); return -1;
            }
            previous = rowind[k];
        }
    }
    AtomFactor *f = calloc(1, sizeof(*f));
    if (!f) { write_error(error,error_capacity,"Factor allocation failed"); return -2; }
    f->n = n;
    AtomCall *call = calloc(1,sizeof(*call));
    if (!call) { atom_superlu_destroy(f); write_error(error,error_capacity,"Call allocation failed"); return -2; }
    AcquireSRWLockExclusive(&factor_lock);
    call->factor = f; active_call = call;
    if (setjmp(call->jump)) {
        active_call = NULL; write_error(error,error_capacity,call->message);
        ReleaseSRWLockExclusive(&factor_lock);
        free(call); atom_superlu_destroy(f); return -3;
    }
    int *columns = atom_slu_malloc(((size_t)n+1)*sizeof(int));
    int *rows = atom_slu_malloc((size_t)nnz*sizeof(int));
    double *data = atom_slu_malloc((size_t)nnz*sizeof(double));
    memcpy(columns,colptr,((size_t)n+1)*sizeof(int));
    memcpy(rows,rowind,(size_t)nnz*sizeof(int));
    memcpy(data,values,(size_t)nnz*sizeof(double));
    f->perm_c = intMalloc(n); f->perm_r = intMalloc(n);
    int *etree = intMalloc(n);
    SuperMatrix A = {0}, AC = {0};
    dCreate_CompCol_Matrix(&A,n,n,nnz,data,rows,columns,SLU_NC,SLU_D,SLU_GE);
    superlu_options_t options = {0};
    ilu_set_default_options(&options);
    options.Equil = NO; options.ColPerm = COLAMD; options.PrintStat = NO;
    options.ILU_DropTol = drop_tol; options.ILU_FillFactor = fill_factor;
    options.DiagPivotThresh = pivot_threshold;
    SuperLUStat_t stat; StatInit(&stat);
    get_perm_c(options.ColPerm, &A, f->perm_c);
    sp_preorder(&options, &A, f->perm_c, etree, &AC);
    GlobalLU_t glu = {0}; int info = 0;
    dgsitrf(&options,&AC,sp_ienv(2),sp_ienv(1),etree,NULL,0,
        f->perm_c,f->perm_r,&f->L,&f->U,&glu,&stat,&info);
    StatFree(&stat); Destroy_CompCol_Permuted(&AC); Destroy_CompCol_Matrix(&A);
    atom_slu_free(etree);
    active_call = NULL; free(call); ReleaseSRWLockExclusive(&factor_lock);
    if (info) {
        char message[128]; snprintf(message,sizeof(message),"SuperLU dgsitrf info=%d",info);
        write_error(error,error_capacity,message); atom_superlu_destroy(f); return info;
    }
    f->nnz = (uint64_t)((SCformat *)f->L.Store)->nnz + (uint64_t)((NCformat *)f->U.Store)->nnz;
    f->valid = 1; *out = f; return 0;
}
ATOM_API int atom_superlu_solve(void *handle, int32_t transpose, int32_t nrhs, double *b,
    int32_t ld, char *error, size_t error_capacity) {
    AtomFactor *f = handle; write_error(error,error_capacity,"");
    if (!f || !f->valid || (transpose != 0 && transpose != 1) || nrhs <= 0 || !b || ld < f->n || active_call) {
        write_error(error,error_capacity,"Invalid SuperLU solve arguments"); return -1;
    }
    AtomCall *call = calloc(1,sizeof(*call));
    if (!call) { write_error(error,error_capacity,"Call allocation failed"); return -2; }
    call->factor = f; active_call = call;
    if (setjmp(call->jump)) {
        active_call = NULL; f->valid = 0;
        write_error(error,error_capacity,call->message); free(call); return -3;
    }
    SuperMatrix B = {0}; SuperLUStat_t stat; int info = 0;
    dCreate_Dense_Matrix(&B,f->n,nrhs,b,ld,SLU_DN,SLU_D,SLU_GE);
    StatInit(&stat);
    dgstrs(transpose ? TRANS : NOTRANS,&f->L,&f->U,f->perm_c,f->perm_r,&B,&stat,&info);
    StatFree(&stat); Destroy_SuperMatrix_Store(&B); active_call = NULL; free(call);
    if (info) {
        char message[128]; snprintf(message,sizeof(message),"SuperLU dgstrs info=%d",info);
        write_error(error,error_capacity,message);
    }
    return info;
}
