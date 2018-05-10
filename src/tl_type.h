#ifndef _TL_TYPE_H_
#define _TL_TYPE_H_

#include <stdint.h>
#include <string.h>
#include <stdio.h>

typedef enum tl_bool_t tl_bool_t;
enum tl_bool_t {
     TL_FALSE = 0,
     TL_TRUE = 1
};

/* keep the size and the enum order in sync with tl_type.c */
#define TL_DTYPE_SIZE 8
typedef enum tl_dtype tl_dtype;
enum tl_dtype {
     TL_FLOAT,
     TL_INT32,
     TL_INT16,
     TL_INT8,
     TL_UINT32,
     TL_UINT16,
     TL_UINT8,
     TL_BOOL,
};

/* keep the size and the enum order in sync with tl_type.c */
#define TL_ELEW_OP_SIZE 7
typedef enum tl_elew_op tl_elew_op;
enum tl_elew_op {
     TL_MUL,
     TL_DIV,
     TL_SUM,
     TL_SUB,
     TL_MAX,
     TL_MIN,
     TL_POW
};

/* pointer subtraction and pointer addition */
#define tl_psub(p1, p2, dsize)                                  \
     (((uint8_t *)(p1) - (uint8_t *)(p2)) / ((ptrdiff_t)dsize))
#define tl_padd(p, offset, dsize)               \
     ((uint8_t *)(p) + (offset) * (dsize))

/* array element assignment */
#define tl_passign(pd, offd, ps, offs, dsize)           \
     memmove(tl_padd((pd), (offd), (dsize)),            \
             tl_padd((ps), (offs), (dsize)), (dsize))

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*tl_fprintf_func) (FILE *fp, const char *fmt, void *p);
typedef int (*tl_cmp_func) (void *p1, void *p2);
typedef void (*tl_elew_func) (void *p1, void *p2, void *r, tl_elew_op elew_op);

size_t tl_size_of(tl_dtype dtype);
const char *tl_fmt(tl_dtype dtype);

#define tl_pointer_sub(p1, p2, dtype)           \
     tl_psub((p1), (p2), tl_size_of(dtype))
#define tl_pointer_add(p, offset, dtype)        \
     tl_padd((p), (offset), tl_size_of(dtype))
#define tl_pointer_assign(pd, offd, ps, offs, dtype)            \
     tl_passign((pd), (offd), (ps), (offs), tl_size_of(dtype))

int tl_fprintf(FILE* fp, const char* fmt,void* p, tl_dtype dtype);
tl_fprintf_func tl_fprintf_getfunc(tl_dtype dtype);
int tl_cmp(void *p1, void *p2, tl_dtype dtype);
tl_cmp_func tl_cmp_getfunc(tl_dtype dtype);
void tl_elew(void *p1, void *p2, void *r, tl_elew_op elew_op, tl_dtype dtype);
tl_elew_func tl_elew_getfunc(tl_dtype dtype);

#ifdef __cplusplus
}
#endif

#endif  /* _TL_TYPE_H_ */
