#ifndef _TL_TYPE_H_
#define _TL_TYPE_H_

#include <stdint.h>
#include <string.h>
#include <stdio.h>

enum tl_bool_t {
     TL_FALSE = 0,
     TL_TRUE = 1
};
typedef enum tl_bool_t tl_bool_t;

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
typedef enum tl_dtype tl_dtype;

#define TL_DTYPE_SIZE 8

/* pointer subtraction and pointer addition */
#define tl_psub(p1, p2, dsize)                      \
     (((uint8_t *)(p1) - (uint8_t *)(p2) / (dsize))
#define tl_padd(p, offset, dsize)               \
     ((uint8_t *)(p) + (offset) * (dsize))

/* array element assignment */
#define tl_passign(pd, offd, ps, offs, dsize)           \
     memmove(tl_padd((pd), (offd), (dsize)),            \
             tl_padd((ps), (offs), (dsize)), (dsize))

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*tl_gfprintf_func) (FILE *fp, const char *fmt, void *p);
typedef int (*tl_gcmp_func) (void *p1, void *p2);
typedef void (*tl_gmul_func) (void *p1, void *p2, void *r);

size_t tl_size_of(tl_dtype dtype);
char *tl_fmt(tl_dtype dtype);

#define tl_pointer_sub(p1, p2, dtype)           \
     tl_psub((p1), (p2), tl_size_of(dtype))
#define tl_pointer_add(p, offset, dtype)        \
     tl_padd((p), (offset), tl_size_of(dtype))
#define tl_pointer_assign(pd, offd, ps, offs, dtype)            \
     tl_passign((pd), (offd), (ps), (offs), tl_size_of(dtype))

void tl_gfprintf(FILE* fp,const char* fmt,void* p, tl_dtype dtype);
tl_gfprintf_func tl_gfprintf_getfunc(tl_dtype dtype);
int tl_gcmp(void *p1, void *p2, tl_dtype dtype);
tl_gcmp_func tl_gcmp_getfunc(tl_dtype dtype);
void tl_gmul(void *p1, void *p2, void *r, tl_dtype dtype);
tl_gmul_func tl_gmul_getfunc(tl_dtype dtype);

#ifdef __cplusplus
}
#endif

#endif  /* _TL_TYPE_H_ */
