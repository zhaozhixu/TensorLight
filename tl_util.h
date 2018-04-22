#ifndef _TL_UTIL_H_
#define _TL_UTIL_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

typedef enum tl_bool_t tl_bool_t;
enum tl_bool_t {
	TL_FALSE = 0,
	TL_TRUE = 1
};

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
#define TL_DTYPE_SIZE 8

#define TL_MAXLINE 4096

#define tl_free free

/* pointer subtraction and pointer addition */
#define tl_psub(p1, p2, dsize)                  \
     (((uint8_t *)(p1) - (uint8_t *)(p2) / (dsize))
#define tl_padd(p, offset, dsize)               \
     ((uint8_t *)(p) + (offset) * (dsize))

/* array element assignment */
#define tl_passign(pd, offd, ps, offs, dsize)   \
     memmove(tl_padd((pd), (offd), (dsize)),    \
              tl_padd((ps), (offs), (dsize)), (dsize))

#ifdef __cplusplus
extern "C" {
#endif

void *tl_alloc(size_t size);
void *tl_clone(const void *src, size_t size);
void *tl_repeat(void *data, size_t size, int times);
int tl_compute_length(int ndim, int *dims);
size_t tl_size_of(tl_dtype dtype);
char *tl_fmt(tl_dtype dtype);

#define tl_pointer_sub(p1, p2, dtype)           \
     tl_psub((p1), (p2), tl_size_of(dtype))
#define tl_pointer_add(p, offset, dtype)        \
     tl_padd((p), (offset), tl_size_of(dtype))
#define tl_pointer_assign(pd, offd, ps, offs, dtype)            \
     tl_passign((pd), (offd), (ps), (offs), tl_size_of(dtype))

int tl_pointer_cmp(void *p1, void *p2, tl_dtype dtype);
void tl_err_msg(const char *fmt, ...);
void tl_err_cont(int error, const char *fmt, ...);
void tl_err_ret(const char *fmt, ...);
void tl_err_quit(const char *fmt, ...);
void tl_err_bt(const char *fmt, ...);
void tl_err_exit(int error, const char *fmt, ...);
void tl_err_sys(const char *fmt, ...);
void tl_err_dump(const char *fmt, ...);

#ifdef __cplusplus
}
#endif

#endif	/* _TL_UTIL_H_ */
