#ifndef _TL_UTIL_H_
#define _TL_UTIL_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

typedef enum tl_bool_t tl_bool_t;
enum tl_bool_t {
	TL_FALSE = 0,
	TL_TRUE = 1
};

typedef enum tl_dtype tl_dtype;
enum tl_dtype {
	TL_BOOL,
	TL_FLOAT,
	TL_INT32,
	TL_INT16,
	TL_INT8,
	TL_UINT32,
	TL_UINT16,
	TL_UINT8
};

typedef int (*tl_cmp_func)(void *, void *);
typedef void (* tl_fprint_func) (FILE *fp, void *data);

#define TL_MAXLINE 4096

#define tl_free free

#ifdef __cplusplus
extern "C" {
#endif

void *tl_alloc(size_t size);
void *tl_clone(const void *src, size_t size);
void *tl_repeat(void *data, size_t size, int times);
int tl_compute_length(int ndim, int *dims);
size_t tl_size_of(tl_dtype dtype);
int tl_pointer_sub(void *p1, void *p2, tl_dtype dtype);
void *tl_pointer_add(void *p, int offset, tl_dtype dtype);
void tl_err_msg(const char *fmt, ...);
void tl_err_cont(int error, const char *fmt, ...);
void tl_err_ret(const char *fmt, ...);
void tl_err_quit(const char *fmt, ...);
void tl_err_exit(int error, const char *fmt, ...);
void tl_err_sys(const char *fmt, ...);
void tl_err_dump(const char *fmt, ...);

#ifdef __cplusplus
}
#endif

#endif	/* _TL_UTIL_H_ */
