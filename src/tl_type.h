/*
 * Copyright (c) 2018 Zhao Zhixu
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef _TL_TYPE_H_
#define _TL_TYPE_H_

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <stddef.h>
#include "tl_util.h"

enum tl_bool_t {
    TL_FALSE = 0,
    TL_TRUE = 1
};
typedef enum tl_bool_t tl_bool_t;

/* keep the size and the enum order in sync with tl_type.c */
enum tl_dtype {
    TL_DTYPE_INVALID = -1,
    TL_DOUBLE = 0,
    TL_FLOAT,
    TL_INT32,
    TL_INT16,
    TL_INT8,
    TL_UINT32,
    TL_UINT16,
    TL_UINT8,
    TL_BOOL,
    TL_DTYPE_SIZE
};
typedef enum tl_dtype tl_dtype;

/* keep the size and the enum order in sync with tl_type.c */
enum tl_elew_op {
    TL_ELEW_OP_INVALID = -1,
    TL_MUL = 0,
    TL_DIV,
    TL_SUM,
    TL_SUB,
    TL_MAX,
    TL_MIN,
    TL_POW,
    TL_ELEW_OP_SIZE
};
typedef enum tl_elew_op tl_elew_op;

/* keep the size and the enum order in sync with tl_type.c */
enum tl_resize_type {
    TL_RESIZE_TYPE_INVALID = -1,
    TL_NEAREST = 0,
    TL_LINEAR,
    TL_RESIZE_TYPE_SIZE
};
typedef enum tl_resize_type tl_resize_type;

enum tl_sort_dir {
    TL_SORT_DIR_INVALID = -1,
    TL_SORT_DIR_ASCENDING = 0,
    TL_SORT_DIR_DESCENDING,
    TL_SORT_DIR_SIZE
};
typedef enum tl_sort_dir tl_sort_dir;

#define tl_check_resize_type(rtype)                     \
    assert(rtype >= 0 && rtype < TL_RESIZE_TYPE_SIZE)

typedef int (*tl_fprintf_func) (FILE *fp, const char *fmt, void *p);
typedef int (*tl_cmp_func) (void *p1, void *p2);
typedef void (*tl_elew_func) (void *p1, void *p2, void *r, tl_elew_op elew_op);

#define tl_check_dtype(dtype)                   \
    assert(dtype >= 0 && dtype < TL_DTYPE_SIZE)

#define tl_check_elew_op(op)                    \
    assert(op >= 0 && op < TL_ELEW_OP_SIZE)

#define tl_check_sort_dir(dir)                  \
    assert(dir >= 0 && dir < TL_SORT_DIR_SIZE)

#ifdef __cplusplus
TL_CPPSTART
#endif

/* pointer subtraction and pointer addition */
static inline ptrdiff_t tl_psub(void *p1, void *p2, size_t dsize)
{
    return (((uint8_t *)(p1) - (uint8_t *)(p2)) / ((ptrdiff_t)dsize));
}

static inline void *tl_padd(void *p, ptrdiff_t offset, size_t dsize)
{
    return ((uint8_t *)(p) + (offset) * (dsize));
}

/* array element assignment */
static inline void tl_passign(void *pd, ptrdiff_t offd,
                              void *ps, ptrdiff_t offs, size_t dsize)
{
    memmove(tl_padd((pd), (offd), (dsize)),
            tl_padd((ps), (offs), (dsize)), (dsize));

}

static inline void tl_pmove(void *pd, ptrdiff_t offd,
                            void *ps, ptrdiff_t offs, size_t dsize, size_t n)
{
    memmove(tl_padd((pd), (offd), (dsize)),
            tl_padd((ps), (offs), (dsize)), (dsize)*(n));
}

size_t tl_size_of(tl_dtype dtype);
const char *tl_dtype_fmt(tl_dtype dtype);
const char *tl_dtype_name(tl_dtype dtype);
tl_dtype tl_dtype_from_str(const char *str);
void tl_dtype_max(tl_dtype dtype, void *ret);
void tl_dtype_min(tl_dtype dtype, void *ret);
double tl_dtype_max_double(tl_dtype dtype);
double tl_dtype_min_double(tl_dtype dtype);
void tl_lrelu(void *pd, const void *ps, float negslope, tl_dtype dtype);
void tl_convert(void *pd, tl_dtype dtype_d, const void *ps, tl_dtype dtype_s);

int tl_fprintf(FILE *fp, const char *fmt,void *p, tl_dtype dtype);
tl_fprintf_func tl_fprintf_getfunc(tl_dtype dtype);
int tl_cmp(void *p1, void *p2, tl_dtype dtype);
tl_cmp_func tl_cmp_getfunc(tl_dtype dtype);
tl_elew_op tl_elew_op_from_str(char *str);
const char *tl_elew_op_name(tl_elew_op op);
void tl_elew(void *p1, void *p2, void *res, tl_elew_op elew_op, tl_dtype dtype);
tl_elew_func tl_elew_getfunc(tl_dtype dtype);

const char *tl_resize_type_name(tl_resize_type rtype);
tl_resize_type tl_resize_type_from_str(const char *str);

const char *tl_sort_dir_name(tl_sort_dir dir);
tl_sort_dir tl_sort_dir_from_str(const char *str);

static inline ptrdiff_t tl_pointer_sub(void *p1, void *p2, tl_dtype dtype)
{
    return tl_psub((p1), (p2), tl_size_of(dtype));
}

static inline void *tl_pointer_add(void *p, ptrdiff_t offset, tl_dtype dtype)
{
    return tl_padd((p), (offset), tl_size_of(dtype));
}

static inline void tl_pointer_assign(void *pd, ptrdiff_t offd,
                                     void *ps, ptrdiff_t offs, tl_dtype dtype)
{
    tl_passign((pd), (offd), (ps), (offs), tl_size_of(dtype));
}

#ifdef __cplusplus
TL_CPPEND
#endif

#ifdef TL_CUDA
#ifdef __cplusplus
TL_CPPSTART
#endif

static inline void tl_passign_h2d(void *pd, ptrdiff_t offd,
                                  void *ps, ptrdiff_t offs, size_t dsize)
{
    tl_memcpy_h2d(tl_padd((pd), (offd), (dsize)),
                  tl_padd((ps), (offs), (dsize)), (dsize));
}

static inline void tl_passign_d2h(void *pd, ptrdiff_t offd,
                                  void *ps, ptrdiff_t offs, size_t dsize)
{
    tl_memcpy_d2h(tl_padd((pd), (offd), (dsize)),
                  tl_padd((ps), (offs), (dsize)), (dsize));
}

static inline void tl_passign_d2d(void *pd, ptrdiff_t offd,
                                  void *ps, ptrdiff_t offs, size_t dsize)
{
    tl_memcpy_d2d(tl_padd((pd), (offd), (dsize)),
                  tl_padd((ps), (offs), (dsize)), (dsize));
}

static inline void tl_pointer_assign_h2d(void *pd, ptrdiff_t offd,
                                         void *ps, ptrdiff_t offs, tl_dtype dtype)
{
    tl_passign_h2d((pd), (offd), (ps), (offs), tl_size_of(dtype));
}

static inline void tl_pointer_assign_d2h(void *pd, ptrdiff_t offd,
                                         void *ps, ptrdiff_t offs, tl_dtype dtype)
{
    tl_passign_d2h((pd), (offd), (ps), (offs), tl_size_of(dtype));
}

static inline void tl_pointer_assign_d2d(void *pd, ptrdiff_t offd,
                                         void *ps, ptrdiff_t offs, tl_dtype dtype)
{
    tl_passign_d2d((pd), (offd), (ps), (offs), tl_size_of(dtype));
}

int tl_fprintf_cuda(FILE *fp, const char *fmt,void *p, tl_dtype dtype);

#ifdef __cplusplus
TL_CPPEND
#endif

#endif  /* TL_CUDA */

#endif  /* _TL_TYPE_H_ */
