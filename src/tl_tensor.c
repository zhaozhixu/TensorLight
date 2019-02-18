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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>
#include <math.h>

#include "tl_tensor.h"
#include "tl_util.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

static inline int get_index(const int *ids, int ndim, const int *dims)
{
    int i, id;
    for (i = 0, id = ids[0]; i < ndim-1; i++)
        id = dims[i+1] * id + ids[i+1];
    return id;
}

static inline void get_coords(int id, int *ids, int ndim, const int *dims)
{
    for (int i = ndim-1; i >= 0; i--) {
        ids[i] = id % dims[i];
        id /= dims[i];
    }
}

static inline void get_strides(tl_tensor *t, int *strides)
{
    int i;

    assert(strides);
    strides[t->ndim-1] = 1;
    if (t->ndim == 1)
        return;
    for (i = t->dims[t->ndim-2]; i >= 0; i--)
        strides[i] = strides[i+1] * t->dims[i+1];
}

static inline void check_dim(int ndim, const int *dims)
{
    int i;

    assert(ndim > 0);
    assert(dims);
    for (i = 0; i < ndim; i++)
        assert(dims[i] > 0);
}

static inline void check_tensor(const tl_tensor *t)
{
    assert(t && t->data);
    assert(t->dtype >= 0 && t->dtype < TL_DTYPE_SIZE);
    assert(t->len == tl_compute_length(t->ndim, t->dims));
}

int tl_tensor_index(const tl_tensor *t, int *coords)
{
    assert(t);
    assert(coords);
#ifdef NDEBUG
    for (int i = 0; i < t->ndim; i++)
        assert(coords[i] >= 0 && coords[i] < t->dims[i]);
#endif
    return get_index(coords, t->ndim, t->dims);
}

void tl_tensor_coords(const tl_tensor *t, int index, int *coords)
{
    assert(t);
    assert(index >= 0 && index < t->len);
    assert(coords);
    get_coords(index, coords, t->ndim, t->dims);
}

int tl_tensor_issameshape(const tl_tensor *t1, const tl_tensor *t2)
{
    int ndim;

    assert(t1 && t2);
    if (t1->ndim == t2->ndim) {
        ndim = t1->ndim;
        while (--ndim >= 0)
            if (t1->dims[ndim] != t2->dims[ndim])
                return 0;
        return 1;
    }
    return 0;
}

tl_tensor *tl_tensor_create(void *data, int ndim, const int *dims,
                            tl_dtype dtype)
{
    tl_tensor *t;

    assert(ndim > 0 && ndim <= TL_MAXDIM);
    for (int i = 0; i < ndim; i++)
        assert(dims[i] > 0);
    tl_check_dtype(dtype);

    t = (tl_tensor *)tl_alloc(sizeof(tl_tensor));
    t->len = tl_compute_length(ndim, dims);
    t->ndim = ndim;
    t->dims = (int *)tl_clone(dims, sizeof(int) * ndim);
    t->dtype = dtype;
    t->backend_data = NULL;
    t->data = data;
    t->owner = NULL;

    return t;
}

void tl_tensor_free(tl_tensor *t)
{
    if (!t)
        return;
    tl_free(t->dims);
    tl_free(t);
}

void tl_tensor_free_data_too(tl_tensor *t)
{
    if (!t)
        return;
    tl_free(t->data);
    tl_tensor_free(t);
}

tl_tensor *tl_tensor_zeros(int ndim, const int *dims, tl_dtype dtype)
{
    tl_tensor *t;
    size_t size;

    t = tl_tensor_create(NULL, ndim, dims, dtype);
    t->owner = t;
    size = t->len * tl_size_of(dtype);
    t->data = tl_alloc(size);
    memset(t->data, 0, size);
    return t;
}

size_t tl_tensor_size(tl_tensor *t)
{
    return t->len * tl_size_of(t->dtype);
}

tl_tensor *tl_tensor_clone(const tl_tensor *src)
{
    void *data;
    tl_tensor *dst;

    assert(src);
    data = tl_clone(src->data, src->len*tl_size_of(src->dtype));
    dst = tl_tensor_create(data, src->ndim, src->dims, src->dtype);
    dst->owner = dst;
    return dst;
}

tl_tensor *tl_tensor_repeat(const tl_tensor *src, int times)
{
    void *data;
    int *dims;
    tl_tensor *dst;

    assert(src);
    data = tl_repeat(src->data, src->len*tl_size_of(src->dtype), times);
    dims = (int *)tl_alloc(sizeof(int)*(src->ndim+1));
    memmove(dims+1, src->dims, sizeof(int)*(src->ndim));
    dims[0] = times;
    dst = tl_tensor_create(data, src->ndim+1, dims, src->dtype);
    dst->owner = dst;
    tl_free(dims);
    return dst;
}

tl_tensor *tl_tensor_arange(double start, double stop, double step,
                            tl_dtype dtype)
{
    int dims[1];
    tl_tensor *dst;
    double len, elem;
    size_t dsize;

    dsize = tl_size_of(dtype);
#ifdef TL_DEBUG
    double max_d, min_d;
    max_d = tl_dtype_max_double(dtype);
    min_d = tl_dtype_min_double(dtype);
    assert(start >= min_d && start <= max_d);
    assert(stop >= min_d && stop <= max_d);
    assert(step >= min_d && step <= max_d);
    assert(step != 0);
    assert(stop > start);      /* TODO: expand to all possibilities */
#endif

    len = ceil((stop - start) / step);
    if (len > INT32_MAX)
        return NULL;

    dims[0] = (int)len;
    dst = tl_tensor_zeros(1, dims, dtype);
    for (int i = 0; i < dims[0]; i++) {
        elem = start + step * i;
        tl_convert(tl_padd(dst->data, i, dsize), dtype, &elem, TL_DOUBLE);
    }

    return dst;
}

void tl_tensor_rearange(tl_tensor *src, double start, double stop,
                        double step)
{
    double len, elem;
    size_t dsize;

#ifdef TL_DEBUG
    double max_d, min_d;
    max_d = tl_dtype_max_double(src->dtype);
    min_d = tl_dtype_min_double(src->dtype);
    assert(start >= min_d && start <= max_d);
    assert(stop >= min_d && stop <= max_d);
    assert(step >= min_d && step <= max_d);
    assert(step != 0);
    assert(stop > start);      /* TODO: expand to all possibilities */
#endif

    len = ceil((stop - start) / step);
    dsize = tl_size_of(src->dtype);

    assert(len <= INT32_MAX);
    assert(src->ndim == 1);
    assert(src->len == (int)len);
    assert(src->data);

    for (int i = 0; i < src->len; i++) {
        elem = start + step * i;
        tl_convert(tl_padd(src->data, i, dsize), src->dtype, &elem, TL_DOUBLE);
    }
}

void tl_tensor_fprint(FILE *stream, const tl_tensor *t, const char *fmt)
{
    int ndim, len, *dims; /* pointer short cut */
    void *data;
    tl_dtype dtype;
    size_t dsize;

    /* dimision size and how deep current chars go */
    int *dim_sizes, *dim_levels;
    /* buffer for brackets */
    char *left_buf, *right_buf;
    char *lp, *rp;
    size_t right_len;
    int i, j, k;

    assert(stream && t);
    ndim = t->ndim;
    len = t->len;
    dims = t->dims;
    data = t->data;
    dtype = t->dtype;
    dsize = tl_size_of(dtype);

    dim_sizes = (int *)tl_alloc(sizeof(int) * ndim);
    dim_levels = (int *)tl_alloc(sizeof(int) * ndim);
    dim_sizes[ndim-1] = dims[ndim-1];
    dim_levels[ndim-1] = 0;
    left_buf = (char *)tl_alloc(sizeof(char) * (ndim+1));
    right_buf = (char *)tl_alloc(sizeof(char) * (ndim+1));
    lp = left_buf;
    rp = right_buf;

    for (i = ndim-2; i >= 0; i--) {
        dim_sizes[i] = dims[i] * dim_sizes[i+1];
        dim_levels[i] = 0;
    }
    for (i = 0; i < len; i++) {
        for (j = 0; j < ndim; j++) {
            if (i % dim_sizes[j] == 0)
                dim_levels[j]++;
            if (dim_levels[j] == 1) {
                *lp++ = '[';
                dim_levels[j]++;
            }
            if (dim_levels[j] == 3) {
                *rp++ = ']';
                if (j != 0 && dim_levels[j] > dim_levels[j-1]) {
                    *lp++ = '[';
                    dim_levels[j] = 2;
                } else
                    dim_levels[j] = 0;
            }
        }
        *lp = *rp = '\0';
        fprintf(stream, "%s", right_buf);
        if (*right_buf != '\0') {
            fprintf(stream, "\n");
            right_len = strlen(right_buf);
            for (k = ndim-right_len; k > 0; k--)
                fprintf(stream, " ");
        }
        fprintf(stream, "%s", left_buf);
        if (*left_buf == '\0')
            fprintf(stream, " ");
        tl_fprintf(stream, fmt, tl_padd(data, i, dsize), dtype);
        lp = left_buf, rp = right_buf;
    }
    for (j = 0; j < ndim; j++)
        fprintf(stream, "]");
    fprintf(stream, "\n");

    tl_free(dim_sizes);
    tl_free(dim_levels);
    tl_free(left_buf);
    tl_free(right_buf);
}

void tl_tensor_print(const tl_tensor *t, const char *fmt)
{
    tl_tensor_fprint(stdout, t, fmt);
}

int tl_tensor_save(const char *file_name, const tl_tensor *t, const char *fmt)
{
    FILE *fp;

    fp = fopen(file_name, "w");
    if (!fp) {
        tl_warn_ret("ERROR: cannot open %s", file_name);
        return -1;
    }
    tl_tensor_fprint(fp, t, fmt);
    fclose(fp);
    return 0;
}

tl_tensor *tl_tensor_create_slice(void *data, const tl_tensor *src, int axis,
                                  int len, tl_dtype dtype)
{
    tl_tensor *dst;
    int *dims;

    assert(src);
    assert(axis < src->ndim && axis >= 0);
    assert(len <= src->dims[axis] && len > 0);

    dims = (int *)tl_clone(src->dims, sizeof(int) * src->ndim);
    dims[axis] = len;
    dst = tl_tensor_create(data, src->ndim, dims, dtype);
    tl_free(dims);

    return dst;
}

tl_tensor *tl_tensor_zeros_slice(const tl_tensor *src, int axis, int len,
                                 tl_dtype dtype)
{
    tl_tensor *dst;
    int *dims;

    assert(src);
    assert(axis < src->ndim && axis >= 0);
    assert(len <= src->dims[axis] && len > 0);

    dims = (int *)tl_clone(src->dims, sizeof(int) * src->ndim);
    dims[axis] = len;
    dst = tl_tensor_zeros(src->ndim, dims, dtype);
    tl_free(dims);

    return dst;
}

tl_tensor *tl_tensor_slice(const tl_tensor *src, tl_tensor *dst, int axis,
                           int start, int len)
{
    int i;
    int d_vol, s_vol, vol;
    int thread_num;
    int si, di;
    size_t dsize;

    assert(src && src->data);
    assert(axis < src->ndim && axis >= 0);
    assert(len <= src->dims[axis] && len > 0);
    assert(start < src->dims[axis] && start >= 0);
    assert(len + start <= src->dims[axis]);
    if (dst) {
#ifndef NDEBUG
        assert(dst->data);
        assert(src->dtype == dst->dtype);
        assert(dst->ndim == src->ndim);
        for (i = 0; i < src->ndim; i++)
            assert(i == axis ? dst->dims[i] == len :
                   dst->dims[i] == src->dims[i]);
#endif
    } else {
        dst = tl_tensor_zeros_slice(src, axis, len, src->dtype);
    }

    for (i = axis+1, vol = 1; i < dst->ndim; i++)
        vol *= dst->dims[i];
    d_vol = vol * dst->dims[axis];
    s_vol = vol * src->dims[axis];
    thread_num = dst->len;

    dsize = tl_size_of(src->dtype);
    for (di = 0; di < thread_num; di++) {
        si = di / d_vol * s_vol + di % d_vol + start * vol;
        tl_passign(dst->data, di, src->data, si, dsize);
    }

    return dst;
}

tl_tensor *tl_tensor_slice_nocopy(tl_tensor *src, tl_tensor *dst,
                                  int axis, int start, int len)
{
    int i, volumn;

    assert(src && src->data);
    assert(axis == 0);
    assert(len <= src->dims[axis] && len > 0);
    assert(start < src->dims[axis] && start >= 0);
    assert(len + start <= src->dims[axis]);
    if (dst) {
#ifndef NDEBUG
        assert(src->dtype == dst->dtype);
        assert(dst->ndim == src->ndim);
        for (i = 0; i < src->ndim; i++)
            assert(i == axis ? dst->dims[i] == len :
                   dst->dims[i] == src->dims[i]);
#endif
    } else {
        dst = tl_tensor_create_slice(NULL, src, axis, len, src->dtype);
    }

    dst->owner = src;
    for (i = axis+1, volumn = 1; i < dst->ndim; i++)
        volumn *= dst->dims[i];
    dst->data = tl_padd(src->data, start*volumn, tl_size_of(src->dtype));

    return dst;
}

tl_tensor *tl_tensor_concat(const tl_tensor *src1, const tl_tensor *src2,
                            tl_tensor *dst, int axis)
{
    int i;
    int s1_nvol, s2_nvol, vol;
    int di, s1i, s2i;
    int thread_num;
    int *dims;
    size_t dsize;

    assert(src1 && src1->data);
    assert(src2 && src2->data);
    assert(src1->dtype == src2->dtype);
    assert(src1->ndim == src2->ndim);
    assert(axis >= 0 && axis < src1->ndim);
    for (i = 0; i < src1->ndim; i++)
        assert(i == axis ? 1 : src1->dims[i] == src2->dims[i]);

    if (dst) {
        assert(dst->data);
        assert(src1->dtype == dst->dtype);
        assert(src1->ndim == dst->ndim);
        assert(dst->dims[axis] == src1->dims[axis] + src2->dims[axis]);
    } else {
        dims = tl_clone(src1->dims, sizeof(int)*src1->ndim);
        dims[axis] = src1->dims[axis] + src2->dims[axis];
        dst = tl_tensor_zeros(src1->ndim, dims, src1->dtype);
        tl_free(dims);
    }

    for (i = axis+1, vol = 1; i < dst->ndim; i++)
        vol *= dst->dims[i];
    s1_nvol = src1->dims[axis];
    s2_nvol = src2->dims[axis];
    thread_num = 1;
    for (i = 0; i <= axis; i++)
        thread_num *= dst->dims[i];

    dsize = tl_size_of(src1->dtype) * vol;
    for (di = 0, s1i = 0, s2i = 0; di < thread_num;) {
        tl_pmove(dst->data, di, src1->data, s1i, dsize, s1_nvol);
        di += s1_nvol;
        s1i += s1_nvol;
        tl_pmove(dst->data, di, src2->data, s2i, dsize, s2_nvol);
        di += s2_nvol;
        s2i += s2_nvol;
    }

    return dst;
}

/* reshape tensor without copy */
tl_tensor *tl_tensor_reshape(tl_tensor *src, int ndim, const int *dims)
{
    tl_tensor *dst;

    assert(src);
    assert(src->len == tl_compute_length(ndim, dims));
    dst = tl_tensor_create(src->data, ndim, dims, src->dtype);
    dst->owner = src;
    return dst;
}

void tl_tensor_reshape_src(tl_tensor *src, int ndim, const int *dims)
{
    assert(src);
    assert(src->len == tl_compute_length(ndim, dims));
    src->ndim = ndim;
    tl_free(src->dims);
    src->dims = tl_clone(dims, sizeof(int)*ndim);
}

/* tl_tensor *tl_tensor_vreshape(const tl_tensor *src, int ndim, ...) */
/* { */
/*      tl_tensor *dst; */
/*      int *dims; */
/*      va_list ap; */
/*      int i; */

/*      assert(src && src->data); */
/*      assert(ndim > 0); */
/*      dims = (int *)tl_alloc(sizeof(int) * ndim); */
/*      va_start(ap, ndim); */
/*      for (i = 0; i < ndim; i++) { */
/*           dims[i] = va_arg(ap, int); */
/*           assert(dims[i] > 0); */
/*      } */
/*      va_end(ap); */
/*      assert(src->len == tl_compute_length(ndim, dims)); */
/*      dst = tl_tensor_create(src->data, ndim, dims, src->dtype); */
/*      tl_free(dims); */
/*      return dst; */
/* } */

tl_tensor *tl_tensor_maxreduce(const tl_tensor *src, tl_tensor *dst,
                               tl_tensor *arg, int axis)
{
    /* suppose the shape of src is [N, C, H, W], dim = 1, then thread_num is N x H x W
       reduce_vol is H x W, batch_vol is C x H x W */
    int thread_num, reduce_vol, batch_vol;
    int i, di, si, maxi;
    int dim_size;
    void *data_s, *data_d, *data_a, *nowp, *maxp;
    size_t dsize;
    tl_dtype dtype;
    tl_cmp_func cmp;

    assert(src && src->data);
    assert(axis < src->ndim && axis >= 0);
    if (dst) {
#ifndef NDEBUG
        assert(dst->data);
        assert(src->dtype == dst->dtype);
        for (i = 0; i < dst->ndim; i++)
            assert(i == axis ? dst->dims[i] == 1 :
                   dst->dims[i] == src->dims[i]);
#endif
    } else {
        dst = tl_tensor_zeros_slice(src, axis, 1, src->dtype);
    }
    if (arg) {
#ifndef NDEBUG
        assert(arg->data);
        assert(arg->dtype == TL_INT32);
        for (i = 0; i < arg->ndim; i++)
            assert(i == axis ? arg->dims[i] == 1 :
                   arg->dims[i] == src->dims[i]);
#endif
    }

    for (i = axis+1, thread_num = 1; i < dst->ndim; i++)
        thread_num *= dst->dims[i];
    reduce_vol = thread_num;
    batch_vol = thread_num * src->dims[axis];
    for (i = 0; i < axis; i++)
        thread_num *= dst->dims[i];

    dtype = src->dtype;
    cmp = tl_cmp_getfunc(dtype);
    dsize = tl_size_of(dtype);
    dim_size = src->dims[axis];
    nowp = tl_alloc(dsize);
    maxp = tl_alloc(dsize);
    data_s = src->data;
    data_d = dst->data;
    if (arg)
        data_a = arg->data;
    for (di = 0; di < thread_num; di++) {
        /* src[si] is the first element in this thread to be compared, then
           si = batch_vol * batch + (di - reduce_vol * batch),
           where batch = di / reduce_vol (in range [0, N-1] in [N, C, H, W]),
           which is the same as the following code: */
        si = (batch_vol - reduce_vol) * (di / reduce_vol) + di;
        tl_passign(nowp, 0, data_s, si, dsize);
        tl_passign(maxp, 0, nowp, 0, dsize);
        for (i = 1, maxi = 0; i < dim_size; i++) {
            tl_passign(nowp, 0, data_s, si+i*reduce_vol, dsize);
            if (cmp(nowp, maxp) > 0) {
                tl_passign(maxp, 0, nowp, 0, dsize);
                maxi = i;
            }
        }
        tl_passign(data_d, di, maxp, 0, dsize);
        if (arg)
            ((int32_t *)data_a)[di] = maxi;
    }
    tl_free(nowp);
    tl_free(maxp);

    return dst;
}

tl_tensor *tl_tensor_elew(const tl_tensor *src1, const tl_tensor *src2,
                          tl_tensor *dst, tl_elew_op elew_op)
{
    int thread_num;
    int di;
    size_t dsize;
    tl_dtype dtype;
    void *s1_data, *s2_data, *d_data;
    void *elew_res;
    tl_elew_func elew;

    assert(tl_tensor_issameshape(src1, src2));
    assert(src1->data && src2->data);
    assert(src1->dtype == src2->dtype);
    if (dst) {
        assert(dst->data);
        assert(tl_tensor_issameshape(src1, dst));
        assert(src1->dtype == dst->dtype);
    } else {
        dst = tl_tensor_zeros(src1->ndim, src2->dims, src1->dtype);
    }

    thread_num = dst->len;
    s1_data = src1->data;
    s2_data = src2->data;
    d_data = dst->data;
    dtype = src1->dtype;
    dsize = tl_size_of(dtype);
    elew = tl_elew_getfunc(dtype);
    elew_res = tl_alloc(dsize);
    for (di = 0; di < thread_num; di++) {
        elew(tl_padd(s1_data, di, dsize),
             tl_padd(s2_data, di, dsize), elew_res, elew_op);
        tl_passign(d_data, di, elew_res, 0, dsize);
    }
    tl_free(elew_res);

    return dst;
}

tl_tensor *tl_tensor_elew_param(const tl_tensor *src, double param,
                                tl_tensor *dst, tl_elew_op elew_op)
{
    int thread_num;
    int di;
    size_t dsize;
    tl_dtype dtype;
    void *s_data, *d_data;
    void *elew_res, *param_data;
    tl_elew_func elew;

    assert(src && src->data);
    if (dst) {
        assert(dst->data);
        assert(tl_tensor_issameshape(src, dst));
        assert(src->dtype == dst->dtype);
    } else {
        dst = tl_tensor_zeros(src->ndim, src->dims, src->dtype);
    }

    thread_num = dst->len;
    s_data = src->data;
    d_data = dst->data;
    dtype = src->dtype;
    dsize = tl_size_of(dtype);
    elew = tl_elew_getfunc(dtype);
    elew_res = tl_alloc(dsize);
    param_data = tl_alloc(dsize);
    tl_convert(param_data, dtype, &param, TL_DOUBLE);
    for (di = 0; di < thread_num; di++) {
        elew(tl_padd(s_data, di, dsize), param_data, elew_res, elew_op);
        tl_passign(d_data, di, elew_res, 0, dsize);
    }
    tl_free(elew_res);
    tl_free(param_data);

    return dst;
}

tl_tensor *tl_tensor_transpose(const tl_tensor *src, tl_tensor *dst,
                               const int *axes)
{
    int i;

#ifndef NDEBUG
    int tmp[TL_MAXDIM] = {0};
    for (i = 0; i < src->ndim; i++)
        tmp[axes[i]] = 1;
    for (i = 0; i < src->ndim; i++)
        assert(tmp[i] && "axes don't match src tensor's shape");
    assert(src && src->data);
#endif
    if (dst) {
#ifndef NDEBUG
        assert(dst->data);
        assert(src->dtype == dst->dtype);
        assert(src->len == dst->len);
        assert(src->ndim == dst->ndim);
        for (i = 0; i < dst->ndim; i++)
            assert(src->dims[axes[i]] = dst->dims[i]);
#endif
    } else {
        int d_dims[TL_MAXDIM];
        for (i = 0; i < src->ndim; i++)
            d_dims[i] = src->dims[axes[i]];
        dst = tl_tensor_zeros(src->ndim, d_dims, src->dtype);
    }

    int di, si;
    int s_ids[TL_MAXDIM], d_ids[TL_MAXDIM];
    size_t dsize = tl_size_of(src->dtype);
    int ndim = dst->ndim;

    for (di = 0; di < dst->len; di++) {
        get_coords(di, d_ids, ndim, dst->dims);
        for (i = 0; i < ndim; i++)
            s_ids[axes[i]] = d_ids[i];
        si = get_index(s_ids, ndim, src->dims);

        tl_passign(dst->data, di, src->data, si, dsize);
    }

    return dst;
}

tl_tensor *tl_tensor_lrelu(const tl_tensor *src, tl_tensor *dst, float negslope)
{
    assert(src && src->data);
    if (dst) {
        assert(dst && dst->data);
        assert(tl_tensor_issameshape(dst, src));
        assert(dst->dtype == src->dtype);
    } else {
        dst = tl_tensor_zeros(src->ndim, src->dims, src->dtype);
    }

    tl_dtype dtype = src->dtype;
    size_t dsize = tl_size_of(dtype);
    for (int i = 0; i < src->len; i++)
        tl_lrelu(tl_padd(dst->data, i, dsize),
                 tl_padd(src->data, i, dsize), negslope, dtype);

    return dst;
}

tl_tensor *tl_tensor_convert(const tl_tensor *src, tl_tensor *dst,
                             tl_dtype dtype_d)
{
    size_t dsize_d, dsize_s;
    void *s_data, *d_data;
    tl_dtype dtype_s;
    int thread_num;
    int di;

    assert(src && src->data);
    if (dst) {
        assert(dst->data);
        assert(tl_tensor_issameshape(src, dst));
        assert(dst->dtype == dtype_d);
    } else {
        dst = tl_tensor_zeros(src->ndim, src->dims, dtype_d);
    }

    dtype_s = src->dtype;
    s_data = src->data;
    d_data = dst->data;
    dsize_d = tl_size_of(dtype_d);
    dsize_s = tl_size_of(dtype_s);
    thread_num = dst->len;
    for (di = 0; di < thread_num; di++)
        tl_convert(tl_padd(d_data, di, dsize_d), dtype_d,
                   tl_padd(s_data, di, dsize_s), dtype_s);

    return dst;
}

static void nearest_resize(const tl_tensor *src, tl_tensor *dst,
                           const int *new_dims)
{
    int src_id, dst_id, i;
    int src_coords[TL_MAXDIM], dst_coords[TL_MAXDIM];
    size_t dsize = tl_size_of(src->dtype);
    float rounded, scales[TL_MAXDIM];

    for (i = 0; i < src->ndim; i++)
        scales[i] = (float)src->dims[i] / (float)new_dims[i];
    for (dst_id = 0; dst_id < dst->len; dst_id++) {
        get_coords(dst_id, dst_coords, src->ndim, new_dims);
        for (i = 0; i < src->ndim; i++) {
            rounded = roundf(((float)dst_coords[i] + 0.5) * scales[i] - 0.5);
            tl_convert(&src_coords[i], TL_INT32, &rounded, TL_FLOAT);
        }
        src_id = get_index(src_coords, src->ndim, src->dims);
        tl_passign(dst->data, dst_id, src->data, src_id, dsize);
    }
}

static void linear_resize(const tl_tensor *src, tl_tensor *dst,
                          const int *new_dims)
{
    assert(0 && "not support TL_LINEAR yet");
}

tl_tensor *tl_tensor_resize(const tl_tensor *src, tl_tensor *dst,
                            const int *new_dims, tl_resize_type rtype)
{
    assert(src && src->data);
    assert(new_dims);
    tl_check_resize_type(rtype);
    if (dst) {
        assert(dst->data);
        assert(dst->dtype == src->dtype);
        assert(dst->ndim == src->ndim);
    } else {
        dst = tl_tensor_zeros(src->ndim, new_dims, src->dtype);
    }

    switch (rtype) {
    case TL_NEAREST:
        nearest_resize(src, dst, new_dims);
        break;
    case TL_LINEAR:
        linear_resize(src, dst, new_dims);
        break;
    default:
        assert(0 && "unsupported tl_resize_type");
        break;
    }
    return dst;
}

/* static void top1(void *src, void *dst, int32_t *arg, int len, int stride, */
/*                  tl_dtype dtype, int largest) */
/* { */
/*     void *tmp; */
/*     void *elem; */
/*     int32_t idx; */
/*     int i; */
/*     tl_cmp_func cmp; */
/*     size_t dsize; */

/*     assert(src); */
/*     assert(dst); */
/*     dsize = tl_size_of(dtype); */
/*     tmp = tl_alloc(dsize); */
/*     cmp = tl_cmp_getfunc(dtype); */

/*     if (largest) { */
/*         tl_dtype_min(dtype, tmp); */
/*         for (i = 0; i < len; i += stride) { */
/*             elem = tl_padd(src, i, dsize); */
/*             if (cmp(tmp, elem) < 0) { */
/*                 tl_passign(tmp, 0, elem, 0, dsize); */
/*                 idx = i; */
/*             } */
/*         } */
/*     } else { */
/*         tl_dtype_max(dtype, tmp); */
/*         for (i = 0; i < len; i += stride) { */
/*             elem = tl_padd(src, i, dsize); */
/*             if (cmp(tmp, elem) > 0) { */
/*                 tl_passign(tmp, 0, elem, 0, dsize); */
/*                 idx = i; */
/*             } */
/*         } */
/*     } */
/*     tl_passign(dst, 0, tmp, 0, dsize); */
/*     tl_free(tmp); */
/*     *arg = idx; */
/* } */

/* tl_tensor *tl_tensor_topk(const tl_tensor *src, tl_tensor *dst, tl_tensor *arg, */
/*                           int axis, int k, int sorted, int largest) */
/* { */
/*     int i; */
/*     int strides[TL_MAXDIM]; */
/*     int cmp_seq; */

/*     assert(src && src->data); */
/*     assert(k > 0 && k <= src->dims[src->ndim]); */
/*     assert(axis < src->ndim && axis >= 0); */

/*     if (dst) { */
/* #ifndef NDEBUG */
/*         assert(dst->data); */
/*         assert(src->dtype == dst->dtype); */
/*         for (i = 0; i < dst->ndim; i++) */
/*             assert(i == axis ? dst->dims[i] == k : */
/*                    dst->dims[i] == src->dims[i]); */
/* #endif */
/*     } else { */
/*         dst = tl_tensor_zeros_slice(src, axis, k, src->dtype); */
/*     } */
/*     if (arg) { */
/* #ifndef NDEBUG */
/*         assert(arg->data); */
/*         assert(arg->dtype == TL_INT32); */
/*         for (i = 0; i < arg->ndim; i++) */
/*             assert(i == axis ? arg->dims[i] == k : */
/*                    arg->dims[i] == src->dims[i]); */
/* #endif */
/*     } */

/*     get_strides(src, strides); */
/*     if (k == 1) { */
/*         for (i = ) */
/*     } */

/* } */
