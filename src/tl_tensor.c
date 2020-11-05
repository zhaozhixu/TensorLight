/*
 * Copyright (c) 2018-2020 Zhixu Zhao
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

#include "tl_tensor_internal.h"

TL_EXPORT int tl_tensor_index(const tl_tensor *t, int *coords)
{
    assert(t);
    assert(coords);
#ifdef NDEBUG
    for (int i = 0; i < t->ndim; i++)
        assert(coords[i] >= 0 && coords[i] < t->dims[i]);
#endif
    return tl_get_index(coords, t->ndim, t->dims);
}

TL_EXPORT void tl_tensor_coords(const tl_tensor *t, int index, int *coords)
{
    assert(t);
    assert(index >= 0 && index < t->len);
    assert(coords);
    tl_get_coords(index, coords, t->ndim, t->dims);
}

TL_EXPORT int tl_tensor_issameshape(const tl_tensor *t1, const tl_tensor *t2)
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

TL_EXPORT tl_tensor *tl_tensor_create(void *data, int ndim, const int *dims, tl_dtype dtype)
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

TL_EXPORT void tl_tensor_free(tl_tensor *t)
{
    if (!t)
        return;
    tl_free(t->dims);
    tl_free(t);
}

TL_EXPORT void tl_tensor_free_data_too(tl_tensor *t)
{
    if (!t)
        return;
    tl_free(t->data);
    tl_tensor_free(t);
}

TL_EXPORT tl_tensor *tl_tensor_zeros(int ndim, const int *dims, tl_dtype dtype)
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

TL_EXPORT size_t tl_tensor_size(tl_tensor *t)
{
    return t->len * tl_size_of(t->dtype);
}

TL_EXPORT tl_tensor *tl_tensor_clone(const tl_tensor *src)
{
    void *data;
    tl_tensor *dst;

    assert(src);
    data = tl_clone(src->data, src->len * tl_size_of(src->dtype));
    dst = tl_tensor_create(data, src->ndim, src->dims, src->dtype);
    dst->owner = dst;
    return dst;
}

TL_EXPORT tl_tensor *tl_tensor_repeat(const tl_tensor *src, int times)
{
    void *data;
    int *dims;
    tl_tensor *dst;

    assert(src);
    data = tl_repeat(src->data, src->len * tl_size_of(src->dtype), times);
    dims = (int *)tl_alloc(sizeof(int) * (src->ndim + 1));
    memmove(dims + 1, src->dims, sizeof(int) * (src->ndim));
    dims[0] = times;
    dst = tl_tensor_create(data, src->ndim + 1, dims, src->dtype);
    dst->owner = dst;
    tl_free(dims);
    return dst;
}

TL_EXPORT tl_tensor *tl_tensor_arange(double start, double stop, double step, tl_dtype dtype)
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
    assert(stop > start); /* TODO: expand to all possibilities */
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

TL_EXPORT void tl_tensor_rearange(tl_tensor *src, double start, double stop, double step)
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
    assert(stop > start); /* TODO: expand to all possibilities */
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

TL_EXPORT void tl_tensor_fprint(FILE *stream, const tl_tensor *t, const char *fmt)
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
    dim_sizes[ndim - 1] = dims[ndim - 1];
    dim_levels[ndim - 1] = 0;
    left_buf = (char *)tl_alloc(sizeof(char) * (ndim + 1));
    right_buf = (char *)tl_alloc(sizeof(char) * (ndim + 1));
    lp = left_buf;
    rp = right_buf;

    for (i = ndim - 2; i >= 0; i--) {
        dim_sizes[i] = dims[i] * dim_sizes[i + 1];
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
                if (j != 0 && dim_levels[j] > dim_levels[j - 1]) {
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
            for (k = ndim - right_len; k > 0; k--)
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

TL_EXPORT void tl_tensor_print(const tl_tensor *t, const char *fmt)
{
    tl_tensor_fprint(stdout, t, fmt);
}

TL_EXPORT int tl_tensor_save(const char *file_name, const tl_tensor *t, const char *fmt)
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
