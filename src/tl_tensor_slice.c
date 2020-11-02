/*
 * Copyright (c) 2018-2020 Zhao Zhixu
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

TL_EXPORT tl_tensor *tl_tensor_create_slice(void *data, const tl_tensor *src, int axis, int len,
                                            tl_dtype dtype)
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

TL_EXPORT tl_tensor *tl_tensor_zeros_slice(const tl_tensor *src, int axis, int len, tl_dtype dtype)
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

TL_EXPORT tl_tensor *tl_tensor_slice(const tl_tensor *src, tl_tensor *dst, int axis, int start,
                                     int len)
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
            assert(i == axis ? dst->dims[i] == len : dst->dims[i] == src->dims[i]);
#endif
    } else {
        dst = tl_tensor_zeros_slice(src, axis, len, src->dtype);
    }

    for (i = axis + 1, vol = 1; i < dst->ndim; i++)
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

TL_EXPORT tl_tensor *tl_tensor_slice_nocopy(tl_tensor *src, tl_tensor *dst, int axis, int start,
                                            int len)
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
            assert(i == axis ? dst->dims[i] == len : dst->dims[i] == src->dims[i]);
#endif
    } else {
        dst = tl_tensor_create_slice(NULL, src, axis, len, src->dtype);
    }

    dst->owner = src;
    for (i = axis + 1, volumn = 1; i < dst->ndim; i++)
        volumn *= dst->dims[i];
    dst->data = tl_padd(src->data, start * volumn, tl_size_of(src->dtype));

    return dst;
}
