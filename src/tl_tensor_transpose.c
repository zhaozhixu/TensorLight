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

TL_EXPORT tl_tensor *tl_tensor_transpose(const tl_tensor *src, tl_tensor *dst, const int *axes)
{
    int i;

#ifndef NDEBUG
    int tmp[TL_MAXDIM] = { 0 };
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
            assert(src->dims[axes[i]] == dst->dims[i]);
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
        tl_get_coords(di, d_ids, ndim, dst->dims);
        for (i = 0; i < ndim; i++)
            s_ids[axes[i]] = d_ids[i];
        si = tl_get_index(s_ids, ndim, src->dims);

        tl_passign(dst->data, di, src->data, si, dsize);
    }

    return dst;
}
