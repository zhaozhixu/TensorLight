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

static void nearest_resize(const tl_tensor *src, tl_tensor *dst, const int *new_dims)
{
    int src_id, dst_id, i;
    int src_coords[TL_MAXDIM], dst_coords[TL_MAXDIM];
    size_t dsize = tl_size_of(src->dtype);
    float rounded, scales[TL_MAXDIM];

    for (i = 0; i < src->ndim; i++)
        scales[i] = (float)src->dims[i] / (float)new_dims[i];
    for (dst_id = 0; dst_id < dst->len; dst_id++) {
        tl_get_coords(dst_id, dst_coords, src->ndim, new_dims);
        for (i = 0; i < src->ndim; i++) {
            rounded = roundf(((float)dst_coords[i] + 0.5) * scales[i] - 0.5);
            tl_convert(&src_coords[i], TL_INT32, &rounded, TL_FLOAT);
        }
        src_id = tl_get_index(src_coords, src->ndim, src->dims);
        tl_passign(dst->data, dst_id, src->data, src_id, dsize);
    }
}

static void linear_resize(const tl_tensor *src, tl_tensor *dst, const int *new_dims)
{
    assert(0 && "not support TL_LINEAR yet");
}

TL_EXPORT tl_tensor *tl_tensor_resize(const tl_tensor *src, tl_tensor *dst, const int *new_dims,
                                      tl_resize_type rtype)
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
