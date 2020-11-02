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

TL_EXPORT tl_tensor *tl_tensor_maxreduce(const tl_tensor *src, tl_tensor *dst, tl_tensor *arg,
                                         int axis)
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
            assert(i == axis ? dst->dims[i] == 1 : dst->dims[i] == src->dims[i]);
#endif
    } else {
        dst = tl_tensor_zeros_slice(src, axis, 1, src->dtype);
    }
    if (arg) {
#ifndef NDEBUG
        assert(arg->data);
        assert(arg->dtype == TL_INT32);
        for (i = 0; i < arg->ndim; i++)
            assert(i == axis ? arg->dims[i] == 1 : arg->dims[i] == src->dims[i]);
#endif
    }

    for (i = axis + 1, thread_num = 1; i < dst->ndim; i++)
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
            tl_passign(nowp, 0, data_s, si + i * reduce_vol, dsize);
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
