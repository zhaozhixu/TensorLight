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

TL_EXPORT tl_tensor *tl_tensor_elew(const tl_tensor *src1, const tl_tensor *src2, tl_tensor *dst,
                                    tl_elew_op elew_op)
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
        elew(tl_padd(s1_data, di, dsize), tl_padd(s2_data, di, dsize), elew_res, elew_op);
        tl_passign(d_data, di, elew_res, 0, dsize);
    }
    tl_free(elew_res);

    return dst;
}

TL_EXPORT tl_tensor *tl_tensor_elew_param(const tl_tensor *src, double param, tl_tensor *dst,
                                          tl_elew_op elew_op)
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
