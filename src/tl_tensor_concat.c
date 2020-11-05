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

TL_EXPORT tl_tensor *tl_tensor_concat(const tl_tensor *src1, const tl_tensor *src2, tl_tensor *dst,
                                      int axis)
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
        dims = tl_clone(src1->dims, sizeof(int) * src1->ndim);
        dims[axis] = src1->dims[axis] + src2->dims[axis];
        dst = tl_tensor_zeros(src1->ndim, dims, src1->dtype);
        tl_free(dims);
    }

    for (i = axis + 1, vol = 1; i < dst->ndim; i++)
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
