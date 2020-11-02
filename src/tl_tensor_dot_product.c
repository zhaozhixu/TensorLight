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

TL_EXPORT tl_tensor *tl_tensor_dot_product(const tl_tensor *src1, const tl_tensor *src2,
                                           tl_tensor *dst)
{
    int di;
    size_t dsize;
    tl_dtype dtype;
    void *s1_data, *s2_data, *d_data;
    char elew_prod[TL_DTYPE_MAX_SIZE];
    tl_elew_func elew;

    assert(tl_tensor_issameshape(src1, src2));
    assert(src1->data && src2->data);
    assert(src1->dtype == src2->dtype);
    if (dst) {
        assert(dst->data);
        assert(dst->ndim == 1);
        assert(dst->dims[0] == 1);
        assert(src1->dtype == dst->dtype);
    } else {
        dst = tl_tensor_zeros(1, (int[]){ 1 }, src1->dtype);
    }

    s1_data = src1->data;
    s2_data = src2->data;
    d_data = dst->data;
    dtype = src1->dtype;
    dsize = tl_size_of(dtype);
    elew = tl_elew_getfunc(dtype);
    memset(dst->data, 0, tl_tensor_size(dst));
    for (di = 0; di < src1->len; di++) {
        elew(tl_padd(s1_data, di, dsize), tl_padd(s2_data, di, dsize), elew_prod, TL_MUL);
        elew(elew_prod, d_data, d_data, TL_SUM);
    }

    return dst;
}
