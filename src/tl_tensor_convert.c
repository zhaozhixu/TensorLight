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

TL_EXPORT tl_tensor *tl_tensor_convert(const tl_tensor *src, tl_tensor *dst, tl_dtype dtype_d)
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
        tl_convert(tl_padd(d_data, di, dsize_d), dtype_d, tl_padd(s_data, di, dsize_s), dtype_s);

    return dst;
}
