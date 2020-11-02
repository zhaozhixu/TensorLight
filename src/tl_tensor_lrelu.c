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

TL_EXPORT tl_tensor *tl_tensor_lrelu(const tl_tensor *src, tl_tensor *dst, float negslope)
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
        tl_lrelu(tl_padd(dst->data, i, dsize), tl_padd(src->data, i, dsize), negslope, dtype);

    return dst;
}
