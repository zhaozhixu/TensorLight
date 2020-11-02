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

TL_EXPORT tl_tensor *tl_tensor_split(const tl_tensor *src, tl_tensor *dst1, tl_tensor *dst2,
                                     int axis, const int *splits)
{
    assert(src && src->data);
    assert(dst1 && dst1->data);
    assert(dst1 && dst2->data);
    assert(src->dtype == dst1->dtype);
    assert(src->dtype == dst2->dtype);
    assert(axis >= 0 && axis < src->ndim);
    assert(splits[0] + splits[1] == src->dims[axis]);
    assert(splits[0] == dst1->dims[axis]);
    assert(splits[1] == dst2->dims[axis]);
#ifndef NDEBUG
    for (int i = 0; i < src->ndim; i++) {
        if (i != axis) {
            assert(src->dims[i] == dst1->dims[i]);
            assert(src->dims[i] == dst2->dims[i]);
        }
    }
#endif /* NDEBUG */
}
