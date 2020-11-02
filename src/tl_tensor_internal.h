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

#ifndef _TL_TENSOR_INTERNAL_H_
#define _TL_TENSOR_INTERNAL_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>
#include <math.h>

#include "tl_type.h"
#include "tl_tensor.h"
#include "tl_util.h"

static inline int tl_get_index(const int *ids, int ndim, const int *dims)
{
    int i, id;
    for (i = 0, id = ids[0]; i < ndim - 1; i++)
        id = dims[i + 1] * id + ids[i + 1];
    return id;
}

static inline void tl_get_coords(int id, int *ids, int ndim, const int *dims)
{
    for (int i = ndim - 1; i >= 0; i--) {
        ids[i] = id % dims[i];
        id /= dims[i];
    }
}

static inline void tl_get_strides(tl_tensor *t, int *strides)
{
    int i;

    assert(strides);
    strides[t->ndim - 1] = 1;
    if (t->ndim == 1)
        return;
    for (i = t->dims[t->ndim - 2]; i >= 0; i--)
        strides[i] = strides[i + 1] * t->dims[i + 1];
}

static inline void tl_check_dim(int ndim, const int *dims)
{
    int i;

    assert(ndim > 0);
    assert(dims);
    for (i = 0; i < ndim; i++)
        assert(dims[i] > 0);
}

static inline void tl_check_tensor(const tl_tensor *t)
{
    assert(t && t->data);
    assert(t->dtype >= 0 && t->dtype < TL_DTYPE_SIZE);
    assert(t->len == tl_compute_length(t->ndim, t->dims));
}

#endif /* _TL_TENSOR_INTERNAL_H_ */
