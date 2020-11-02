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

/* reshape tensor without copy */
TL_EXPORT tl_tensor *tl_tensor_reshape(tl_tensor *src, int ndim, const int *dims)
{
    tl_tensor *dst;

    assert(src);
    assert(src->len == tl_compute_length(ndim, dims));
    dst = tl_tensor_create(src->data, ndim, dims, src->dtype);
    dst->owner = src;
    return dst;
}

TL_EXPORT void tl_tensor_reshape_src(tl_tensor *src, int ndim, const int *dims)
{
    assert(src);
    assert(src->len == tl_compute_length(ndim, dims));
    src->ndim = ndim;
    tl_free(src->dims);
    src->dims = tl_clone(dims, sizeof(int) * ndim);
}

/* tl_tensor *tl_tensor_vreshape(const tl_tensor *src, int ndim, ...) */
/* { */
/*      tl_tensor *dst; */
/*      int *dims; */
/*      va_list ap; */
/*      int i; */

/*      assert(src && src->data); */
/*      assert(ndim > 0); */
/*      dims = (int *)tl_alloc(sizeof(int) * ndim); */
/*      va_start(ap, ndim); */
/*      for (i = 0; i < ndim; i++) { */
/*           dims[i] = va_arg(ap, int); */
/*           assert(dims[i] > 0); */
/*      } */
/*      va_end(ap); */
/*      assert(src->len == tl_compute_length(ndim, dims)); */
/*      dst = tl_tensor_create(src->data, ndim, dims, src->dtype); */
/*      tl_free(dims); */
/*      return dst; */
/* } */
