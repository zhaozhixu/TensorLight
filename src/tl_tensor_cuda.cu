/*
 * Copyright (c) 2018 Zhao Zhixu
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>

#include "tl_tensor.h"
#include "tl_util.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

static inline __device__ int get_index(int *ids, int ndim, int *dims)
{
     int i, id;
     for (i = 0, id = ids[0]; i < ndim-1; i++)
          id = dims[i+1] * id + ids[i+1];
     return id;
}

static inline __device__ void get_indexes(int id, int *ids, int ndim, int *dims)
{
     for (int i = ndim-1; i >=0; i--) {
          ids[i] = id % dims[i];
          id = id / dims[i];
     }
}

static inline __device__ int compute_length(int ndim, const int *dims)
{
     int i, len;

     assert(ndim > 0);
     assert(dims);
     for (i = 0, len = 1; i < ndim; i++) {
          assert(dims[i] > 0);
          len *= dims[i];
     }
     return len;
}

static inline __device__ void check_dim(int ndim, const int *dims)
{
     int i;

     assert(ndim > 0);
     assert(dims);
     for (i = 0; i < ndim; i++)
          assert(dims[i] > 0);
}

static inline void check_tensor(const tl_tensor *t)
{
     assert(t);
     assert(t->data);
     assert(is_device_mem(t->data));
     assert(t->dtype >= 0 && t->dtype < TL_DTYPE_SIZE);
     assert(t->len == compute_length(t->ndim, t->dims));
}
