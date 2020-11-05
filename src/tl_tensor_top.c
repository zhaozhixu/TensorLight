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

/* static void top1(void *src, void *dst, int32_t *arg, int len, int stride, */
/*                  tl_dtype dtype, int largest) */
/* { */
/*     void *tmp; */
/*     void *elem; */
/*     int32_t idx; */
/*     int i; */
/*     tl_cmp_func cmp; */
/*     size_t dsize; */

/*     assert(src); */
/*     assert(dst); */
/*     dsize = tl_size_of(dtype); */
/*     tmp = tl_alloc(dsize); */
/*     cmp = tl_cmp_getfunc(dtype); */

/*     if (largest) { */
/*         tl_dtype_min(dtype, tmp); */
/*         for (i = 0; i < len; i += stride) { */
/*             elem = tl_padd(src, i, dsize); */
/*             if (cmp(tmp, elem) < 0) { */
/*                 tl_passign(tmp, 0, elem, 0, dsize); */
/*                 idx = i; */
/*             } */
/*         } */
/*     } else { */
/*         tl_dtype_max(dtype, tmp); */
/*         for (i = 0; i < len; i += stride) { */
/*             elem = tl_padd(src, i, dsize); */
/*             if (cmp(tmp, elem) > 0) { */
/*                 tl_passign(tmp, 0, elem, 0, dsize); */
/*                 idx = i; */
/*             } */
/*         } */
/*     } */
/*     tl_passign(dst, 0, tmp, 0, dsize); */
/*     tl_free(tmp); */
/*     *arg = idx; */
/* } */

/* tl_tensor *tl_tensor_topk(const tl_tensor *src, tl_tensor *dst, tl_tensor *arg, */
/*                           int axis, int k, int sorted, int largest) */
/* { */
/*     int i; */
/*     int strides[TL_MAXDIM]; */
/*     int cmp_seq; */

/*     assert(src && src->data); */
/*     assert(k > 0 && k <= src->dims[src->ndim]); */
/*     assert(axis < src->ndim && axis >= 0); */

/*     if (dst) { */
/* #ifndef NDEBUG */
/*         assert(dst->data); */
/*         assert(src->dtype == dst->dtype); */
/*         for (i = 0; i < dst->ndim; i++) */
/*             assert(i == axis ? dst->dims[i] == k : */
/*                    dst->dims[i] == src->dims[i]); */
/* #endif */
/*     } else { */
/*         dst = tl_tensor_zeros_slice(src, axis, k, src->dtype); */
/*     } */
/*     if (arg) { */
/* #ifndef NDEBUG */
/*         assert(arg->data); */
/*         assert(arg->dtype == TL_INT32); */
/*         for (i = 0; i < arg->ndim; i++) */
/*             assert(i == axis ? arg->dims[i] == k : */
/*                    arg->dims[i] == src->dims[i]); */
/* #endif */
/*     } */

/*     tl_get_strides(src, strides); */
/*     if (k == 1) { */
/*         for (i = ) */
/*     } */

/* } */
