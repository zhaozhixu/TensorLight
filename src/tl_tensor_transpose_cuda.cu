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

#include "tl_tensor_internal_cuda.h"

template <typename T>
static __global__ void transpose_kernel(T *src, T *dst, int ndim, int *s_dims, int *d_dims,
                                        int *axes, int block_size, int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;

    int s_ids[TL_MAXDIM], d_ids[TL_MAXDIM];
    tl_get_coords_cu(di, d_ids, ndim, d_dims);
    for (int i = 0; i < ndim; i++)
        s_ids[axes[i]] = d_ids[i];
    int si = tl_get_index_cu(s_ids, ndim, s_dims);

    dst[di] = src[si];
}

TL_EXPORT tl_tensor *tl_tensor_transpose_cuda(const tl_tensor *src, tl_tensor *dst, const int *axes)
{
    int i;

#ifndef NDEBUG
    int tmp[TL_MAXDIM] = { 0 };
    for (i = 0; i < src->ndim; i++)
        tmp[axes[i]] = 1;
    for (i = 0; i < src->ndim; i++)
        assert(tmp[i] && "axes don't match src tensor's shape");
    assert(src && tl_is_device_mem(src->data));
#endif
    if (dst) {
#ifndef NDEBUG
        assert(tl_is_device_mem(dst->data));
        assert(src->dtype == dst->dtype);
        assert(src->len == dst->len);
        assert(src->ndim == dst->ndim);
        for (i = 0; i < dst->ndim; i++)
            assert(src->dims[axes[i]] = dst->dims[i]);
#endif
    } else {
        int d_dims[TL_MAXDIM];
        for (i = 0; i < src->ndim; i++)
            d_dims[i] = src->dims[axes[i]];
        dst = tl_tensor_zeros_cuda(src->ndim, d_dims, src->dtype);
    }

    int *axes_device;
    int *s_dims, *d_dims;
    int thread_num, block_num;

    thread_num = dst->len;
    block_num = BLOCK_NUM(BLOCK_SIZE, thread_num);
    s_dims = (int *)tl_clone_h2d(src->dims, sizeof(int) * src->ndim);
    d_dims = (int *)tl_clone_h2d(dst->dims, sizeof(int) * dst->ndim);
    axes_device = (int *)tl_clone_h2d(axes, sizeof(int) * src->ndim);

    switch (src->dtype) {
    case TL_DOUBLE:
        transpose_kernel<double>
            <<<block_num, BLOCK_SIZE>>>((double *)src->data, (double *)dst->data, dst->ndim,
                                          s_dims, d_dims, axes_device, BLOCK_SIZE, thread_num);
        break;
    case TL_FLOAT:
        transpose_kernel<float><<<block_num, BLOCK_SIZE>>>((float *)src->data, (float *)dst->data,
                                                             dst->ndim, s_dims, d_dims, axes_device,
                                                             BLOCK_SIZE, thread_num);
        break;
    case TL_INT32:
        transpose_kernel<int32_t>
            <<<block_num, BLOCK_SIZE>>>((int32_t *)src->data, (int32_t *)dst->data, dst->ndim,
                                          s_dims, d_dims, axes_device, BLOCK_SIZE, thread_num);
        break;
    case TL_INT16:
        transpose_kernel<int16_t>
            <<<block_num, BLOCK_SIZE>>>((int16_t *)src->data, (int16_t *)dst->data, dst->ndim,
                                          s_dims, d_dims, axes_device, BLOCK_SIZE, thread_num);
        break;
    case TL_INT8:
        transpose_kernel<int8_t>
            <<<block_num, BLOCK_SIZE>>>((int8_t *)src->data, (int8_t *)dst->data, dst->ndim,
                                          s_dims, d_dims, axes_device, BLOCK_SIZE, thread_num);
        break;
    case TL_UINT32:
        transpose_kernel<uint32_t>
            <<<block_num, BLOCK_SIZE>>>((uint32_t *)src->data, (uint32_t *)dst->data, dst->ndim,
                                          s_dims, d_dims, axes_device, BLOCK_SIZE, thread_num);
        break;
    case TL_UINT16:
        transpose_kernel<uint16_t>
            <<<block_num, BLOCK_SIZE>>>((uint16_t *)src->data, (uint16_t *)dst->data, dst->ndim,
                                          s_dims, d_dims, axes_device, BLOCK_SIZE, thread_num);
        break;
    case TL_UINT8:
        transpose_kernel<uint8_t>
            <<<block_num, BLOCK_SIZE>>>((uint8_t *)src->data, (uint8_t *)dst->data, dst->ndim,
                                          s_dims, d_dims, axes_device, BLOCK_SIZE, thread_num);
        break;
    case TL_BOOL:
        transpose_kernel<tl_bool_t>
            <<<block_num, BLOCK_SIZE>>>((tl_bool_t *)src->data, (tl_bool_t *)dst->data, dst->ndim,
                                          s_dims, d_dims, axes_device, BLOCK_SIZE, thread_num);
        break;
    default:
        assert(0 && "unsupported tl_dtype");
        break;
    }
    tl_cuda_device_sync();

    tl_free_cuda(s_dims);
    tl_free_cuda(d_dims);
    tl_free_cuda(axes_device);

    return dst;
}
