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

#include "tl_tensor_internal_cuda.h"

/* TODO: strided access should be avoid */
template <typename T>
static __global__ void pick1d_kernel(T *src, T *dst, int *idx, int stride, int block_size,
                                     int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;
    int si = idx[di];
    for (int i = 0; i < stride; i++)
        dst[di * stride + i] = src[si * stride + i];
}

TL_EXPORT tl_tensor *tl_tensor_pick1d_cuda(const tl_tensor *src, const tl_tensor *index,
                                           tl_tensor *dst, int stride, int len)
{
    assert(src);
    assert(tl_is_device_mem(src->data));
    assert(src->ndim == 1);
    assert(index);
    assert(tl_is_device_mem(index->data));
    assert(index->dtype == TL_INT32);
    assert(index->ndim == 1);
    assert(index->len >= len);
    assert(stride >= 1);
    if (dst) {
        assert(dst);
        assert(tl_is_device_mem(dst->data));
        assert(dst->ndim == 1);
        assert(dst->len == len * stride);
        assert(dst->dtype == src->dtype);
    } else {
        int dims[1];
        dims[0] = len;
        dst = tl_tensor_zeros_cuda(1, dims, src->dtype);
    }

    int thread_num, block_num;
    thread_num = len;
    block_num = thread_num / BLOCK_SIZE + 1;

    switch (src->dtype) {
    case TL_DOUBLE:
        pick1d_kernel<double><<<block_num, BLOCK_SIZE>>>((double *)src->data, (double *)dst->data,
                                                           (int *)index->data, stride, BLOCK_SIZE,
                                                           thread_num);
        break;
    case TL_FLOAT:
        pick1d_kernel<float><<<block_num, BLOCK_SIZE>>>((float *)src->data, (float *)dst->data,
                                                          (int *)index->data, stride, BLOCK_SIZE,
                                                          thread_num);
        break;
    case TL_INT32:
        pick1d_kernel<int32_t>
            <<<block_num, BLOCK_SIZE>>>((int32_t *)src->data, (int32_t *)dst->data,
                                          (int *)index->data, stride, BLOCK_SIZE, thread_num);
        break;
    case TL_INT16:
        pick1d_kernel<int16_t>
            <<<block_num, BLOCK_SIZE>>>((int16_t *)src->data, (int16_t *)dst->data,
                                          (int *)index->data, stride, BLOCK_SIZE, thread_num);
        break;
    case TL_INT8:
        pick1d_kernel<int8_t><<<block_num, BLOCK_SIZE>>>((int8_t *)src->data, (int8_t *)dst->data,
                                                           (int *)index->data, stride, BLOCK_SIZE,
                                                           thread_num);
        break;
    case TL_UINT32:
        pick1d_kernel<uint32_t>
            <<<block_num, BLOCK_SIZE>>>((uint32_t *)src->data, (uint32_t *)dst->data,
                                          (int *)index->data, stride, BLOCK_SIZE, thread_num);
        break;
    case TL_UINT16:
        pick1d_kernel<uint16_t>
            <<<block_num, BLOCK_SIZE>>>((uint16_t *)src->data, (uint16_t *)dst->data,
                                          (int *)index->data, stride, BLOCK_SIZE, thread_num);
        break;
    case TL_UINT8:
        pick1d_kernel<uint8_t>
            <<<block_num, BLOCK_SIZE>>>((uint8_t *)src->data, (uint8_t *)dst->data,
                                          (int *)index->data, stride, BLOCK_SIZE, thread_num);
        break;
    case TL_BOOL:
        pick1d_kernel<int><<<block_num, BLOCK_SIZE>>>(
            (int *)src->data, (int *)dst->data, (int *)index->data, stride, BLOCK_SIZE, thread_num);
        break;
    default:
        assert(0 && "unsupported tl_dtype");
        break;
    }
    tl_cuda_device_sync();

    return dst;
}
