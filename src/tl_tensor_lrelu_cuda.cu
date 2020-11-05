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

template <typename T>
static __global__ void lrelu_kernel(const T *src, T *dst, float negslope, int block_size, int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di > total)
        return;

    T s = src[di];
    dst[di] = s >= 0 ? s : s * (T)negslope;
}

#define LRELU_CUDA(ps, pd, ns, bn, bs, tn, type)                                                   \
    lrelu_kernel<type><<<(bn), (bs)>>>((type *)(ps), (type *)(pd), (ns), (bs), (tn))

TL_EXPORT tl_tensor *tl_tensor_lrelu_cuda(const tl_tensor *src, tl_tensor *dst, float negslope)
{
    assert(src && tl_is_device_mem(src->data));
    if (dst) {
        assert(dst && tl_is_device_mem(dst->data));
        assert(tl_tensor_issameshape(dst, src));
        assert(dst->dtype == src->dtype);
    } else {
        dst = tl_tensor_zeros_cuda(src->ndim, src->dims, src->dtype);
    }

    int thread_num, block_num;

    thread_num = dst->len;
    block_num = BLOCK_NUM(BLOCK_SIZE, thread_num);
    switch (src->dtype) {
    case TL_DOUBLE:
        LRELU_CUDA(src->data, dst->data, negslope, block_num, BLOCK_SIZE, thread_num, double);
        break;
    case TL_FLOAT:
        LRELU_CUDA(src->data, dst->data, negslope, block_num, BLOCK_SIZE, thread_num, float);
        break;
    case TL_INT32:
        LRELU_CUDA(src->data, dst->data, negslope, block_num, BLOCK_SIZE, thread_num, int32_t);
        break;
    case TL_INT16:
        LRELU_CUDA(src->data, dst->data, negslope, block_num, BLOCK_SIZE, thread_num, int16_t);
        break;
    case TL_INT8:
        LRELU_CUDA(src->data, dst->data, negslope, block_num, BLOCK_SIZE, thread_num, int8_t);
        break;
    case TL_UINT32:
        tl_memcpy_d2d(dst->data, src->data, tl_tensor_size(dst));
        break;
    case TL_UINT16:
        tl_memcpy_d2d(dst->data, src->data, tl_tensor_size(dst));
        break;
    case TL_UINT8:
        tl_memcpy_d2d(dst->data, src->data, tl_tensor_size(dst));
        break;
    case TL_BOOL:
        LRELU_CUDA(src->data, dst->data, negslope, block_num, BLOCK_SIZE, thread_num, int);
        break;
    default:
        assert(0 && "unsupported tl_dtype");
        break;
    }
    tl_cuda_device_sync();

    return dst;
}
#undef LRELU_CUDA
