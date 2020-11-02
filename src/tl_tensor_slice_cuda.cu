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

TL_EXPORT tl_tensor *tl_tensor_zeros_slice_cuda(const tl_tensor *src, int axis, int len,
                                                tl_dtype dtype)
{
    tl_tensor *dst;
    int *dims;

    assert(src);
    assert(axis < src->ndim && axis >= 0);
    assert(len <= src->dims[axis] && len > 0);

    dims = (int *)tl_clone(src->dims, sizeof(int) * src->ndim);
    dims[axis] = len;
    dst = tl_tensor_zeros_cuda(src->ndim, dims, dtype);
    tl_free(dims);

    return dst;
}

template <typename T>
static __global__ void slice_kernel(T *src, T *dst, int start, int s_vol, int d_vol, int vol,
                                    int block_size, int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;
    int si = di / d_vol * s_vol + di % d_vol + start * vol;
    dst[di] = src[si];
}

TL_EXPORT tl_tensor *tl_tensor_slice_cuda(const tl_tensor *src, tl_tensor *dst, int axis, int start,
                                          int len)
{
    int i;
    int d_vol, s_vol, vol;
    int thread_num, block_num;

    assert(src && tl_is_device_mem(src->data));
    assert(axis < src->ndim && axis >= 0);
    assert(len <= src->dims[axis] && len > 0);
    assert(start < src->dims[axis] && start >= 0);
    assert(len + start <= src->dims[axis]);
    if (dst) {
#ifndef NDEBUG
        assert(tl_is_device_mem(dst->data));
        assert(src->dtype == dst->dtype);
        assert(dst->ndim == src->ndim);
        for (i = 0; i < src->ndim; i++)
            assert(i == axis ? dst->dims[i] == len : dst->dims[i] == src->dims[i]);
#endif
    } else {
        dst = tl_tensor_zeros_slice_cuda(src, axis, len, src->dtype);
    }

    for (i = axis + 1, vol = 1; i < dst->ndim; i++)
        vol *= dst->dims[i];
    d_vol = vol * dst->dims[axis];
    s_vol = vol * src->dims[axis];
    thread_num = dst->len;
    block_num = BLOCK_NUM(BLOCK_SIZE, thread_num);

    switch (src->dtype) {
    case TL_DOUBLE:
        slice_kernel<double><<<block_num, BLOCK_SIZE>>>((double *)src->data, (double *)dst->data,
                                                          start, s_vol, d_vol, vol, BLOCK_SIZE,
                                                          thread_num);
        break;
    case TL_FLOAT:
        slice_kernel<float><<<block_num, BLOCK_SIZE>>>((float *)src->data, (float *)dst->data,
                                                         start, s_vol, d_vol, vol, BLOCK_SIZE,
                                                         thread_num);
        break;
    case TL_INT32:
        slice_kernel<int32_t><<<block_num, BLOCK_SIZE>>>((int32_t *)src->data,
                                                           (int32_t *)dst->data, start, s_vol,
                                                           d_vol, vol, BLOCK_SIZE, thread_num);
        break;
    case TL_INT16:
        slice_kernel<int16_t><<<block_num, BLOCK_SIZE>>>((int16_t *)src->data,
                                                           (int16_t *)dst->data, start, s_vol,
                                                           d_vol, vol, BLOCK_SIZE, thread_num);
        break;
    case TL_INT8:
        slice_kernel<int8_t><<<block_num, BLOCK_SIZE>>>((int8_t *)src->data, (int8_t *)dst->data,
                                                          start, s_vol, d_vol, vol, BLOCK_SIZE,
                                                          thread_num);
        break;
    case TL_UINT32:
        slice_kernel<uint32_t><<<block_num, BLOCK_SIZE>>>((uint32_t *)src->data,
                                                            (uint32_t *)dst->data, start, s_vol,
                                                            d_vol, vol, BLOCK_SIZE, thread_num);
        break;
    case TL_UINT16:
        slice_kernel<uint16_t><<<block_num, BLOCK_SIZE>>>((uint16_t *)src->data,
                                                            (uint16_t *)dst->data, start, s_vol,
                                                            d_vol, vol, BLOCK_SIZE, thread_num);
        break;
    case TL_UINT8:
        slice_kernel<uint8_t><<<block_num, BLOCK_SIZE>>>((uint8_t *)src->data,
                                                           (uint8_t *)dst->data, start, s_vol,
                                                           d_vol, vol, BLOCK_SIZE, thread_num);
        break;
    case TL_BOOL:
        slice_kernel<tl_bool_t><<<block_num, BLOCK_SIZE>>>((tl_bool_t *)src->data,
                                                             (tl_bool_t *)dst->data, start, s_vol,
                                                             d_vol, vol, BLOCK_SIZE, thread_num);
        break;
    default:
        assert(0 && "unsupported tl_dtype");
        break;
    }
    tl_cuda_device_sync();
    return dst;
}
