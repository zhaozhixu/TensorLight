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
static __global__ void maxreduce_kernel(T *src, T *dst, int32_t *arg, int dim_size, int reduce_vol,
                                        int batch_vol, int block_size, int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;

    /* src[si] is the first element in this thread to be compared, then
       si = batch_vol * batch + (di - reduce_vol * batch),
       where batch = di / reduce_vol,
       which is the same as the following code: */
    int si = (batch_vol - reduce_vol) * (di / reduce_vol) + di;
    T now = src[si], max = now;
    int maxi = 0;
    for (int i = 1; i < dim_size; i++) {
        now = src[si + i * reduce_vol];
        if (now > max) {
            max = now;
            maxi = i;
        }
    }
    dst[di] = max;
    if (arg)
        arg[di] = maxi;
}

TL_EXPORT tl_tensor *tl_tensor_maxreduce_cuda(const tl_tensor *src, tl_tensor *dst, tl_tensor *arg,
                                              int axis)
{
    /* suppose the shape of src is [N, C, H, W], dim = 1, then thread_num is N x H x W
       reduce_vol is H x W, index_vol is C x H x W */
    int thread_num, block_num, reduce_vol, index_vol;
    void *arg_data;
    int i;

    tl_check_dtype(src->dtype);
    assert(src && tl_is_device_mem(src->data));
    assert(axis < src->ndim && axis >= 0);
    if (dst) {
#ifndef NDEBUG
        assert(tl_is_device_mem(dst->data));
        assert(src->dtype == dst->dtype);
        for (i = 0; i < dst->ndim; i++)
            assert(i == axis ? dst->dims[i] == 1 : dst->dims[i] == src->dims[i]);
#endif
    } else {
        dst = tl_tensor_zeros_slice_cuda(src, axis, 1, src->dtype);
    }
    if (arg) {
#ifndef NDEBUG
        assert(tl_is_device_mem(arg->data));
        assert(arg->dtype == TL_INT32);
        for (i = 0; i < arg->ndim; i++)
            assert(i == axis ? arg->dims[i] == 1 : arg->dims[i] == src->dims[i]);
#endif
        arg_data = arg->data;
    } else {
        arg_data = NULL;
    }

    for (i = axis + 1, thread_num = 1; i < dst->ndim; i++)
        thread_num *= dst->dims[i];
    reduce_vol = thread_num;
    index_vol = thread_num * src->dims[axis];
    for (i = 0; i < axis; i++)
        thread_num *= dst->dims[i];
    block_num = BLOCK_NUM(BLOCK_SIZE, thread_num);

    switch (src->dtype) {
    case TL_DOUBLE:
        maxreduce_kernel<double><<<block_num, BLOCK_SIZE>>>(
            (double *)src->data, (double *)dst->data, (int32_t *)arg_data, src->dims[axis],
            reduce_vol, index_vol, BLOCK_SIZE, thread_num);
        break;
    case TL_FLOAT:
        maxreduce_kernel<float><<<block_num, BLOCK_SIZE>>>((float *)src->data, (float *)dst->data,
                                                             (int32_t *)arg_data, src->dims[axis],
                                                             reduce_vol, index_vol, BLOCK_SIZE,
                                                             thread_num);
        break;
    case TL_INT32:
        maxreduce_kernel<int32_t><<<block_num, BLOCK_SIZE>>>(
            (int32_t *)src->data, (int32_t *)dst->data, (int32_t *)arg_data, src->dims[axis],
            reduce_vol, index_vol, BLOCK_SIZE, thread_num);
        break;
    case TL_INT16:
        maxreduce_kernel<int16_t><<<block_num, BLOCK_SIZE>>>(
            (int16_t *)src->data, (int16_t *)dst->data, (int32_t *)arg_data, src->dims[axis],
            reduce_vol, index_vol, BLOCK_SIZE, thread_num);
        break;
    case TL_INT8:
        maxreduce_kernel<int8_t><<<block_num, BLOCK_SIZE>>>(
            (int8_t *)src->data, (int8_t *)dst->data, (int32_t *)arg_data, src->dims[axis],
            reduce_vol, index_vol, BLOCK_SIZE, thread_num);
        break;
    case TL_UINT32:
        maxreduce_kernel<uint32_t><<<block_num, BLOCK_SIZE>>>(
            (uint32_t *)src->data, (uint32_t *)dst->data, (int32_t *)arg_data, src->dims[axis],
            reduce_vol, index_vol, BLOCK_SIZE, thread_num);
        break;
    case TL_UINT16:
        maxreduce_kernel<uint16_t><<<block_num, BLOCK_SIZE>>>(
            (uint16_t *)src->data, (uint16_t *)dst->data, (int32_t *)arg_data, src->dims[axis],
            reduce_vol, index_vol, BLOCK_SIZE, thread_num);
        break;
    case TL_UINT8:
        maxreduce_kernel<uint8_t><<<block_num, BLOCK_SIZE>>>(
            (uint8_t *)src->data, (uint8_t *)dst->data, (int32_t *)arg_data, src->dims[axis],
            reduce_vol, index_vol, BLOCK_SIZE, thread_num);
        break;
    case TL_BOOL:
        maxreduce_kernel<tl_bool_t><<<block_num, BLOCK_SIZE>>>(
            (tl_bool_t *)src->data, (tl_bool_t *)dst->data, (int32_t *)arg_data, src->dims[axis],
            reduce_vol, index_vol, BLOCK_SIZE, thread_num);
        break;
    default:
        assert(0 && "unsupported tl_dtype");
        break;
    }
    tl_cuda_device_sync();
    return dst;
}
