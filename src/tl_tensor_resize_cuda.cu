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
static __global__ void nearest_resize_kernel(const T *src, T *dst, int ndim, const int *dims,
                                             const int *new_dims, int block_size, int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;

    __shared__ float scales[TL_MAXDIM];
    if (threadIdx.x < ndim) {
        scales[threadIdx.x] = (float)dims[threadIdx.x] / (float)new_dims[threadIdx.x];
    }

    if (di > total)
        return;

    int si;
    float rounded;
    int src_coords[TL_MAXDIM];
    int dst_coords[TL_MAXDIM];
    tl_get_coords_cu(di, dst_coords, ndim, new_dims);
    for (int i = 0; i < ndim; i++) {
        rounded = roundf(((float)dst_coords[i] + 0.5) * scales[i] - 0.5);
        tl_convert_device_cu(&src_coords[i], TL_INT32, &rounded, TL_FLOAT);
    }
    si = tl_get_index_cu(src_coords, ndim, dims);
    dst[di] = src[si];
}

TL_EXPORT tl_tensor *tl_tensor_resize_cuda(const tl_tensor *src, tl_tensor *dst,
                                           const int *new_dims, tl_resize_type rtype)
{
    assert(src);
    assert(tl_is_device_mem(src->data));
    assert(new_dims);
    tl_check_resize_type(rtype);
    if (dst) {
        assert(dst->data);
        assert(dst->dtype == src->dtype);
        assert(dst->ndim == src->ndim);
    } else {
        dst = tl_tensor_zeros_cuda(src->ndim, new_dims, src->dtype);
    }

    int block_num, thread_num;
    int *dims_cuda, *new_dims_cuda;

    dims_cuda = (int *)tl_clone_h2d(src->dims, sizeof(int) * src->ndim);
    new_dims_cuda = (int *)tl_clone_h2d(new_dims, sizeof(int) * src->ndim);

    thread_num = dst->len;
    block_num = BLOCK_NUM(BLOCK_SIZE, thread_num);
    switch (rtype) {
    case TL_NEAREST:
        switch (src->dtype) {
        case TL_DOUBLE:
            nearest_resize_kernel<double>
                <<<block_num, BLOCK_SIZE>>>((double *)src->data, (double *)dst->data, src->ndim,
                                              dims_cuda, new_dims_cuda, BLOCK_SIZE, thread_num);
            break;
        case TL_FLOAT:
            nearest_resize_kernel<float>
                <<<block_num, BLOCK_SIZE>>>((float *)src->data, (float *)dst->data, src->ndim,
                                              dims_cuda, new_dims_cuda, BLOCK_SIZE, thread_num);
            break;
        case TL_INT32:
            nearest_resize_kernel<int32_t>
                <<<block_num, BLOCK_SIZE>>>((int32_t *)src->data, (int32_t *)dst->data, src->ndim,
                                              dims_cuda, new_dims_cuda, BLOCK_SIZE, thread_num);
            break;
        case TL_INT16:
            nearest_resize_kernel<int16_t>
                <<<block_num, BLOCK_SIZE>>>((int16_t *)src->data, (int16_t *)dst->data, src->ndim,
                                              dims_cuda, new_dims_cuda, BLOCK_SIZE, thread_num);
            break;
        case TL_INT8:
            nearest_resize_kernel<int8_t>
                <<<block_num, BLOCK_SIZE>>>((int8_t *)src->data, (int8_t *)dst->data, src->ndim,
                                              dims_cuda, new_dims_cuda, BLOCK_SIZE, thread_num);
            break;
        case TL_UINT32:
            nearest_resize_kernel<uint32_t><<<block_num, BLOCK_SIZE>>>(
                (uint32_t *)src->data, (uint32_t *)dst->data, src->ndim, dims_cuda, new_dims_cuda,
                BLOCK_SIZE, thread_num);
            break;
        case TL_UINT16:
            nearest_resize_kernel<uint16_t><<<block_num, BLOCK_SIZE>>>(
                (uint16_t *)src->data, (uint16_t *)dst->data, src->ndim, dims_cuda, new_dims_cuda,
                BLOCK_SIZE, thread_num);
        case TL_UINT8:
            nearest_resize_kernel<uint8_t>
                <<<block_num, BLOCK_SIZE>>>((uint8_t *)src->data, (uint8_t *)dst->data, src->ndim,
                                              dims_cuda, new_dims_cuda, BLOCK_SIZE, thread_num);
            break;
        case TL_BOOL:
            nearest_resize_kernel<tl_bool_t><<<block_num, BLOCK_SIZE>>>(
                (tl_bool_t *)src->data, (tl_bool_t *)dst->data, src->ndim, dims_cuda, new_dims_cuda,
                BLOCK_SIZE, thread_num);
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_LINEAR:
        assert(0 && "not support TL_LINEAR yet");
        break;
    default:
        assert(0 && "unsupported tl_resize_type");
        break;
    }
    tl_cuda_device_sync();

    tl_free_cuda(dims_cuda);
    tl_free_cuda(new_dims_cuda);
    return dst;
}
