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

template <typename Ts, typename Td>
static __global__ void submean_kernel(const Ts *src, Td *dst, tl_dtype dst_dtype, double mean1,
                                      double mean2, double mean3, int H, int W, int C,
                                      int block_size, int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;

    double mean[] = { mean1, mean2, mean3 };
    int src_coords[TL_MAXDIM];
    int dst_coords[TL_MAXDIM];
    int src_dims[] = { H, W, C };
    int dst_dims[] = { C, H, W };
    tl_get_coords_cu(di, dst_coords, 3, dst_dims);
    src_coords[0] = dst_coords[1];
    src_coords[1] = dst_coords[2];
    src_coords[2] = dst_coords[0];
    int si = tl_get_index_cu(src_coords, 3, src_dims);
    double res = (double)src[si] - mean[src_coords[2]];
    tl_convert_device_cu(&dst[di], dst_dtype, &res, TL_DOUBLE);
}

/* src: H*W*C, dst: C*H*W */
TL_EXPORT tl_tensor *tl_tensor_submean_cuda(const tl_tensor *src, tl_tensor *dst,
                                            const double *mean)
{
    assert(src);
    assert(tl_is_device_mem(src->data));
    assert(mean);
    assert(src->ndim == 3);
    int new_dims[] = { src->dims[2], src->dims[0], src->dims[1] };

    if (dst) {
        assert(tl_is_device_mem(dst->data));
        assert(dst->ndim == src->ndim);
        assert(dst->dims[0] == 3);
    } else {
        dst = tl_tensor_zeros_cuda(src->ndim, new_dims, TL_FLOAT);
    }

    int thread_num, block_num;
    thread_num = dst->len;
    block_num = thread_num / BLOCK_SIZE + 1;

    /*
     * Generated by tools/generic.pl with
     * $switchtype(src->dtype, T1)
     * $switchtype(dst->dtype, T2)
     * $typenoset(T1, TL_BOOL)
     * $typenoset(T2, TL_BOOL)
     * submean_kernel<T1, T2><<<block_num, BLOCK_SIZE>>>((T1 *)src->data, (T2 *)dst->data, dst->dtype, mean[0], mean[1], mean[2], src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
     */
    switch (src->dtype) {
    case TL_DOUBLE:
        switch (dst->dtype) {
        case TL_DOUBLE:
            submean_kernel<double, double><<<block_num, BLOCK_SIZE>>>(
                (double *)src->data, (double *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_FLOAT:
            submean_kernel<double, float><<<block_num, BLOCK_SIZE>>>(
                (double *)src->data, (float *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_INT16:
            submean_kernel<double, int16_t><<<block_num, BLOCK_SIZE>>>(
                (double *)src->data, (int16_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_INT32:
            submean_kernel<double, int32_t><<<block_num, BLOCK_SIZE>>>(
                (double *)src->data, (int32_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_INT8:
            submean_kernel<double, int8_t><<<block_num, BLOCK_SIZE>>>(
                (double *)src->data, (int8_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_UINT16:
            submean_kernel<double, uint16_t><<<block_num, BLOCK_SIZE>>>(
                (double *)src->data, (uint16_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_UINT32:
            submean_kernel<double, uint32_t><<<block_num, BLOCK_SIZE>>>(
                (double *)src->data, (uint32_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_UINT8:
            submean_kernel<double, uint8_t><<<block_num, BLOCK_SIZE>>>(
                (double *)src->data, (uint8_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        default:
            assert(0 && "unsupported dtype for dst->dtype");
            break;
        }
        break;
    case TL_FLOAT:
        switch (dst->dtype) {
        case TL_DOUBLE:
            submean_kernel<float, double><<<block_num, BLOCK_SIZE>>>(
                (float *)src->data, (double *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_FLOAT:
            submean_kernel<float, float><<<block_num, BLOCK_SIZE>>>(
                (float *)src->data, (float *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_INT16:
            submean_kernel<float, int16_t><<<block_num, BLOCK_SIZE>>>(
                (float *)src->data, (int16_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_INT32:
            submean_kernel<float, int32_t><<<block_num, BLOCK_SIZE>>>(
                (float *)src->data, (int32_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_INT8:
            submean_kernel<float, int8_t><<<block_num, BLOCK_SIZE>>>(
                (float *)src->data, (int8_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_UINT16:
            submean_kernel<float, uint16_t><<<block_num, BLOCK_SIZE>>>(
                (float *)src->data, (uint16_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_UINT32:
            submean_kernel<float, uint32_t><<<block_num, BLOCK_SIZE>>>(
                (float *)src->data, (uint32_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_UINT8:
            submean_kernel<float, uint8_t><<<block_num, BLOCK_SIZE>>>(
                (float *)src->data, (uint8_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        default:
            assert(0 && "unsupported dtype for dst->dtype");
            break;
        }
        break;
    case TL_INT16:
        switch (dst->dtype) {
        case TL_DOUBLE:
            submean_kernel<int16_t, double><<<block_num, BLOCK_SIZE>>>(
                (int16_t *)src->data, (double *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_FLOAT:
            submean_kernel<int16_t, float><<<block_num, BLOCK_SIZE>>>(
                (int16_t *)src->data, (float *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_INT16:
            submean_kernel<int16_t, int16_t><<<block_num, BLOCK_SIZE>>>(
                (int16_t *)src->data, (int16_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_INT32:
            submean_kernel<int16_t, int32_t><<<block_num, BLOCK_SIZE>>>(
                (int16_t *)src->data, (int32_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_INT8:
            submean_kernel<int16_t, int8_t><<<block_num, BLOCK_SIZE>>>(
                (int16_t *)src->data, (int8_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_UINT16:
            submean_kernel<int16_t, uint16_t><<<block_num, BLOCK_SIZE>>>(
                (int16_t *)src->data, (uint16_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_UINT32:
            submean_kernel<int16_t, uint32_t><<<block_num, BLOCK_SIZE>>>(
                (int16_t *)src->data, (uint32_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_UINT8:
            submean_kernel<int16_t, uint8_t><<<block_num, BLOCK_SIZE>>>(
                (int16_t *)src->data, (uint8_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        default:
            assert(0 && "unsupported dtype for dst->dtype");
            break;
        }
        break;
    case TL_INT32:
        switch (dst->dtype) {
        case TL_DOUBLE:
            submean_kernel<int32_t, double><<<block_num, BLOCK_SIZE>>>(
                (int32_t *)src->data, (double *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_FLOAT:
            submean_kernel<int32_t, float><<<block_num, BLOCK_SIZE>>>(
                (int32_t *)src->data, (float *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_INT16:
            submean_kernel<int32_t, int16_t><<<block_num, BLOCK_SIZE>>>(
                (int32_t *)src->data, (int16_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_INT32:
            submean_kernel<int32_t, int32_t><<<block_num, BLOCK_SIZE>>>(
                (int32_t *)src->data, (int32_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_INT8:
            submean_kernel<int32_t, int8_t><<<block_num, BLOCK_SIZE>>>(
                (int32_t *)src->data, (int8_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_UINT16:
            submean_kernel<int32_t, uint16_t><<<block_num, BLOCK_SIZE>>>(
                (int32_t *)src->data, (uint16_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_UINT32:
            submean_kernel<int32_t, uint32_t><<<block_num, BLOCK_SIZE>>>(
                (int32_t *)src->data, (uint32_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_UINT8:
            submean_kernel<int32_t, uint8_t><<<block_num, BLOCK_SIZE>>>(
                (int32_t *)src->data, (uint8_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        default:
            assert(0 && "unsupported dtype for dst->dtype");
            break;
        }
        break;
    case TL_INT8:
        switch (dst->dtype) {
        case TL_DOUBLE:
            submean_kernel<int8_t, double><<<block_num, BLOCK_SIZE>>>(
                (int8_t *)src->data, (double *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_FLOAT:
            submean_kernel<int8_t, float><<<block_num, BLOCK_SIZE>>>(
                (int8_t *)src->data, (float *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_INT16:
            submean_kernel<int8_t, int16_t><<<block_num, BLOCK_SIZE>>>(
                (int8_t *)src->data, (int16_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_INT32:
            submean_kernel<int8_t, int32_t><<<block_num, BLOCK_SIZE>>>(
                (int8_t *)src->data, (int32_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_INT8:
            submean_kernel<int8_t, int8_t><<<block_num, BLOCK_SIZE>>>(
                (int8_t *)src->data, (int8_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_UINT16:
            submean_kernel<int8_t, uint16_t><<<block_num, BLOCK_SIZE>>>(
                (int8_t *)src->data, (uint16_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_UINT32:
            submean_kernel<int8_t, uint32_t><<<block_num, BLOCK_SIZE>>>(
                (int8_t *)src->data, (uint32_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_UINT8:
            submean_kernel<int8_t, uint8_t><<<block_num, BLOCK_SIZE>>>(
                (int8_t *)src->data, (uint8_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        default:
            assert(0 && "unsupported dtype for dst->dtype");
            break;
        }
        break;
    case TL_UINT16:
        switch (dst->dtype) {
        case TL_DOUBLE:
            submean_kernel<uint16_t, double><<<block_num, BLOCK_SIZE>>>(
                (uint16_t *)src->data, (double *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_FLOAT:
            submean_kernel<uint16_t, float><<<block_num, BLOCK_SIZE>>>(
                (uint16_t *)src->data, (float *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_INT16:
            submean_kernel<uint16_t, int16_t><<<block_num, BLOCK_SIZE>>>(
                (uint16_t *)src->data, (int16_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_INT32:
            submean_kernel<uint16_t, int32_t><<<block_num, BLOCK_SIZE>>>(
                (uint16_t *)src->data, (int32_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_INT8:
            submean_kernel<uint16_t, int8_t><<<block_num, BLOCK_SIZE>>>(
                (uint16_t *)src->data, (int8_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_UINT16:
            submean_kernel<uint16_t, uint16_t><<<block_num, BLOCK_SIZE>>>(
                (uint16_t *)src->data, (uint16_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_UINT32:
            submean_kernel<uint16_t, uint32_t><<<block_num, BLOCK_SIZE>>>(
                (uint16_t *)src->data, (uint32_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_UINT8:
            submean_kernel<uint16_t, uint8_t><<<block_num, BLOCK_SIZE>>>(
                (uint16_t *)src->data, (uint8_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        default:
            assert(0 && "unsupported dtype for dst->dtype");
            break;
        }
        break;
    case TL_UINT32:
        switch (dst->dtype) {
        case TL_DOUBLE:
            submean_kernel<uint32_t, double><<<block_num, BLOCK_SIZE>>>(
                (uint32_t *)src->data, (double *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_FLOAT:
            submean_kernel<uint32_t, float><<<block_num, BLOCK_SIZE>>>(
                (uint32_t *)src->data, (float *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_INT16:
            submean_kernel<uint32_t, int16_t><<<block_num, BLOCK_SIZE>>>(
                (uint32_t *)src->data, (int16_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_INT32:
            submean_kernel<uint32_t, int32_t><<<block_num, BLOCK_SIZE>>>(
                (uint32_t *)src->data, (int32_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_INT8:
            submean_kernel<uint32_t, int8_t><<<block_num, BLOCK_SIZE>>>(
                (uint32_t *)src->data, (int8_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_UINT16:
            submean_kernel<uint32_t, uint16_t><<<block_num, BLOCK_SIZE>>>(
                (uint32_t *)src->data, (uint16_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_UINT32:
            submean_kernel<uint32_t, uint32_t><<<block_num, BLOCK_SIZE>>>(
                (uint32_t *)src->data, (uint32_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_UINT8:
            submean_kernel<uint32_t, uint8_t><<<block_num, BLOCK_SIZE>>>(
                (uint32_t *)src->data, (uint8_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        default:
            assert(0 && "unsupported dtype for dst->dtype");
            break;
        }
        break;
    case TL_UINT8:
        switch (dst->dtype) {
        case TL_DOUBLE:
            submean_kernel<uint8_t, double><<<block_num, BLOCK_SIZE>>>(
                (uint8_t *)src->data, (double *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_FLOAT:
            submean_kernel<uint8_t, float><<<block_num, BLOCK_SIZE>>>(
                (uint8_t *)src->data, (float *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_INT16:
            submean_kernel<uint8_t, int16_t><<<block_num, BLOCK_SIZE>>>(
                (uint8_t *)src->data, (int16_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_INT32:
            submean_kernel<uint8_t, int32_t><<<block_num, BLOCK_SIZE>>>(
                (uint8_t *)src->data, (int32_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_INT8:
            submean_kernel<uint8_t, int8_t><<<block_num, BLOCK_SIZE>>>(
                (uint8_t *)src->data, (int8_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_UINT16:
            submean_kernel<uint8_t, uint16_t><<<block_num, BLOCK_SIZE>>>(
                (uint8_t *)src->data, (uint16_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_UINT32:
            submean_kernel<uint8_t, uint32_t><<<block_num, BLOCK_SIZE>>>(
                (uint8_t *)src->data, (uint32_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        case TL_UINT8:
            submean_kernel<uint8_t, uint8_t><<<block_num, BLOCK_SIZE>>>(
                (uint8_t *)src->data, (uint8_t *)dst->data, dst->dtype, mean[0], mean[1], mean[2],
                src->dims[0], src->dims[1], src->dims[2], BLOCK_SIZE, thread_num);
            break;
        default:
            assert(0 && "unsupported dtype for dst->dtype");
            break;
        }
        break;
    default:
        assert(0 && "unsupported dtype for src->dtype");
        break;
    }

    tl_cuda_device_sync();

    return dst;
}