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
static __global__ void mul_kernel(T *src1, T *src2, T *dst, int block_size, int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;
    dst[di] = src1[di] * src2[di];
}

static __global__ void mul_bool_kernel(tl_bool_t *src1, tl_bool_t *src2, tl_bool_t *dst,
                                       int block_size, int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;
    int res = src1[di] * src2[di];
    if (res)
        dst[di] = TL_TRUE;
    else
        dst[di] = TL_FALSE;
}

template <typename T>
static __global__ void div_kernel(T *src1, T *src2, T *dst, int block_size, int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;
    // assert(src2[di] && "divided by zero");
    dst[di] = src1[di] / src2[di];
}

static __global__ void div_bool_kernel(tl_bool_t *src1, tl_bool_t *src2, tl_bool_t *dst,
                                       int block_size, int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;
    int res = src1[di] / src2[di];
    if (res)
        dst[di] = TL_TRUE;
    else
        dst[di] = TL_FALSE;
}

template <typename T>
static __global__ void sum_kernel(T *src1, T *src2, T *dst, int block_size, int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;
    dst[di] = src1[di] + src2[di];
}

static __global__ void sum_bool_kernel(tl_bool_t *src1, tl_bool_t *src2, tl_bool_t *dst,
                                       int block_size, int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;
    int res = src1[di] + src2[di];
    if (res)
        dst[di] = TL_TRUE;
    else
        dst[di] = TL_FALSE;
}

template <typename T>
static __global__ void sub_kernel(T *src1, T *src2, T *dst, int block_size, int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;
    dst[di] = src1[di] - src2[di];
}

static __global__ void sub_bool_kernel(tl_bool_t *src1, tl_bool_t *src2, tl_bool_t *dst,
                                       int block_size, int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;
    int res = src1[di] - src2[di];
    if (res)
        dst[di] = TL_TRUE;
    else
        dst[di] = TL_FALSE;
}

template <typename T>
static __global__ void max_kernel(T *src1, T *src2, T *dst, int block_size, int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;
    dst[di] = max(src1[di], src2[di]);
}

template <typename T>
static __global__ void min_kernel(T *src1, T *src2, T *dst, int block_size, int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;
    dst[di] = min(src1[di], src2[di]);
}

template <typename T>
static __global__ void pow_int_kernel(T *src1, T *src2, T *dst, T type_max, T type_min,
                                      int block_size, int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;

    float f1, f2, fd;

    f1 = (float)src1[di];
    f2 = (float)src2[di];
    fd = powf(f1, f2);
    if (fd >= type_max)
        dst[di] = type_max;
    else if (fd <= type_min)
        dst[di] = type_min;
    else
        dst[di] = (T)fd;
}

static __global__ void pow_double_kernel(double *src1, double *src2, double *dst, int block_size,
                                         int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;

    dst[di] = pow(src1[di], src2[di]);
}

static __global__ void pow_float_kernel(float *src1, float *src2, float *dst, int block_size,
                                        int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;

    dst[di] = powf(src1[di], src2[di]);
}

static __global__ void pow_bool_kernel(tl_bool_t *src1, tl_bool_t *src2, tl_bool_t *dst,
                                       int block_size, int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;

    float f1, f2, fd;

    f1 = (float)src1[di];
    f2 = (float)src2[di];
    fd = powf(f1, f2);
    if (fd > 0 || fd < 0)
        dst[di] = TL_TRUE;
    else
        dst[di] = TL_FALSE;
}

TL_EXPORT tl_tensor *tl_tensor_elew_cuda(const tl_tensor *src1, const tl_tensor *src2,
                                         tl_tensor *dst, tl_elew_op elew_op)
{
    assert(tl_tensor_issameshape(src1, src2));
    assert(tl_is_device_mem(src1->data) && tl_is_device_mem(src2->data));
    assert(src1->dtype == src2->dtype);
    if (dst) {
        assert(tl_is_device_mem(dst->data));
        assert(tl_tensor_issameshape(src1, dst));
        assert(src1->dtype == dst->dtype);
    } else {
        dst = tl_tensor_zeros_cuda(src1->ndim, src2->dims, src1->dtype);
    }

    int thread_num = dst->len;
    int block_num = BLOCK_NUM(BLOCK_SIZE, thread_num);

    switch (src1->dtype) {
    case TL_DOUBLE:
        switch (elew_op) {
        case TL_MUL:
            mul_kernel<double>
                <<<block_num, BLOCK_SIZE>>>((double *)src1->data, (double *)src2->data,
                                              (double *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_DIV:
            div_kernel<double>
                <<<block_num, BLOCK_SIZE>>>((double *)src1->data, (double *)src2->data,
                                              (double *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_SUM:
            sum_kernel<double>
                <<<block_num, BLOCK_SIZE>>>((double *)src1->data, (double *)src2->data,
                                              (double *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_SUB:
            sub_kernel<double>
                <<<block_num, BLOCK_SIZE>>>((double *)src1->data, (double *)src2->data,
                                              (double *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_MAX:
            max_kernel<double>
                <<<block_num, BLOCK_SIZE>>>((double *)src1->data, (double *)src2->data,
                                              (double *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_MIN:
            min_kernel<double>
                <<<block_num, BLOCK_SIZE>>>((double *)src1->data, (double *)src2->data,
                                              (double *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_POW:
            pow_double_kernel<<<block_num, BLOCK_SIZE>>>((double *)src1->data,
                                                           (double *)src2->data,
                                                           (double *)dst->data, BLOCK_SIZE,
                                                           thread_num);
            break;
        default:
            assert(0 && "unsopported tl_elew_op");
            break;
        }
        break;
    case TL_FLOAT:
        switch (elew_op) {
        case TL_MUL:
            mul_kernel<float><<<block_num, BLOCK_SIZE>>>((float *)src1->data, (float *)src2->data,
                                                           (float *)dst->data, BLOCK_SIZE,
                                                           thread_num);
            break;
        case TL_DIV:
            div_kernel<float><<<block_num, BLOCK_SIZE>>>((float *)src1->data, (float *)src2->data,
                                                           (float *)dst->data, BLOCK_SIZE,
                                                           thread_num);
            break;
        case TL_SUM:
            sum_kernel<float><<<block_num, BLOCK_SIZE>>>((float *)src1->data, (float *)src2->data,
                                                           (float *)dst->data, BLOCK_SIZE,
                                                           thread_num);
            break;
        case TL_SUB:
            sub_kernel<float><<<block_num, BLOCK_SIZE>>>((float *)src1->data, (float *)src2->data,
                                                           (float *)dst->data, BLOCK_SIZE,
                                                           thread_num);
            break;
        case TL_MAX:
            max_kernel<float><<<block_num, BLOCK_SIZE>>>((float *)src1->data, (float *)src2->data,
                                                           (float *)dst->data, BLOCK_SIZE,
                                                           thread_num);
            break;
        case TL_MIN:
            min_kernel<float><<<block_num, BLOCK_SIZE>>>((float *)src1->data, (float *)src2->data,
                                                           (float *)dst->data, BLOCK_SIZE,
                                                           thread_num);
            break;
        case TL_POW:
            pow_float_kernel<<<block_num, BLOCK_SIZE>>>((float *)src1->data, (float *)src2->data,
                                                          (float *)dst->data, BLOCK_SIZE,
                                                          thread_num);
            break;
        default:
            assert(0 && "unsopported tl_elew_op");
            break;
        }
        break;
    case TL_INT32:
        switch (elew_op) {
        case TL_MUL:
            mul_kernel<int32_t>
                <<<block_num, BLOCK_SIZE>>>((int32_t *)src1->data, (int32_t *)src2->data,
                                              (int32_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_DIV:
            div_kernel<int32_t>
                <<<block_num, BLOCK_SIZE>>>((int32_t *)src1->data, (int32_t *)src2->data,
                                              (int32_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_SUM:
            sum_kernel<int32_t>
                <<<block_num, BLOCK_SIZE>>>((int32_t *)src1->data, (int32_t *)src2->data,
                                              (int32_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_SUB:
            sub_kernel<int32_t>
                <<<block_num, BLOCK_SIZE>>>((int32_t *)src1->data, (int32_t *)src2->data,
                                              (int32_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_MAX:
            max_kernel<int32_t>
                <<<block_num, BLOCK_SIZE>>>((int32_t *)src1->data, (int32_t *)src2->data,
                                              (int32_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_MIN:
            min_kernel<int32_t>
                <<<block_num, BLOCK_SIZE>>>((int32_t *)src1->data, (int32_t *)src2->data,
                                              (int32_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_POW:
            pow_int_kernel<int32_t><<<block_num, BLOCK_SIZE>>>((int32_t *)src1->data,
                                                                 (int32_t *)src2->data,
                                                                 (int32_t *)dst->data, INT32_MAX,
                                                                 INT32_MIN, BLOCK_SIZE, thread_num);
            break;
        default:
            assert(0 && "unsopported tl_elew_op");
            break;
        }
        break;
    case TL_INT16:
        switch (elew_op) {
        case TL_MUL:
            mul_kernel<int16_t>
                <<<block_num, BLOCK_SIZE>>>((int16_t *)src1->data, (int16_t *)src2->data,
                                              (int16_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_DIV:
            div_kernel<int16_t>
                <<<block_num, BLOCK_SIZE>>>((int16_t *)src1->data, (int16_t *)src2->data,
                                              (int16_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_SUM:
            sum_kernel<int16_t>
                <<<block_num, BLOCK_SIZE>>>((int16_t *)src1->data, (int16_t *)src2->data,
                                              (int16_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_SUB:
            sub_kernel<int16_t>
                <<<block_num, BLOCK_SIZE>>>((int16_t *)src1->data, (int16_t *)src2->data,
                                              (int16_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_MAX:
            max_kernel<int16_t>
                <<<block_num, BLOCK_SIZE>>>((int16_t *)src1->data, (int16_t *)src2->data,
                                              (int16_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_MIN:
            min_kernel<int16_t>
                <<<block_num, BLOCK_SIZE>>>((int16_t *)src1->data, (int16_t *)src2->data,
                                              (int16_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_POW:
            pow_int_kernel<int16_t><<<block_num, BLOCK_SIZE>>>((int16_t *)src1->data,
                                                                 (int16_t *)src2->data,
                                                                 (int16_t *)dst->data, INT16_MAX,
                                                                 INT16_MIN, BLOCK_SIZE, thread_num);
            break;
        default:
            assert(0 && "unsopported tl_elew_op");
            break;
        }
        break;
    case TL_INT8:
        switch (elew_op) {
        case TL_MUL:
            mul_kernel<int8_t>
                <<<block_num, BLOCK_SIZE>>>((int8_t *)src1->data, (int8_t *)src2->data,
                                              (int8_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_DIV:
            div_kernel<int8_t>
                <<<block_num, BLOCK_SIZE>>>((int8_t *)src1->data, (int8_t *)src2->data,
                                              (int8_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_SUM:
            sum_kernel<int8_t>
                <<<block_num, BLOCK_SIZE>>>((int8_t *)src1->data, (int8_t *)src2->data,
                                              (int8_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_SUB:
            sub_kernel<int8_t>
                <<<block_num, BLOCK_SIZE>>>((int8_t *)src1->data, (int8_t *)src2->data,
                                              (int8_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_MAX:
            max_kernel<int8_t>
                <<<block_num, BLOCK_SIZE>>>((int8_t *)src1->data, (int8_t *)src2->data,
                                              (int8_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_MIN:
            min_kernel<int8_t>
                <<<block_num, BLOCK_SIZE>>>((int8_t *)src1->data, (int8_t *)src2->data,
                                              (int8_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_POW:
            pow_int_kernel<int8_t><<<block_num, BLOCK_SIZE>>>((int8_t *)src1->data,
                                                                (int8_t *)src2->data,
                                                                (int8_t *)dst->data, INT8_MAX,
                                                                INT8_MIN, BLOCK_SIZE, thread_num);
            break;
        default:
            assert(0 && "unsopported tl_elew_op");
            break;
        }
        break;
    case TL_UINT32:
        switch (elew_op) {
        case TL_MUL:
            mul_kernel<uint32_t>
                <<<block_num, BLOCK_SIZE>>>((uint32_t *)src1->data, (uint32_t *)src2->data,
                                              (uint32_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_DIV:
            div_kernel<uint32_t>
                <<<block_num, BLOCK_SIZE>>>((uint32_t *)src1->data, (uint32_t *)src2->data,
                                              (uint32_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_SUM:
            sum_kernel<uint32_t>
                <<<block_num, BLOCK_SIZE>>>((uint32_t *)src1->data, (uint32_t *)src2->data,
                                              (uint32_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_SUB:
            sub_kernel<uint32_t>
                <<<block_num, BLOCK_SIZE>>>((uint32_t *)src1->data, (uint32_t *)src2->data,
                                              (uint32_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_MAX:
            max_kernel<uint32_t>
                <<<block_num, BLOCK_SIZE>>>((uint32_t *)src1->data, (uint32_t *)src2->data,
                                              (uint32_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_MIN:
            min_kernel<uint32_t>
                <<<block_num, BLOCK_SIZE>>>((uint32_t *)src1->data, (uint32_t *)src2->data,
                                              (uint32_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_POW:
            pow_int_kernel<uint32_t><<<block_num, BLOCK_SIZE>>>((uint32_t *)src1->data,
                                                                  (uint32_t *)src2->data,
                                                                  (uint32_t *)dst->data, UINT32_MAX,
                                                                  0, BLOCK_SIZE, thread_num);
            break;
        default:
            assert(0 && "unsopported tl_elew_op");
            break;
        }
        break;
    case TL_UINT16:
        switch (elew_op) {
        case TL_MUL:
            mul_kernel<uint16_t>
                <<<block_num, BLOCK_SIZE>>>((uint16_t *)src1->data, (uint16_t *)src2->data,
                                              (uint16_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_DIV:
            div_kernel<uint16_t>
                <<<block_num, BLOCK_SIZE>>>((uint16_t *)src1->data, (uint16_t *)src2->data,
                                              (uint16_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_SUM:
            sum_kernel<uint16_t>
                <<<block_num, BLOCK_SIZE>>>((uint16_t *)src1->data, (uint16_t *)src2->data,
                                              (uint16_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_SUB:
            sub_kernel<uint16_t>
                <<<block_num, BLOCK_SIZE>>>((uint16_t *)src1->data, (uint16_t *)src2->data,
                                              (uint16_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_MAX:
            max_kernel<uint16_t>
                <<<block_num, BLOCK_SIZE>>>((uint16_t *)src1->data, (uint16_t *)src2->data,
                                              (uint16_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_MIN:
            min_kernel<uint16_t>
                <<<block_num, BLOCK_SIZE>>>((uint16_t *)src1->data, (uint16_t *)src2->data,
                                              (uint16_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_POW:
            pow_int_kernel<uint16_t><<<block_num, BLOCK_SIZE>>>((uint16_t *)src1->data,
                                                                  (uint16_t *)src2->data,
                                                                  (uint16_t *)dst->data, UINT16_MAX,
                                                                  0, BLOCK_SIZE, thread_num);
            break;
        default:
            assert(0 && "unsopported tl_elew_op");
            break;
        }
        break;
    case TL_UINT8:
        switch (elew_op) {
        case TL_MUL:
            mul_kernel<uint8_t>
                <<<block_num, BLOCK_SIZE>>>((uint8_t *)src1->data, (uint8_t *)src2->data,
                                              (uint8_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_DIV:
            div_kernel<uint8_t>
                <<<block_num, BLOCK_SIZE>>>((uint8_t *)src1->data, (uint8_t *)src2->data,
                                              (uint8_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_SUM:
            sum_kernel<uint8_t>
                <<<block_num, BLOCK_SIZE>>>((uint8_t *)src1->data, (uint8_t *)src2->data,
                                              (uint8_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_SUB:
            sub_kernel<uint8_t>
                <<<block_num, BLOCK_SIZE>>>((uint8_t *)src1->data, (uint8_t *)src2->data,
                                              (uint8_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_MAX:
            max_kernel<uint8_t>
                <<<block_num, BLOCK_SIZE>>>((uint8_t *)src1->data, (uint8_t *)src2->data,
                                              (uint8_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_MIN:
            min_kernel<uint8_t>
                <<<block_num, BLOCK_SIZE>>>((uint8_t *)src1->data, (uint8_t *)src2->data,
                                              (uint8_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_POW:
            pow_int_kernel<uint8_t><<<block_num, BLOCK_SIZE>>>((uint8_t *)src1->data,
                                                                 (uint8_t *)src2->data,
                                                                 (uint8_t *)dst->data, UINT8_MAX, 0,
                                                                 BLOCK_SIZE, thread_num);
            break;
        default:
            assert(0 && "unsopported tl_elew_op");
            break;
        }
        break;
    case TL_BOOL:
        switch (elew_op) {
        case TL_MUL:
            mul_bool_kernel<<<block_num, BLOCK_SIZE>>>((tl_bool_t *)src1->data,
                                                         (tl_bool_t *)src2->data,
                                                         (tl_bool_t *)dst->data, BLOCK_SIZE,
                                                         thread_num);
            break;
        case TL_DIV:
            div_bool_kernel<<<block_num, BLOCK_SIZE>>>((tl_bool_t *)src1->data,
                                                         (tl_bool_t *)src2->data,
                                                         (tl_bool_t *)dst->data, BLOCK_SIZE,
                                                         thread_num);
            break;
        case TL_SUM:
            sum_bool_kernel<<<block_num, BLOCK_SIZE>>>((tl_bool_t *)src1->data,
                                                         (tl_bool_t *)src2->data,
                                                         (tl_bool_t *)dst->data, BLOCK_SIZE,
                                                         thread_num);
            break;
        case TL_SUB:
            sub_bool_kernel<<<block_num, BLOCK_SIZE>>>((tl_bool_t *)src1->data,
                                                         (tl_bool_t *)src2->data,
                                                         (tl_bool_t *)dst->data, BLOCK_SIZE,
                                                         thread_num);
            break;
        case TL_MAX:
            max_kernel<tl_bool_t>
                <<<block_num, BLOCK_SIZE>>>((tl_bool_t *)src1->data, (tl_bool_t *)src2->data,
                                              (tl_bool_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_MIN:
            min_kernel<tl_bool_t>
                <<<block_num, BLOCK_SIZE>>>((tl_bool_t *)src1->data, (tl_bool_t *)src2->data,
                                              (tl_bool_t *)dst->data, BLOCK_SIZE, thread_num);
            break;
        case TL_POW:
            pow_bool_kernel<<<block_num, BLOCK_SIZE>>>((tl_bool_t *)src1->data,
                                                         (tl_bool_t *)src2->data,
                                                         (tl_bool_t *)dst->data, BLOCK_SIZE,
                                                         thread_num);
            break;
        default:
            assert(0 && "unsopported tl_elew_op");
            break;
        }
        break;
    default:
        assert(0 && "unsupported tl_dtype");
        break;
    }
    tl_cuda_device_sync();

    return dst;
}
