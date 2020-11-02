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

static __global__ void convert_kernel(void *src, void *dst, tl_dtype dtype_s, tl_dtype dtype_d,
                                      int block_size, int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;

    double val_d;
    float val_f;
    int32_t val_i32;
    uint32_t val_u32;
    int16_t val_i16;
    uint16_t val_u16;
    int8_t val_i8;
    uint8_t val_u8;

    switch (dtype_d) {
    case TL_DOUBLE:
        switch (dtype_s) {
        case TL_DOUBLE:
            ((double *)dst)[di] = ((double *)src)[di];
            break;
        case TL_FLOAT:
            ((double *)dst)[di] = (double)((float *)src)[di];
            break;
        case TL_INT32:
            ((double *)dst)[di] = (double)((int32_t *)src)[di];
            break;
        case TL_INT16:
            ((double *)dst)[di] = (double)((int16_t *)src)[di];
            break;
        case TL_INT8:
            ((double *)dst)[di] = (double)((int8_t *)src)[di];
            break;
        case TL_UINT32:
            ((double *)dst)[di] = (double)((uint32_t *)src)[di];
            break;
        case TL_UINT16:
            ((double *)dst)[di] = (double)((uint16_t *)src)[di];
            break;
        case TL_UINT8:
            ((double *)dst)[di] = (double)((uint8_t *)src)[di];
            break;
        case TL_BOOL:
            ((double *)dst)[di] = (double)((tl_bool_t *)src)[di];
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_FLOAT:
        switch (dtype_s) {
        case TL_DOUBLE:
            val_d = ((double *)src)[di];
            if (val_d >= FLT_MAX)
                ((float *)dst)[di] = FLT_MAX;
            else if (val_d <= -FLT_MAX)
                ((float *)dst)[di] = -FLT_MAX;
            else
                ((float *)dst)[di] = (float)val_d;
            break;
        case TL_FLOAT:
            ((float *)dst)[di] = ((float *)src)[di];
            break;
        case TL_INT32:
            ((float *)dst)[di] = (float)((int32_t *)src)[di];
            break;
        case TL_INT16:
            ((float *)dst)[di] = (float)((int16_t *)src)[di];
            break;
        case TL_INT8:
            ((float *)dst)[di] = (float)((int8_t *)src)[di];
            break;
        case TL_UINT32:
            ((float *)dst)[di] = (float)((uint32_t *)src)[di];
            break;
        case TL_UINT16:
            ((float *)dst)[di] = (float)((uint16_t *)src)[di];
            break;
        case TL_UINT8:
            ((float *)dst)[di] = (float)((uint8_t *)src)[di];
            break;
        case TL_BOOL:
            ((float *)dst)[di] = (float)((tl_bool_t *)src)[di];
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_INT32:
        switch (dtype_s) {
        case TL_DOUBLE:
            val_d = ((double *)src)[di];
            if (val_d >= INT32_MAX)
                ((int32_t *)dst)[di] = INT32_MAX;
            else if (val_d <= INT32_MIN)
                ((int32_t *)dst)[di] = INT32_MIN;
            else
                ((int32_t *)dst)[di] = (int32_t)val_d;
            break;
        case TL_FLOAT:
            val_f = ((float *)src)[di];
            if (val_f >= INT32_MAX)
                ((int32_t *)dst)[di] = INT32_MAX;
            else if (val_f <= INT32_MIN)
                ((int32_t *)dst)[di] = INT32_MIN;
            else
                ((int32_t *)dst)[di] = (int32_t)val_f;
            break;
        case TL_INT32:
            ((int32_t *)dst)[di] = ((int32_t *)src)[di];
            break;
        case TL_INT16:
            ((int32_t *)dst)[di] = (int32_t)((int16_t *)src)[di];
            break;
        case TL_INT8:
            ((int32_t *)dst)[di] = (int32_t)((int8_t *)src)[di];
            break;
        case TL_UINT32:
            val_u32 = ((uint32_t *)src)[di];
            if (val_u32 >= INT32_MAX)
                ((int32_t *)dst)[di] = INT32_MAX;
            else
                ((int32_t *)dst)[di] = (int32_t)val_u32;
            break;
        case TL_UINT16:
            ((int32_t *)dst)[di] = (int32_t)((uint16_t *)src)[di];
            break;
        case TL_UINT8:
            ((int32_t *)dst)[di] = (int32_t)((uint8_t *)src)[di];
            break;
        case TL_BOOL:
            ((int32_t *)dst)[di] = (int32_t)((tl_bool_t *)src)[di];
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_INT16:
        switch (dtype_s) {
        case TL_DOUBLE:
            val_d = ((double *)src)[di];
            if (val_d >= INT16_MAX)
                ((int16_t *)dst)[di] = INT16_MAX;
            else if (val_d <= INT16_MIN)
                ((int16_t *)dst)[di] = INT16_MIN;
            else
                ((int16_t *)dst)[di] = (int16_t)val_d;
            break;
        case TL_FLOAT:
            val_f = ((float *)src)[di];
            if (val_f >= INT16_MAX)
                ((int16_t *)dst)[di] = INT16_MAX;
            else if (val_f <= INT16_MIN)
                ((int16_t *)dst)[di] = INT16_MIN;
            else
                ((int16_t *)dst)[di] = (int16_t)val_f;
            break;
        case TL_INT32:
            val_i32 = ((int32_t *)src)[di];
            if (val_i32 >= INT16_MAX)
                ((int16_t *)dst)[di] = INT16_MAX;
            else if (val_i32 <= INT16_MIN)
                ((int16_t *)dst)[di] = INT16_MIN;
            else
                ((int16_t *)dst)[di] = (int16_t)val_i32;
            break;
        case TL_INT16:
            ((int16_t *)dst)[di] = ((int16_t *)src)[di];
            break;
        case TL_INT8:
            ((int16_t *)dst)[di] = (int16_t)((int8_t *)src)[di];
            break;
        case TL_UINT32:
            val_u32 = ((uint32_t *)src)[di];
            if (val_u32 >= INT16_MAX)
                ((int16_t *)dst)[di] = INT16_MAX;
            else
                ((int16_t *)dst)[di] = (int16_t)val_u32;
            break;
        case TL_UINT16:
            val_u16 = ((uint16_t *)src)[di];
            if (val_u16 >= INT16_MAX)
                ((int16_t *)dst)[di] = INT16_MAX;
            else
                ((int16_t *)dst)[di] = (int16_t)val_u16;
            break;
        case TL_UINT8:
            ((int16_t *)dst)[di] = (int16_t)((uint8_t *)src)[di];
            break;
        case TL_BOOL:
            ((int16_t *)dst)[di] = (int16_t)((tl_bool_t *)src)[di];
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_INT8:
        switch (dtype_s) {
        case TL_DOUBLE:
            val_d = ((double *)src)[di];
            if (val_d >= INT8_MAX)
                ((int8_t *)dst)[di] = INT8_MAX;
            else if (val_d <= INT8_MIN)
                ((int8_t *)dst)[di] = INT8_MIN;
            else
                ((int8_t *)dst)[di] = (int8_t)val_d;
            break;
        case TL_FLOAT:
            val_f = ((float *)src)[di];
            if (val_f >= INT8_MAX)
                ((int8_t *)dst)[di] = INT8_MAX;
            else if (val_f <= INT8_MIN)
                ((int8_t *)dst)[di] = INT8_MIN;
            else
                ((int8_t *)dst)[di] = (int8_t)val_f;
            break;
        case TL_INT32:
            val_i32 = ((int32_t *)src)[di];
            if (val_i32 >= INT8_MAX)
                ((int8_t *)dst)[di] = INT8_MAX;
            else if (val_i32 <= INT8_MIN)
                ((int8_t *)dst)[di] = INT8_MIN;
            else
                ((int8_t *)dst)[di] = (int8_t)val_i32;
            break;
        case TL_INT16:
            val_i16 = ((int16_t *)src)[di];
            if (val_i16 >= INT8_MAX)
                ((int8_t *)dst)[di] = INT8_MAX;
            else if (val_i16 <= INT8_MIN)
                ((int8_t *)dst)[di] = INT8_MIN;
            else
                ((int8_t *)dst)[di] = (int8_t)val_i16;
            break;
        case TL_INT8:
            ((int8_t *)dst)[di] = ((int8_t *)src)[di];
            break;
        case TL_UINT32:
            val_u32 = ((uint32_t *)src)[di];
            if (val_u32 >= INT8_MAX)
                ((int8_t *)dst)[di] = INT8_MAX;
            else
                ((int8_t *)dst)[di] = (int8_t)val_u32;
            break;
        case TL_UINT16:
            val_u16 = ((uint16_t *)src)[di];
            if (val_u16 >= INT8_MAX)
                ((int8_t *)dst)[di] = INT8_MAX;
            else
                ((int8_t *)dst)[di] = (int8_t)val_u16;
            break;
        case TL_UINT8:
            val_u8 = ((uint8_t *)src)[di];
            if (val_u8 >= INT8_MAX)
                ((int8_t *)dst)[di] = INT8_MAX;
            else
                ((int8_t *)dst)[di] = (int8_t)val_u8;
            break;
        case TL_BOOL:
            ((int8_t *)dst)[di] = (int8_t)((tl_bool_t *)src)[di];
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_UINT32:
        switch (dtype_s) {
        case TL_DOUBLE:
            val_d = ((double *)src)[di];
            if (val_d >= UINT32_MAX)
                ((uint32_t *)dst)[di] = UINT32_MAX;
            else if (val_d < 0)
                ((uint32_t *)dst)[di] = 0;
            else
                ((uint32_t *)dst)[di] = (uint32_t)val_d;
            break;
        case TL_FLOAT:
            val_f = ((float *)src)[di];
            if (val_f >= UINT32_MAX)
                ((uint32_t *)dst)[di] = UINT32_MAX;
            else if (val_f < 0)
                ((uint32_t *)dst)[di] = 0;
            else
                ((uint32_t *)dst)[di] = (uint32_t)val_f;
            break;
        case TL_INT32:
            val_i32 = ((int32_t *)src)[di];
            if (val_i32 >= 0)
                ((uint32_t *)dst)[di] = (uint32_t)val_i32;
            else
                ((uint32_t *)dst)[di] = 0;
            break;
        case TL_INT16:
            val_i16 = ((int16_t *)src)[di];
            if (val_i16 >= 0)
                ((uint32_t *)dst)[di] = (uint32_t)val_i16;
            else
                ((uint32_t *)dst)[di] = 0;
            break;
        case TL_INT8:
            val_i8 = ((int8_t *)src)[di];
            if (val_i8 >= 0)
                ((uint32_t *)dst)[di] = (uint32_t)val_i8;
            else
                ((uint32_t *)dst)[di] = 0;
            break;
        case TL_UINT32:
            ((uint32_t *)dst)[di] = ((uint32_t *)src)[di];
            break;
        case TL_UINT16:
            ((uint32_t *)dst)[di] = (uint32_t)((uint16_t *)src)[di];
            break;
        case TL_UINT8:
            ((uint32_t *)dst)[di] = (uint32_t)((uint8_t *)src)[di];
            break;
        case TL_BOOL:
            ((uint32_t *)dst)[di] = (uint32_t)((tl_bool_t *)src)[di];
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_UINT16:
        switch (dtype_s) {
        case TL_DOUBLE:
            val_d = ((double *)src)[di];
            if (val_d >= UINT16_MAX)
                ((uint16_t *)dst)[di] = UINT16_MAX;
            else if (val_d < 0)
                ((uint16_t *)dst)[di] = 0;
            else
                ((uint16_t *)dst)[di] = (uint16_t)val_d;
            break;
        case TL_FLOAT:
            val_f = ((float *)src)[di];
            if (val_f >= UINT16_MAX)
                ((uint16_t *)dst)[di] = UINT16_MAX;
            else if (val_f < 0)
                ((uint16_t *)dst)[di] = 0;
            else
                ((uint16_t *)dst)[di] = (uint16_t)val_f;
            break;
        case TL_INT32:
            val_i32 = ((int32_t *)src)[di];
            if (val_i32 >= UINT16_MAX)
                ((uint16_t *)dst)[di] = UINT16_MAX;
            else if (val_i32 < 0)
                ((uint16_t *)dst)[di] = 0;
            else
                ((uint16_t *)dst)[di] = (uint16_t)val_i32;
            break;
        case TL_INT16:
            val_i16 = ((int16_t *)src)[di];
            if (val_i16 >= 0)
                ((uint16_t *)dst)[di] = (uint16_t)val_i16;
            else
                ((uint16_t *)dst)[di] = 0;
            break;
        case TL_INT8:
            val_i8 = ((int8_t *)src)[di];
            if (val_i8 >= 0)
                ((uint16_t *)dst)[di] = (uint16_t)val_i8;
            else
                ((uint16_t *)dst)[di] = 0;
            break;
        case TL_UINT32:
            val_u32 = ((uint32_t *)src)[di];
            if (val_u32 >= UINT16_MAX)
                ((uint16_t *)dst)[di] = UINT16_MAX;
            else
                ((uint16_t *)dst)[di] = (uint16_t)val_u32;
            break;
        case TL_UINT16:
            ((uint16_t *)dst)[di] = ((uint16_t *)src)[di];
            break;
        case TL_UINT8:
            ((uint16_t *)dst)[di] = (uint16_t)((uint8_t *)src)[di];
            break;
        case TL_BOOL:
            ((uint16_t *)dst)[di] = (uint16_t)((tl_bool_t *)src)[di];
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_UINT8:
        switch (dtype_s) {
        case TL_DOUBLE:
            val_d = ((double *)src)[di];
            if (val_d >= UINT8_MAX)
                ((uint8_t *)dst)[di] = UINT8_MAX;
            else if (val_d < 0)
                ((uint8_t *)dst)[di] = 0;
            else
                ((uint8_t *)dst)[di] = (uint8_t)val_d;
            break;
        case TL_FLOAT:
            val_f = ((float *)src)[di];
            if (val_f >= UINT8_MAX)
                ((uint8_t *)dst)[di] = UINT8_MAX;
            else if (val_f < 0)
                ((uint8_t *)dst)[di] = 0;
            else
                ((uint8_t *)dst)[di] = (uint8_t)val_f;
            break;
        case TL_INT32:
            val_i32 = ((int32_t *)src)[di];
            if (val_i32 >= UINT8_MAX)
                ((uint8_t *)dst)[di] = UINT8_MAX;
            else if (val_i32 < 0)
                ((uint8_t *)dst)[di] = 0;
            else
                ((uint8_t *)dst)[di] = (uint8_t)val_i32;
            break;
        case TL_INT16:
            val_i16 = ((int16_t *)src)[di];
            if (val_i16 >= UINT8_MAX)
                ((uint8_t *)dst)[di] = UINT8_MAX;
            else if (val_i16 < 0)
                ((uint8_t *)dst)[di] = 0;
            else
                ((uint8_t *)dst)[di] = (uint8_t)val_i16;
            break;
        case TL_INT8:
            val_i8 = ((int8_t *)src)[di];
            if (val_i8 >= 0)
                ((uint8_t *)dst)[di] = (uint8_t)val_i8;
            else
                ((uint8_t *)dst)[di] = 0;
            break;
        case TL_UINT32:
            val_u32 = ((uint32_t *)src)[di];
            if (val_u32 >= UINT8_MAX)
                ((uint8_t *)dst)[di] = UINT8_MAX;
            else
                ((uint8_t *)dst)[di] = (uint8_t)val_u32;
            break;
        case TL_UINT16:
            val_u16 = ((uint16_t *)src)[di];
            if (val_u16 >= UINT8_MAX)
                ((uint8_t *)dst)[di] = UINT8_MAX;
            else
                ((uint8_t *)dst)[di] = (uint8_t)val_u16;
            break;
        case TL_UINT8:
            ((uint8_t *)dst)[di] = ((uint8_t *)src)[di];
            break;
        case TL_BOOL:
            ((uint8_t *)dst)[di] = (uint8_t)((tl_bool_t *)src)[di];
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_BOOL:
        switch (dtype_s) {
        case TL_DOUBLE:
            val_d = ((double *)src)[di];
            if (val_d > 0 || val_d < 0)
                ((tl_bool_t *)dst)[di] = TL_TRUE;
            else
                ((tl_bool_t *)dst)[di] = TL_FALSE;
            break;
        case TL_FLOAT:
            val_f = ((float *)src)[di];
            if (val_f > 0 || val_f < 0)
                ((tl_bool_t *)dst)[di] = TL_TRUE;
            else
                ((tl_bool_t *)dst)[di] = TL_FALSE;
            break;
        case TL_INT32:
            val_i32 = ((int32_t *)src)[di];
            if (val_i32)
                ((tl_bool_t *)dst)[di] = TL_TRUE;
            else
                ((tl_bool_t *)dst)[di] = TL_FALSE;
            break;
        case TL_INT16:
            val_i16 = ((int16_t *)src)[di];
            if (val_i16)
                ((tl_bool_t *)dst)[di] = TL_TRUE;
            else
                ((tl_bool_t *)dst)[di] = TL_FALSE;
            break;
        case TL_INT8:
            val_i8 = ((int8_t *)src)[di];
            if (val_i8)
                ((tl_bool_t *)dst)[di] = TL_TRUE;
            else
                ((tl_bool_t *)dst)[di] = TL_FALSE;
            break;
        case TL_UINT32:
            val_u32 = ((uint32_t *)src)[di];
            if (val_u32)
                ((tl_bool_t *)dst)[di] = TL_TRUE;
            else
                ((tl_bool_t *)dst)[di] = TL_FALSE;
            break;
        case TL_UINT16:
            val_u16 = ((uint16_t *)src)[di];
            if (val_u16)
                ((tl_bool_t *)dst)[di] = TL_TRUE;
            else
                ((tl_bool_t *)dst)[di] = TL_FALSE;
            break;
        case TL_UINT8:
            val_u8 = ((uint8_t *)src)[di];
            if (val_u8)
                ((tl_bool_t *)dst)[di] = TL_TRUE;
            else
                ((tl_bool_t *)dst)[di] = TL_FALSE;
            break;
        case TL_BOOL:
            ((tl_bool_t *)dst)[di] = ((tl_bool_t *)src)[di];
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    default:
        assert(0 && "unsupported tl_dtype");
        break;
    }
}

TL_EXPORT tl_tensor *tl_tensor_convert_cuda(const tl_tensor *src, tl_tensor *dst, tl_dtype dtype_d)
{
    tl_dtype dtype_s;
    int thread_num, block_num;

    assert(src && tl_is_device_mem(src->data));
    if (dst) {
        assert(tl_is_device_mem(dst->data));
        assert(tl_tensor_issameshape(src, dst));
        assert(dst->dtype == dtype_d);
    } else {
        dst = tl_tensor_zeros_cuda(src->ndim, src->dims, dtype_d);
    }

    dtype_s = src->dtype;
    thread_num = dst->len;
    block_num = BLOCK_NUM(BLOCK_SIZE, thread_num);
    convert_kernel<<<block_num, BLOCK_SIZE>>>(src->data, dst->data, dtype_s, dtype_d, BLOCK_SIZE,
                                                thread_num);
    tl_cuda_device_sync();

    return dst;
}
