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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>
#include <math.h>

#include "tl_tensor.h"
#include "tl_util.h"

#define MAX_THREADS 1024
#define BLOCK_SIZE MAX_THREADS
#define BLOCK_NUM(bs, tn) (((tn) + (bs)-1) / (bs))
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

static inline __device__ int tl_get_index_cu(const int *ids, int ndim, const int *dims)
{
    int i, id;
    for (i = 0, id = ids[0]; i < ndim - 1; i++)
        id = dims[i + 1] * id + ids[i + 1];
    return id;
}

static inline __device__ void tl_get_coords_cu(int id, int *ids, int ndim, const int *dims)
{
    for (int i = ndim - 1; i >= 0; i--) {
        ids[i] = id % dims[i];
        id /= dims[i];
    }
}

static __device__ void tl_convert_device_cu(void *pd, tl_dtype dtype_d, const void *ps, tl_dtype dtype_s)
{
    tl_check_dtype(dtype_d);
    tl_check_dtype(dtype_s);

    double val_d;
    float val_f;
    int64_t val_i64;
    int32_t val_i32;
    uint64_t val_u64;
    uint32_t val_u32;
    int16_t val_i16;
    uint16_t val_u16;
    int8_t val_i8;
    uint8_t val_u8;

    switch (dtype_d) {
    case TL_DOUBLE:
        switch (dtype_s) {
        case TL_DOUBLE:
            *(double *)pd = *(double *)ps;
            break;
        case TL_FLOAT:
            *(double *)pd = (double)*(float *)ps;
            break;
        case TL_INT64:
            *(double *)pd = (double)*(int64_t *)ps;
            break;
        case TL_INT32:
            *(double *)pd = (double)*(int32_t *)ps;
            break;
        case TL_INT16:
            *(double *)pd = (double)*(int16_t *)ps;
            break;
        case TL_INT8:
            *(double *)pd = (double)*(int8_t *)ps;
            break;
        case TL_UINT64:
            *(double *)pd = (double)*(uint64_t *)ps;
            break;
        case TL_UINT32:
            *(double *)pd = (double)*(uint32_t *)ps;
            break;
        case TL_UINT16:
            *(double *)pd = (double)*(uint16_t *)ps;
            break;
        case TL_UINT8:
            *(double *)pd = (double)*(uint8_t *)ps;
            break;
        case TL_BOOL:
            *(double *)pd = (double)*(tl_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_FLOAT:
        switch (dtype_s) {
        case TL_DOUBLE:
            val_d = *(double *)ps;
            if (val_d >= FLT_MAX)
                *(float *)pd = FLT_MAX;
            else if (val_d <= -FLT_MAX)
                *(float *)pd = -FLT_MAX;
            else
                *(float *)pd = (float)val_d;
            break;
        case TL_FLOAT:
            *(float *)pd = *(float *)ps;
            break;
        case TL_INT64:
            *(float *)pd = (float)*(int64_t *)ps;
            break;
        case TL_INT32:
            *(float *)pd = (float)*(int32_t *)ps;
            break;
        case TL_INT16:
            *(float *)pd = (float)*(int16_t *)ps;
            break;
        case TL_INT8:
            *(float *)pd = (float)*(int8_t *)ps;
            break;
        case TL_UINT64:
            *(float *)pd = (float)*(uint64_t *)ps;
            break;
        case TL_UINT32:
            *(float *)pd = (float)*(uint32_t *)ps;
            break;
        case TL_UINT16:
            *(float *)pd = (float)*(uint16_t *)ps;
            break;
        case TL_UINT8:
            *(float *)pd = (float)*(uint8_t *)ps;
            break;
        case TL_BOOL:
            *(float *)pd = (float)*(tl_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_INT64:
        switch (dtype_s) {
        case TL_DOUBLE:
            val_d = *(double *)ps;
            if (val_d >= INT64_MAX)
                *(int64_t *)pd = INT64_MAX;
            else if (val_d <= INT64_MIN)
                *(int64_t *)pd = INT64_MIN;
            else
                *(int64_t *)pd = (int64_t)val_d;
            break;
        case TL_FLOAT:
            val_f = *(float *)ps;
            if (val_f >= INT64_MAX)
                *(int64_t *)pd = INT64_MAX;
            else if (val_f <= INT64_MIN)
                *(int64_t *)pd = INT64_MIN;
            else
                *(int64_t *)pd = (int64_t)val_f;
            break;
        case TL_INT64:
            *(int64_t *)pd = *(int64_t *)ps;
            break;
        case TL_INT32:
            *(int64_t *)pd = (int64_t) * (int32_t *)ps;
            break;
        case TL_INT16:
            *(int64_t *)pd = (int64_t) * (int16_t *)ps;
            break;
        case TL_INT8:
            *(int64_t *)pd = (int64_t) * (int8_t *)ps;
            break;
        case TL_UINT64:
            val_u64 = *(uint64_t *)ps;
            if (val_u64 >= INT64_MAX)
                *(int64_t *)pd = INT64_MAX;
            else
                *(int64_t *)pd = (int64_t)val_u64;
            break;
        case TL_UINT32:
            *(int64_t *)pd = (int64_t) * (uint32_t *)ps;
            break;
        case TL_UINT16:
            *(int64_t *)pd = (int64_t) * (uint16_t *)ps;
            break;
        case TL_UINT8:
            *(int64_t *)pd = (int64_t) * (uint8_t *)ps;
            break;
        case TL_BOOL:
            *(int64_t *)pd = (int64_t) * (tl_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_INT32:
        switch (dtype_s) {
        case TL_DOUBLE:
            val_d = *(double *)ps;
            if (val_d >= INT32_MAX)
                *(int32_t *)pd = INT32_MAX;
            else if (val_d <= INT32_MIN)
                *(int32_t *)pd = INT32_MIN;
            else
                *(int32_t *)pd = (int32_t)val_d;
            break;
        case TL_FLOAT:
            val_f = *(float *)ps;
            if (val_f >= INT32_MAX)
                *(int32_t *)pd = INT32_MAX;
            else if (val_f <= INT32_MIN)
                *(int32_t *)pd = INT32_MIN;
            else
                *(int32_t *)pd = (int32_t)val_f;
            break;
        case TL_INT64:
            val_i64 = *(int64_t *)ps;
            if (val_i64 >= INT32_MAX)
                *(int32_t *)pd = INT32_MAX;
            else
                *(int32_t *)pd = (int32_t)val_u64;
            break;
        case TL_INT32:
            *(int32_t *)pd = *(int32_t *)ps;
            break;
        case TL_INT16:
            *(int32_t *)pd = (int32_t) * (int16_t *)ps;
            break;
        case TL_INT8:
            *(int32_t *)pd = (int32_t) * (int8_t *)ps;
            break;
        case TL_UINT64:
            val_u64 = *(uint64_t *)ps;
            if (val_u64 >= INT32_MAX)
                *(int32_t *)pd = INT32_MAX;
            else
                *(int32_t *)pd = (int32_t)val_u64;
            break;
        case TL_UINT32:
            val_u32 = *(uint32_t *)ps;
            if (val_u32 >= INT32_MAX)
                *(int32_t *)pd = INT32_MAX;
            else
                *(int32_t *)pd = (int32_t)val_u32;
            break;
        case TL_UINT16:
            /* printf("*ps = %d\n", *(uint16_t *)ps); */
            *(int32_t *)pd = (int32_t) * (uint16_t *)ps;
            /* printf("*pd = %d\n", *(int32_t *)pd); */
            break;
        case TL_UINT8:
            *(int32_t *)pd = (int32_t) * (uint8_t *)ps;
            break;
        case TL_BOOL:
            *(int32_t *)pd = (int32_t) * (tl_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_INT16:
        switch (dtype_s) {
        case TL_DOUBLE:
            val_d = *(double *)ps;
            if (val_d >= INT16_MAX)
                *(int16_t *)pd = INT16_MAX;
            else if (val_d <= INT16_MIN)
                *(int16_t *)pd = INT16_MIN;
            else
                *(int16_t *)pd = (int16_t)val_d;
            break;
        case TL_FLOAT:
            val_f = *(float *)ps;
            if (val_f >= INT16_MAX)
                *(int16_t *)pd = INT16_MAX;
            else if (val_f <= INT16_MIN)
                *(int16_t *)pd = INT16_MIN;
            else
                *(int16_t *)pd = (int16_t)val_f;
            break;
        case TL_INT64:
            val_i64 = *(int64_t *)ps;
            if (val_i64 >= INT16_MAX)
                *(int16_t *)pd = INT16_MAX;
            else if (val_i64 <= INT16_MIN)
                *(int16_t *)pd = INT16_MIN;
            else
                *(int16_t *)pd = (int16_t)val_i64;
            break;
        case TL_INT32:
            val_i32 = *(int32_t *)ps;
            if (val_i32 >= INT16_MAX)
                *(int16_t *)pd = INT16_MAX;
            else if (val_i32 <= INT16_MIN)
                *(int16_t *)pd = INT16_MIN;
            else
                *(int16_t *)pd = (int16_t)val_i32;
            break;
        case TL_INT16:
            *(int16_t *)pd = *(int16_t *)ps;
            break;
        case TL_INT8:
            *(int16_t *)pd = (int16_t) * (int8_t *)ps;
            break;
        case TL_UINT64:
            val_u64 = *(uint64_t *)ps;
            if (val_u64 >= INT16_MAX)
                *(int16_t *)pd = INT16_MAX;
            else
                *(int16_t *)pd = (int16_t)val_u64;
            break;
        case TL_UINT32:
            val_u32 = *(uint32_t *)ps;
            if (val_u32 >= INT16_MAX)
                *(int16_t *)pd = INT16_MAX;
            else
                *(int16_t *)pd = (int16_t)val_u32;
            break;
        case TL_UINT16:
            val_u16 = *(uint16_t *)ps;
            if (val_u16 >= INT16_MAX)
                *(int16_t *)pd = INT16_MAX;
            else
                *(int16_t *)pd = (int16_t)val_u16;
            break;
        case TL_UINT8:
            *(int16_t *)pd = (int16_t) * (uint8_t *)ps;
            break;
        case TL_BOOL:
            *(int16_t *)pd = (int16_t) * (tl_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_INT8:
        switch (dtype_s) {
        case TL_DOUBLE:
            val_d = *(double *)ps;
            if (val_d >= INT8_MAX)
                *(int8_t *)pd = INT8_MAX;
            else if (val_d <= INT8_MIN)
                *(int8_t *)pd = INT8_MIN;
            else
                *(int8_t *)pd = (int8_t)val_d;
            break;
        case TL_FLOAT:
            val_f = *(float *)ps;
            if (val_f >= INT8_MAX)
                *(int8_t *)pd = INT8_MAX;
            else if (val_f <= INT8_MIN)
                *(int8_t *)pd = INT8_MIN;
            else
                *(int8_t *)pd = (int8_t)val_f;
            break;
        case TL_INT64:
            val_i64 = *(int64_t *)ps;
            if (val_i64 >= INT8_MAX)
                *(int8_t *)pd = INT8_MAX;
            else if (val_i64 <= INT8_MIN)
                *(int8_t *)pd = INT8_MIN;
            else
                *(int8_t *)pd = (int8_t)val_i64;
            break;
        case TL_INT32:
            val_i32 = *(int32_t *)ps;
            if (val_i32 >= INT8_MAX)
                *(int8_t *)pd = INT8_MAX;
            else if (val_i32 <= INT8_MIN)
                *(int8_t *)pd = INT8_MIN;
            else
                *(int8_t *)pd = (int8_t)val_i32;
            break;
        case TL_INT16:
            val_i16 = *(int16_t *)ps;
            if (val_i16 >= INT8_MAX)
                *(int8_t *)pd = INT8_MAX;
            else if (val_i16 <= INT8_MIN)
                *(int8_t *)pd = INT8_MIN;
            else
                *(int8_t *)pd = (int8_t)val_i16;
            break;
        case TL_INT8:
            *(int8_t *)pd = *(int8_t *)ps;
            break;
        case TL_UINT32:
            val_u32 = *(uint32_t *)ps;
            if (val_u32 >= INT8_MAX)
                *(int8_t *)pd = INT8_MAX;
            else
                *(int8_t *)pd = (int8_t)val_u32;
            break;
        case TL_UINT64:
            val_u64 = *(uint64_t *)ps;
            if (val_u64 >= INT8_MAX)
                *(int8_t *)pd = INT8_MAX;
            else
                *(int8_t *)pd = (int8_t)val_u64;
            break;
        case TL_UINT16:
            val_u16 = *(uint16_t *)ps;
            if (val_u16 >= INT8_MAX)
                *(int8_t *)pd = INT8_MAX;
            else
                *(int8_t *)pd = (int8_t)val_u16;
            break;
        case TL_UINT8:
            val_u8 = *(uint8_t *)ps;
            if (val_u8 >= INT8_MAX)
                *(int8_t *)pd = INT8_MAX;
            else
                *(int8_t *)pd = (int8_t)val_u8;
            break;
        case TL_BOOL:
            *(int8_t *)pd = (int8_t) * (tl_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_UINT64:
        switch (dtype_s) {
        case TL_DOUBLE:
            val_d = *(double *)ps;
            if (val_d >= UINT64_MAX)
                *(uint64_t *)pd = UINT64_MAX;
            else if (val_d < 0)
                *(uint64_t *)pd = 0;
            else
                *(uint64_t *)pd = (uint64_t)val_d;
            break;
        case TL_FLOAT:
            val_f = *(float *)ps;
            if (val_f >= UINT64_MAX)
                *(uint64_t *)pd = UINT64_MAX;
            else if (val_f < 0)
                *(uint64_t *)pd = 0;
            else
                *(uint64_t *)pd = (uint64_t)val_f;
            break;
        case TL_INT64:
            val_i64 = *(int64_t *)ps;
            if (val_i64 >= 0)
                *(uint64_t *)pd = (uint64_t)val_i64;
            else
                *(uint64_t *)pd = 0;
            break;
        case TL_INT32:
            val_i32 = *(int32_t *)ps;
            if (val_i32 >= 0)
                *(uint64_t *)pd = (uint64_t)val_i32;
            else
                *(uint64_t *)pd = 0;
            break;
        case TL_INT16:
            val_i16 = *(int16_t *)ps;
            if (val_i16 >= 0)
                *(uint64_t *)pd = (uint64_t)val_i16;
            else
                *(uint64_t *)pd = 0;
            break;
        case TL_INT8:
            val_i8 = *(int8_t *)ps;
            if (val_i8 >= 0)
                *(uint64_t *)pd = (uint64_t)val_i8;
            else
                *(uint64_t *)pd = 0;
            break;
        case TL_UINT64:
            *(uint64_t *)pd = *(uint64_t *)ps;
            break;
        case TL_UINT32:
            *(uint64_t *)pd = *(uint32_t *)ps;
            break;
        case TL_UINT16:
            *(uint64_t *)pd = (uint64_t) * (uint16_t *)ps;
            break;
        case TL_UINT8:
            *(uint64_t *)pd = (uint64_t) * (uint8_t *)ps;
            break;
        case TL_BOOL:
            *(uint64_t *)pd = (uint64_t) * (tl_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_UINT32:
        switch (dtype_s) {
        case TL_DOUBLE:
            val_d = *(double *)ps;
            if (val_d >= UINT32_MAX)
                *(uint32_t *)pd = UINT32_MAX;
            else if (val_d < 0)
                *(uint32_t *)pd = 0;
            else
                *(uint32_t *)pd = (uint32_t)val_d;
            break;
        case TL_FLOAT:
            val_f = *(float *)ps;
            if (val_f >= UINT32_MAX)
                *(uint32_t *)pd = UINT32_MAX;
            else if (val_f < 0)
                *(uint32_t *)pd = 0;
            else
                *(uint32_t *)pd = (uint32_t)val_f;
            break;
        case TL_INT64:
            val_i64 = *(int64_t *)ps;
            if (val_i64 >= UINT32_MAX)
                *(uint32_t *)pd = UINT32_MAX;
            else if (val_i64 >= 0)
                *(uint32_t *)pd = (uint32_t)val_i64;
            else
                *(uint32_t *)pd = 0;
            break;
        case TL_INT32:
            val_i32 = *(int32_t *)ps;
            if (val_i32 >= 0)
                *(uint32_t *)pd = (uint32_t)val_i32;
            else
                *(uint32_t *)pd = 0;
            break;
        case TL_INT16:
            val_i16 = *(int16_t *)ps;
            if (val_i16 >= 0)
                *(uint32_t *)pd = (uint32_t)val_i16;
            else
                *(uint32_t *)pd = 0;
            break;
        case TL_INT8:
            val_i8 = *(int8_t *)ps;
            if (val_i8 >= 0)
                *(uint32_t *)pd = (uint32_t)val_i8;
            else
                *(uint32_t *)pd = 0;
            break;
        case TL_UINT64:
            val_u64 = *(uint64_t *)ps;
            if (val_u64 >= UINT32_MAX)
                *(uint32_t *)pd = UINT32_MAX;
            else
                *(uint32_t *)pd = (uint32_t)val_u64;
            break;
        case TL_UINT32:
            *(uint32_t *)pd = *(uint32_t *)ps;
            break;
        case TL_UINT16:
            *(uint32_t *)pd = (uint32_t) * (uint16_t *)ps;
            break;
        case TL_UINT8:
            *(uint32_t *)pd = (uint32_t) * (uint8_t *)ps;
            break;
        case TL_BOOL:
            *(uint32_t *)pd = (uint32_t) * (tl_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_UINT16:
        switch (dtype_s) {
        case TL_DOUBLE:
            val_d = *(double *)ps;
            if (val_d >= UINT16_MAX)
                *(uint16_t *)pd = UINT16_MAX;
            else if (val_d < 0)
                *(uint16_t *)pd = 0;
            else
                *(uint16_t *)pd = (uint16_t)val_d;
            break;
        case TL_FLOAT:
            val_f = *(float *)ps;
            if (val_f >= UINT16_MAX)
                *(uint16_t *)pd = UINT16_MAX;
            else if (val_f < 0)
                *(uint16_t *)pd = 0;
            else
                *(uint16_t *)pd = (uint16_t)val_f;
            break;
        case TL_INT64:
            val_i64 = *(int64_t *)ps;
            if (val_i64 >= UINT16_MAX)
                *(uint16_t *)pd = UINT16_MAX;
            else if (val_i64 < 0)
                *(uint16_t *)pd = 0;
            else
                *(uint16_t *)pd = (uint16_t)val_i64;
            break;
        case TL_INT32:
            val_i32 = *(int32_t *)ps;
            if (val_i32 >= UINT16_MAX)
                *(uint16_t *)pd = UINT16_MAX;
            else if (val_i32 < 0)
                *(uint16_t *)pd = 0;
            else
                *(uint16_t *)pd = (uint16_t)val_i32;
            break;
        case TL_INT16:
            val_i16 = *(int16_t *)ps;
            if (val_i16 >= 0)
                *(uint16_t *)pd = (uint16_t)val_i16;
            else
                *(uint16_t *)pd = 0;
            break;
        case TL_INT8:
            val_i8 = *(int8_t *)ps;
            if (val_i8 >= 0)
                *(uint16_t *)pd = (uint16_t)val_i8;
            else
                *(uint16_t *)pd = 0;
            break;
        case TL_UINT64:
            val_u64 = *(uint64_t *)ps;
            if (val_u64 >= UINT16_MAX)
                *(uint16_t *)pd = UINT16_MAX;
            else
                *(uint16_t *)pd = (uint16_t)val_u64;
            break;
        case TL_UINT32:
            val_u32 = *(uint32_t *)ps;
            if (val_u32 >= UINT16_MAX)
                *(uint16_t *)pd = UINT16_MAX;
            else
                *(uint16_t *)pd = (uint16_t)val_u32;
            break;
        case TL_UINT16:
            *(uint16_t *)pd = *(uint16_t *)ps;
            break;
        case TL_UINT8:
            *(uint16_t *)pd = (uint16_t) * (uint8_t *)ps;
            break;
        case TL_BOOL:
            *(uint16_t *)pd = (uint16_t) * (tl_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_UINT8:
        switch (dtype_s) {
        case TL_DOUBLE:
            val_d = *(double *)ps;
            if (val_d >= UINT8_MAX)
                *(uint8_t *)pd = UINT8_MAX;
            else if (val_d < 0)
                *(uint8_t *)pd = 0;
            else
                *(uint8_t *)pd = (uint8_t)val_d;
            break;
        case TL_FLOAT:
            val_f = *(float *)ps;
            if (val_f >= UINT8_MAX)
                *(uint8_t *)pd = UINT8_MAX;
            else if (val_f < 0)
                *(uint8_t *)pd = 0;
            else
                *(uint8_t *)pd = (uint8_t)val_f;
            break;
        case TL_INT64:
            val_i64 = *(int64_t *)ps;
            if (val_i64 >= UINT8_MAX)
                *(uint8_t *)pd = UINT8_MAX;
            else if (val_i64 < 0)
                *(uint8_t *)pd = 0;
            else
                *(uint8_t *)pd = (uint8_t)val_i64;
            break;
        case TL_INT32:
            val_i32 = *(int32_t *)ps;
            if (val_i32 >= UINT8_MAX)
                *(uint8_t *)pd = UINT8_MAX;
            else if (val_i32 < 0)
                *(uint8_t *)pd = 0;
            else
                *(uint8_t *)pd = (uint8_t)val_i32;
            break;
        case TL_INT16:
            val_i16 = *(int16_t *)ps;
            if (val_i16 >= UINT8_MAX)
                *(uint8_t *)pd = UINT8_MAX;
            else if (val_i16 < 0)
                *(uint8_t *)pd = 0;
            else
                *(uint8_t *)pd = (uint8_t)val_i16;
            break;
        case TL_INT8:
            val_i8 = *(int8_t *)ps;
            if (val_i8 >= 0)
                *(uint8_t *)pd = (uint8_t)val_i8;
            else
                *(uint8_t *)pd = 0;
            break;
        case TL_UINT64:
            val_u64 = *(uint64_t *)ps;
            if (val_u64 >= UINT8_MAX)
                *(uint8_t *)pd = UINT8_MAX;
            else
                *(uint8_t *)pd = (uint8_t)val_u64;
            break;
        case TL_UINT32:
            val_u32 = *(uint32_t *)ps;
            if (val_u32 >= UINT8_MAX)
                *(uint8_t *)pd = UINT8_MAX;
            else
                *(uint8_t *)pd = (uint8_t)val_u32;
            break;
        case TL_UINT16:
            val_u16 = *(uint16_t *)ps;
            if (val_u16 >= UINT8_MAX)
                *(uint8_t *)pd = UINT8_MAX;
            else
                *(uint8_t *)pd = (uint8_t)val_u16;
            break;
        case TL_UINT8:
            *(uint8_t *)pd = *(uint8_t *)ps;
            break;
        case TL_BOOL:
            *(uint8_t *)pd = (uint8_t) * (tl_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_BOOL:
        switch (dtype_s) {
        case TL_DOUBLE:
            val_d = *(double *)ps;
            if (val_d > 0 || val_d < 0)
                *(tl_bool_t *)pd = TL_TRUE;
            else
                *(tl_bool_t *)pd = TL_FALSE;
            break;
        case TL_FLOAT:
            val_f = *(float *)ps;
            if (val_f > 0 || val_f < 0)
                *(tl_bool_t *)pd = TL_TRUE;
            else
                *(tl_bool_t *)pd = TL_FALSE;
            break;
        case TL_INT64:
            val_i64 = *(int64_t *)ps;
            if (val_i64)
                *(tl_bool_t *)pd = TL_TRUE;
            else
                *(tl_bool_t *)pd = TL_FALSE;
            break;
        case TL_INT32:
            val_i32 = *(int32_t *)ps;
            if (val_i32)
                *(tl_bool_t *)pd = TL_TRUE;
            else
                *(tl_bool_t *)pd = TL_FALSE;
            break;
        case TL_INT16:
            val_i16 = *(int16_t *)ps;
            if (val_i16)
                *(tl_bool_t *)pd = TL_TRUE;
            else
                *(tl_bool_t *)pd = TL_FALSE;
            break;
        case TL_INT8:
            val_i8 = *(int8_t *)ps;
            if (val_i8)
                *(tl_bool_t *)pd = TL_TRUE;
            else
                *(tl_bool_t *)pd = TL_FALSE;
            break;
        case TL_UINT64:
            val_u64 = *(uint64_t *)ps;
            if (val_u64)
                *(tl_bool_t *)pd = TL_TRUE;
            else
                *(tl_bool_t *)pd = TL_FALSE;
            break;
        case TL_UINT32:
            val_u32 = *(uint32_t *)ps;
            if (val_u32)
                *(tl_bool_t *)pd = TL_TRUE;
            else
                *(tl_bool_t *)pd = TL_FALSE;
            break;
        case TL_UINT16:
            val_u16 = *(uint16_t *)ps;
            if (val_u16)
                *(tl_bool_t *)pd = TL_TRUE;
            else
                *(tl_bool_t *)pd = TL_FALSE;
            break;
        case TL_UINT8:
            val_u8 = *(uint8_t *)ps;
            if (val_u8)
                *(tl_bool_t *)pd = TL_TRUE;
            else
                *(tl_bool_t *)pd = TL_FALSE;
            break;
        case TL_BOOL:
            *(tl_bool_t *)pd = *(tl_bool_t *)ps;
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
