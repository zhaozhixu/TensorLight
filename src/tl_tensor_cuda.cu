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
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "tl_tensor.h"
#include "tl_util.h"

#define MAX_THREADS 1024
#define BLOCK_SIZE MAX_THREADS
#define BLOCK_NUM(bs, tn) (((tn) + (bs)-1) / (bs))
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

static unsigned int next_pow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

static inline __device__ int get_index(const int *ids, int ndim, const int *dims)
{
    int i, id;
    for (i = 0, id = ids[0]; i < ndim - 1; i++)
        id = dims[i + 1] * id + ids[i + 1];
    return id;
}

static inline __device__ void get_coords(int id, int *ids, int ndim, const int *dims)
{
    for (int i = ndim - 1; i >= 0; i--) {
        ids[i] = id % dims[i];
        id /= dims[i];
    }
}

static __device__ void convert_device(void *pd, tl_dtype dtype_d, const void *ps, tl_dtype dtype_s)
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

void tl_tensor_free_data_too_cuda(tl_tensor *t)
{
    if (!t)
        return;
    tl_free_cuda(t->data);
    tl_tensor_free(t);
}

tl_tensor *tl_tensor_zeros_cuda(int ndim, const int *dims, tl_dtype dtype)
{
    tl_tensor *t;
    size_t size;

    t = tl_tensor_create(NULL, ndim, dims, dtype);
    t->owner = t;
    size = t->len * tl_size_of(dtype);
    t->data = tl_alloc_cuda(size);
    tl_memset_cuda(t->data, 0, size);
    return t;
}

tl_tensor *tl_tensor_clone_h2d(const tl_tensor *src)
{
    void *data;
    tl_tensor *dst;

    assert(src);
    data = tl_clone_h2d(src->data, src->len * tl_size_of(src->dtype));
    dst = tl_tensor_create(data, src->ndim, src->dims, src->dtype);
    dst->owner = dst;
    return dst;
}

tl_tensor *tl_tensor_clone_d2h(const tl_tensor *src)
{
    void *data;
    tl_tensor *dst;

    assert(src);
    data = tl_clone_d2h(src->data, src->len * tl_size_of(src->dtype));
    dst = tl_tensor_create(data, src->ndim, src->dims, src->dtype);
    dst->owner = dst;
    return dst;
}

tl_tensor *tl_tensor_clone_d2d(const tl_tensor *src)
{
    void *data;
    tl_tensor *dst;

    assert(src);
    data = tl_clone_d2d(src->data, src->len * tl_size_of(src->dtype));
    dst = tl_tensor_create(data, src->ndim, src->dims, src->dtype);
    dst->owner = dst;
    return dst;
}

tl_tensor *tl_tensor_repeat_h2d(const tl_tensor *src, int times)
{
    void *data;
    int *dims;
    tl_tensor *dst;

    assert(src);
    data = tl_repeat_h2d(src->data, src->len * tl_size_of(src->dtype), times);
    dims = (int *)tl_alloc(sizeof(int) * (src->ndim + 1));
    memmove(dims + 1, src->dims, sizeof(int) * (src->ndim));
    dims[0] = times;
    dst = tl_tensor_create(data, src->ndim + 1, dims, src->dtype);
    dst->owner = dst;
    tl_free(dims);
    return dst;
}

tl_tensor *tl_tensor_repeat_d2h(const tl_tensor *src, int times)
{
    void *data;
    int *dims;
    tl_tensor *dst;

    assert(src);
    data = tl_repeat_d2h(src->data, src->len * tl_size_of(src->dtype), times);
    dims = (int *)tl_alloc(sizeof(int) * (src->ndim + 1));
    memmove(dims + 1, src->dims, sizeof(int) * (src->ndim));
    dims[0] = times;
    dst = tl_tensor_create(data, src->ndim + 1, dims, src->dtype);
    dst->owner = dst;
    tl_free(dims);
    return dst;
}

tl_tensor *tl_tensor_repeat_d2d(const tl_tensor *src, int times)
{
    void *data;
    int *dims;
    tl_tensor *dst;

    assert(src);
    data = tl_repeat_d2d(src->data, src->len * tl_size_of(src->dtype), times);
    dims = (int *)tl_alloc(sizeof(int) * (src->ndim + 1));
    memmove(dims + 1, src->dims, sizeof(int) * (src->ndim));
    dims[0] = times;
    dst = tl_tensor_create(data, src->ndim + 1, dims, src->dtype);
    dst->owner = dst;
    tl_free(dims);
    return dst;
}

/* arrange at host, copy to device */
tl_tensor *tl_tensor_arange_cuda(double start, double stop, double step, tl_dtype dtype)
{
    int dims[1];
    void *data;
    tl_tensor *dst;
    double len, elem;
    size_t dsize;

    dsize = tl_size_of(dtype);
#ifdef TL_DEBUG
    double max_d, min_d;
    max_d = tl_dtype_max_double(dtype);
    min_d = tl_dtype_min_double(dtype);
    assert(start >= min_d && start <= max_d);
    assert(stop >= min_d && stop <= max_d);
    assert(step >= min_d && step <= max_d);
    assert(step != 0);
    assert(stop > start); /* TODO: expand to all possibilities */
#endif

    len = ceil((stop - start) / step);
    if (len > INT32_MAX)
        return NULL;

    dims[0] = (int)len;
    dst = tl_tensor_zeros_cuda(1, dims, dtype);
    data = tl_tensor_zeros(1, dims, dtype);
    for (int i = 0; i < dims[0]; i++) {
        elem = start + step * i;
        tl_convert(tl_padd(data, i, dsize), dtype, &elem, TL_DOUBLE);
    }
    tl_memcpy_h2d(dst->data, data, tl_size_of(dst->dtype) * dst->len);

    return dst;
}

template <typename T>
__global__ void rearange_kernel(T *dst, int len, double start, double step, int block_size,
                                int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;

    dst[di] = (T)(start + step * di);
}

void tl_tensor_rearange_cuda(tl_tensor *src, double start, double stop, double step)
{
    double len;

#ifdef TL_DEBUG
    double max_d, min_d;
    max_d = tl_dtype_max_double(dtype);
    min_d = tl_dtype_min_double(dtype);
    assert(start >= min_d && start <= max_d);
    assert(stop >= min_d && stop <= max_d);
    assert(step >= min_d && step <= max_d);
    assert(step != 0);
    assert(stop > start); /* TODO: expand to all possibilities */
#endif

    len = ceil((stop - start) / step);

    assert(len <= INT32_MAX);
    assert(src->ndim == 1);
    assert(src->len == (int)len);
    assert(src->data);

    int thread_num, block_num;
    thread_num = src->len;
    block_num = BLOCK_NUM(BLOCK_SIZE, thread_num);

    switch (src->dtype) {
    case TL_DOUBLE:
        rearange_kernel<<<block_num, BLOCK_SIZE>>>((double *)src->data, src->len, start, step,
                                                     BLOCK_SIZE, thread_num);
        break;
    case TL_FLOAT:
        rearange_kernel<<<block_num, BLOCK_SIZE>>>((float *)src->data, src->len, start, step,
                                                     BLOCK_SIZE, thread_num);
        break;
    case TL_INT32:
        rearange_kernel<<<block_num, BLOCK_SIZE>>>((int32_t *)src->data, src->len, start, step,
                                                     BLOCK_SIZE, thread_num);
        break;
    case TL_INT16:
        rearange_kernel<<<block_num, BLOCK_SIZE>>>((int16_t *)src->data, src->len, start, step,
                                                     BLOCK_SIZE, thread_num);
        break;
    case TL_INT8:
        rearange_kernel<<<block_num, BLOCK_SIZE>>>((int8_t *)src->data, src->len, start, step,
                                                     BLOCK_SIZE, thread_num);
        break;
    case TL_UINT32:
        rearange_kernel<<<block_num, BLOCK_SIZE>>>((uint32_t *)src->data, src->len, start, step,
                                                     BLOCK_SIZE, thread_num);
        break;
    case TL_UINT16:
        rearange_kernel<<<block_num, BLOCK_SIZE>>>((uint16_t *)src->data, src->len, start, step,
                                                     BLOCK_SIZE, thread_num);
        break;
    case TL_UINT8:
        rearange_kernel<<<block_num, BLOCK_SIZE>>>((uint8_t *)src->data, src->len, start, step,
                                                     BLOCK_SIZE, thread_num);
        break;
    case TL_BOOL:
        rearange_kernel<<<block_num, BLOCK_SIZE>>>((int *)src->data, src->len, start, step,
                                                     BLOCK_SIZE, thread_num);
        break;
    default:
        assert(0 && "unsupported tl_dtype");
        break;
    }
    tl_cuda_device_sync();
}

void tl_tensor_fprint_cuda(FILE *stream, const tl_tensor *t, const char *fmt)
{
    tl_tensor *t_host;

    t_host = tl_tensor_clone_d2h(t);
    tl_tensor_fprint(stream, t_host, fmt);
    tl_tensor_free_data_too(t_host);
}

void tl_tensor_print_cuda(const tl_tensor *t, const char *fmt)
{
    tl_tensor_fprint_cuda(stdout, t, fmt);
}

int tl_tensor_save_cuda(const char *file_name, const tl_tensor *t, const char *fmt)
{
    tl_tensor *t_host;
    int ret;

    t_host = tl_tensor_clone_d2h(t);
    ret = tl_tensor_save(file_name, t_host, fmt);
    tl_tensor_free_data_too(t_host);
    return ret;
}

tl_tensor *tl_tensor_zeros_slice_cuda(const tl_tensor *src, int axis, int len, tl_dtype dtype)
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

tl_tensor *tl_tensor_slice_cuda(const tl_tensor *src, tl_tensor *dst, int axis, int start, int len)
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

tl_tensor *tl_tensor_maxreduce_cuda(const tl_tensor *src, tl_tensor *dst, tl_tensor *arg, int axis)
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

tl_tensor *tl_tensor_elew_cuda(const tl_tensor *src1, const tl_tensor *src2, tl_tensor *dst,
                               tl_elew_op elew_op)
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

template <typename T>
static __global__ void dot_product_kernel(const T *src1, const T *src2, T *dst)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = src1[i] * src2[i] + src1[i + blockDim.x] * src2[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        dst[blockIdx.x] = sdata[0];
}

template <typename T>
static void call_dot_product_kernel(const T *src1, const T *src2, T *dst, int n, int threads,
                                    int blocks)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    int smem_size = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    dot_product_kernel<T><<<dimGrid, dimBlock, smem_size>>>(src1, src2, dst);
    tl_cuda_device_sync();
}

template <typename T> static __global__ void reduce_sum_kernel(const T *src, T *dst)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = src[i] + src[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        dst[blockIdx.x] = sdata[0];
}

template <typename T>
static void call_reduce_sum_kernel(const T *src, T *dst, int n, int threads, int blocks)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    int smem_size = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    reduce_sum_kernel<T><<<dimGrid, dimBlock, smem_size>>>(src, dst);
    tl_cuda_device_sync();
}

static void reduce_get_num_blocks_and_threads(int n, int *blocks, int *threads)
{
    *threads = (n < MAX_THREADS * 2) ? next_pow2((n + 1) / 2) : MAX_THREADS;
    *blocks = BLOCK_NUM(*threads * 2, n);
}

int tl_tensor_dot_product_cuda_ws_len(const tl_tensor *src)
{
    int num_threads, num_blocks;

    reduce_get_num_blocks_and_threads(src->len, &num_blocks, &num_threads);
    return num_blocks;
}

tl_tensor *tl_tensor_dot_product_cuda(const tl_tensor *src1, const tl_tensor *src2, tl_tensor *ws1,
                                      tl_tensor *ws2, tl_tensor *dst)
{
    int tmp_dim[1];

    assert(tl_tensor_issameshape(src1, src2));
    assert(tl_is_device_mem(src1->data) && tl_is_device_mem(src2->data));
    assert(src1->dtype == src2->dtype);
    if (dst) {
        assert(tl_is_device_mem(dst->data));
        assert(src1->dtype == dst->dtype);
    } else {
        tmp_dim[0] = 1;
        dst = tl_tensor_zeros_cuda(1, tmp_dim, src1->dtype);
    }

    int num_threads, num_blocks;
    reduce_get_num_blocks_and_threads(src1->len, &num_blocks, &num_threads);
    if (ws1) {
        assert(tl_is_device_mem(ws1->data));
        assert(ws1->ndim == 1);
        assert(ws1->dims[0] == num_blocks);
        assert(ws1->dtype == src1->dtype);
    } else {
        tmp_dim[0] = num_blocks;
        ws1 = tl_tensor_zeros_cuda(1, tmp_dim, src1->dtype);
    }
    if (ws2) {
        assert(tl_is_device_mem(ws2->data));
        assert(ws2->ndim == 1);
        assert(ws2->dims[0] == num_blocks);
        assert(ws2->dtype == src1->dtype);
    } else {
        tmp_dim[0] = num_blocks;
        ws2 = tl_tensor_zeros_cuda(1, tmp_dim, src1->dtype);
    }

    /*
     * Generated by tools/generic.pl with
     * $switchtype(src1->dtype, T)
     * $typenoset(T, TL_BOOL)
     * call_dot_product_kernel<T>((const T *)src1->data, (const T *)src2->data, (T *)ws1->data, src1->len, num_threads,
     * num_blocks);
     */
    switch (src1->dtype) {
    case TL_DOUBLE:
        call_dot_product_kernel<double>((const double *)src1->data, (const double *)src2->data,
                                        (double *)ws1->data, src1->len, num_threads, num_blocks);
        break;
    case TL_FLOAT:
        call_dot_product_kernel<float>((const float *)src1->data, (const float *)src2->data,
                                       (float *)ws1->data, src1->len, num_threads, num_blocks);
        break;
    case TL_INT16:
        call_dot_product_kernel<int16_t>((const int16_t *)src1->data, (const int16_t *)src2->data,
                                         (int16_t *)ws1->data, src1->len, num_threads, num_blocks);
        break;
    case TL_INT32:
        call_dot_product_kernel<int32_t>((const int32_t *)src1->data, (const int32_t *)src2->data,
                                         (int32_t *)ws1->data, src1->len, num_threads, num_blocks);
        break;
    case TL_INT8:
        call_dot_product_kernel<int8_t>((const int8_t *)src1->data, (const int8_t *)src2->data,
                                        (int8_t *)ws1->data, src1->len, num_threads, num_blocks);
        break;
    case TL_UINT16:
        call_dot_product_kernel<uint16_t>((const uint16_t *)src1->data,
                                          (const uint16_t *)src2->data, (uint16_t *)ws1->data,
                                          src1->len, num_threads, num_blocks);
        break;
    case TL_UINT32:
        call_dot_product_kernel<uint32_t>((const uint32_t *)src1->data,
                                          (const uint32_t *)src2->data, (uint32_t *)ws1->data,
                                          src1->len, num_threads, num_blocks);
        break;
    case TL_UINT8:
        call_dot_product_kernel<uint8_t>((const uint8_t *)src1->data, (const uint8_t *)src2->data,
                                         (uint8_t *)ws1->data, src1->len, num_threads, num_blocks);
        break;
    default:
        assert(0 && "unsupported dtype for src1->dtype");
        break;
    }

    int s = num_blocks;
    while (s > 1) {
        int threads = 0, blocks = 0;
        reduce_get_num_blocks_and_threads(s, &blocks, &threads);
        tl_memcpy_d2d(ws2->data, ws1->data, s * tl_size_of(ws1->dtype));
        /*
         * Generated by tools/generic.pl with
         * $switchtype(src1->dtype,T)
         * $typenoset(T, TL_BOOL)
         * call_reduce_sum_kernel<T>((const T *)ws2->data, (T *)ws1->data, s, threads, blocks);
         */
        switch (src1->dtype) {
        case TL_DOUBLE:
            call_reduce_sum_kernel<double>((const double *)ws2->data, (double *)ws1->data, s,
                                           threads, blocks);
            break;
        case TL_FLOAT:
            call_reduce_sum_kernel<float>((const float *)ws2->data, (float *)ws1->data, s, threads,
                                          blocks);
            break;
        case TL_INT16:
            call_reduce_sum_kernel<int16_t>((const int16_t *)ws2->data, (int16_t *)ws1->data, s,
                                            threads, blocks);
            break;
        case TL_INT32:
            call_reduce_sum_kernel<int32_t>((const int32_t *)ws2->data, (int32_t *)ws1->data, s,
                                            threads, blocks);
            break;
        case TL_INT8:
            call_reduce_sum_kernel<int8_t>((const int8_t *)ws2->data, (int8_t *)ws1->data, s,
                                           threads, blocks);
            break;
        case TL_UINT16:
            call_reduce_sum_kernel<uint16_t>((const uint16_t *)ws2->data, (uint16_t *)ws1->data, s,
                                             threads, blocks);
            break;
        case TL_UINT32:
            call_reduce_sum_kernel<uint32_t>((const uint32_t *)ws2->data, (uint32_t *)ws1->data, s,
                                             threads, blocks);
            break;
        case TL_UINT8:
            call_reduce_sum_kernel<uint8_t>((const uint8_t *)ws2->data, (uint8_t *)ws1->data, s,
                                            threads, blocks);
            break;
        default:
            assert(0 && "unsupported dtype for src1->dtype");
            break;
        }
        s = (s + (threads * 2 - 1)) / (threads * 2);
    }
    tl_memcpy_d2d(dst->data, ws1->data, tl_size_of(src1->dtype));

    if (!ws1)
        tl_tensor_free_data_too_cuda(ws1);
    if (!ws2)
        tl_tensor_free_data_too_cuda(ws2);

    return dst;
}

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

tl_tensor *tl_tensor_convert_cuda(const tl_tensor *src, tl_tensor *dst, tl_dtype dtype_d)
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

template <typename T>
static __global__ void transpose_kernel(T *src, T *dst, int ndim, int *s_dims, int *d_dims,
                                        int *axes, int block_size, int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;

    int s_ids[TL_MAXDIM], d_ids[TL_MAXDIM];
    get_coords(di, d_ids, ndim, d_dims);
    for (int i = 0; i < ndim; i++)
        s_ids[axes[i]] = d_ids[i];
    int si = get_index(s_ids, ndim, s_dims);

    dst[di] = src[si];
}

tl_tensor *tl_tensor_transpose_cuda(const tl_tensor *src, tl_tensor *dst, const int *axes)
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

tl_tensor *tl_tensor_lrelu_cuda(const tl_tensor *src, tl_tensor *dst, float negslope)
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
    get_coords(di, dst_coords, ndim, new_dims);
    for (int i = 0; i < ndim; i++) {
        rounded = roundf(((float)dst_coords[i] + 0.5) * scales[i] - 0.5);
        convert_device(&src_coords[i], TL_INT32, &rounded, TL_FLOAT);
    }
    si = get_index(src_coords, ndim, dims);
    dst[di] = src[si];
}

tl_tensor *tl_tensor_resize_cuda(const tl_tensor *src, tl_tensor *dst, const int *new_dims,
                                 tl_resize_type rtype)
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

__global__ void transform_bboxSQD_kernel(float *delta, float *anchor, float *res, int width,
                                         int height, int img_width, int img_height, int x_shift,
                                         int y_shift, int block_size, int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;

    /* TODO: FIXME: only support batch_size = 1 */
    float x_scale = 1.0 * img_width / width;
    float y_scale = 1.0 * img_height / height;
    float E = 2.718281828;

    /* take 4 elements from each of delta and anchor */
    int si = di * 4;
    float d[4] = { delta[si], delta[si + 1], delta[si + 2], delta[si + 3] };
    float a[4] = { anchor[si], anchor[si + 1], anchor[si + 2], anchor[si + 3] };
    /* compute and put 4 result elements to res,
       according to SqueezeDet's source code */

    /* TODO: don't know why (maybe the resize),
       always has some shift compared to groundtruth*/
    float cx = (a[0] + d[0] * a[2]) * x_scale + x_shift;
    float cy = (a[1] + d[1] * a[3]) * y_scale + y_shift;
    float w = (a[2] * (d[2] < 1 ? expf(d[2]) : d[2] * E)) * x_scale;
    float h = (a[3] * (d[3] < 1 ? expf(d[3]) : d[3] * E)) * y_scale;
    res[si] = min(max(cx - w * 0.5, 0), img_width - 1);
    res[si + 1] = min(max(cy - h * 0.5, 0), img_height - 1);
    res[si + 2] = max(min(cx + w * 0.5, img_width - 1), 0);
    res[si + 3] = max(min(cy + h * 0.5, img_height - 1), 0);
}

tl_tensor *tl_tensor_transform_bboxSQD_cuda(const tl_tensor *delta, const tl_tensor *anchor,
                                            tl_tensor *dst, int width, int height, int img_width,
                                            int img_height, int x_shift, int y_shift)
{
    assert(delta && anchor);
    assert(tl_is_device_mem(delta->data));
    assert(tl_tensor_issameshape(delta, anchor));
    assert(delta->dtype == TL_FLOAT);
    assert(delta->dtype == anchor->dtype);
    assert(delta->ndim == 5);
    assert(delta->dims[4] == 4);
    assert(width > 0 && height > 0 && img_width > 0 && img_height > 0);
    if (dst) {
        assert(dst->data);
        assert(tl_is_device_mem(dst->data));
        assert(tl_tensor_issameshape(delta, dst));
        assert(dst->dtype == delta->dtype);
    } else {
        dst = tl_tensor_zeros_cuda(delta->ndim, delta->dims, delta->dtype);
    }

    int i, thread_num, block_num;
    for (i = 0, thread_num = 1; i < dst->ndim - 1; i++)
        thread_num *= dst->dims[i];
    block_num = BLOCK_NUM(BLOCK_SIZE, thread_num);

    transform_bboxSQD_kernel<<<block_num, BLOCK_SIZE>>>((float *)delta->data,
                                                          (float *)anchor->data, (float *)dst->data,
                                                          width, height, img_width, img_height,
                                                          x_shift, y_shift, BLOCK_SIZE, thread_num);
    tl_cuda_device_sync();
    return dst;
}

/* TODO: strided access should be avoid */
template <typename T>
__global__ void pick1d_kernel(T *src, T *dst, int *idx, int stride, int block_size, int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;
    int si = idx[di];
    for (int i = 0; i < stride; i++)
        dst[di * stride + i] = src[si * stride + i];
}

tl_tensor *tl_tensor_pick1d_cuda(const tl_tensor *src, const tl_tensor *index, tl_tensor *dst,
                                 int stride, int len)
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

template <typename Ts, typename Td>
__global__ void submean_kernel(const Ts *src, Td *dst, tl_dtype dst_dtype, double mean1,
                               double mean2, double mean3, int H, int W, int C, int block_size,
                               int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;

    double mean[] = { mean1, mean2, mean3 };
    int src_coords[TL_MAXDIM];
    int dst_coords[TL_MAXDIM];
    int src_dims[] = { H, W, C };
    int dst_dims[] = { C, H, W };
    get_coords(di, dst_coords, 3, dst_dims);
    src_coords[0] = dst_coords[1];
    src_coords[1] = dst_coords[2];
    src_coords[2] = dst_coords[0];
    int si = get_index(src_coords, 3, src_dims);
    double res = (double)src[si] - mean[src_coords[2]];
    convert_device(&dst[di], dst_dtype, &res, TL_DOUBLE);
}

/* src: H*W*C, dst: C*H*W */
tl_tensor *tl_tensor_submean_cuda(const tl_tensor *src, tl_tensor *dst, const double *mean)
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
