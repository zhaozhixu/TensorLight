/*
 * Copyright (c) 2018 Zhao Zhixu
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

#include <stdint.h>
#include <float.h>
#include "test_tensorlight.h"
#include "../src/tl_type.h"

static void setup(void)
{
}

static void teardown(void)
{
}

START_TEST(test_tl_cast)
{
     double val_d;
     float val_f;
     int32_t val_i32;
     uint32_t val_u32;
     int16_t val_i16;
     uint16_t val_u16;
     int8_t val_i8;
     uint8_t val_u8;
     tl_bool_t val_b;

     const double val_d_max = DBL_MAX;
     const double val_d_min = -DBL_MAX;
     const double val_d_normal = 1.0;
     const float val_f_max = FLT_MAX;
     const float val_f_min = -FLT_MAX;
     const float val_f_normal = 1.0;
     const int32_t val_i32_max = INT32_MAX;
     const int32_t val_i32_min = INT32_MIN;
     const int32_t val_i32_normal = 1;
     const uint32_t val_u32_max = UINT32_MAX;
     const uint32_t val_u32_min = 0;
     const uint32_t val_u32_normal = 1;
     const int16_t val_i16_max = INT16_MAX;
     const int16_t val_i16_min = INT16_MIN;
     const int16_t val_i16_normal = 1;
     const uint16_t val_u16_max = UINT16_MAX;
     const uint16_t val_u16_min = 0;
     const uint16_t val_u16_normal = 1;
     const int8_t val_i8_max = INT8_MAX;
     const int8_t val_i8_min = INT8_MIN;
     const int8_t val_i8_normal = 1;
     const uint8_t val_u8_max = UINT8_MAX;
     const uint8_t val_u8_min = 0;
     const uint8_t val_u8_normal = 1;
     const tl_bool_t val_b_true = TL_TRUE;
     const tl_bool_t val_b_false = TL_FALSE;

     /* TL_DOUBLE */
     tl_cast(&val_d, TL_DOUBLE, &val_d_max, TL_DOUBLE);
     ck_assert(val_d == val_d_max);
     tl_cast(&val_d, TL_DOUBLE, &val_d_min, TL_DOUBLE);
     ck_assert(val_d == val_d_min);
     tl_cast(&val_d, TL_DOUBLE, &val_d_normal, TL_DOUBLE);
     ck_assert(val_d == val_d_normal);
     tl_cast(&val_d, TL_DOUBLE, &val_f_max, TL_FLOAT);
     ck_assert(val_d == (double)val_f_max);
     tl_cast(&val_d, TL_DOUBLE, &val_f_min, TL_FLOAT);
     ck_assert(val_d == (double)val_f_min);
     tl_cast(&val_d, TL_DOUBLE, &val_f_normal, TL_FLOAT);
     ck_assert(val_d == (double)val_f_normal);
     tl_cast(&val_d, TL_DOUBLE, &val_i32_max, TL_INT32);
     ck_assert(val_d == (double)val_i32_max);
     tl_cast(&val_d, TL_DOUBLE, &val_i32_min, TL_INT32);
     ck_assert(val_d == (double)val_i32_min);
     tl_cast(&val_d, TL_DOUBLE, &val_i32_normal, TL_INT32);
     ck_assert(val_d == (double)val_i32_normal);
     tl_cast(&val_d, TL_DOUBLE, &val_i16_max, TL_INT16);
     ck_assert(val_d == (double)val_i16_max);
     tl_cast(&val_d, TL_DOUBLE, &val_i16_min, TL_INT16);
     ck_assert(val_d == (double)val_i16_min);
     tl_cast(&val_d, TL_DOUBLE, &val_i16_normal, TL_INT16);
     ck_assert(val_d == (double)val_i16_normal);
     tl_cast(&val_d, TL_DOUBLE, &val_i8_max, TL_INT8);
     ck_assert(val_d == (double)val_i8_max);
     tl_cast(&val_d, TL_DOUBLE, &val_i8_min, TL_INT8);
     ck_assert(val_d == (double)val_i8_min);
     tl_cast(&val_d, TL_DOUBLE, &val_i8_normal, TL_INT8);
     ck_assert(val_d == (double)val_i8_normal);
     tl_cast(&val_d, TL_DOUBLE, &val_u32_max, TL_UINT32);
     ck_assert(val_d == (double)val_u32_max);
     tl_cast(&val_d, TL_DOUBLE, &val_u32_normal, TL_UINT32);
     ck_assert(val_d == (double)val_u32_normal);
     tl_cast(&val_d, TL_DOUBLE, &val_u16_max, TL_UINT16);
     ck_assert(val_d == (double)val_u16_max);
     tl_cast(&val_d, TL_DOUBLE, &val_u16_normal, TL_UINT16);
     ck_assert(val_d == (double)val_u16_normal);
     tl_cast(&val_d, TL_DOUBLE, &val_u8_max, TL_UINT8);
     ck_assert(val_d == (double)val_u8_max);
     tl_cast(&val_d, TL_DOUBLE, &val_u8_normal, TL_UINT8);
     ck_assert(val_d == (double)val_u8_normal);
     tl_cast(&val_d, TL_DOUBLE, &val_b_true, TL_BOOL);
     ck_assert(val_d == (double)val_b_true);
     tl_cast(&val_d, TL_DOUBLE, &val_b_false, TL_BOOL);
     ck_assert(val_d == (double)val_b_false);

     /* TL_FLOAT */
     tl_cast(&val_f, TL_FLOAT, &val_d_max, TL_DOUBLE);
     ck_assert(val_f == val_f_max);
     tl_cast(&val_f, TL_FLOAT, &val_d_min, TL_DOUBLE);
     ck_assert(val_f == val_f_min);
     tl_cast(&val_f, TL_FLOAT, &val_d_normal, TL_DOUBLE);
     ck_assert(val_f == (float)val_d_normal);
     tl_cast(&val_f, TL_FLOAT, &val_f_max, TL_FLOAT);
     ck_assert(val_f == val_f_max);
     tl_cast(&val_f, TL_FLOAT, &val_f_min, TL_FLOAT);
     ck_assert(val_f == val_f_min);
     tl_cast(&val_f, TL_FLOAT, &val_f_normal, TL_FLOAT);
     ck_assert(val_f == val_f_normal);
     tl_cast(&val_f, TL_FLOAT, &val_i32_max, TL_INT32);
     ck_assert(val_f == (float)val_i32_max);
     tl_cast(&val_f, TL_FLOAT, &val_i32_min, TL_INT32);
     ck_assert(val_f == (float)val_i32_min);
     tl_cast(&val_f, TL_FLOAT, &val_i32_normal, TL_INT32);
     ck_assert(val_f == (float)val_i32_normal);
     tl_cast(&val_f, TL_FLOAT, &val_i16_max, TL_INT16);
     ck_assert(val_f == (float)val_i16_max);
     tl_cast(&val_f, TL_FLOAT, &val_i16_min, TL_INT16);
     ck_assert(val_f == (float)val_i16_min);
     tl_cast(&val_f, TL_FLOAT, &val_i16_normal, TL_INT16);
     ck_assert(val_f == (float)val_i16_normal);
     tl_cast(&val_f, TL_FLOAT, &val_i8_max, TL_INT8);
     ck_assert(val_f == (float)val_i8_max);
     tl_cast(&val_f, TL_FLOAT, &val_i8_min, TL_INT8);
     ck_assert(val_f == (float)val_i8_min);
     tl_cast(&val_f, TL_FLOAT, &val_i8_normal, TL_INT8);
     ck_assert(val_f == (float)val_i8_normal);
     tl_cast(&val_f, TL_FLOAT, &val_u32_max, TL_UINT32);
     ck_assert(val_f == (float)val_u32_max);
     tl_cast(&val_f, TL_FLOAT, &val_u32_normal, TL_UINT32);
     ck_assert(val_f == (float)val_u32_normal);
     tl_cast(&val_f, TL_FLOAT, &val_u16_max, TL_UINT16);
     ck_assert(val_f == (float)val_u16_max);
     tl_cast(&val_f, TL_FLOAT, &val_u16_normal, TL_UINT16);
     ck_assert(val_f == (float)val_u16_normal);
     tl_cast(&val_f, TL_FLOAT, &val_u8_max, TL_UINT8);
     ck_assert(val_f == (float)val_u8_max);
     tl_cast(&val_f, TL_FLOAT, &val_u8_normal, TL_UINT8);
     ck_assert(val_f == (float)val_u8_normal);
     tl_cast(&val_f, TL_FLOAT, &val_b_true, TL_BOOL);
     ck_assert(val_f == (float)val_b_true);
     tl_cast(&val_f, TL_FLOAT, &val_b_false, TL_BOOL);
     ck_assert(val_f == (float)val_b_false);

     /* TL_INT32 */
     tl_cast(&val_i32, TL_INT32, &val_d_max, TL_DOUBLE);
     ck_assert(val_i32 == val_i32_max);
     tl_cast(&val_i32, TL_INT32, &val_d_min, TL_DOUBLE);
     ck_assert(val_i32 == val_i32_min);
     tl_cast(&val_i32, TL_INT32, &val_d_normal, TL_DOUBLE);
     ck_assert(val_i32 == (int32_t)val_d_normal);
     tl_cast(&val_i32, TL_INT32, &val_f_max, TL_FLOAT);
     ck_assert(val_i32 == val_i32_max);
     tl_cast(&val_i32, TL_INT32, &val_f_min, TL_FLOAT);
     ck_assert(val_i32 == val_i32_min);
     tl_cast(&val_i32, TL_INT32, &val_f_normal, TL_FLOAT);
     ck_assert(val_i32 == (int32_t)val_f_normal);
     tl_cast(&val_i32, TL_INT32, &val_i32_max, TL_INT32);
     ck_assert(val_i32 == val_i32_max);
     tl_cast(&val_i32, TL_INT32, &val_i32_min, TL_INT32);
     ck_assert(val_i32 == val_i32_min);
     tl_cast(&val_i32, TL_INT32, &val_i32_normal, TL_INT32);
     ck_assert(val_i32 == val_i32_normal);
     tl_cast(&val_i32, TL_INT32, &val_i16_max, TL_INT16);
     ck_assert(val_i32 == (int32_t)val_i16_max);
     tl_cast(&val_i32, TL_INT32, &val_i16_min, TL_INT16);
     ck_assert(val_i32 == (int32_t)val_i16_min);
     tl_cast(&val_i32, TL_INT32, &val_i16_normal, TL_INT16);
     ck_assert(val_i32 == (int32_t)val_i16_normal);
     tl_cast(&val_i32, TL_INT32, &val_i8_max, TL_INT8);
     ck_assert(val_i32 == (int32_t)val_i8_max);
     tl_cast(&val_i32, TL_INT32, &val_i8_min, TL_INT8);
     ck_assert(val_i32 == (int32_t)val_i8_min);
     tl_cast(&val_i32, TL_INT32, &val_i8_normal, TL_INT8);
     ck_assert(val_i32 == (int32_t)val_i8_normal);
     tl_cast(&val_i32, TL_INT32, &val_u32_max, TL_UINT32);
     ck_assert(val_i32 == val_i32_max);
     tl_cast(&val_i32, TL_INT32, &val_u32_normal, TL_UINT32);
     ck_assert(val_i32 == (int32_t)val_u32_normal);
     tl_cast(&val_i32, TL_INT32, &val_u16_max, TL_UINT16);
     ck_assert(val_i32 == (int32_t)val_u16_max);
     tl_cast(&val_i32, TL_INT32, &val_u16_normal, TL_UINT16);
     ck_assert(val_i32 == (int32_t)val_u16_normal);
     tl_cast(&val_i32, TL_INT32, &val_u8_max, TL_UINT8);
     ck_assert(val_i32 == (int32_t)val_u8_max);
     tl_cast(&val_i32, TL_INT32, &val_u8_normal, TL_UINT8);
     ck_assert(val_i32 == (int32_t)val_u8_normal);
     tl_cast(&val_i32, TL_INT32, &val_b_true, TL_BOOL);
     ck_assert(val_i32 == (int32_t)val_b_true);
     tl_cast(&val_i32, TL_INT32, &val_b_false, TL_BOOL);
     ck_assert(val_i32 == (int32_t)val_b_false);

     /* TL_INT16 */
     tl_cast(&val_i16, TL_INT16, &val_d_max, TL_DOUBLE);
     ck_assert(val_i16 == val_i16_max);
     tl_cast(&val_i16, TL_INT16, &val_d_min, TL_DOUBLE);
     ck_assert(val_i16 == val_i16_min);
     tl_cast(&val_i16, TL_INT16, &val_d_normal, TL_DOUBLE);
     ck_assert(val_i16 == (int16_t)val_d_normal);
     tl_cast(&val_i16, TL_INT16, &val_f_max, TL_FLOAT);
     ck_assert(val_i16 == val_i16_max);
     tl_cast(&val_i16, TL_INT16, &val_f_min, TL_FLOAT);
     ck_assert(val_i16 == val_i16_min);
     tl_cast(&val_i16, TL_INT16, &val_f_normal, TL_FLOAT);
     ck_assert(val_i16 == (int16_t)val_f_normal);
     tl_cast(&val_i16, TL_INT16, &val_i32_max, TL_INT32);
     ck_assert(val_i16 == val_i16_max);
     tl_cast(&val_i16, TL_INT16, &val_i32_min, TL_INT32);
     ck_assert(val_i16 == val_i16_min);
     tl_cast(&val_i16, TL_INT16, &val_i32_normal, TL_INT32);
     ck_assert(val_i16 == (int16_t)val_i32_normal);
     tl_cast(&val_i16, TL_INT16, &val_i16_max, TL_INT16);
     ck_assert(val_i16 == val_i16_max);
     tl_cast(&val_i16, TL_INT16, &val_i16_min, TL_INT16);
     ck_assert(val_i16 == val_i16_min);
     tl_cast(&val_i16, TL_INT16, &val_i16_normal, TL_INT16);
     ck_assert(val_i16 == val_i16_normal);
     tl_cast(&val_i16, TL_INT16, &val_i8_max, TL_INT8);
     ck_assert(val_i16 == (int16_t)val_i8_max);
     tl_cast(&val_i16, TL_INT16, &val_i8_min, TL_INT8);
     ck_assert(val_i16 == (int16_t)val_i8_min);
     tl_cast(&val_i16, TL_INT16, &val_i8_normal, TL_INT8);
     ck_assert(val_i16 == (int16_t)val_i8_normal);
     tl_cast(&val_i16, TL_INT16, &val_u32_max, TL_UINT32);
     ck_assert(val_i16 == val_i16_max);
     tl_cast(&val_i16, TL_INT16, &val_u32_normal, TL_UINT32);
     ck_assert(val_i16 == (int16_t)val_u32_normal);
     tl_cast(&val_i16, TL_INT16, &val_u16_max, TL_UINT16);
     ck_assert(val_i16 == val_i16_max);
     tl_cast(&val_i16, TL_INT16, &val_u16_normal, TL_UINT16);
     ck_assert(val_i16 == (int16_t)val_u16_normal);
     tl_cast(&val_i16, TL_INT16, &val_u8_max, TL_UINT8);
     ck_assert(val_i16 == (int16_t)val_u8_max);
     tl_cast(&val_i16, TL_INT16, &val_u8_normal, TL_UINT8);
     ck_assert(val_i16 == (int16_t)val_u8_normal);
     tl_cast(&val_i16, TL_INT16, &val_b_true, TL_BOOL);
     ck_assert(val_i16 == (int16_t)val_b_true);
     tl_cast(&val_i16, TL_INT16, &val_b_false, TL_BOOL);
     ck_assert(val_i16 == (int16_t)val_b_false);

     /* TL_INT8 */
     tl_cast(&val_i8, TL_INT8, &val_d_max, TL_DOUBLE);
     ck_assert(val_i8 == val_i8_max);
     tl_cast(&val_i8, TL_INT8, &val_d_min, TL_DOUBLE);
     ck_assert(val_i8 == val_i8_min);
     tl_cast(&val_i8, TL_INT8, &val_d_normal, TL_DOUBLE);
     ck_assert(val_i8 == (int8_t)val_d_normal);
     tl_cast(&val_i8, TL_INT8, &val_f_max, TL_FLOAT);
     ck_assert(val_i8 == val_i8_max);
     tl_cast(&val_i8, TL_INT8, &val_f_min, TL_FLOAT);
     ck_assert(val_i8 == val_i8_min);
     tl_cast(&val_i8, TL_INT8, &val_f_normal, TL_FLOAT);
     ck_assert(val_i8 == (int8_t)val_f_normal);
     tl_cast(&val_i8, TL_INT8, &val_i32_max, TL_INT32);
     ck_assert(val_i8 == val_i8_max);
     tl_cast(&val_i8, TL_INT8, &val_i32_min, TL_INT32);
     ck_assert(val_i8 == val_i8_min);
     tl_cast(&val_i8, TL_INT8, &val_i32_normal, TL_INT32);
     ck_assert(val_i8 == (int8_t)val_i32_normal);
     tl_cast(&val_i8, TL_INT8, &val_i16_max, TL_INT16);
     ck_assert(val_i8 == val_i8_max);
     tl_cast(&val_i8, TL_INT8, &val_i16_min, TL_INT16);
     ck_assert(val_i8 == val_i8_min);
     tl_cast(&val_i8, TL_INT8, &val_i16_normal, TL_INT16);
     ck_assert(val_i8 == val_i8_normal);
     tl_cast(&val_i8, TL_INT8, &val_i8_max, TL_INT8);
     ck_assert(val_i8 == val_i8_max);
     tl_cast(&val_i8, TL_INT8, &val_i8_min, TL_INT8);
     ck_assert(val_i8 == val_i8_min);
     tl_cast(&val_i8, TL_INT8, &val_i8_normal, TL_INT8);
     ck_assert(val_i8 == val_i8_normal);
     tl_cast(&val_i8, TL_INT8, &val_u32_max, TL_UINT32);
     ck_assert(val_i8 == val_i8_max);
     tl_cast(&val_i8, TL_INT8, &val_u32_normal, TL_UINT32);
     ck_assert(val_i8 == (int8_t)val_u32_normal);
     tl_cast(&val_i8, TL_INT8, &val_u16_max, TL_UINT16);
     ck_assert(val_i8 == val_i8_max);
     tl_cast(&val_i8, TL_INT8, &val_u16_normal, TL_UINT16);
     ck_assert(val_i8 == (int8_t)val_u16_normal);
     tl_cast(&val_i8, TL_INT8, &val_u8_max, TL_UINT8);
     ck_assert(val_i8 == val_i8_max);
     tl_cast(&val_i8, TL_INT8, &val_u8_normal, TL_UINT8);
     ck_assert(val_i8 == (int8_t)val_u8_normal);
     tl_cast(&val_i8, TL_INT8, &val_b_true, TL_BOOL);
     ck_assert(val_i8 == (int8_t)val_b_true);
     tl_cast(&val_i8, TL_INT8, &val_b_false, TL_BOOL);
     ck_assert(val_i8 == (int8_t)val_b_false);

     /* TL_UINT32 */
     tl_cast(&val_u32, TL_UINT32, &val_d_max, TL_DOUBLE);
     ck_assert(val_u32 == val_u32_max);
     tl_cast(&val_u32, TL_UINT32, &val_d_min, TL_DOUBLE);
     ck_assert(val_u32 == val_u32_min);
     tl_cast(&val_u32, TL_UINT32, &val_d_normal, TL_DOUBLE);
     ck_assert(val_u32 == (uint32_t)val_d_normal);
     tl_cast(&val_u32, TL_UINT32, &val_f_max, TL_FLOAT);
     ck_assert(val_u32 == val_u32_max);
     tl_cast(&val_u32, TL_UINT32, &val_f_min, TL_FLOAT);
     ck_assert(val_u32 == val_u32_min);
     tl_cast(&val_u32, TL_UINT32, &val_f_normal, TL_FLOAT);
     ck_assert(val_u32 == (uint32_t)val_f_normal);
     tl_cast(&val_u32, TL_UINT32, &val_i32_max, TL_INT32);
     ck_assert(val_u32 == (uint32_t)val_i32_max);
     tl_cast(&val_u32, TL_UINT32, &val_i32_min, TL_INT32);
     ck_assert(val_u32 == val_u32_min);
     tl_cast(&val_u32, TL_UINT32, &val_i32_normal, TL_INT32);
     ck_assert(val_u32 == (uint32_t)val_i32_normal);
     tl_cast(&val_u32, TL_UINT32, &val_i16_max, TL_INT16);
     ck_assert(val_u32 == (uint32_t)val_i16_max);
     tl_cast(&val_u32, TL_UINT32, &val_i16_min, TL_INT16);
     ck_assert(val_u32 == val_u32_min);
     tl_cast(&val_u32, TL_UINT32, &val_i16_normal, TL_INT16);
     ck_assert(val_u32 == (uint32_t)val_i16_normal);
     tl_cast(&val_u32, TL_UINT32, &val_i8_max, TL_INT8);
     ck_assert(val_u32 == (uint32_t)val_i8_max);
     tl_cast(&val_u32, TL_UINT32, &val_i8_min, TL_INT8);
     ck_assert(val_u32 == val_u32_min);
     tl_cast(&val_u32, TL_UINT32, &val_i8_normal, TL_INT8);
     ck_assert(val_u32 == (uint32_t)val_i8_normal);
     tl_cast(&val_u32, TL_UINT32, &val_u32_max, TL_UINT32);
     ck_assert(val_u32 == val_u32_max);
     tl_cast(&val_u32, TL_UINT32, &val_u32_normal, TL_UINT32);
     ck_assert(val_u32 == val_u32_normal);
     tl_cast(&val_u32, TL_UINT32, &val_u16_max, TL_UINT16);
     ck_assert(val_u32 == (uint32_t)val_u16_max);
     tl_cast(&val_u32, TL_UINT32, &val_u16_normal, TL_UINT16);
     ck_assert(val_u32 == (uint32_t)val_u16_normal);
     tl_cast(&val_u32, TL_UINT32, &val_u8_max, TL_UINT8);
     ck_assert(val_u32 == (uint32_t)val_u8_max);
     tl_cast(&val_u32, TL_UINT32, &val_u8_normal, TL_UINT8);
     ck_assert(val_u32 == (uint32_t)val_u8_normal);
     tl_cast(&val_u32, TL_UINT32, &val_b_true, TL_BOOL);
     ck_assert(val_u32 == (uint32_t)val_b_true);
     tl_cast(&val_u32, TL_UINT32, &val_b_false, TL_BOOL);
     ck_assert(val_u32 == (uint32_t)val_b_false);

     /* TL_UINT16 */
     tl_cast(&val_u16, TL_UINT16, &val_d_max, TL_DOUBLE);
     ck_assert(val_u16 == val_u16_max);
     tl_cast(&val_u16, TL_UINT16, &val_d_min, TL_DOUBLE);
     ck_assert(val_u16 == val_u16_min);
     tl_cast(&val_u16, TL_UINT16, &val_d_normal, TL_DOUBLE);
     ck_assert(val_u16 == (uint16_t)val_d_normal);
     tl_cast(&val_u16, TL_UINT16, &val_f_max, TL_FLOAT);
     ck_assert(val_u16 == val_u16_max);
     tl_cast(&val_u16, TL_UINT16, &val_f_min, TL_FLOAT);
     ck_assert(val_u16 == val_u16_min);
     tl_cast(&val_u16, TL_UINT16, &val_f_normal, TL_FLOAT);
     ck_assert(val_u16 == (uint16_t)val_f_normal);
     tl_cast(&val_u16, TL_UINT16, &val_i32_max, TL_INT32);
     ck_assert(val_u16 == val_u16_max);
     tl_cast(&val_u16, TL_UINT16, &val_i32_min, TL_INT32);
     ck_assert(val_u16 == val_u16_min);
     tl_cast(&val_u16, TL_UINT16, &val_i32_normal, TL_INT32);
     ck_assert(val_u16 == (uint16_t)val_i32_normal);
     tl_cast(&val_u16, TL_UINT16, &val_i16_max, TL_INT16);
     ck_assert(val_u16 == (uint16_t)val_i16_max);
     tl_cast(&val_u16, TL_UINT16, &val_i16_min, TL_INT16);
     ck_assert(val_u16 == val_u16_min);
     tl_cast(&val_u16, TL_UINT16, &val_i16_normal, TL_INT16);
     ck_assert(val_u16 == (uint16_t)val_i16_normal);
     tl_cast(&val_u16, TL_UINT16, &val_i8_max, TL_INT8);
     ck_assert(val_u16 == (uint16_t)val_i8_max);
     tl_cast(&val_u16, TL_UINT16, &val_i8_min, TL_INT8);
     ck_assert(val_u16 == val_u16_min);
     tl_cast(&val_u16, TL_UINT16, &val_i8_normal, TL_INT8);
     ck_assert(val_u16 == (uint16_t)val_i8_normal);
     tl_cast(&val_u16, TL_UINT16, &val_u32_max, TL_UINT32);
     ck_assert(val_u16 == val_u16_max);
     tl_cast(&val_u16, TL_UINT16, &val_u32_normal, TL_UINT32);
     ck_assert(val_u16 == (uint16_t)val_u32_normal);
     tl_cast(&val_u16, TL_UINT16, &val_u16_max, TL_UINT16);
     ck_assert(val_u16 == val_u16_max);
     tl_cast(&val_u16, TL_UINT16, &val_u16_normal, TL_UINT16);
     ck_assert(val_u16 == val_u16_normal);
     tl_cast(&val_u16, TL_UINT16, &val_u8_max, TL_UINT8);
     ck_assert(val_u16 == (uint16_t)val_u8_max);
     tl_cast(&val_u16, TL_UINT16, &val_u8_normal, TL_UINT8);
     ck_assert(val_u16 == (uint16_t)val_u8_normal);
     tl_cast(&val_u16, TL_UINT16, &val_b_true, TL_BOOL);
     ck_assert(val_u16 == (uint16_t)val_b_true);
     tl_cast(&val_u16, TL_UINT16, &val_b_false, TL_BOOL);
     ck_assert(val_u16 == (uint16_t)val_b_false);

     /* TL_UINT8 */
     tl_cast(&val_u8, TL_UINT8, &val_d_max, TL_DOUBLE);
     ck_assert(val_u8 == val_u8_max);
     tl_cast(&val_u8, TL_UINT8, &val_d_min, TL_DOUBLE);
     ck_assert(val_u8 == val_u8_min);
     tl_cast(&val_u8, TL_UINT8, &val_d_normal, TL_DOUBLE);
     ck_assert(val_u8 == (uint8_t)val_d_normal);
     tl_cast(&val_u8, TL_UINT8, &val_f_max, TL_FLOAT);
     ck_assert(val_u8 == val_u8_max);
     tl_cast(&val_u8, TL_UINT8, &val_f_min, TL_FLOAT);
     ck_assert(val_u8 == val_u8_min);
     tl_cast(&val_u8, TL_UINT8, &val_f_normal, TL_FLOAT);
     ck_assert(val_u8 == (uint8_t)val_f_normal);
     tl_cast(&val_u8, TL_UINT8, &val_i32_max, TL_INT32);
     ck_assert(val_u8 == val_u8_max);
     tl_cast(&val_u8, TL_UINT8, &val_i32_min, TL_INT32);
     ck_assert(val_u8 == val_u8_min);
     tl_cast(&val_u8, TL_UINT8, &val_i32_normal, TL_INT32);
     ck_assert(val_u8 == (uint8_t)val_i32_normal);
     tl_cast(&val_u8, TL_UINT8, &val_i16_max, TL_INT16);
     ck_assert(val_u8 == val_u8_max);
     tl_cast(&val_u8, TL_UINT8, &val_i16_min, TL_INT16);
     ck_assert(val_u8 == val_u8_min);
     tl_cast(&val_u8, TL_UINT8, &val_i16_normal, TL_INT16);
     ck_assert(val_u8 == (uint8_t)val_i16_normal);
     tl_cast(&val_u8, TL_UINT8, &val_i8_max, TL_INT8);
     ck_assert(val_u8 == (uint8_t)val_i8_max);
     tl_cast(&val_u8, TL_UINT8, &val_i8_min, TL_INT8);
     ck_assert(val_u8 == val_u8_min);
     tl_cast(&val_u8, TL_UINT8, &val_i8_normal, TL_INT8);
     ck_assert(val_u8 == (uint8_t)val_i8_normal);
     tl_cast(&val_u8, TL_UINT8, &val_u32_max, TL_UINT32);
     ck_assert(val_u8 == val_u8_max);
     tl_cast(&val_u8, TL_UINT8, &val_u32_normal, TL_UINT32);
     ck_assert(val_u8 == (uint8_t)val_u32_normal);
     tl_cast(&val_u8, TL_UINT8, &val_u16_max, TL_UINT16);
     ck_assert(val_u8 == val_u8_max);
     tl_cast(&val_u8, TL_UINT8, &val_u16_normal, TL_UINT16);
     ck_assert(val_u8 == (uint8_t)val_u16_normal);
     tl_cast(&val_u8, TL_UINT8, &val_u8_max, TL_UINT8);
     ck_assert(val_u8 == val_u8_max);
     tl_cast(&val_u8, TL_UINT8, &val_u8_normal, TL_UINT8);
     ck_assert(val_u8 == val_u8_normal);
     tl_cast(&val_u8, TL_UINT8, &val_b_true, TL_BOOL);
     ck_assert(val_u8 == (uint8_t)val_b_true);
     tl_cast(&val_u8, TL_UINT8, &val_b_false, TL_BOOL);
     ck_assert(val_u8 == (uint8_t)val_b_false);

     /* TL_BOOL */
     tl_cast(&val_b, TL_BOOL, &val_d_max, TL_DOUBLE);
     ck_assert(val_b == val_b_true);
     tl_cast(&val_b, TL_BOOL, &val_d_min, TL_DOUBLE);
     ck_assert(val_b == val_b_true);
     tl_cast(&val_b, TL_BOOL, &val_d_normal, TL_DOUBLE);
     ck_assert(val_b == val_b_true);
     tl_cast(&val_b, TL_BOOL, &val_f_max, TL_FLOAT);
     ck_assert(val_b == val_b_true);
     tl_cast(&val_b, TL_BOOL, &val_f_min, TL_FLOAT);
     ck_assert(val_b == val_b_true);
     tl_cast(&val_b, TL_BOOL, &val_f_normal, TL_FLOAT);
     ck_assert(val_b == val_b_true);
     tl_cast(&val_b, TL_BOOL, &val_i32_max, TL_INT32);
     ck_assert(val_b == val_b_true);
     tl_cast(&val_b, TL_BOOL, &val_i32_min, TL_INT32);
     ck_assert(val_b == val_b_true);
     tl_cast(&val_b, TL_BOOL, &val_i32_normal, TL_INT32);
     ck_assert(val_b == val_b_true);
     tl_cast(&val_b, TL_BOOL, &val_i16_max, TL_INT16);
     ck_assert(val_b == val_b_true);
     tl_cast(&val_b, TL_BOOL, &val_i16_min, TL_INT16);
     ck_assert(val_b == val_b_true);
     tl_cast(&val_b, TL_BOOL, &val_i16_normal, TL_INT16);
     ck_assert(val_b == val_b_true);
     tl_cast(&val_b, TL_BOOL, &val_i8_max, TL_INT8);
     ck_assert(val_b == val_b_true);
     tl_cast(&val_b, TL_BOOL, &val_i8_min, TL_INT8);
     ck_assert(val_b == val_b_true);
     tl_cast(&val_b, TL_BOOL, &val_i8_normal, TL_INT8);
     ck_assert(val_b == val_b_true);
     tl_cast(&val_b, TL_BOOL, &val_u32_max, TL_UINT32);
     ck_assert(val_b == val_b_true);
     tl_cast(&val_b, TL_BOOL, &val_u32_normal, TL_UINT32);
     ck_assert(val_b == val_b_true);
     tl_cast(&val_b, TL_BOOL, &val_u16_max, TL_UINT16);
     ck_assert(val_b == val_b_true);
     tl_cast(&val_b, TL_BOOL, &val_u16_normal, TL_UINT16);
     ck_assert(val_b == val_b_true);
     tl_cast(&val_b, TL_BOOL, &val_u8_max, TL_UINT8);
     ck_assert(val_b == val_b_true);
     tl_cast(&val_b, TL_BOOL, &val_u8_normal, TL_UINT8);
     ck_assert(val_b == val_b_true);
     tl_cast(&val_b, TL_BOOL, &val_b_true, TL_BOOL);
     ck_assert(val_b == val_b_true);
     tl_cast(&val_b, TL_BOOL, &val_b_false, TL_BOOL);
     ck_assert(val_b == val_b_false);
}
END_TEST
/* end of tests */

Suite *make_cast_suite(void)
{
     Suite *s;
     TCase *tc_cast;

     s = suite_create("cast");
     tc_cast = tcase_create("cast");
     tcase_add_checked_fixture(tc_cast, setup, teardown);

     tcase_add_test(tc_cast, test_tl_cast);
     /* end of adding tests */

     suite_add_tcase(s, tc_cast);

     return s;
}
