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

#include <float.h>
#include "test_tensorlight.h"
#include "../src/tl_type.h"
#include "../src/tl_util.h"

static void setup(void)
{
}

static void teardown(void)
{
}

START_TEST(test_tl_size_of)
{
     ck_assert_int_eq(tl_size_of(TL_DOUBLE), sizeof(double));
     ck_assert_int_eq(tl_size_of(TL_FLOAT), sizeof(float));
     ck_assert_int_eq(tl_size_of(TL_INT32), sizeof(int32_t));
     ck_assert_int_eq(tl_size_of(TL_INT16), sizeof(int16_t));
     ck_assert_int_eq(tl_size_of(TL_INT8), sizeof(int8_t));
     ck_assert_int_eq(tl_size_of(TL_UINT32), sizeof(uint32_t));
     ck_assert_int_eq(tl_size_of(TL_UINT16), sizeof(uint16_t));
     ck_assert_int_eq(tl_size_of(TL_UINT8), sizeof(uint8_t));
     ck_assert_int_eq(tl_size_of(TL_BOOL), sizeof(tl_bool_t));
}
END_TEST

START_TEST(test_tl_psub)
{
     double p_double[2];
     float p_float[2];
     int32_t p_int32[2];
     int16_t p_int16[2];
     int8_t p_int8[2];
     uint32_t p_uint32[2];
     uint16_t p_uint16[2];
     uint8_t p_uint8[2];
     tl_bool_t p_bool[2];

     ck_assert_int_eq(tl_psub(&p_double[1], &p_double[0], tl_size_of(TL_DOUBLE)), 1);
     ck_assert_int_eq(tl_psub(&p_double[0], &p_double[1], tl_size_of(TL_DOUBLE)), -1);
     ck_assert_int_eq(tl_psub(&p_double[0], &p_double[0], tl_size_of(TL_DOUBLE)), 0);

     ck_assert_int_eq(tl_psub(&p_float[1], &p_float[0], tl_size_of(TL_FLOAT)), 1);
     ck_assert_int_eq(tl_psub(&p_float[0], &p_float[1], tl_size_of(TL_FLOAT)), -1);
     ck_assert_int_eq(tl_psub(&p_float[0], &p_float[0], tl_size_of(TL_FLOAT)), 0);

     ck_assert_int_eq(tl_psub(&p_int32[1], &p_int32[0], tl_size_of(TL_INT32)), 1);
     ck_assert_int_eq(tl_psub(&p_int32[0], &p_int32[1], tl_size_of(TL_INT32)), -1);
     ck_assert_int_eq(tl_psub(&p_int32[0], &p_int32[0], tl_size_of(TL_INT32)), 0);

     ck_assert_int_eq(tl_psub(&p_int16[1], &p_int16[0], tl_size_of(TL_INT16)), 1);
     ck_assert_int_eq(tl_psub(&p_int16[0], &p_int16[1], tl_size_of(TL_INT16)), -1);
     ck_assert_int_eq(tl_psub(&p_int16[0], &p_int16[0], tl_size_of(TL_INT16)), 0);

     ck_assert_int_eq(tl_psub(&p_int8[1], &p_int8[0], tl_size_of(TL_INT8)), 1);
     ck_assert_int_eq(tl_psub(&p_int8[0], &p_int8[1], tl_size_of(TL_INT8)), -1);
     ck_assert_int_eq(tl_psub(&p_int8[0], &p_int8[0], tl_size_of(TL_INT8)), 0);

     ck_assert_int_eq(tl_psub(&p_uint32[1], &p_uint32[0], tl_size_of(TL_UINT32)), 1);
     ck_assert_int_eq(tl_psub(&p_uint32[0], &p_uint32[1], tl_size_of(TL_UINT32)), -1);
     ck_assert_int_eq(tl_psub(&p_uint32[0], &p_uint32[0], tl_size_of(TL_UINT32)), 0);

     ck_assert_int_eq(tl_psub(&p_uint16[1], &p_uint16[0], tl_size_of(TL_UINT16)), 1);
     ck_assert_int_eq(tl_psub(&p_uint16[0], &p_uint16[1], tl_size_of(TL_UINT16)), -1);
     ck_assert_int_eq(tl_psub(&p_uint16[0], &p_uint16[0], tl_size_of(TL_UINT16)), 0);

     ck_assert_int_eq(tl_psub(&p_uint8[1], &p_uint8[0], tl_size_of(TL_UINT8)), 1);
     ck_assert_int_eq(tl_psub(&p_uint8[0], &p_uint8[1], tl_size_of(TL_UINT8)), -1);
     ck_assert_int_eq(tl_psub(&p_uint8[0], &p_uint8[0], tl_size_of(TL_UINT8)), 0);

     ck_assert_int_eq(tl_psub(&p_bool[1], &p_bool[0], tl_size_of(TL_BOOL)), 1);
     ck_assert_int_eq(tl_psub(&p_bool[0], &p_bool[1], tl_size_of(TL_BOOL)), -1);
     ck_assert_int_eq(tl_psub(&p_bool[0], &p_bool[0], tl_size_of(TL_BOOL)), 0);
}
END_TEST

START_TEST(test_tl_padd)
{
     double p_double[3];
     float p_float[3];
     int32_t p_int32[3];
     int16_t p_int16[3];
     int8_t p_int8[3];
     uint32_t p_uint32[3];
     uint16_t p_uint16[3];
     uint8_t p_uint8[3];
     tl_bool_t p_bool[3];

     ck_assert_ptr_eq(tl_padd(&p_double[1], 1, tl_size_of(TL_DOUBLE)), &p_double[2]);
     ck_assert_ptr_eq(tl_padd(&p_double[1], -1, tl_size_of(TL_DOUBLE)), &p_double[0]);
     ck_assert_ptr_eq(tl_padd(&p_double[1], 0, tl_size_of(TL_DOUBLE)), &p_double[1]);

     ck_assert_ptr_eq(tl_padd(&p_float[1], 1, tl_size_of(TL_FLOAT)), &p_float[2]);
     ck_assert_ptr_eq(tl_padd(&p_float[1], -1, tl_size_of(TL_FLOAT)), &p_float[0]);
     ck_assert_ptr_eq(tl_padd(&p_float[1], 0, tl_size_of(TL_FLOAT)), &p_float[1]);

     ck_assert_ptr_eq(tl_padd(&p_int32[1], 1, tl_size_of(TL_INT32)), &p_int32[2]);
     ck_assert_ptr_eq(tl_padd(&p_int32[1], -1, tl_size_of(TL_INT32)), &p_int32[0]);
     ck_assert_ptr_eq(tl_padd(&p_int32[1], 0, tl_size_of(TL_INT32)), &p_int32[1]);

     ck_assert_ptr_eq(tl_padd(&p_int16[1], 1, tl_size_of(TL_INT16)), &p_int16[2]);
     ck_assert_ptr_eq(tl_padd(&p_int16[1], -1, tl_size_of(TL_INT16)), &p_int16[0]);
     ck_assert_ptr_eq(tl_padd(&p_int16[1], 0, tl_size_of(TL_INT16)), &p_int16[1]);

     ck_assert_ptr_eq(tl_padd(&p_int8[1], 1, tl_size_of(TL_INT8)), &p_int8[2]);
     ck_assert_ptr_eq(tl_padd(&p_int8[1], -1, tl_size_of(TL_INT8)), &p_int8[0]);
     ck_assert_ptr_eq(tl_padd(&p_int8[1], 0, tl_size_of(TL_INT8)), &p_int8[1]);

     ck_assert_ptr_eq(tl_padd(&p_uint32[1], 1, tl_size_of(TL_UINT32)), &p_uint32[2]);
     ck_assert_ptr_eq(tl_padd(&p_uint32[1], -1, tl_size_of(TL_UINT32)), &p_uint32[0]);
     ck_assert_ptr_eq(tl_padd(&p_uint32[1], 0, tl_size_of(TL_UINT32)), &p_uint32[1]);

     ck_assert_ptr_eq(tl_padd(&p_uint16[1], 1, tl_size_of(TL_UINT16)), &p_uint16[2]);
     ck_assert_ptr_eq(tl_padd(&p_uint16[1], -1, tl_size_of(TL_UINT16)), &p_uint16[0]);
     ck_assert_ptr_eq(tl_padd(&p_uint16[1], 0, tl_size_of(TL_UINT16)), &p_uint16[1]);

     ck_assert_ptr_eq(tl_padd(&p_uint8[1], 1, tl_size_of(TL_UINT8)), &p_uint8[2]);
     ck_assert_ptr_eq(tl_padd(&p_uint8[1], -1, tl_size_of(TL_UINT8)), &p_uint8[0]);
     ck_assert_ptr_eq(tl_padd(&p_uint8[1], 0, tl_size_of(TL_UINT8)), &p_uint8[1]);

     ck_assert_ptr_eq(tl_padd(&p_bool[1], 1, tl_size_of(TL_BOOL)), &p_bool[2]);
     ck_assert_ptr_eq(tl_padd(&p_bool[1], -1, tl_size_of(TL_BOOL)), &p_bool[0]);
     ck_assert_ptr_eq(tl_padd(&p_bool[1], 0, tl_size_of(TL_BOOL)), &p_bool[1]);
}
END_TEST

START_TEST(test_tl_passign)
{
     double p_double[2] = {0, 1};
     float p_float[2] = {0, 1};
     int32_t p_int32[2] = {0, 1};
     int16_t p_int16[2] = {0, 1};
     int8_t p_int8[2] = {0, 1};
     uint32_t p_uint32[2] = {0, 1};
     uint16_t p_uint16[2] = {0, 1};
     uint8_t p_uint8[2] = {0, 1};
     tl_bool_t p_bool[2] = {0, 1};

     tl_passign(p_double, 0, p_double, 1, tl_size_of(TL_DOUBLE));
     ck_assert(p_double[0] == p_double[1]);

     tl_passign(p_float, 0, p_float, 1, tl_size_of(TL_FLOAT));
     ck_assert(p_float[0] == p_float[1]);

     tl_passign(p_int32, 0, p_int32, 1, tl_size_of(TL_INT32));
     ck_assert(p_int32[0] == p_int32[1]);

     tl_passign(p_int16, 0, p_int16, 1, tl_size_of(TL_INT16));
     ck_assert(p_int16[0] == p_int16[1]);

     tl_passign(p_int8, 0, p_int8, 1, tl_size_of(TL_INT8));
     ck_assert(p_int8[0] == p_int8[1]);

     tl_passign(p_uint32, 0, p_uint32, 1, tl_size_of(TL_UINT32));
     ck_assert(p_uint32[0] == p_uint32[1]);

     tl_passign(p_uint16, 0, p_uint16, 1, tl_size_of(TL_UINT16));
     ck_assert(p_uint16[0] == p_uint16[1]);

     tl_passign(p_uint8, 0, p_uint8, 1, tl_size_of(TL_UINT8));
     ck_assert(p_uint8[0] == p_uint8[1]);

     tl_passign(p_bool, 0, p_bool, 1, tl_size_of(TL_BOOL));
     ck_assert(p_bool[0] == p_bool[1]);
}
END_TEST

START_TEST(test_tl_dtype_fmt)
{
     const char *fmt;

     fmt = tl_dtype_fmt(TL_DOUBLE);
     ck_assert_str_eq(fmt, "%.3f");

     fmt = tl_dtype_fmt(TL_FLOAT);
     ck_assert_str_eq(fmt, "%.3f");

     fmt = tl_dtype_fmt(TL_INT32);
     ck_assert_str_eq(fmt, "%d");

     fmt = tl_dtype_fmt(TL_INT16);
     ck_assert_str_eq(fmt, "%d");

     fmt = tl_dtype_fmt(TL_INT8);
     ck_assert_str_eq(fmt, "%d");

     fmt = tl_dtype_fmt(TL_UINT32);
     ck_assert_str_eq(fmt, "%u");

     fmt = tl_dtype_fmt(TL_UINT16);
     ck_assert_str_eq(fmt, "%u");

     fmt = tl_dtype_fmt(TL_UINT8);
     ck_assert_str_eq(fmt, "%u");

     fmt = tl_dtype_fmt(TL_BOOL);
     ck_assert_str_eq(fmt, "%d");
}
END_TEST

START_TEST(test_tl_pointer_sub)
{
     double p_double[2];
     float p_float[2];
     int32_t p_int32[2];
     int16_t p_int16[2];
     int8_t p_int8[2];
     uint32_t p_uint32[2];
     uint16_t p_uint16[2];
     uint8_t p_uint8[2];
     tl_bool_t p_bool[2];

     ck_assert_int_eq(tl_pointer_sub(&p_double[1], &p_double[0], (TL_DOUBLE)), 1);
     ck_assert_int_eq(tl_pointer_sub(&p_double[0], &p_double[1], (TL_DOUBLE)), -1);
     ck_assert_int_eq(tl_pointer_sub(&p_double[0], &p_double[0], (TL_DOUBLE)), 0);

     ck_assert_int_eq(tl_pointer_sub(&p_float[1], &p_float[0], (TL_FLOAT)), 1);
     ck_assert_int_eq(tl_pointer_sub(&p_float[0], &p_float[1], (TL_FLOAT)), -1);
     ck_assert_int_eq(tl_pointer_sub(&p_float[0], &p_float[0], (TL_FLOAT)), 0);

     ck_assert_int_eq(tl_pointer_sub(&p_int32[1], &p_int32[0], (TL_INT32)), 1);
     ck_assert_int_eq(tl_pointer_sub(&p_int32[0], &p_int32[1], (TL_INT32)), -1);
     ck_assert_int_eq(tl_pointer_sub(&p_int32[0], &p_int32[0], (TL_INT32)), 0);

     ck_assert_int_eq(tl_pointer_sub(&p_int16[1], &p_int16[0], (TL_INT16)), 1);
     ck_assert_int_eq(tl_pointer_sub(&p_int16[0], &p_int16[1], (TL_INT16)), -1);
     ck_assert_int_eq(tl_pointer_sub(&p_int16[0], &p_int16[0], (TL_INT16)), 0);

     ck_assert_int_eq(tl_pointer_sub(&p_int8[1], &p_int8[0], (TL_INT8)), 1);
     ck_assert_int_eq(tl_pointer_sub(&p_int8[0], &p_int8[1], (TL_INT8)), -1);
     ck_assert_int_eq(tl_pointer_sub(&p_int8[0], &p_int8[0], (TL_INT8)), 0);

     ck_assert_int_eq(tl_pointer_sub(&p_uint32[1], &p_uint32[0], (TL_UINT32)), 1);
     ck_assert_int_eq(tl_pointer_sub(&p_uint32[0], &p_uint32[1], (TL_UINT32)), -1);
     ck_assert_int_eq(tl_pointer_sub(&p_uint32[0], &p_uint32[0], (TL_UINT32)), 0);

     ck_assert_int_eq(tl_pointer_sub(&p_uint16[1], &p_uint16[0], (TL_UINT16)), 1);
     ck_assert_int_eq(tl_pointer_sub(&p_uint16[0], &p_uint16[1], (TL_UINT16)), -1);
     ck_assert_int_eq(tl_pointer_sub(&p_uint16[0], &p_uint16[0], (TL_UINT16)), 0);

     ck_assert_int_eq(tl_pointer_sub(&p_uint8[1], &p_uint8[0], (TL_UINT8)), 1);
     ck_assert_int_eq(tl_pointer_sub(&p_uint8[0], &p_uint8[1], (TL_UINT8)), -1);
     ck_assert_int_eq(tl_pointer_sub(&p_uint8[0], &p_uint8[0], (TL_UINT8)), 0);

     ck_assert_int_eq(tl_pointer_sub(&p_bool[1], &p_bool[0], (TL_BOOL)), 1);
     ck_assert_int_eq(tl_pointer_sub(&p_bool[0], &p_bool[1], (TL_BOOL)), -1);
     ck_assert_int_eq(tl_pointer_sub(&p_bool[0], &p_bool[0], (TL_BOOL)), 0);
}
END_TEST

START_TEST(test_tl_pointer_add)
{
     double p_double[3];
     float p_float[3];
     int32_t p_int32[3];
     int16_t p_int16[3];
     int8_t p_int8[3];
     uint32_t p_uint32[3];
     uint16_t p_uint16[3];
     uint8_t p_uint8[3];
     tl_bool_t p_bool[3];

     ck_assert_ptr_eq(tl_pointer_add(&p_double[1], 1, (TL_DOUBLE)), &p_double[2]);
     ck_assert_ptr_eq(tl_pointer_add(&p_double[1], -1, (TL_DOUBLE)), &p_double[0]);
     ck_assert_ptr_eq(tl_pointer_add(&p_double[1], 0, (TL_DOUBLE)), &p_double[1]);

     ck_assert_ptr_eq(tl_pointer_add(&p_float[1], 1, (TL_FLOAT)), &p_float[2]);
     ck_assert_ptr_eq(tl_pointer_add(&p_float[1], -1, (TL_FLOAT)), &p_float[0]);
     ck_assert_ptr_eq(tl_pointer_add(&p_float[1], 0, (TL_FLOAT)), &p_float[1]);

     ck_assert_ptr_eq(tl_pointer_add(&p_int32[1], 1, (TL_INT32)), &p_int32[2]);
     ck_assert_ptr_eq(tl_pointer_add(&p_int32[1], -1, (TL_INT32)), &p_int32[0]);
     ck_assert_ptr_eq(tl_pointer_add(&p_int32[1], 0, (TL_INT32)), &p_int32[1]);

     ck_assert_ptr_eq(tl_pointer_add(&p_int16[1], 1, (TL_INT16)), &p_int16[2]);
     ck_assert_ptr_eq(tl_pointer_add(&p_int16[1], -1, (TL_INT16)), &p_int16[0]);
     ck_assert_ptr_eq(tl_pointer_add(&p_int16[1], 0, (TL_INT16)), &p_int16[1]);

     ck_assert_ptr_eq(tl_pointer_add(&p_int8[1], 1, (TL_INT8)), &p_int8[2]);
     ck_assert_ptr_eq(tl_pointer_add(&p_int8[1], -1, (TL_INT8)), &p_int8[0]);
     ck_assert_ptr_eq(tl_pointer_add(&p_int8[1], 0, (TL_INT8)), &p_int8[1]);

     ck_assert_ptr_eq(tl_pointer_add(&p_uint32[1], 1, (TL_UINT32)), &p_uint32[2]);
     ck_assert_ptr_eq(tl_pointer_add(&p_uint32[1], -1, (TL_UINT32)), &p_uint32[0]);
     ck_assert_ptr_eq(tl_pointer_add(&p_uint32[1], 0, (TL_UINT32)), &p_uint32[1]);

     ck_assert_ptr_eq(tl_pointer_add(&p_uint16[1], 1, (TL_UINT16)), &p_uint16[2]);
     ck_assert_ptr_eq(tl_pointer_add(&p_uint16[1], -1, (TL_UINT16)), &p_uint16[0]);
     ck_assert_ptr_eq(tl_pointer_add(&p_uint16[1], 0, (TL_UINT16)), &p_uint16[1]);

     ck_assert_ptr_eq(tl_pointer_add(&p_uint8[1], 1, (TL_UINT8)), &p_uint8[2]);
     ck_assert_ptr_eq(tl_pointer_add(&p_uint8[1], -1, (TL_UINT8)), &p_uint8[0]);
     ck_assert_ptr_eq(tl_pointer_add(&p_uint8[1], 0, (TL_UINT8)), &p_uint8[1]);

     ck_assert_ptr_eq(tl_pointer_add(&p_bool[1], 1, (TL_BOOL)), &p_bool[2]);
     ck_assert_ptr_eq(tl_pointer_add(&p_bool[1], -1, (TL_BOOL)), &p_bool[0]);
     ck_assert_ptr_eq(tl_pointer_add(&p_bool[1], 0, (TL_BOOL)), &p_bool[1]);
}
END_TEST

START_TEST(test_tl_pointer_assign)
{
     double p_double[2] = {0, 1};
     float p_float[2] = {0, 1};
     int32_t p_int32[2] = {0, 1};
     int16_t p_int16[2] = {0, 1};
     int8_t p_int8[2] = {0, 1};
     uint32_t p_uint32[2] = {0, 1};
     uint16_t p_uint16[2] = {0, 1};
     uint8_t p_uint8[2] = {0, 1};
     tl_bool_t p_bool[2] = {0, 1};

     tl_pointer_assign(p_double, 0, p_double, 1, (TL_DOUBLE));
     ck_assert(p_double[0] == p_double[1]);

     tl_pointer_assign(p_float, 0, p_float, 1, (TL_FLOAT));
     ck_assert(p_float[0] == p_float[1]);

     tl_pointer_assign(p_int32, 0, p_int32, 1, (TL_INT32));
     ck_assert(p_int32[0] == p_int32[1]);

     tl_pointer_assign(p_int16, 0, p_int16, 1, (TL_INT16));
     ck_assert(p_int16[0] == p_int16[1]);

     tl_pointer_assign(p_int8, 0, p_int8, 1, (TL_INT8));
     ck_assert(p_int8[0] == p_int8[1]);

     tl_pointer_assign(p_uint32, 0, p_uint32, 1, (TL_UINT32));
     ck_assert(p_uint32[0] == p_uint32[1]);

     tl_pointer_assign(p_uint16, 0, p_uint16, 1, (TL_UINT16));
     ck_assert(p_uint16[0] == p_uint16[1]);

     tl_pointer_assign(p_uint8, 0, p_uint8, 1, (TL_UINT8));
     ck_assert(p_uint8[0] == p_uint8[1]);

     tl_pointer_assign(p_bool, 0, p_bool, 1, (TL_BOOL));
     ck_assert(p_bool[0] == p_bool[1]);
}
END_TEST

START_TEST(test_tl_fprintf)
{
     FILE *fp;
     double val_double = 0.12345;
     float val_float = 0.12345;
     int32_t val_int32 = -1;
     int16_t val_int16 = -1;
     int8_t val_int8 = -1;
     uint32_t val_uint32 = 1;
     uint16_t val_uint16 = 1;
     uint8_t val_uint8 = 1;
     tl_bool_t val_bool = TL_TRUE;
     char s[10];

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     ck_assert_int_ge(tl_fprintf(fp, NULL, &val_double, TL_DOUBLE), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "0.123");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     ck_assert_int_ge(tl_fprintf(fp, NULL, &val_float, TL_FLOAT), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "0.123");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     ck_assert_int_ge(tl_fprintf(fp, "%.1f", &val_float, TL_FLOAT), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "0.1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     ck_assert_int_ge(tl_fprintf(fp, "%.1f", &val_double, TL_DOUBLE), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "0.1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     ck_assert_int_ge(tl_fprintf(fp, NULL, &val_int32, TL_INT32), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "-1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     ck_assert_int_ge(tl_fprintf(fp, NULL, &val_int16, TL_INT16), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "-1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     ck_assert_int_ge(tl_fprintf(fp, NULL, &val_int8, TL_INT8), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "-1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     ck_assert_int_ge(tl_fprintf(fp, NULL, &val_uint32, TL_UINT32), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     ck_assert_int_ge(tl_fprintf(fp, NULL, &val_uint16, TL_UINT16), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     ck_assert_int_ge(tl_fprintf(fp, NULL, &val_uint8, TL_UINT8), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     ck_assert_int_ge(tl_fprintf(fp, NULL, &val_bool, TL_BOOL), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "1");
     fclose(fp);
}
END_TEST

START_TEST(test_tl_fprintf_getfunc)
{
     FILE *fp;
     double val_double = 0.12345;
     float val_float = 0.12345;
     int32_t val_int32 = -1;
     int16_t val_int16 = -1;
     int8_t val_int8 = -1;
     uint32_t val_uint32 = 1;
     uint16_t val_uint16 = 1;
     uint8_t val_uint8 = 1;
     tl_bool_t val_bool = TL_TRUE;
     tl_fprintf_func gfprintf_func;
     char s[10];

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     gfprintf_func = tl_fprintf_getfunc(TL_DOUBLE);
     ck_assert_int_ge(gfprintf_func(fp, NULL, &val_double), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "0.123");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     gfprintf_func = tl_fprintf_getfunc(TL_FLOAT);
     ck_assert_int_ge(gfprintf_func(fp, NULL, &val_float), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "0.123");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     gfprintf_func = tl_fprintf_getfunc(TL_FLOAT);
     ck_assert_int_ge(gfprintf_func(fp, "%.1f", &val_float), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "0.1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     gfprintf_func = tl_fprintf_getfunc(TL_DOUBLE);
     ck_assert_int_ge(gfprintf_func(fp, "%.1f", &val_double), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "0.1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     gfprintf_func = tl_fprintf_getfunc(TL_INT32);
     ck_assert_int_ge(gfprintf_func(fp, NULL, &val_int32), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "-1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     gfprintf_func = tl_fprintf_getfunc(TL_INT16);
     ck_assert_int_ge(gfprintf_func(fp, NULL, &val_int16), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "-1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     gfprintf_func = tl_fprintf_getfunc(TL_INT8);
     ck_assert_int_ge(gfprintf_func(fp, NULL, &val_int8), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "-1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     gfprintf_func = tl_fprintf_getfunc(TL_UINT32);
     ck_assert_int_ge(gfprintf_func(fp, NULL, &val_uint32), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     gfprintf_func = tl_fprintf_getfunc(TL_UINT16);
     ck_assert_int_ge(gfprintf_func(fp, NULL, &val_uint16), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     gfprintf_func = tl_fprintf_getfunc(TL_UINT8);
     ck_assert_int_ge(gfprintf_func(fp, NULL, &val_uint8), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     gfprintf_func = tl_fprintf_getfunc(TL_BOOL);
     ck_assert_int_ge(gfprintf_func(fp, NULL, &val_bool), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "1");
     fclose(fp);
}
END_TEST

START_TEST(test_tl_cmp)
{
     double val1_double = 1, val2_double = 2;
     float val1_float = 1, val2_float = 2;
     int32_t val1_int32 = 1, val2_int32 = 2;
     int16_t val1_int16 = 1, val2_int16 = 2;
     int8_t val1_int8 = 1, val2_int8 = 2;
     uint32_t val1_uint32 = 1, val2_uint32 = 2;
     uint16_t val1_uint16 = 1, val2_uint16 = 2;
     uint8_t val1_uint8 = 1, val2_uint8 = 2;
     tl_bool_t val1_bool = TL_FALSE, val2_bool = TL_TRUE;

     ck_assert(tl_cmp(&val1_double, &val2_double, TL_DOUBLE) < 0);
     ck_assert(tl_cmp(&val2_double, &val1_double, TL_DOUBLE) > 0);
     ck_assert(tl_cmp(&val1_double, &val1_double, TL_DOUBLE) == 0);

     ck_assert(tl_cmp(&val1_float, &val2_float, TL_FLOAT) < 0);
     ck_assert(tl_cmp(&val2_float, &val1_float, TL_FLOAT) > 0);
     ck_assert(tl_cmp(&val1_float, &val1_float, TL_FLOAT) == 0);

     ck_assert(tl_cmp(&val1_int32, &val2_int32, TL_INT32) < 0);
     ck_assert(tl_cmp(&val2_int32, &val1_int32, TL_INT32) > 0);
     ck_assert(tl_cmp(&val1_int32, &val1_int32, TL_INT32) == 0);

     ck_assert(tl_cmp(&val1_int16, &val2_int16, TL_INT16) < 0);
     ck_assert(tl_cmp(&val2_int16, &val1_int16, TL_INT16) > 0);
     ck_assert(tl_cmp(&val1_int16, &val1_int16, TL_INT16) == 0);

     ck_assert(tl_cmp(&val1_int8, &val2_int8, TL_INT8) < 0);
     ck_assert(tl_cmp(&val2_int8, &val1_int8, TL_INT8) > 0);
     ck_assert(tl_cmp(&val1_int8, &val1_int8, TL_INT8) == 0);

     ck_assert(tl_cmp(&val1_uint32, &val2_uint32, TL_UINT32) < 0);
     ck_assert(tl_cmp(&val2_uint32, &val1_uint32, TL_UINT32) > 0);
     ck_assert(tl_cmp(&val1_uint32, &val1_uint32, TL_UINT32) == 0);

     ck_assert(tl_cmp(&val1_uint16, &val2_uint16, TL_UINT16) < 0);
     ck_assert(tl_cmp(&val2_uint16, &val1_uint16, TL_UINT16) > 0);
     ck_assert(tl_cmp(&val1_uint16, &val1_uint16, TL_UINT16) == 0);

     ck_assert(tl_cmp(&val1_uint8, &val2_uint8, TL_UINT8) < 0);
     ck_assert(tl_cmp(&val2_uint8, &val1_uint8, TL_UINT8) > 0);
     ck_assert(tl_cmp(&val1_uint8, &val1_uint8, TL_UINT8) == 0);

     ck_assert(tl_cmp(&val1_bool, &val2_bool, TL_BOOL) < 0);
     ck_assert(tl_cmp(&val2_bool, &val1_bool, TL_BOOL) > 0);
     ck_assert(tl_cmp(&val1_bool, &val1_bool, TL_BOOL) == 0);
}
END_TEST

START_TEST(test_tl_cmp_getfunc)
{
     double val1_double = 1, val2_double = 2;
     float val1_float = 1, val2_float = 2;
     int32_t val1_int32 = 1, val2_int32 = 2;
     int16_t val1_int16 = 1, val2_int16 = 2;
     int8_t val1_int8 = 1, val2_int8 = 2;
     uint32_t val1_uint32 = 1, val2_uint32 = 2;
     uint16_t val1_uint16 = 1, val2_uint16 = 2;
     uint8_t val1_uint8 = 1, val2_uint8 = 2;
     tl_bool_t val1_bool = TL_FALSE, val2_bool = TL_TRUE;
     tl_cmp_func gcmp_func;

     gcmp_func = tl_cmp_getfunc(TL_DOUBLE);
     ck_assert(gcmp_func(&val1_double, &val2_double) < 0);
     ck_assert(gcmp_func(&val2_double, &val1_double) > 0);
     ck_assert(gcmp_func(&val1_double, &val1_double) == 0);

     gcmp_func = tl_cmp_getfunc(TL_FLOAT);
     ck_assert(gcmp_func(&val1_float, &val2_float) < 0);
     ck_assert(gcmp_func(&val2_float, &val1_float) > 0);
     ck_assert(gcmp_func(&val1_float, &val1_float) == 0);

     gcmp_func = tl_cmp_getfunc(TL_INT32);
     ck_assert(gcmp_func(&val1_int32, &val2_int32) < 0);
     ck_assert(gcmp_func(&val2_int32, &val1_int32) > 0);
     ck_assert(gcmp_func(&val1_int32, &val1_int32) == 0);

     gcmp_func = tl_cmp_getfunc(TL_INT16);
     ck_assert(gcmp_func(&val1_int16, &val2_int16) < 0);
     ck_assert(gcmp_func(&val2_int16, &val1_int16) > 0);
     ck_assert(gcmp_func(&val1_int16, &val1_int16) == 0);

     gcmp_func = tl_cmp_getfunc(TL_INT8);
     ck_assert(gcmp_func(&val1_int8, &val2_int8) < 0);
     ck_assert(gcmp_func(&val2_int8, &val1_int8) > 0);
     ck_assert(gcmp_func(&val1_int8, &val1_int8) == 0);

     gcmp_func = tl_cmp_getfunc(TL_UINT32);
     ck_assert(gcmp_func(&val1_uint32, &val2_uint32) < 0);
     ck_assert(gcmp_func(&val2_uint32, &val1_uint32) > 0);
     ck_assert(gcmp_func(&val1_uint32, &val1_uint32) == 0);

     gcmp_func = tl_cmp_getfunc(TL_UINT16);
     ck_assert(gcmp_func(&val1_uint16, &val2_uint16) < 0);
     ck_assert(gcmp_func(&val2_uint16, &val1_uint16) > 0);
     ck_assert(gcmp_func(&val1_uint16, &val1_uint16) == 0);

     gcmp_func = tl_cmp_getfunc(TL_UINT8);
     ck_assert(gcmp_func(&val1_uint8, &val2_uint8) < 0);
     ck_assert(gcmp_func(&val2_uint8, &val1_uint8) > 0);
     ck_assert(gcmp_func(&val1_uint8, &val1_uint8) == 0);

     gcmp_func = tl_cmp_getfunc(TL_BOOL);
     ck_assert(gcmp_func(&val1_bool, &val2_bool) < 0);
     ck_assert(gcmp_func(&val2_bool, &val1_bool) > 0);
     ck_assert(gcmp_func(&val1_bool, &val1_bool) == 0);
}
END_TEST

START_TEST(test_tl_elew)
{
     double val1_double = 1, val2_double = 2, val3_double;
     float val1_float = 1, val2_float = 2, val3_float;
     int32_t val1_int32 = 1, val2_int32 = 2, val3_int32;
     int16_t val1_int16 = 1, val2_int16 = 2, val3_int16;
     int8_t val1_int8 = 1, val2_int8 = 2, val3_int8;
     uint32_t val1_uint32 = 1, val2_uint32 = 2, val3_uint32;
     uint16_t val1_uint16 = 1, val2_uint16 = 2, val3_uint16;
     uint8_t val1_uint8 = 1, val2_uint8 = 2, val3_uint8;
     tl_bool_t val1_bool = TL_FALSE, val2_bool = TL_TRUE, val3_bool;

     /* TL_MUL */
     tl_elew(&val1_double, &val2_double, &val3_double, TL_MUL, TL_DOUBLE);
     ck_assert(val3_double == 2);

     tl_elew(&val1_float, &val2_float, &val3_float, TL_MUL, TL_FLOAT);
     ck_assert(val3_float == 2);

     tl_elew(&val1_int32, &val2_int32, &val3_int32, TL_MUL, TL_INT32);
     ck_assert(val3_int32 == 2);

     tl_elew(&val1_int16, &val2_int16, &val3_int16, TL_MUL, TL_INT16);
     ck_assert(val3_int16 == 2);

     tl_elew(&val1_int8, &val2_int8, &val3_int8, TL_MUL, TL_INT8);
     ck_assert(val3_int8 == 2);

     tl_elew(&val1_uint32, &val2_uint32, &val3_uint32, TL_MUL, TL_UINT32);
     ck_assert(val3_uint32 == 2);

     tl_elew(&val1_uint16, &val2_uint16, &val3_uint16, TL_MUL, TL_UINT16);
     ck_assert(val3_uint16 == 2);

     tl_elew(&val1_uint8, &val2_uint8, &val3_uint8, TL_MUL, TL_UINT8);
     ck_assert(val3_uint8 == 2);

     tl_elew(&val1_bool, &val2_bool, &val3_bool, TL_MUL, TL_BOOL);
     ck_assert(val3_bool == TL_FALSE);

     /* TL_DIV */
     tl_elew(&val2_double, &val1_double, &val3_double, TL_DIV, TL_DOUBLE);
     ck_assert(val3_double == 2);

     tl_elew(&val2_float, &val1_float, &val3_float, TL_DIV, TL_FLOAT);
     ck_assert(val3_float == 2);

     tl_elew(&val2_int32, &val1_int32, &val3_int32, TL_DIV, TL_INT32);
     ck_assert(val3_int32 == 2);

     tl_elew(&val2_int16, &val1_int16, &val3_int16, TL_DIV, TL_INT16);
     ck_assert(val3_int16 == 2);

     tl_elew(&val2_int8, &val1_int8, &val3_int8, TL_DIV, TL_INT8);
     ck_assert(val3_int8 == 2);

     tl_elew(&val2_uint32, &val1_uint32, &val3_uint32, TL_DIV, TL_UINT32);
     ck_assert(val3_uint32 == 2);

     tl_elew(&val2_uint16, &val1_uint16, &val3_uint16, TL_DIV, TL_UINT16);
     ck_assert(val3_uint16 == 2);

     tl_elew(&val2_uint8, &val1_uint8, &val3_uint8, TL_DIV, TL_UINT8);
     ck_assert(val3_uint8 == 2);

     /* tl_elew(&val2_bool, &val1_bool, &val3_bool, TL_DIV, TL_BOOL); */
     /* ck_assert(val3_bool == 2); */

     /* TL_SUM */
     tl_elew(&val2_double, &val1_double, &val3_double, TL_SUM, TL_DOUBLE);
     ck_assert(val3_double == 3);

     tl_elew(&val2_float, &val1_float, &val3_float, TL_SUM, TL_FLOAT);
     ck_assert(val3_float == 3);

     tl_elew(&val2_int32, &val1_int32, &val3_int32, TL_SUM, TL_INT32);
     ck_assert(val3_int32 == 3);

     tl_elew(&val2_int16, &val1_int16, &val3_int16, TL_SUM, TL_INT16);
     ck_assert(val3_int16 == 3);

     tl_elew(&val2_int8, &val1_int8, &val3_int8, TL_SUM, TL_INT8);
     ck_assert(val3_int8 == 3);

     tl_elew(&val2_uint32, &val1_uint32, &val3_uint32, TL_SUM, TL_UINT32);
     ck_assert(val3_uint32 == 3);

     tl_elew(&val2_uint16, &val1_uint16, &val3_uint16, TL_SUM, TL_UINT16);
     ck_assert(val3_uint16 == 3);

     tl_elew(&val2_uint8, &val1_uint8, &val3_uint8, TL_SUM, TL_UINT8);
     ck_assert(val3_uint8 == 3);

     tl_elew(&val2_bool, &val1_bool, &val3_bool, TL_SUM, TL_BOOL);
     ck_assert(val3_bool == 1);

     /* TL_SUB */
     tl_elew(&val2_double, &val1_double, &val3_double, TL_SUB, TL_DOUBLE);
     ck_assert(val3_double == 1);

     tl_elew(&val2_float, &val1_float, &val3_float, TL_SUB, TL_FLOAT);
     ck_assert(val3_float == 1);

     tl_elew(&val2_int32, &val1_int32, &val3_int32, TL_SUB, TL_INT32);
     ck_assert(val3_int32 == 1);

     tl_elew(&val2_int16, &val1_int16, &val3_int16, TL_SUB, TL_INT16);
     ck_assert(val3_int16 == 1);

     tl_elew(&val2_int8, &val1_int8, &val3_int8, TL_SUB, TL_INT8);
     ck_assert(val3_int8 == 1);

     tl_elew(&val2_uint32, &val1_uint32, &val3_uint32, TL_SUB, TL_UINT32);
     ck_assert(val3_uint32 == 1);

     tl_elew(&val2_uint16, &val1_uint16, &val3_uint16, TL_SUB, TL_UINT16);
     ck_assert(val3_uint16 == 1);

     tl_elew(&val2_uint8, &val1_uint8, &val3_uint8, TL_SUB, TL_UINT8);
     ck_assert(val3_uint8 == 1);

     tl_elew(&val2_bool, &val1_bool, &val3_bool, TL_SUB, TL_BOOL);
     ck_assert(val3_bool == 1);

     /* TL_MAX */
     tl_elew(&val2_double, &val1_double, &val3_double, TL_MAX, TL_DOUBLE);
     ck_assert(val3_double == 2);

     tl_elew(&val2_float, &val1_float, &val3_float, TL_MAX, TL_FLOAT);
     ck_assert(val3_float == 2);

     tl_elew(&val2_int32, &val1_int32, &val3_int32, TL_MAX, TL_INT32);
     ck_assert(val3_int32 == 2);

     tl_elew(&val2_int16, &val1_int16, &val3_int16, TL_MAX, TL_INT16);
     ck_assert(val3_int16 == 2);

     tl_elew(&val2_int8, &val1_int8, &val3_int8, TL_MAX, TL_INT8);
     ck_assert(val3_int8 == 2);

     tl_elew(&val2_uint32, &val1_uint32, &val3_uint32, TL_MAX, TL_UINT32);
     ck_assert(val3_uint32 == 2);

     tl_elew(&val2_uint16, &val1_uint16, &val3_uint16, TL_MAX, TL_UINT16);
     ck_assert(val3_uint16 == 2);

     tl_elew(&val2_uint8, &val1_uint8, &val3_uint8, TL_MAX, TL_UINT8);
     ck_assert(val3_uint8 == 2);

     tl_elew(&val2_bool, &val1_bool, &val3_bool, TL_MAX, TL_BOOL);
     ck_assert(val3_bool == 1);

     /* TL_MIN */
     tl_elew(&val2_double, &val1_double, &val3_double, TL_MIN, TL_DOUBLE);
     ck_assert(val3_double == 1);

     tl_elew(&val2_float, &val1_float, &val3_float, TL_MIN, TL_FLOAT);
     ck_assert(val3_float == 1);

     tl_elew(&val2_int32, &val1_int32, &val3_int32, TL_MIN, TL_INT32);
     ck_assert(val3_int32 == 1);

     tl_elew(&val2_int16, &val1_int16, &val3_int16, TL_MIN, TL_INT16);
     ck_assert(val3_int16 == 1);

     tl_elew(&val2_int8, &val1_int8, &val3_int8, TL_MIN, TL_INT8);
     ck_assert(val3_int8 == 1);

     tl_elew(&val2_uint32, &val1_uint32, &val3_uint32, TL_MIN, TL_UINT32);
     ck_assert(val3_uint32 == 1);

     tl_elew(&val2_uint16, &val1_uint16, &val3_uint16, TL_MIN, TL_UINT16);
     ck_assert(val3_uint16 == 1);

     tl_elew(&val2_uint8, &val1_uint8, &val3_uint8, TL_MIN, TL_UINT8);
     ck_assert(val3_uint8 == 1);

     tl_elew(&val2_bool, &val1_bool, &val3_bool, TL_MIN, TL_BOOL);
     ck_assert(val3_bool == 0);

     /* TL_POW */
     tl_elew(&val2_double, &val1_double, &val3_double, TL_POW, TL_DOUBLE);
     ck_assert(val3_double == 2);

     tl_elew(&val2_float, &val1_float, &val3_float, TL_POW, TL_FLOAT);
     ck_assert(val3_float == 2);

     tl_elew(&val2_int32, &val1_int32, &val3_int32, TL_POW, TL_INT32);
     ck_assert(val3_int32 == 2);

     tl_elew(&val2_int16, &val1_int16, &val3_int16, TL_POW, TL_INT16);
     ck_assert(val3_int16 == 2);

     tl_elew(&val2_int8, &val1_int8, &val3_int8, TL_POW, TL_INT8);
     ck_assert(val3_int8 == 2);

     tl_elew(&val2_uint32, &val1_uint32, &val3_uint32, TL_POW, TL_UINT32);
     ck_assert(val3_uint32 == 2);

     tl_elew(&val2_uint16, &val1_uint16, &val3_uint16, TL_POW, TL_UINT16);
     ck_assert(val3_uint16 == 2);

     tl_elew(&val2_uint8, &val1_uint8, &val3_uint8, TL_POW, TL_UINT8);
     ck_assert(val3_uint8 == 2);

     tl_elew(&val2_bool, &val1_bool, &val3_bool, TL_POW, TL_BOOL);
     ck_assert(val3_bool == 1);
}
END_TEST

START_TEST(test_tl_elew_getfunc)
{
     double val1_double = 1, val2_double = 2, val3_double;
     float val1_float = 1, val2_float = 2, val3_float;
     int32_t val1_int32 = 1, val2_int32 = 2, val3_int32;
     int16_t val1_int16 = 1, val2_int16 = 2, val3_int16;
     int8_t val1_int8 = 1, val2_int8 = 2, val3_int8;
     uint32_t val1_uint32 = 1, val2_uint32 = 2, val3_uint32;
     uint16_t val1_uint16 = 1, val2_uint16 = 2, val3_uint16;
     uint8_t val1_uint8 = 1, val2_uint8 = 2, val3_uint8;
     tl_bool_t val1_bool = TL_FALSE, val2_bool = TL_TRUE, val3_bool;
     tl_elew_func elew_func;

     elew_func = tl_elew_getfunc(TL_DOUBLE);
     elew_func(&val2_double, &val1_double, &val3_double, TL_MUL);
     ck_assert(val3_double == 2);

     elew_func = tl_elew_getfunc(TL_FLOAT);
     elew_func(&val2_float, &val1_float, &val3_float, TL_MUL);
     ck_assert(val3_float == 2);

     elew_func = tl_elew_getfunc(TL_INT32);
     elew_func(&val2_int32, &val1_int32, &val3_int32, TL_DIV);
     ck_assert(val3_int32 == 2);

     elew_func = tl_elew_getfunc(TL_INT16);
     elew_func(&val2_int16, &val1_int16, &val3_int16, TL_SUM);
     ck_assert(val3_int16 == 3);

     elew_func = tl_elew_getfunc(TL_INT8);
     elew_func(&val2_int8, &val1_int8, &val3_int8, TL_MAX);
     ck_assert(val3_int8 == 2);

     elew_func = tl_elew_getfunc(TL_UINT32);
     elew_func(&val2_uint32, &val1_uint32, &val3_uint32, TL_MIN);
     ck_assert(val3_uint32 == 1);

     elew_func = tl_elew_getfunc(TL_UINT16);
     elew_func(&val2_uint16, &val1_uint16, &val3_uint16, TL_POW);
     ck_assert(val3_uint16 == 2);

     elew_func = tl_elew_getfunc(TL_UINT8);
     elew_func(&val2_uint8, &val1_uint8, &val3_uint8, TL_MUL);
     ck_assert(val3_uint8 == 2);

     elew_func = tl_elew_getfunc(TL_BOOL);
     elew_func(&val2_bool, &val1_bool, &val3_bool, TL_MUL);
     ck_assert(val3_bool == TL_FALSE);
}
END_TEST

START_TEST(test_tl_convert)
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
     tl_convert(&val_d, TL_DOUBLE, &val_d_max, TL_DOUBLE);
     ck_assert(val_d == val_d_max);
     tl_convert(&val_d, TL_DOUBLE, &val_d_min, TL_DOUBLE);
     ck_assert(val_d == val_d_min);
     tl_convert(&val_d, TL_DOUBLE, &val_d_normal, TL_DOUBLE);
     ck_assert(val_d == val_d_normal);
     tl_convert(&val_d, TL_DOUBLE, &val_f_max, TL_FLOAT);
     ck_assert(val_d == (double)val_f_max);
     tl_convert(&val_d, TL_DOUBLE, &val_f_min, TL_FLOAT);
     ck_assert(val_d == (double)val_f_min);
     tl_convert(&val_d, TL_DOUBLE, &val_f_normal, TL_FLOAT);
     ck_assert(val_d == (double)val_f_normal);
     tl_convert(&val_d, TL_DOUBLE, &val_i32_max, TL_INT32);
     ck_assert(val_d == (double)val_i32_max);
     tl_convert(&val_d, TL_DOUBLE, &val_i32_min, TL_INT32);
     ck_assert(val_d == (double)val_i32_min);
     tl_convert(&val_d, TL_DOUBLE, &val_i32_normal, TL_INT32);
     ck_assert(val_d == (double)val_i32_normal);
     tl_convert(&val_d, TL_DOUBLE, &val_i16_max, TL_INT16);
     ck_assert(val_d == (double)val_i16_max);
     tl_convert(&val_d, TL_DOUBLE, &val_i16_min, TL_INT16);
     ck_assert(val_d == (double)val_i16_min);
     tl_convert(&val_d, TL_DOUBLE, &val_i16_normal, TL_INT16);
     ck_assert(val_d == (double)val_i16_normal);
     tl_convert(&val_d, TL_DOUBLE, &val_i8_max, TL_INT8);
     ck_assert(val_d == (double)val_i8_max);
     tl_convert(&val_d, TL_DOUBLE, &val_i8_min, TL_INT8);
     ck_assert(val_d == (double)val_i8_min);
     tl_convert(&val_d, TL_DOUBLE, &val_i8_normal, TL_INT8);
     ck_assert(val_d == (double)val_i8_normal);
     tl_convert(&val_d, TL_DOUBLE, &val_u32_max, TL_UINT32);
     ck_assert(val_d == (double)val_u32_max);
     tl_convert(&val_d, TL_DOUBLE, &val_u32_normal, TL_UINT32);
     ck_assert(val_d == (double)val_u32_normal);
     tl_convert(&val_d, TL_DOUBLE, &val_u16_max, TL_UINT16);
     ck_assert(val_d == (double)val_u16_max);
     tl_convert(&val_d, TL_DOUBLE, &val_u16_normal, TL_UINT16);
     ck_assert(val_d == (double)val_u16_normal);
     tl_convert(&val_d, TL_DOUBLE, &val_u8_max, TL_UINT8);
     ck_assert(val_d == (double)val_u8_max);
     tl_convert(&val_d, TL_DOUBLE, &val_u8_normal, TL_UINT8);
     ck_assert(val_d == (double)val_u8_normal);
     tl_convert(&val_d, TL_DOUBLE, &val_b_true, TL_BOOL);
     ck_assert(val_d == (double)val_b_true);
     tl_convert(&val_d, TL_DOUBLE, &val_b_false, TL_BOOL);
     ck_assert(val_d == (double)val_b_false);

     /* TL_FLOAT */
     tl_convert(&val_f, TL_FLOAT, &val_d_max, TL_DOUBLE);
     ck_assert(val_f == val_f_max);
     tl_convert(&val_f, TL_FLOAT, &val_d_min, TL_DOUBLE);
     ck_assert(val_f == val_f_min);
     tl_convert(&val_f, TL_FLOAT, &val_d_normal, TL_DOUBLE);
     ck_assert(val_f == (float)val_d_normal);
     tl_convert(&val_f, TL_FLOAT, &val_f_max, TL_FLOAT);
     ck_assert(val_f == val_f_max);
     tl_convert(&val_f, TL_FLOAT, &val_f_min, TL_FLOAT);
     ck_assert(val_f == val_f_min);
     tl_convert(&val_f, TL_FLOAT, &val_f_normal, TL_FLOAT);
     ck_assert(val_f == val_f_normal);
     tl_convert(&val_f, TL_FLOAT, &val_i32_max, TL_INT32);
     ck_assert(val_f == (float)val_i32_max);
     tl_convert(&val_f, TL_FLOAT, &val_i32_min, TL_INT32);
     ck_assert(val_f == (float)val_i32_min);
     tl_convert(&val_f, TL_FLOAT, &val_i32_normal, TL_INT32);
     ck_assert(val_f == (float)val_i32_normal);
     tl_convert(&val_f, TL_FLOAT, &val_i16_max, TL_INT16);
     ck_assert(val_f == (float)val_i16_max);
     tl_convert(&val_f, TL_FLOAT, &val_i16_min, TL_INT16);
     ck_assert(val_f == (float)val_i16_min);
     tl_convert(&val_f, TL_FLOAT, &val_i16_normal, TL_INT16);
     ck_assert(val_f == (float)val_i16_normal);
     tl_convert(&val_f, TL_FLOAT, &val_i8_max, TL_INT8);
     ck_assert(val_f == (float)val_i8_max);
     tl_convert(&val_f, TL_FLOAT, &val_i8_min, TL_INT8);
     ck_assert(val_f == (float)val_i8_min);
     tl_convert(&val_f, TL_FLOAT, &val_i8_normal, TL_INT8);
     ck_assert(val_f == (float)val_i8_normal);
     tl_convert(&val_f, TL_FLOAT, &val_u32_max, TL_UINT32);
     ck_assert(val_f == (float)val_u32_max);
     tl_convert(&val_f, TL_FLOAT, &val_u32_normal, TL_UINT32);
     ck_assert(val_f == (float)val_u32_normal);
     tl_convert(&val_f, TL_FLOAT, &val_u16_max, TL_UINT16);
     ck_assert(val_f == (float)val_u16_max);
     tl_convert(&val_f, TL_FLOAT, &val_u16_normal, TL_UINT16);
     ck_assert(val_f == (float)val_u16_normal);
     tl_convert(&val_f, TL_FLOAT, &val_u8_max, TL_UINT8);
     ck_assert(val_f == (float)val_u8_max);
     tl_convert(&val_f, TL_FLOAT, &val_u8_normal, TL_UINT8);
     ck_assert(val_f == (float)val_u8_normal);
     tl_convert(&val_f, TL_FLOAT, &val_b_true, TL_BOOL);
     ck_assert(val_f == (float)val_b_true);
     tl_convert(&val_f, TL_FLOAT, &val_b_false, TL_BOOL);
     ck_assert(val_f == (float)val_b_false);

     /* TL_INT32 */
     tl_convert(&val_i32, TL_INT32, &val_d_max, TL_DOUBLE);
     ck_assert(val_i32 == val_i32_max);
     tl_convert(&val_i32, TL_INT32, &val_d_min, TL_DOUBLE);
     ck_assert(val_i32 == val_i32_min);
     tl_convert(&val_i32, TL_INT32, &val_d_normal, TL_DOUBLE);
     ck_assert(val_i32 == (int32_t)val_d_normal);
     tl_convert(&val_i32, TL_INT32, &val_f_max, TL_FLOAT);
     ck_assert(val_i32 == val_i32_max);
     tl_convert(&val_i32, TL_INT32, &val_f_min, TL_FLOAT);
     ck_assert(val_i32 == val_i32_min);
     tl_convert(&val_i32, TL_INT32, &val_f_normal, TL_FLOAT);
     ck_assert(val_i32 == (int32_t)val_f_normal);
     tl_convert(&val_i32, TL_INT32, &val_i32_max, TL_INT32);
     ck_assert(val_i32 == val_i32_max);
     tl_convert(&val_i32, TL_INT32, &val_i32_min, TL_INT32);
     ck_assert(val_i32 == val_i32_min);
     tl_convert(&val_i32, TL_INT32, &val_i32_normal, TL_INT32);
     ck_assert(val_i32 == val_i32_normal);
     tl_convert(&val_i32, TL_INT32, &val_i16_max, TL_INT16);
     ck_assert(val_i32 == (int32_t)val_i16_max);
     tl_convert(&val_i32, TL_INT32, &val_i16_min, TL_INT16);
     ck_assert(val_i32 == (int32_t)val_i16_min);
     tl_convert(&val_i32, TL_INT32, &val_i16_normal, TL_INT16);
     ck_assert(val_i32 == (int32_t)val_i16_normal);
     tl_convert(&val_i32, TL_INT32, &val_i8_max, TL_INT8);
     ck_assert(val_i32 == (int32_t)val_i8_max);
     tl_convert(&val_i32, TL_INT32, &val_i8_min, TL_INT8);
     ck_assert(val_i32 == (int32_t)val_i8_min);
     tl_convert(&val_i32, TL_INT32, &val_i8_normal, TL_INT8);
     ck_assert(val_i32 == (int32_t)val_i8_normal);
     tl_convert(&val_i32, TL_INT32, &val_u32_max, TL_UINT32);
     ck_assert(val_i32 == val_i32_max);
     tl_convert(&val_i32, TL_INT32, &val_u32_normal, TL_UINT32);
     ck_assert(val_i32 == (int32_t)val_u32_normal);
     tl_convert(&val_i32, TL_INT32, &val_u16_max, TL_UINT16);
     ck_assert(val_i32 == (int32_t)val_u16_max);
     tl_convert(&val_i32, TL_INT32, &val_u16_normal, TL_UINT16);
     ck_assert(val_i32 == (int32_t)val_u16_normal);
     tl_convert(&val_i32, TL_INT32, &val_u8_max, TL_UINT8);
     ck_assert(val_i32 == (int32_t)val_u8_max);
     tl_convert(&val_i32, TL_INT32, &val_u8_normal, TL_UINT8);
     ck_assert(val_i32 == (int32_t)val_u8_normal);
     tl_convert(&val_i32, TL_INT32, &val_b_true, TL_BOOL);
     ck_assert(val_i32 == (int32_t)val_b_true);
     tl_convert(&val_i32, TL_INT32, &val_b_false, TL_BOOL);
     ck_assert(val_i32 == (int32_t)val_b_false);

     /* TL_INT16 */
     tl_convert(&val_i16, TL_INT16, &val_d_max, TL_DOUBLE);
     ck_assert(val_i16 == val_i16_max);
     tl_convert(&val_i16, TL_INT16, &val_d_min, TL_DOUBLE);
     ck_assert(val_i16 == val_i16_min);
     tl_convert(&val_i16, TL_INT16, &val_d_normal, TL_DOUBLE);
     ck_assert(val_i16 == (int16_t)val_d_normal);
     tl_convert(&val_i16, TL_INT16, &val_f_max, TL_FLOAT);
     ck_assert(val_i16 == val_i16_max);
     tl_convert(&val_i16, TL_INT16, &val_f_min, TL_FLOAT);
     ck_assert(val_i16 == val_i16_min);
     tl_convert(&val_i16, TL_INT16, &val_f_normal, TL_FLOAT);
     ck_assert(val_i16 == (int16_t)val_f_normal);
     tl_convert(&val_i16, TL_INT16, &val_i32_max, TL_INT32);
     ck_assert(val_i16 == val_i16_max);
     tl_convert(&val_i16, TL_INT16, &val_i32_min, TL_INT32);
     ck_assert(val_i16 == val_i16_min);
     tl_convert(&val_i16, TL_INT16, &val_i32_normal, TL_INT32);
     ck_assert(val_i16 == (int16_t)val_i32_normal);
     tl_convert(&val_i16, TL_INT16, &val_i16_max, TL_INT16);
     ck_assert(val_i16 == val_i16_max);
     tl_convert(&val_i16, TL_INT16, &val_i16_min, TL_INT16);
     ck_assert(val_i16 == val_i16_min);
     tl_convert(&val_i16, TL_INT16, &val_i16_normal, TL_INT16);
     ck_assert(val_i16 == val_i16_normal);
     tl_convert(&val_i16, TL_INT16, &val_i8_max, TL_INT8);
     ck_assert(val_i16 == (int16_t)val_i8_max);
     tl_convert(&val_i16, TL_INT16, &val_i8_min, TL_INT8);
     ck_assert(val_i16 == (int16_t)val_i8_min);
     tl_convert(&val_i16, TL_INT16, &val_i8_normal, TL_INT8);
     ck_assert(val_i16 == (int16_t)val_i8_normal);
     tl_convert(&val_i16, TL_INT16, &val_u32_max, TL_UINT32);
     ck_assert(val_i16 == val_i16_max);
     tl_convert(&val_i16, TL_INT16, &val_u32_normal, TL_UINT32);
     ck_assert(val_i16 == (int16_t)val_u32_normal);
     tl_convert(&val_i16, TL_INT16, &val_u16_max, TL_UINT16);
     ck_assert(val_i16 == val_i16_max);
     tl_convert(&val_i16, TL_INT16, &val_u16_normal, TL_UINT16);
     ck_assert(val_i16 == (int16_t)val_u16_normal);
     tl_convert(&val_i16, TL_INT16, &val_u8_max, TL_UINT8);
     ck_assert(val_i16 == (int16_t)val_u8_max);
     tl_convert(&val_i16, TL_INT16, &val_u8_normal, TL_UINT8);
     ck_assert(val_i16 == (int16_t)val_u8_normal);
     tl_convert(&val_i16, TL_INT16, &val_b_true, TL_BOOL);
     ck_assert(val_i16 == (int16_t)val_b_true);
     tl_convert(&val_i16, TL_INT16, &val_b_false, TL_BOOL);
     ck_assert(val_i16 == (int16_t)val_b_false);

     /* TL_INT8 */
     tl_convert(&val_i8, TL_INT8, &val_d_max, TL_DOUBLE);
     ck_assert(val_i8 == val_i8_max);
     tl_convert(&val_i8, TL_INT8, &val_d_min, TL_DOUBLE);
     ck_assert(val_i8 == val_i8_min);
     tl_convert(&val_i8, TL_INT8, &val_d_normal, TL_DOUBLE);
     ck_assert(val_i8 == (int8_t)val_d_normal);
     tl_convert(&val_i8, TL_INT8, &val_f_max, TL_FLOAT);
     ck_assert(val_i8 == val_i8_max);
     tl_convert(&val_i8, TL_INT8, &val_f_min, TL_FLOAT);
     ck_assert(val_i8 == val_i8_min);
     tl_convert(&val_i8, TL_INT8, &val_f_normal, TL_FLOAT);
     ck_assert(val_i8 == (int8_t)val_f_normal);
     tl_convert(&val_i8, TL_INT8, &val_i32_max, TL_INT32);
     ck_assert(val_i8 == val_i8_max);
     tl_convert(&val_i8, TL_INT8, &val_i32_min, TL_INT32);
     ck_assert(val_i8 == val_i8_min);
     tl_convert(&val_i8, TL_INT8, &val_i32_normal, TL_INT32);
     ck_assert(val_i8 == (int8_t)val_i32_normal);
     tl_convert(&val_i8, TL_INT8, &val_i16_max, TL_INT16);
     ck_assert(val_i8 == val_i8_max);
     tl_convert(&val_i8, TL_INT8, &val_i16_min, TL_INT16);
     ck_assert(val_i8 == val_i8_min);
     tl_convert(&val_i8, TL_INT8, &val_i16_normal, TL_INT16);
     ck_assert(val_i8 == val_i8_normal);
     tl_convert(&val_i8, TL_INT8, &val_i8_max, TL_INT8);
     ck_assert(val_i8 == val_i8_max);
     tl_convert(&val_i8, TL_INT8, &val_i8_min, TL_INT8);
     ck_assert(val_i8 == val_i8_min);
     tl_convert(&val_i8, TL_INT8, &val_i8_normal, TL_INT8);
     ck_assert(val_i8 == val_i8_normal);
     tl_convert(&val_i8, TL_INT8, &val_u32_max, TL_UINT32);
     ck_assert(val_i8 == val_i8_max);
     tl_convert(&val_i8, TL_INT8, &val_u32_normal, TL_UINT32);
     ck_assert(val_i8 == (int8_t)val_u32_normal);
     tl_convert(&val_i8, TL_INT8, &val_u16_max, TL_UINT16);
     ck_assert(val_i8 == val_i8_max);
     tl_convert(&val_i8, TL_INT8, &val_u16_normal, TL_UINT16);
     ck_assert(val_i8 == (int8_t)val_u16_normal);
     tl_convert(&val_i8, TL_INT8, &val_u8_max, TL_UINT8);
     ck_assert(val_i8 == val_i8_max);
     tl_convert(&val_i8, TL_INT8, &val_u8_normal, TL_UINT8);
     ck_assert(val_i8 == (int8_t)val_u8_normal);
     tl_convert(&val_i8, TL_INT8, &val_b_true, TL_BOOL);
     ck_assert(val_i8 == (int8_t)val_b_true);
     tl_convert(&val_i8, TL_INT8, &val_b_false, TL_BOOL);
     ck_assert(val_i8 == (int8_t)val_b_false);

     /* TL_UINT32 */
     tl_convert(&val_u32, TL_UINT32, &val_d_max, TL_DOUBLE);
     ck_assert(val_u32 == val_u32_max);
     tl_convert(&val_u32, TL_UINT32, &val_d_min, TL_DOUBLE);
     ck_assert(val_u32 == val_u32_min);
     tl_convert(&val_u32, TL_UINT32, &val_d_normal, TL_DOUBLE);
     ck_assert(val_u32 == (uint32_t)val_d_normal);
     tl_convert(&val_u32, TL_UINT32, &val_f_max, TL_FLOAT);
     ck_assert(val_u32 == val_u32_max);
     tl_convert(&val_u32, TL_UINT32, &val_f_min, TL_FLOAT);
     ck_assert(val_u32 == val_u32_min);
     tl_convert(&val_u32, TL_UINT32, &val_f_normal, TL_FLOAT);
     ck_assert(val_u32 == (uint32_t)val_f_normal);
     tl_convert(&val_u32, TL_UINT32, &val_i32_max, TL_INT32);
     ck_assert(val_u32 == (uint32_t)val_i32_max);
     tl_convert(&val_u32, TL_UINT32, &val_i32_min, TL_INT32);
     ck_assert(val_u32 == val_u32_min);
     tl_convert(&val_u32, TL_UINT32, &val_i32_normal, TL_INT32);
     ck_assert(val_u32 == (uint32_t)val_i32_normal);
     tl_convert(&val_u32, TL_UINT32, &val_i16_max, TL_INT16);
     ck_assert(val_u32 == (uint32_t)val_i16_max);
     tl_convert(&val_u32, TL_UINT32, &val_i16_min, TL_INT16);
     ck_assert(val_u32 == val_u32_min);
     tl_convert(&val_u32, TL_UINT32, &val_i16_normal, TL_INT16);
     ck_assert(val_u32 == (uint32_t)val_i16_normal);
     tl_convert(&val_u32, TL_UINT32, &val_i8_max, TL_INT8);
     ck_assert(val_u32 == (uint32_t)val_i8_max);
     tl_convert(&val_u32, TL_UINT32, &val_i8_min, TL_INT8);
     ck_assert(val_u32 == val_u32_min);
     tl_convert(&val_u32, TL_UINT32, &val_i8_normal, TL_INT8);
     ck_assert(val_u32 == (uint32_t)val_i8_normal);
     tl_convert(&val_u32, TL_UINT32, &val_u32_max, TL_UINT32);
     ck_assert(val_u32 == val_u32_max);
     tl_convert(&val_u32, TL_UINT32, &val_u32_normal, TL_UINT32);
     ck_assert(val_u32 == val_u32_normal);
     tl_convert(&val_u32, TL_UINT32, &val_u16_max, TL_UINT16);
     ck_assert(val_u32 == (uint32_t)val_u16_max);
     tl_convert(&val_u32, TL_UINT32, &val_u16_normal, TL_UINT16);
     ck_assert(val_u32 == (uint32_t)val_u16_normal);
     tl_convert(&val_u32, TL_UINT32, &val_u8_max, TL_UINT8);
     ck_assert(val_u32 == (uint32_t)val_u8_max);
     tl_convert(&val_u32, TL_UINT32, &val_u8_normal, TL_UINT8);
     ck_assert(val_u32 == (uint32_t)val_u8_normal);
     tl_convert(&val_u32, TL_UINT32, &val_b_true, TL_BOOL);
     ck_assert(val_u32 == (uint32_t)val_b_true);
     tl_convert(&val_u32, TL_UINT32, &val_b_false, TL_BOOL);
     ck_assert(val_u32 == (uint32_t)val_b_false);

     /* TL_UINT16 */
     tl_convert(&val_u16, TL_UINT16, &val_d_max, TL_DOUBLE);
     ck_assert(val_u16 == val_u16_max);
     tl_convert(&val_u16, TL_UINT16, &val_d_min, TL_DOUBLE);
     ck_assert(val_u16 == val_u16_min);
     tl_convert(&val_u16, TL_UINT16, &val_d_normal, TL_DOUBLE);
     ck_assert(val_u16 == (uint16_t)val_d_normal);
     tl_convert(&val_u16, TL_UINT16, &val_f_max, TL_FLOAT);
     ck_assert(val_u16 == val_u16_max);
     tl_convert(&val_u16, TL_UINT16, &val_f_min, TL_FLOAT);
     ck_assert(val_u16 == val_u16_min);
     tl_convert(&val_u16, TL_UINT16, &val_f_normal, TL_FLOAT);
     ck_assert(val_u16 == (uint16_t)val_f_normal);
     tl_convert(&val_u16, TL_UINT16, &val_i32_max, TL_INT32);
     ck_assert(val_u16 == val_u16_max);
     tl_convert(&val_u16, TL_UINT16, &val_i32_min, TL_INT32);
     ck_assert(val_u16 == val_u16_min);
     tl_convert(&val_u16, TL_UINT16, &val_i32_normal, TL_INT32);
     ck_assert(val_u16 == (uint16_t)val_i32_normal);
     tl_convert(&val_u16, TL_UINT16, &val_i16_max, TL_INT16);
     ck_assert(val_u16 == (uint16_t)val_i16_max);
     tl_convert(&val_u16, TL_UINT16, &val_i16_min, TL_INT16);
     ck_assert(val_u16 == val_u16_min);
     tl_convert(&val_u16, TL_UINT16, &val_i16_normal, TL_INT16);
     ck_assert(val_u16 == (uint16_t)val_i16_normal);
     tl_convert(&val_u16, TL_UINT16, &val_i8_max, TL_INT8);
     ck_assert(val_u16 == (uint16_t)val_i8_max);
     tl_convert(&val_u16, TL_UINT16, &val_i8_min, TL_INT8);
     ck_assert(val_u16 == val_u16_min);
     tl_convert(&val_u16, TL_UINT16, &val_i8_normal, TL_INT8);
     ck_assert(val_u16 == (uint16_t)val_i8_normal);
     tl_convert(&val_u16, TL_UINT16, &val_u32_max, TL_UINT32);
     ck_assert(val_u16 == val_u16_max);
     tl_convert(&val_u16, TL_UINT16, &val_u32_normal, TL_UINT32);
     ck_assert(val_u16 == (uint16_t)val_u32_normal);
     tl_convert(&val_u16, TL_UINT16, &val_u16_max, TL_UINT16);
     ck_assert(val_u16 == val_u16_max);
     tl_convert(&val_u16, TL_UINT16, &val_u16_normal, TL_UINT16);
     ck_assert(val_u16 == val_u16_normal);
     tl_convert(&val_u16, TL_UINT16, &val_u8_max, TL_UINT8);
     ck_assert(val_u16 == (uint16_t)val_u8_max);
     tl_convert(&val_u16, TL_UINT16, &val_u8_normal, TL_UINT8);
     ck_assert(val_u16 == (uint16_t)val_u8_normal);
     tl_convert(&val_u16, TL_UINT16, &val_b_true, TL_BOOL);
     ck_assert(val_u16 == (uint16_t)val_b_true);
     tl_convert(&val_u16, TL_UINT16, &val_b_false, TL_BOOL);
     ck_assert(val_u16 == (uint16_t)val_b_false);

     /* TL_UINT8 */
     tl_convert(&val_u8, TL_UINT8, &val_d_max, TL_DOUBLE);
     ck_assert(val_u8 == val_u8_max);
     tl_convert(&val_u8, TL_UINT8, &val_d_min, TL_DOUBLE);
     ck_assert(val_u8 == val_u8_min);
     tl_convert(&val_u8, TL_UINT8, &val_d_normal, TL_DOUBLE);
     ck_assert(val_u8 == (uint8_t)val_d_normal);
     tl_convert(&val_u8, TL_UINT8, &val_f_max, TL_FLOAT);
     ck_assert(val_u8 == val_u8_max);
     tl_convert(&val_u8, TL_UINT8, &val_f_min, TL_FLOAT);
     ck_assert(val_u8 == val_u8_min);
     tl_convert(&val_u8, TL_UINT8, &val_f_normal, TL_FLOAT);
     ck_assert(val_u8 == (uint8_t)val_f_normal);
     tl_convert(&val_u8, TL_UINT8, &val_i32_max, TL_INT32);
     ck_assert(val_u8 == val_u8_max);
     tl_convert(&val_u8, TL_UINT8, &val_i32_min, TL_INT32);
     ck_assert(val_u8 == val_u8_min);
     tl_convert(&val_u8, TL_UINT8, &val_i32_normal, TL_INT32);
     ck_assert(val_u8 == (uint8_t)val_i32_normal);
     tl_convert(&val_u8, TL_UINT8, &val_i16_max, TL_INT16);
     ck_assert(val_u8 == val_u8_max);
     tl_convert(&val_u8, TL_UINT8, &val_i16_min, TL_INT16);
     ck_assert(val_u8 == val_u8_min);
     tl_convert(&val_u8, TL_UINT8, &val_i16_normal, TL_INT16);
     ck_assert(val_u8 == (uint8_t)val_i16_normal);
     tl_convert(&val_u8, TL_UINT8, &val_i8_max, TL_INT8);
     ck_assert(val_u8 == (uint8_t)val_i8_max);
     tl_convert(&val_u8, TL_UINT8, &val_i8_min, TL_INT8);
     ck_assert(val_u8 == val_u8_min);
     tl_convert(&val_u8, TL_UINT8, &val_i8_normal, TL_INT8);
     ck_assert(val_u8 == (uint8_t)val_i8_normal);
     tl_convert(&val_u8, TL_UINT8, &val_u32_max, TL_UINT32);
     ck_assert(val_u8 == val_u8_max);
     tl_convert(&val_u8, TL_UINT8, &val_u32_normal, TL_UINT32);
     ck_assert(val_u8 == (uint8_t)val_u32_normal);
     tl_convert(&val_u8, TL_UINT8, &val_u16_max, TL_UINT16);
     ck_assert(val_u8 == val_u8_max);
     tl_convert(&val_u8, TL_UINT8, &val_u16_normal, TL_UINT16);
     ck_assert(val_u8 == (uint8_t)val_u16_normal);
     tl_convert(&val_u8, TL_UINT8, &val_u8_max, TL_UINT8);
     ck_assert(val_u8 == val_u8_max);
     tl_convert(&val_u8, TL_UINT8, &val_u8_normal, TL_UINT8);
     ck_assert(val_u8 == val_u8_normal);
     tl_convert(&val_u8, TL_UINT8, &val_b_true, TL_BOOL);
     ck_assert(val_u8 == (uint8_t)val_b_true);
     tl_convert(&val_u8, TL_UINT8, &val_b_false, TL_BOOL);
     ck_assert(val_u8 == (uint8_t)val_b_false);

     /* TL_BOOL */
     tl_convert(&val_b, TL_BOOL, &val_d_max, TL_DOUBLE);
     ck_assert(val_b == val_b_true);
     tl_convert(&val_b, TL_BOOL, &val_d_min, TL_DOUBLE);
     ck_assert(val_b == val_b_true);
     tl_convert(&val_b, TL_BOOL, &val_d_normal, TL_DOUBLE);
     ck_assert(val_b == val_b_true);
     tl_convert(&val_b, TL_BOOL, &val_f_max, TL_FLOAT);
     ck_assert(val_b == val_b_true);
     tl_convert(&val_b, TL_BOOL, &val_f_min, TL_FLOAT);
     ck_assert(val_b == val_b_true);
     tl_convert(&val_b, TL_BOOL, &val_f_normal, TL_FLOAT);
     ck_assert(val_b == val_b_true);
     tl_convert(&val_b, TL_BOOL, &val_i32_max, TL_INT32);
     ck_assert(val_b == val_b_true);
     tl_convert(&val_b, TL_BOOL, &val_i32_min, TL_INT32);
     ck_assert(val_b == val_b_true);
     tl_convert(&val_b, TL_BOOL, &val_i32_normal, TL_INT32);
     ck_assert(val_b == val_b_true);
     tl_convert(&val_b, TL_BOOL, &val_i16_max, TL_INT16);
     ck_assert(val_b == val_b_true);
     tl_convert(&val_b, TL_BOOL, &val_i16_min, TL_INT16);
     ck_assert(val_b == val_b_true);
     tl_convert(&val_b, TL_BOOL, &val_i16_normal, TL_INT16);
     ck_assert(val_b == val_b_true);
     tl_convert(&val_b, TL_BOOL, &val_i8_max, TL_INT8);
     ck_assert(val_b == val_b_true);
     tl_convert(&val_b, TL_BOOL, &val_i8_min, TL_INT8);
     ck_assert(val_b == val_b_true);
     tl_convert(&val_b, TL_BOOL, &val_i8_normal, TL_INT8);
     ck_assert(val_b == val_b_true);
     tl_convert(&val_b, TL_BOOL, &val_u32_max, TL_UINT32);
     ck_assert(val_b == val_b_true);
     tl_convert(&val_b, TL_BOOL, &val_u32_normal, TL_UINT32);
     ck_assert(val_b == val_b_true);
     tl_convert(&val_b, TL_BOOL, &val_u16_max, TL_UINT16);
     ck_assert(val_b == val_b_true);
     tl_convert(&val_b, TL_BOOL, &val_u16_normal, TL_UINT16);
     ck_assert(val_b == val_b_true);
     tl_convert(&val_b, TL_BOOL, &val_u8_max, TL_UINT8);
     ck_assert(val_b == val_b_true);
     tl_convert(&val_b, TL_BOOL, &val_u8_normal, TL_UINT8);
     ck_assert(val_b == val_b_true);
     tl_convert(&val_b, TL_BOOL, &val_b_true, TL_BOOL);
     ck_assert(val_b == val_b_true);
     tl_convert(&val_b, TL_BOOL, &val_b_false, TL_BOOL);
     ck_assert(val_b == val_b_false);
}
END_TEST
/* end of tests */

Suite *make_type_suite(void)
{
     Suite *s;
     TCase *tc_type;

     s = suite_create("type");
     tc_type = tcase_create("type");
     tcase_add_checked_fixture(tc_type, setup, teardown);

     tcase_add_test(tc_type, test_tl_size_of);
     tcase_add_test(tc_type, test_tl_psub);
     tcase_add_test(tc_type, test_tl_padd);
     tcase_add_test(tc_type, test_tl_passign);
     tcase_add_test(tc_type, test_tl_dtype_fmt);
     tcase_add_test(tc_type, test_tl_pointer_sub);
     tcase_add_test(tc_type, test_tl_pointer_add);
     tcase_add_test(tc_type, test_tl_pointer_assign);
     tcase_add_test(tc_type, test_tl_fprintf);
     tcase_add_test(tc_type, test_tl_fprintf_getfunc);
     tcase_add_test(tc_type, test_tl_cmp);
     tcase_add_test(tc_type, test_tl_cmp_getfunc);
     tcase_add_test(tc_type, test_tl_elew);
     tcase_add_test(tc_type, test_tl_elew_getfunc);
     tcase_add_test(tc_type, test_tl_convert);
     /* end of adding tests */

     suite_add_tcase(s, tc_type);

     return s;
}
