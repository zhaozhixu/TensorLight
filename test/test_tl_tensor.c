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

#include "test_tensorlight.h"
#include "../src/tl_tensor.h"
#include "../src/tl_util.h"
#include "../src/tl_check.h"

#define ARR(type, varg...) (type[]){varg}

static void setup(void)
{
}

static void teardown(void)
{
}

START_TEST(test_tl_tensor_create)
{
     tl_tensor *t;
     int dims[3] = {1, 2, 3};
     int32_t data[6] = {1, 2, 3, 4, 5, 6};
     int i;

     t = tl_tensor_create(NULL, 3, (int[]){1, 2, 3}, TL_DOUBLE);
     ck_assert_int_eq(t->ndim, 3);
     ck_assert_int_eq(t->dtype, TL_DOUBLE);
     ck_assert_int_eq(t->len, 6);
     for (i = 0; i < t->ndim; i++)
          ck_assert(t->dims[i] == dims[i]);
     ck_assert_ptr_eq(t->data, NULL);
     tl_tensor_free_data_too(t);

     t = tl_tensor_create(NULL, 3, dims, TL_FLOAT);
     ck_assert_int_eq(t->ndim, 3);
     ck_assert_int_eq(t->dtype, TL_FLOAT);
     ck_assert_int_eq(t->len, 6);
     for (i = 0; i < t->ndim; i++)
          ck_assert(t->dims[i] == dims[i]);
     ck_assert_ptr_eq(t->data, NULL);
     tl_tensor_free_data_too(t);

     t = tl_tensor_create(data, 3, dims, TL_INT32);
     ck_assert_int_eq(t->ndim, 3);
     ck_assert_int_eq(t->dtype, TL_INT32);
     ck_assert_int_eq(t->len, 6);
     for (i = 0; i < t->ndim; i++)
          ck_assert(t->dims[i] == dims[i]);
     for (i = 0; i < t->len; i++)
          ck_assert(((int32_t *)t->data)[i] == data[i]);
     tl_tensor_free(t);

     t = tl_tensor_create(data, 3, dims, TL_INT32);
     ck_assert_int_eq(t->ndim, 3);
     ck_assert_int_eq(t->dtype, TL_INT32);
     ck_assert_int_eq(t->len, 6);
     for (i = 0; i < t->ndim; i++)
          ck_assert(t->dims[i] == dims[i]);
     for (i = 0; i < t->len; i++)
          ck_assert(((int32_t *)t->data)[i] == data[i]);
     tl_tensor_free(t);
}
END_TEST

START_TEST(test_tl_tensor_free)
{
}
END_TEST

START_TEST(test_tl_tensor_clone)
{
     tl_tensor *t1, *t2;
     int dims[3] = {1, 2, 3};
     int32_t data[6] = {1, 2, 3, 4, 5, 6};
     int i;

     t1 = tl_tensor_create(data, 3, dims, TL_INT32);
     t2 = tl_tensor_clone(t1);
     ck_assert_int_eq(t2->ndim, 3);
     ck_assert_int_eq(t2->dtype, TL_INT32);
     ck_assert_int_eq(t2->len, 6);
     for (i = 0; i < t2->ndim; i++)
          ck_assert(t2->dims[i] == dims[i]);
     for (i = 0; i < t2->len; i++)
          ck_assert(((int32_t *)t2->data)[i] == data[i]);
     tl_tensor_free(t1);
     tl_tensor_free_data_too(t2);
}
END_TEST

START_TEST(test_tl_tensor_repeat)
{
     int dims[] = {2, 3};
     float data[] = {1, 2, 3, 4, 5, 6};
     int dims1[] = {1, 2, 3};
     float data1[] = {1, 2, 3, 4, 5, 6};
     int dims2[] = {2, 2, 3};
     float data2[] = {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
     int dims3[] = {3, 2, 3};
     float data3[] = {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
     tl_tensor *t, *t1, *t2, *t3;

     t = tl_tensor_create(data, 2, dims, TL_FLOAT);

     t1 = tl_tensor_repeat(t, 1);
     ck_assert_array_int_eq(t1->dims, dims1, 3);
     ck_assert_array_float_eq_tol((float*)t1->data, data1, t1->len, 0);
     tl_tensor_free_data_too(t1);

     t2 = tl_tensor_repeat(t, 2);
     ck_assert_array_int_eq(t2->dims, dims2, 3);
     ck_assert_array_float_eq_tol((float*)t2->data, data2, t2->len, 0);
     tl_tensor_free_data_too(t2);

     t3 = tl_tensor_repeat(t, 3);
     ck_assert_array_int_eq(t3->dims, dims3, 3);
     ck_assert_array_float_eq_tol((float*)t3->data, data3, t3->len, 0);
     tl_tensor_free_data_too(t3);

     tl_tensor_free(t);
}
END_TEST

START_TEST(test_tl_tensor_arange)
{
     tl_tensor *dst, *t;

     t = tl_tensor_create(ARR(int16_t,0,1,2), 1, ARR(int,3), TL_INT16);
     dst = tl_tensor_arange(0, 3, 1, TL_INT16);
     tl_assert_tensor_eq(dst, t);
     tl_tensor_free(t);
     tl_tensor_free_data_too(dst);

     t = tl_tensor_create(ARR(float,0.1,1.6,3.1), 1, ARR(int,3), TL_FLOAT);
     dst = tl_tensor_arange(0.1, 3.2, 1.5, TL_FLOAT);
     tl_assert_tensor_eq(dst, t);
     tl_tensor_free(t);
     tl_tensor_free_data_too(dst);

     t = tl_tensor_create(ARR(float,0.1,1.6), 1, ARR(int,2), TL_FLOAT);
     dst = tl_tensor_arange(0.1, 3.1, 1.5, TL_FLOAT);
     tl_assert_tensor_eq(dst, t);
     tl_tensor_free(t);
     tl_tensor_free_data_too(dst);
}
END_TEST

START_TEST(test_tl_tensor_rearange)
{
     tl_tensor *src, *t;

     t = tl_tensor_create(ARR(int16_t,0,1,2), 1, ARR(int,3), TL_INT16);
     src = tl_tensor_zeros(1, ARR(int, 3), TL_INT16);
     tl_tensor_rearange(src, 0, 3, 1);
     tl_assert_tensor_eq(src, t);
     tl_tensor_free(t);
     tl_tensor_free_data_too(src);

     t = tl_tensor_create(ARR(float,0.1,1.6,3.1), 1, ARR(int,3), TL_FLOAT);
     src = tl_tensor_zeros(1, ARR(int, 3), TL_FLOAT);
     tl_tensor_rearange(src, 0.1, 3.2, 1.5);
     tl_assert_tensor_eq(src, t);
     tl_tensor_free(t);
     tl_tensor_free_data_too(src);

     t = tl_tensor_create(ARR(float,0.1,1.6), 1, ARR(int,2), TL_FLOAT);
     src = tl_tensor_zeros(1, ARR(int, 2), TL_FLOAT);
     tl_tensor_rearange(src, 0.1, 3.1, 1.5);
     tl_assert_tensor_eq(src, t);
     tl_tensor_free(t);
     tl_tensor_free_data_too(src);
}
END_TEST

START_TEST(test_tl_tensor_issameshape)
{
     tl_tensor *t1;
     tl_tensor *t2;

     t1 = tl_tensor_zeros(2, (int[]){3, 3}, TL_FLOAT);
     t2 = tl_tensor_zeros(2, (int[]){3, 3}, TL_FLOAT);
     ck_assert_int_eq(tl_tensor_issameshape(t1, t2), 1);
     tl_tensor_free_data_too(t1);
     tl_tensor_free_data_too(t2);
}
END_TEST

START_TEST(test_tl_tensor_fprint)
{
     tl_tensor *t;
     FILE *fp;
     char s[BUFSIZ];

     t = tl_tensor_zeros(3, (int[]){1, 2, 3}, TL_FLOAT);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     tl_tensor_fprint(fp, t, NULL);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 100, fp), NULL);
     ck_assert_ptr_ne(fgets(s+strlen(s), 100, fp), NULL);
     ck_assert_str_eq(s, "[[[0.000 0.000 0.000]\n"
                         "  [0.000 0.000 0.000]]]\n");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     tl_tensor_fprint(fp, t, "%.4f");
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 100, fp), NULL);
     ck_assert_ptr_ne(fgets(s+strlen(s), 100, fp), NULL);
     ck_assert_str_eq(s, "[[[0.0000 0.0000 0.0000]\n"
                         "  [0.0000 0.0000 0.0000]]]\n");
     fclose(fp);

     tl_tensor_free_data_too(t);
}
END_TEST

START_TEST(test_tl_tensor_print)
{
}
END_TEST

START_TEST(test_tl_tensor_save)
{
     tl_tensor *t;
     FILE *fp;
     char s[BUFSIZ];

     t = tl_tensor_zeros(3, (int[]){1, 2, 3}, TL_FLOAT);

     tl_tensor_save("__test_tensor_save_tmp", t, NULL);
     fp = fopen("__test_tensor_save_tmp", "r");
     ck_assert_ptr_ne(fp, NULL);
     ck_assert_ptr_ne(fgets(s, 100, fp), NULL);
     ck_assert_ptr_ne(fgets(s+strlen(s), 100, fp), NULL);
     ck_assert_str_eq(s, "[[[0.000 0.000 0.000]\n"
                         "  [0.000 0.000 0.000]]]\n");
     fclose(fp);
     ck_assert_int_eq(remove("__test_tensor_save_tmp"), 0);

     /* ck_assert_int_lt(tl_tensor_save("__non_exist_dir/tmp", t, NULL), 0); */
     tl_tensor_free_data_too(t);
}
END_TEST

START_TEST(test_tl_tensor_zeros_slice)
{
     tl_tensor *t1, *t2;
     int i;

     t1 = tl_tensor_zeros(1, (int[]){1}, TL_INT8);
     t2 = tl_tensor_zeros_slice(t1, 0, 1, TL_INT8);
     ck_assert_int_eq(t2->ndim, 1);
     ck_assert_int_eq(t2->dtype, TL_INT8);
     ck_assert_int_eq(t2->len, 1);
     ck_assert(t2->dims[0] == 1);
     for (i = 0; i < t2->len; i++)
          ck_assert(((int8_t *)t2->data)[i] == 0);

     tl_tensor_free_data_too(t1);
     tl_tensor_free_data_too(t2);

     t1 = tl_tensor_zeros(3, (int[]){1, 2, 3}, TL_INT16);
     t2 = tl_tensor_zeros_slice(t1, 2, 2, TL_UINT8);
     ck_assert_int_eq(t2->ndim, 3);
     ck_assert_int_eq(t2->dtype, TL_UINT8);
     ck_assert_int_eq(t2->len, 4);
     ck_assert(t2->dims[0] == 1);
     ck_assert(t2->dims[1] == 2);
     ck_assert(t2->dims[2] == 2);
     for (i = 0; i < t2->len; i++)
          ck_assert(((uint8_t *)t2->data)[i] == 0);
     tl_tensor_free_data_too(t1);
     tl_tensor_free_data_too(t2);
}
END_TEST

START_TEST(test_tl_tensor_slice)
{
     tl_tensor *t1, *t2;
     int ndim = 3;
     int dims[3] = {1, 2, 3};
     uint16_t data[6] = {1, 2, 3, 4, 5, 6};
     uint16_t data_slice1[4] = {2, 3, 5, 6};
     uint16_t data_slice2[3] = {1, 2, 3};
     int i;

     t1 = tl_tensor_create(data, ndim, dims, TL_UINT16);
     t2 = tl_tensor_slice(t1, NULL, 2, 1, 2);
     ck_assert_int_eq(t2->ndim, 3);
     ck_assert_int_eq(t2->dtype, TL_UINT16);
     ck_assert_int_eq(t2->len, 4);
     ck_assert(t2->dims[0] == 1);
     ck_assert(t2->dims[1] == 2);
     ck_assert(t2->dims[2] == 2);
     for (i = 0; i < t2->len; i++)
          ck_assert(((uint16_t *)t2->data)[i] == data_slice1[i]);
     tl_tensor_free(t1);
     tl_tensor_free_data_too(t2);

     t1 = tl_tensor_create(data, ndim, dims, TL_UINT16);
     t2 = tl_tensor_zeros_slice(t1, 1, 1, TL_UINT16);
     t2 = tl_tensor_slice(t1, t2, 1, 0, 1);
     ck_assert_int_eq(t2->ndim, 3);
     ck_assert_int_eq(t2->dtype, TL_UINT16);
     ck_assert_int_eq(t2->len, 3);
     ck_assert(t2->dims[0] == 1);
     ck_assert(t2->dims[1] == 1);
     ck_assert(t2->dims[2] == 3);
     for (i = 0; i < t2->len; i++)
          ck_assert(((uint16_t *)t2->data)[i] == data_slice2[i]);
     tl_tensor_free(t1);
     tl_tensor_free_data_too(t2);
}
END_TEST

START_TEST(test_tl_tensor_slice_nocopy)
{
     tl_tensor *t1, *t2;
     int ndim = 2;
     int dims[] = {3, 2};
     uint16_t data[6] = {1, 2, 3, 4, 5, 6};
     uint16_t data_slice1[] = {1, 2, 3, 4};
     uint16_t data_slice2[] = {3, 4, 5, 6};
     int i;

     t1 = tl_tensor_create(data, ndim, dims, TL_UINT16);
     t2 = tl_tensor_slice_nocopy(t1, NULL, 0, 0, 2);
     ck_assert_int_eq(t2->ndim, 2);
     ck_assert_int_eq(t2->dtype, TL_UINT16);
     ck_assert_int_eq(t2->len, 4);
     ck_assert(t2->dims[0] == 2);
     ck_assert(t2->dims[1] == 2);
     ck_assert(t2->owner == t1);
     for (i = 0; i < t2->len; i++)
          ck_assert(((uint16_t *)t2->data)[i] == data_slice1[i]);
     tl_tensor_free(t1);
     tl_tensor_free(t2);

     t1 = tl_tensor_create(data, ndim, dims, TL_UINT16);
     t2 = tl_tensor_create_slice(NULL, t1, 0, 2, TL_UINT16);
     t2 = tl_tensor_slice_nocopy(t1, t2, 0, 1, 2);
     ck_assert_int_eq(t2->ndim, 2);
     ck_assert_int_eq(t2->dtype, TL_UINT16);
     ck_assert_int_eq(t2->len, 4);
     ck_assert(t2->dims[0] == 2);
     ck_assert(t2->dims[1] == 2);
     ck_assert(t2->owner == t1);
     for (i = 0; i < t2->len; i++)
          ck_assert(((uint16_t *)t2->data)[i] == data_slice2[i]);
     tl_tensor_free(t1);
     tl_tensor_free(t2);
}
END_TEST

START_TEST(test_tl_tensor_concat)
{
     tl_tensor *t, *t1, *t2, *t3, *t4, *t5, *t6;
     int ndim = 3;
     int dims[] = {2, 2, 3};
     uint16_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
     int dims1[] = {1, 2, 3};
     uint16_t data_slice1[] = {1, 2, 3, 4, 5, 6};
     int dims2[] = {1, 2, 3};
     uint16_t data_slice2[] = {7, 8, 9, 10, 11, 12};
     int dims3[] = {2, 1, 3};
     uint16_t data_slice3[] = {1, 2, 3, 7, 8, 9};
     int dims4[] = {2, 1, 3};
     uint16_t data_slice4[] = {4, 5, 6, 10, 11, 12};
     int dims5[] = {2, 2, 1};
     uint16_t data_slice5[] = {1, 4, 7, 10};
     int dims6[] = {2, 2, 2};
     uint16_t data_slice6[] = {2, 3, 5, 6, 8, 9, 11, 12};

     t = tl_tensor_zeros(ndim, dims, TL_UINT16);
     t1 = tl_tensor_create(data_slice1, ndim, dims1, TL_UINT16);
     t2 = tl_tensor_create(data_slice2, ndim, dims2, TL_UINT16);
     tl_tensor_concat(t1, t2, t, 0);
     ck_assert_array_uint_eq((uint16_t*)t->data, data, t->len);
     tl_tensor_free_data_too(t);
     tl_tensor_free(t1);
     tl_tensor_free(t2);

     t = tl_tensor_zeros(ndim, dims, TL_UINT16);
     t3 = tl_tensor_create(data_slice3, ndim, dims3, TL_UINT16);
     t4 = tl_tensor_create(data_slice4, ndim, dims4, TL_UINT16);
     tl_tensor_concat(t3, t4, t, 1);
     ck_assert_array_uint_eq((uint16_t*)t->data, data, t->len);
     tl_tensor_free_data_too(t);
     tl_tensor_free(t3);
     tl_tensor_free(t4);

     t5 = tl_tensor_create(data_slice5, ndim, dims5, TL_UINT16);
     t6 = tl_tensor_create(data_slice6, ndim, dims6, TL_UINT16);
     t = tl_tensor_concat(t5, t6, NULL, 2);
     ck_assert_array_uint_eq((uint16_t*)t->data, data, t->len);
     tl_tensor_free_data_too(t);
     tl_tensor_free(t5);
     tl_tensor_free(t6);
}
END_TEST

START_TEST(test_tl_tensor_reshape)
{
     tl_tensor *t1, *t2;
     int dims1[3] = {1, 2, 3};
     int dims2[2] = {1, 6};
     int data[6] = {1, 2, 3, 4, 5, 6};
     int i;

     t1 = tl_tensor_create(data, 3, dims1, TL_UINT32);
     t2 = tl_tensor_reshape(t1, 2, dims2);
     ck_assert_int_eq(t2->ndim, 2);
     ck_assert_int_eq(t2->dtype, TL_UINT32);
     ck_assert_int_eq(t2->len, 6);
     ck_assert(t2->dims[0] == 1);
     ck_assert(t2->dims[1] == 6);
     for (i = 0; i < t2->len; i++)
          ck_assert(((uint32_t *)t2->data)[i] == data[i]);
     tl_tensor_free(t1);
     tl_tensor_free(t2);
}
END_TEST

START_TEST(test_tl_tensor_maxreduce)
{
     tl_tensor *src, *dst, *arg;
     int dims[3] = {2, 3, 2};
     int32_t data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
     int dst_data1[4] = {5, 6, 11, 12};
     int dst_data2[6] = {7, 8, 9, 10, 11, 12};
     int dst_data3[6] = {2, 4, 6, 8, 10, 12};
     int arg_data[6] = {1, 1, 1, 1, 1, 1};
     int i;

     src = tl_tensor_create(data, 3, dims, TL_INT32);
     dst = tl_tensor_maxreduce(src, NULL, NULL, 1);
     ck_assert_int_eq(dst->ndim, 3);
     ck_assert_int_eq(dst->dtype, TL_INT32);
     ck_assert_int_eq(dst->len, 4);
     ck_assert(dst->dims[0] == 2);
     ck_assert(dst->dims[1] == 1);
     ck_assert(dst->dims[2] == 2);
     for (i = 0; i < dst->len; i++)
          ck_assert(((int32_t *)dst->data)[i] == dst_data1[i]);
     tl_tensor_free_data_too(dst);

     dst = tl_tensor_zeros_slice(src, 0, 1, TL_INT32);
     dst = tl_tensor_maxreduce(src, dst, NULL, 0);
     ck_assert_int_eq(dst->ndim, 3);
     ck_assert_int_eq(dst->dtype, TL_INT32);
     ck_assert_int_eq(dst->len, 6);
     ck_assert(dst->dims[0] == 1);
     ck_assert(dst->dims[1] == 3);
     ck_assert(dst->dims[2] == 2);
     for (i = 0; i < dst->len; i++)
          ck_assert(((int32_t *)dst->data)[i] == dst_data2[i]);
     tl_tensor_free_data_too(dst);

     dst = tl_tensor_zeros_slice(src, 2, 1, TL_INT32);
     arg = tl_tensor_zeros_slice(src, 2, 1, TL_INT32);
     dst = tl_tensor_maxreduce(src, dst, arg, 2);
     ck_assert_int_eq(dst->ndim, 3);
     ck_assert_int_eq(dst->dtype, TL_INT32);
     ck_assert_int_eq(dst->len, 6);
     ck_assert(dst->dims[0] == 2);
     ck_assert(dst->dims[1] == 3);
     ck_assert(dst->dims[2] == 1);
     for (i = 0; i < dst->len; i++)
          ck_assert(((int32_t *)dst->data)[i] == dst_data3[i]);
     ck_assert_int_eq(arg->ndim, 3);
     ck_assert_int_eq(arg->dtype, TL_INT32);
     ck_assert_int_eq(arg->len, 6);
     ck_assert(arg->dims[0] == 2);
     ck_assert(arg->dims[1] == 3);
     ck_assert(arg->dims[2] == 1);
     for (i = 0; i < arg->len; i++)
          ck_assert(((int32_t *)arg->data)[i] == arg_data[i]);
     tl_tensor_free_data_too(dst);
     tl_tensor_free_data_too(arg);

     tl_tensor_free(src);
}
END_TEST

START_TEST(test_tl_tensor_elew)
{
     tl_tensor *src1, *src2, *dst;
     int8_t src1_data[6] = {1, 1, 2, 2, 3, 3};
     int8_t src2_data[6] = {1, 2, 3, 4, 5, 6};
     int8_t dst_data[6] = {1, 2, 6, 8, 15, 18};
     int dims[2] = {2, 3};
     int i;

     src1 = tl_tensor_create(src1_data, 2, dims, TL_INT8);
     src2 = tl_tensor_create(src2_data, 2, dims, TL_INT8);
     dst = tl_tensor_elew(src1, src2, NULL, TL_MUL);
     ck_assert_int_eq(dst->ndim, 2);
     ck_assert_int_eq(dst->dtype, TL_INT8);
     ck_assert_int_eq(dst->len, 6);
     ck_assert(dst->dims[0] == 2);
     ck_assert(dst->dims[1] == 3);
     for (i = 0; i < dst->len; i++)
          ck_assert(((int8_t *)dst->data)[i] == dst_data[i]);
     tl_tensor_free_data_too(dst);

     src1 = tl_tensor_create(src1_data, 2, dims, TL_INT8);
     src2 = tl_tensor_create(src2_data, 2, dims, TL_INT8);
     dst = tl_tensor_zeros(2, dims, TL_INT8);
     dst = tl_tensor_elew(src1, src2, dst, TL_MUL);
     ck_assert_int_eq(dst->ndim, 2);
     ck_assert_int_eq(dst->dtype, TL_INT8);
     ck_assert_int_eq(dst->len, 6);
     ck_assert(dst->dims[0] == 2);
     ck_assert(dst->dims[1] == 3);
     for (i = 0; i < dst->len; i++)
          ck_assert(((int8_t *)dst->data)[i] == dst_data[i]);
     tl_tensor_free_data_too(dst);

     tl_tensor_free(src1);
     tl_tensor_free(src2);
}
END_TEST

START_TEST(test_tl_tensor_elew_param)
{
     tl_tensor *src, *dst;
     int8_t src_data[6] = {1, 1, 2, 2, 3, 3};
     double param = 2;
     int8_t dst_data[6] = {2, 2, 4, 4, 6, 6};
     int dims[2] = {2, 3};
     int i;

     src = tl_tensor_create(src_data, 2, dims, TL_INT8);
     dst = tl_tensor_elew_param(src, param, NULL, TL_MUL);
     ck_assert_int_eq(dst->ndim, 2);
     ck_assert_int_eq(dst->dtype, TL_INT8);
     ck_assert_int_eq(dst->len, 6);
     ck_assert(dst->dims[0] == 2);
     ck_assert(dst->dims[1] == 3);
     for (i = 0; i < dst->len; i++)
          ck_assert(((int8_t *)dst->data)[i] == dst_data[i]);
     tl_tensor_free_data_too(dst);

     src = tl_tensor_create(src_data, 2, dims, TL_INT8);
     dst = tl_tensor_zeros(2, dims, TL_INT8);
     dst = tl_tensor_elew_param(src, param, dst, TL_MUL);
     ck_assert_int_eq(dst->ndim, 2);
     ck_assert_int_eq(dst->dtype, TL_INT8);
     ck_assert_int_eq(dst->len, 6);
     ck_assert(dst->dims[0] == 2);
     ck_assert(dst->dims[1] == 3);
     for (i = 0; i < dst->len; i++)
          ck_assert(((int8_t *)dst->data)[i] == dst_data[i]);
     tl_tensor_free_data_too(dst);

     tl_tensor_free(src);
}
END_TEST

START_TEST(test_tl_tensor_dot_product)
{
     tl_tensor *src1, *src2, *dst;
     int8_t src1_data[6] = {1, 1, 2, 2, 3, 3};
     int8_t src2_data[6] = {1, 2, 3, 4, 5, 6};
     int8_t dst_data[1] = {50};
     int dims[2] = {2, 3};

     src1 = tl_tensor_create(src1_data, 2, dims, TL_INT8);
     src2 = tl_tensor_create(src2_data, 2, dims, TL_INT8);
     dst = tl_tensor_dot_product(src1, src2, NULL);
     ck_assert_int_eq(dst->ndim, 1);
     ck_assert_int_eq(dst->dtype, TL_INT8);
     ck_assert_int_eq(dst->len, 1);
     ck_assert(dst->dims[0] == 1);
     ck_assert_array_int_eq((int8_t*)dst->data, dst_data, dst->len);
     tl_tensor_free_data_too(dst);

     src1 = tl_tensor_create(src1_data, 2, dims, TL_INT8);
     src2 = tl_tensor_create(src2_data, 2, dims, TL_INT8);
     dst = tl_tensor_zeros(1, ARR(int,1), TL_INT8);
     dst = tl_tensor_dot_product(src1, src2, dst);
     ck_assert_int_eq(dst->ndim, 1);
     ck_assert_int_eq(dst->dtype, TL_INT8);
     ck_assert_int_eq(dst->len, 1);
     ck_assert(dst->dims[0] == 1);
     ck_assert_array_int_eq((int8_t *)dst->data, dst_data, dst->len);
     tl_tensor_free_data_too(dst);

     tl_tensor_free(src1);
     tl_tensor_free(src2);
}
END_TEST

START_TEST(test_tl_tensor_transpose)
{
     tl_tensor *src, *dst;
     int dims1[3] = {2, 3, 2};
     int dims2[3] = {3, 2, 2};
     uint8_t data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
     int axes1[3] = {0, 2, 1};
     uint8_t dst_data1[12] = {1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12};
     int axes2[3] = {1, 2, 0};
     uint8_t dst_data2[12] = {1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12};
     int i;

     src = tl_tensor_create(data, 3, dims1, TL_UINT8);

     dst = tl_tensor_transpose(src, NULL, axes1);
     ck_assert_int_eq(dst->ndim, 3);
     ck_assert_int_eq(dst->dtype, TL_UINT8);
     ck_assert_int_eq(dst->len, 12);
     ck_assert(dst->dims[0] == 2);
     ck_assert(dst->dims[1] == 2);
     ck_assert(dst->dims[2] == 3);
     for (i = 0; i < dst->len; i++)
          ck_assert(((int8_t *)dst->data)[i] == dst_data1[i]);
     tl_tensor_free_data_too(dst);

     dst = tl_tensor_zeros(3, dims2, TL_UINT8);
     dst = tl_tensor_transpose(src, dst, axes2);
     ck_assert_int_eq(dst->ndim, 3);
     ck_assert_int_eq(dst->dtype, TL_UINT8);
     ck_assert_int_eq(dst->len, 12);
     ck_assert(dst->dims[0] == 3);
     ck_assert(dst->dims[1] == 2);
     ck_assert(dst->dims[2] == 2);
     for (i = 0; i < dst->len; i++)
          ck_assert(((int8_t *)dst->data)[i] == dst_data2[i]);
     tl_tensor_free_data_too(dst);

     tl_tensor_free(src);
}
END_TEST

START_TEST(test_tl_tensor_lrelu)
{
    float data_f[5] = {-1, 0, 1, 255, -256};
    float data_lrelu_f[5] = {-0.1, 0, 1, 255, -25.6};
    float negslope = 0.1;
    tl_tensor *t1, *t2;

    t1 = tl_tensor_create(data_f, 1, (int[]){5}, TL_FLOAT);
    t2 = tl_tensor_lrelu(t1, NULL, negslope);
    ck_assert(tl_tensor_issameshape(t1, t2));
    ck_assert(t1->dtype == t2->dtype);
    ck_assert_array_float_eq_tol((float *)t2->data, data_lrelu_f, t2->len, 0);
    tl_tensor_free_data_too(t2);

    t2 = tl_tensor_zeros(1, (int[]){5}, TL_FLOAT);
    t2 = tl_tensor_lrelu(t1, t2, negslope);
    ck_assert_array_float_eq_tol((float *)t2->data, data_lrelu_f, t2->len, 0);
    tl_tensor_free_data_too(t2);

    tl_tensor_free(t1);
}
END_TEST

START_TEST(test_tl_tensor_convert)
{
     float data_f[5] = {-1, 0, 1, 255, 256};
     uint8_t data_ui8[5] = {0, 0, 1, 255, 255};
     tl_tensor *t1, *t2;

     t1 = tl_tensor_create(data_f, 1, (int[]){5}, TL_FLOAT);

     t2 = tl_tensor_convert(t1, NULL, TL_UINT8);
     ck_assert_int_eq(t2->ndim, 1);
     ck_assert_int_eq(t2->dtype, TL_UINT8);
     ck_assert_int_eq(t2->len, t1->len);
     ck_assert(t2->dims[0] == t1->dims[0]);
     for (int i = 0; i < 5; i++)
          ck_assert_uint_eq(((uint8_t*)t2->data)[i], data_ui8[i]);
     tl_tensor_free_data_too(t2);

     t2 = tl_tensor_zeros(1, (int[]){5}, TL_UINT8);
     t2 = tl_tensor_convert(t1, t2, TL_UINT8);
     ck_assert_int_eq(t2->ndim, 1);
     ck_assert_int_eq(t2->dtype, TL_UINT8);
     ck_assert_int_eq(t2->len, t1->len);
     ck_assert(t2->dims[0] == t1->dims[0]);
     for (int i = 0; i < 5; i++)
          ck_assert_uint_eq(((uint8_t*)t2->data)[i], data_ui8[i]);
     tl_tensor_free_data_too(t2);

     tl_tensor_free(t1);
}
END_TEST

START_TEST(test_tl_tensor_resize)
{
     float src_data[] = {1, 2, 3, 4};
     float true_data[] = {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4};
     float dst_data[16];
     tl_tensor *src, *dst, *true_tensor;

     true_tensor = tl_tensor_create(true_data, 2, ARR(int,4,4), TL_FLOAT);
     src = tl_tensor_create(src_data, 2, ARR(int,2,2), TL_FLOAT);
     dst = tl_tensor_resize(src, NULL, ARR(int,4,4), TL_NEAREST);
     tl_assert_tensor_eq(dst, true_tensor);
     tl_tensor_free_data_too(dst);

     dst = tl_tensor_create(dst_data, 2, ARR(int,4,4), TL_FLOAT);
     dst = tl_tensor_resize(src, dst, ARR(int,4,4), TL_NEAREST);
     tl_assert_tensor_eq(dst, true_tensor);
     tl_tensor_free(dst);

     tl_tensor_free(src);
     tl_tensor_free(true_tensor);
}
END_TEST

START_TEST(test_tl_tensor_submean)
{
    uint8_t src_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float dst_data[12];
    float true_data[] = {0, 3, 6, 9, 0, 3, 6, 9, 0, 3, 6, 9};
    double mean[] = {1, 2, 3};
    tl_tensor *src, *dst, *true_tensor;

    true_tensor = tl_tensor_create(true_data, 3, ARR(int,3,2,2), TL_FLOAT);
    src = tl_tensor_create(src_data, 3, ARR(int,2,2,3), TL_UINT8);
    dst = tl_tensor_submean(src, NULL, mean);
    ck_assert_int_eq(dst->dtype, TL_FLOAT);
    ck_assert_int_eq(dst->ndim, 3);
    ck_assert_array_int_eq(dst->dims, ARR(int,3,2,2), 3);
    tl_assert_tensor_eq(dst, true_tensor);
    tl_tensor_free_data_too(dst);

    dst = tl_tensor_create(dst_data, 3, ARR(int,3,2,2), TL_FLOAT);
    tl_tensor_submean(src, dst, mean);
    tl_assert_tensor_eq(dst, true_tensor);
    tl_tensor_free(dst);

    tl_tensor_free(src);
    tl_tensor_free(true_tensor);
}
END_TEST
/* end of tests */

Suite *make_tensor_suite(void)
{
     Suite *s;
     TCase *tc_tensor;

     s = suite_create("tensor");
     tc_tensor = tcase_create("tensor");
     tcase_add_checked_fixture(tc_tensor, setup, teardown);

     tcase_add_test(tc_tensor, test_tl_tensor_create);
     tcase_add_test(tc_tensor, test_tl_tensor_free);
     tcase_add_test(tc_tensor, test_tl_tensor_clone);
     tcase_add_test(tc_tensor, test_tl_tensor_repeat);
     tcase_add_test(tc_tensor, test_tl_tensor_arange);
     tcase_add_test(tc_tensor, test_tl_tensor_rearange);
     tcase_add_test(tc_tensor, test_tl_tensor_issameshape);
     tcase_add_test(tc_tensor, test_tl_tensor_fprint);
     tcase_add_test(tc_tensor, test_tl_tensor_print);
     tcase_add_test(tc_tensor, test_tl_tensor_save);
     tcase_add_test(tc_tensor, test_tl_tensor_zeros_slice);
     tcase_add_test(tc_tensor, test_tl_tensor_slice);
     tcase_add_test(tc_tensor, test_tl_tensor_slice_nocopy);
     tcase_add_test(tc_tensor, test_tl_tensor_concat);
     tcase_add_test(tc_tensor, test_tl_tensor_reshape);
     tcase_add_test(tc_tensor, test_tl_tensor_maxreduce);
     tcase_add_test(tc_tensor, test_tl_tensor_elew);
     tcase_add_test(tc_tensor, test_tl_tensor_elew_param);
     tcase_add_test(tc_tensor, test_tl_tensor_dot_product);
     tcase_add_test(tc_tensor, test_tl_tensor_transpose);
     tcase_add_test(tc_tensor, test_tl_tensor_lrelu);
     tcase_add_test(tc_tensor, test_tl_tensor_convert);
     tcase_add_test(tc_tensor, test_tl_tensor_resize);
     tcase_add_test(tc_tensor, test_tl_tensor_submean);
     /* end of adding tests */

     suite_add_tcase(s, tc_tensor);

     return s;
}
