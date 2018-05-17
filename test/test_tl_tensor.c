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

#include "test_tsl.h"
#include "../src/tl_tensor.h"
#include "../src/tl_util.h"

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

     t = tl_tensor_create(NULL, 1, dims, TL_DOUBLE);
     ck_assert_int_eq(t->ndim, 1);
     ck_assert_int_eq(t->dtype, TL_DOUBLE);
     ck_assert_int_eq(t->len, 1);
     for (i = 0; i < t->ndim; i++)
          ck_assert(t->dims[i] == dims[i]);
     for (i = 0; i < t->len; i++)
          ck_assert(((double *)t->data)[i] == 0);
     tl_tensor_free_data_too(t);

     t = tl_tensor_create(NULL, 1, dims, TL_FLOAT);
     ck_assert_int_eq(t->ndim, 1);
     ck_assert_int_eq(t->dtype, TL_FLOAT);
     ck_assert_int_eq(t->len, 1);
     for (i = 0; i < t->ndim; i++)
          ck_assert(t->dims[i] == dims[i]);
     for (i = 0; i < t->len; i++)
          ck_assert(((float *)t->data)[i] == 0);
     tl_tensor_free_data_too(t);

     t = tl_tensor_create(data, 1, dims, TL_INT32);
     ck_assert_int_eq(t->ndim, 1);
     ck_assert_int_eq(t->dtype, TL_INT32);
     ck_assert_int_eq(t->len, 1);
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

START_TEST(test_tl_tensor_zeros)
{
     tl_tensor *t;
     int dims[3] = {1, 2, 3};
     int i;

     t = tl_tensor_zeros(TL_FLOAT, 1, 1);
     ck_assert_int_eq(t->ndim, 1);
     ck_assert_int_eq(t->dtype, TL_FLOAT);
     ck_assert_int_eq(t->len, 1);
     for (i = 0; i < t->ndim; i++)
          ck_assert(t->dims[i] == dims[i]);
     for (i = 0; i < t->len; i++)
          ck_assert(((float *)t->data)[i] == 0);
     tl_tensor_free_data_too(t);

     t = tl_tensor_zeros(TL_INT32, 3, 1, 2, 3);
     ck_assert_int_eq(t->ndim, 3);
     ck_assert_int_eq(t->dtype, TL_INT32);
     ck_assert_int_eq(t->len, 6);
     for (i = 0; i < t->ndim; i++)
          ck_assert(t->dims[i] == dims[i]);
     for (i = 0; i < t->len; i++)
          ck_assert(((int32_t *)t->data)[i] == 0);
     tl_tensor_free_data_too(t);
}
END_TEST


START_TEST(test_tl_tensor_vcreate)
{
     tl_tensor *t;
     int dims[3] = {1, 2, 3};
     int i;

     t = tl_tensor_vcreate(TL_FLOAT, 1, 1);
     ck_assert_int_eq(t->ndim, 1);
     ck_assert_int_eq(t->dtype, TL_FLOAT);
     ck_assert_int_eq(t->len, 1);
     for (i = 0; i < t->ndim; i++)
          ck_assert(t->dims[i] == dims[i]);
     for (i = 0; i < t->len; i++)
          ck_assert(((float *)t->data)[i] == 0);
     tl_tensor_free_data_too(t);

     t = tl_tensor_vcreate(TL_INT32, 3, 1, 2, 3);
     ck_assert_int_eq(t->ndim, 3);
     ck_assert_int_eq(t->dtype, TL_INT32);
     ck_assert_int_eq(t->len, 6);
     for (i = 0; i < t->ndim; i++)
          ck_assert(t->dims[i] == dims[i]);
     for (i = 0; i < t->len; i++)
          ck_assert(((int32_t *)t->data)[i] == 0);
     tl_tensor_free_data_too(t);
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

START_TEST(test_tl_tensor_issameshape)
{
     tl_tensor *t1;
     tl_tensor *t2;

     t1 = tl_tensor_zeros(TL_FLOAT, 2, 3, 3);
     t2 = tl_tensor_zeros(TL_FLOAT, 2, 3, 3);
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

     t = tl_tensor_zeros(TL_FLOAT, 3, 1, 2, 3);

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

     t = tl_tensor_zeros(TL_FLOAT, 3, 1, 2, 3);

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

START_TEST(test_tl_tensor_create_slice)
{
     tl_tensor *t1, *t2;
     int i;

     t1 = tl_tensor_zeros(TL_INT8, 1, 1);
     t2 = tl_tensor_create_slice(t1, 0, 1, TL_INT8);
     ck_assert_int_eq(t2->ndim, 1);
     ck_assert_int_eq(t2->dtype, TL_INT8);
     ck_assert_int_eq(t2->len, 1);
     ck_assert(t2->dims[0] == 1);
     for (i = 0; i < t2->len; i++)
          ck_assert(((int8_t *)t2->data)[i] == 0);
     tl_tensor_free_data_too(t1);
     tl_tensor_free_data_too(t2);

     t1 = tl_tensor_zeros(TL_INT16, 3, 1, 2, 3);
     t2 = tl_tensor_create_slice(t1, 2, 2, TL_UINT8);
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
     t2 = tl_tensor_create_slice(t1, 1, 1, TL_UINT16);
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

START_TEST(test_tl_tensor_vreshape)
{
     tl_tensor *t1, *t2;
     int dims[3] = {1, 2, 3};
     int data[6] = {1, 2, 3, 4, 5, 6};
     int i;

     t1 = tl_tensor_create(data, 3, dims, TL_UINT32);
     t2 = tl_tensor_vreshape(t1, 2, 1, 6);
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

     dst = tl_tensor_create_slice(src, 0, 1, TL_INT32);
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

     dst = tl_tensor_create_slice(src, 2, 1, TL_INT32);
     arg = tl_tensor_create_slice(src, 2, 1, TL_INT32);
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
     dst = tl_tensor_create(NULL, 2, dims, TL_INT8);
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
     int *ws[2];
     int i;

     src = tl_tensor_create(data, 3, dims1, TL_UINT8);

     dst = tl_tensor_transpose(src, NULL, axes1, NULL);
     ck_assert_int_eq(dst->ndim, 3);
     ck_assert_int_eq(dst->dtype, TL_UINT8);
     ck_assert_int_eq(dst->len, 12);
     ck_assert(dst->dims[0] == 2);
     ck_assert(dst->dims[1] == 2);
     ck_assert(dst->dims[2] == 3);
     for (i = 0; i < dst->len; i++)
          ck_assert(((int8_t *)dst->data)[i] == dst_data1[i]);
     tl_tensor_free_data_too(dst);

     dst = tl_tensor_create(NULL, 3, dims2, TL_UINT8);
     dst = tl_tensor_transpose(src, dst, axes2, NULL);
     ck_assert_int_eq(dst->ndim, 3);
     ck_assert_int_eq(dst->dtype, TL_UINT8);
     ck_assert_int_eq(dst->len, 12);
     ck_assert(dst->dims[0] == 3);
     ck_assert(dst->dims[1] == 2);
     ck_assert(dst->dims[2] == 2);
     for (i = 0; i < dst->len; i++)
          ck_assert(((int8_t *)dst->data)[i] == dst_data2[i]);
     tl_tensor_free_data_too(dst);

     dst = tl_tensor_create(NULL, 3, dims2, TL_UINT8);
     ws[0] = (int *)tl_alloc(sizeof(int) * dst->ndim * dst->len);
     ws[1] = (int *)tl_alloc(sizeof(int) * dst->ndim * dst->len);
     dst = tl_tensor_transpose(src, dst, axes2, ws);
     ck_assert_int_eq(dst->ndim, 3);
     ck_assert_int_eq(dst->dtype, TL_UINT8);
     ck_assert_int_eq(dst->len, 12);
     ck_assert(dst->dims[0] == 3);
     ck_assert(dst->dims[1] == 2);
     ck_assert(dst->dims[2] == 2);
     for (i = 0; i < dst->len; i++)
          ck_assert(((int8_t *)dst->data)[i] == dst_data2[i]);
     tl_tensor_free_data_too(dst);
     tl_free(ws[0]);
     tl_free(ws[1]);

     tl_tensor_free(src);
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
     tcase_add_test(tc_tensor, test_tl_tensor_zeros);
     tcase_add_test(tc_tensor, test_tl_tensor_vcreate);
     tcase_add_test(tc_tensor, test_tl_tensor_clone);
     tcase_add_test(tc_tensor, test_tl_tensor_issameshape);
     tcase_add_test(tc_tensor, test_tl_tensor_fprint);
     tcase_add_test(tc_tensor, test_tl_tensor_print);
     tcase_add_test(tc_tensor, test_tl_tensor_save);
     tcase_add_test(tc_tensor, test_tl_tensor_create_slice);
     tcase_add_test(tc_tensor, test_tl_tensor_slice);
     tcase_add_test(tc_tensor, test_tl_tensor_reshape);
     tcase_add_test(tc_tensor, test_tl_tensor_vreshape);
     tcase_add_test(tc_tensor, test_tl_tensor_maxreduce);
     tcase_add_test(tc_tensor, test_tl_tensor_elew);
     tcase_add_test(tc_tensor, test_tl_tensor_transpose);
     /* end of adding tests */

     suite_add_tcase(s, tc_tensor);

     return s;
}
