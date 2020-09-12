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

#ifdef TL_CUDA

#include <time.h>

#include "test_tensorlight.h"
#include "../src/tl_tensor.h"
#include "../src/tl_check.h"

#define ARR(type, varg...) (type[]){varg}

static void setup(void)
{
}

static void teardown(void)
{
}

START_TEST(test_tl_tensor_zeros_cuda)
{
    tl_tensor *t;
    int dims[3] = {1, 2, 3};
    int32_t data[6] = {1, 2, 3, 4, 5, 6};
    void *data_h, *data_d;
    int i;

    t = tl_tensor_zeros_cuda(3, (int[]){1, 2, 3}, TL_DOUBLE);
    data_h = tl_clone_d2h(t->data, sizeof(double)*t->len);
    ck_assert_int_eq(t->ndim, 3);
    ck_assert_int_eq(t->dtype, TL_DOUBLE);
    ck_assert_int_eq(t->len, 6);
    for (i = 0; i < t->ndim; i++)
        ck_assert(t->dims[i] == dims[i]);
    for (i = 0; i < t->len; i++)
        ck_assert(((double *)data_h)[i] == 0);
    tl_tensor_free_data_too_cuda(t);
    tl_free(data_h);

    t = tl_tensor_zeros_cuda(3, dims, TL_FLOAT);
    data_h = tl_clone_d2h(t->data, sizeof(float)*t->len);
    ck_assert_int_eq(t->ndim, 3);
    ck_assert_int_eq(t->dtype, TL_FLOAT);
    ck_assert_int_eq(t->len, 6);
    for (i = 0; i < t->ndim; i++)
        ck_assert(t->dims[i] == dims[i]);
    for (i = 0; i < t->len; i++)
        ck_assert(((float *)data_h)[i] == 0);
    tl_tensor_free_data_too_cuda(t);
    tl_free(data_h);

    data_d = tl_clone_h2d(data, sizeof(int32_t)*6);
    t = tl_tensor_create(data_d, 3, dims, TL_INT32);
    data_h = tl_clone_d2h(t->data, sizeof(int32_t)*t->len);
    ck_assert_int_eq(t->ndim, 3);
    ck_assert_int_eq(t->dtype, TL_INT32);
    ck_assert_int_eq(t->len, 6);
    for (i = 0; i < t->ndim; i++)
        ck_assert(t->dims[i] == dims[i]);
    for (i = 0; i < t->len; i++)
        ck_assert(((int32_t *)data_h)[i] == data[i]);
    tl_tensor_free(t);
    tl_free(data_h);

    t = tl_tensor_create(data_d, 3, dims, TL_INT32);
    data_h = tl_clone_d2h(t->data, sizeof(int32_t)*t->len);
    ck_assert_int_eq(t->ndim, 3);
    ck_assert_int_eq(t->dtype, TL_INT32);
    ck_assert_int_eq(t->len, 6);
    for (i = 0; i < t->ndim; i++)
        ck_assert(t->dims[i] == dims[i]);
    for (i = 0; i < t->len; i++)
        ck_assert(((int32_t *)data_h)[i] == data[i]);
    tl_tensor_free(t);
    tl_free(data_h);
}
END_TEST

START_TEST(test_tl_tensor_free_data_too_cuda)
{
}
END_TEST

START_TEST(test_tl_tensor_clone_h2d)
{
    tl_tensor *t1, *t2;
    int dims[3] = {1, 2, 3};
    int32_t data[6] = {1, 2, 3, 4, 5, 6};
    int32_t *data_h;
    int i;

    t1 = tl_tensor_create(data, 3, dims, TL_INT32);
    t2 = tl_tensor_clone_h2d(t1);
    data_h = tl_clone_d2h(t2->data, sizeof(int32_t)*t2->len);
    ck_assert_int_eq(t2->ndim, 3);
    ck_assert_int_eq(t2->dtype, TL_INT32);
    ck_assert_int_eq(t2->len, 6);
    for (i = 0; i < t2->ndim; i++)
        ck_assert(t2->dims[i] == dims[i]);
    for (i = 0; i < t2->len; i++)
        ck_assert(data_h[i] == data[i]);
    tl_tensor_free(t1);
    tl_tensor_free_data_too_cuda(t2);
    tl_free(data_h);
}
END_TEST

START_TEST(test_tl_tensor_clone_d2h)
{
    tl_tensor *t1, *t2;
    int dims[3] = {1, 2, 3};
    int32_t data[6] = {1, 2, 3, 4, 5, 6};
    int32_t *data_d;
    int i;

    data_d = tl_clone_h2d(data, sizeof(int32_t)*6);
    t1 = tl_tensor_create(data_d, 3, dims, TL_INT32);
    t2 = tl_tensor_clone_d2h(t1);
    ck_assert_int_eq(t2->ndim, 3);
    ck_assert_int_eq(t2->dtype, TL_INT32);
    ck_assert_int_eq(t2->len, 6);
    for (i = 0; i < t2->ndim; i++)
        ck_assert(t2->dims[i] == dims[i]);
    for (i = 0; i < t2->len; i++)
        ck_assert(((int32_t *)t2->data)[i] == data[i]);
    tl_tensor_free(t1);
    tl_tensor_free_data_too(t2);
    tl_free_cuda(data_d);
}
END_TEST

START_TEST(test_tl_tensor_clone_d2d)
{
    tl_tensor *t1, *t2, *t3;
    int dims[3] = {1, 2, 3};
    int32_t data[6] = {1, 2, 3, 4, 5, 6};
    int32_t *data_h;
    int i;

    t1 = tl_tensor_create(data, 3, dims, TL_INT32);
    t2 = tl_tensor_clone_h2d(t1);
    t3 = tl_tensor_clone_d2d(t2);
    data_h = tl_clone_d2h(t3->data, sizeof(int32_t)*t3->len);
    ck_assert_int_eq(t3->ndim, 3);
    ck_assert_int_eq(t3->dtype, TL_INT32);
    ck_assert_int_eq(t3->len, 6);
    for (i = 0; i < t3->ndim; i++)
        ck_assert(t3->dims[i] == dims[i]);
    for (i = 0; i < t3->len; i++)
        ck_assert(data_h[i] == data[i]);
    tl_tensor_free(t1);
    tl_tensor_free_data_too_cuda(t2);
    tl_tensor_free_data_too_cuda(t3);
    tl_free(data_h);
}
END_TEST

START_TEST(test_tl_tensor_repeat_h2d)
{
     int dims[] = {2, 3};
     float data[] = {1, 2, 3, 4, 5, 6};
     int dims1[] = {1, 2, 3};
     float data1[] = {1, 2, 3, 4, 5, 6};
     int dims2[] = {2, 2, 3};
     float data2[] = {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
     int dims3[] = {3, 2, 3};
     float data3[] = {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
     tl_tensor *t, *t1_h, *t2_h, *t3_h, *t1_d, *t2_d, *t3_d;

     t = tl_tensor_create(data, 2, dims, TL_FLOAT);

     t1_d = tl_tensor_repeat_h2d(t, 1);
     ck_assert_array_int_eq(t1_d->dims, dims1, 3);
     t1_h = tl_tensor_clone_d2h(t1_d);
     ck_assert_array_float_eq_tol((float*)t1_h->data, data1, t1_h->len, 0);
     tl_tensor_free_data_too(t1_h);
     tl_tensor_free_data_too_cuda(t1_d);

     t2_d = tl_tensor_repeat_h2d(t, 2);
     ck_assert_array_int_eq(t2_d->dims, dims2, 3);
     t2_h = tl_tensor_clone_d2h(t2_d);
     ck_assert_array_float_eq_tol((float*)t2_h->data, data2, t2_h->len, 0);
     tl_tensor_free_data_too(t2_h);
     tl_tensor_free_data_too_cuda(t2_d);

     t3_d = tl_tensor_repeat_h2d(t, 3);
     ck_assert_array_int_eq(t3_d->dims, dims3, 3);
     t3_h = tl_tensor_clone_d2h(t3_d);
     ck_assert_array_float_eq_tol((float*)t3_h->data, data3, t3_h->len, 0);
     tl_tensor_free_data_too(t3_h);
     tl_tensor_free_data_too_cuda(t3_d);

     tl_tensor_free(t);
}
END_TEST

START_TEST(test_tl_tensor_repeat_d2d)
{
     int dims[] = {2, 3};
     float data[] = {1, 2, 3, 4, 5, 6};
     int dims1[] = {1, 2, 3};
     float data1[] = {1, 2, 3, 4, 5, 6};
     int dims2[] = {2, 2, 3};
     float data2[] = {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
     int dims3[] = {3, 2, 3};
     float data3[] = {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
     tl_tensor *t_h, *t_d, *t1_h, *t2_h, *t3_h, *t1_d, *t2_d, *t3_d;

     t_h = tl_tensor_create(data, 2, dims, TL_FLOAT);
     t_d = tl_tensor_clone_h2d(t_h);

     t1_d = tl_tensor_repeat_d2d(t_d, 1);
     ck_assert_array_int_eq(t1_d->dims, dims1, 3);
     t1_h = tl_tensor_clone_d2h(t1_d);
     ck_assert_array_float_eq_tol((float*)t1_h->data, data1, t1_h->len, 0);
     tl_tensor_free_data_too(t1_h);
     tl_tensor_free_data_too_cuda(t1_d);

     t2_d = tl_tensor_repeat_d2d(t_d, 2);
     ck_assert_array_int_eq(t2_d->dims, dims2, 3);
     t2_h = tl_tensor_clone_d2h(t2_d);
     ck_assert_array_float_eq_tol((float*)t2_h->data, data2, t2_h->len, 0);
     tl_tensor_free_data_too(t2_h);
     tl_tensor_free_data_too_cuda(t2_d);

     t3_d = tl_tensor_repeat_d2d(t_d, 3);
     ck_assert_array_int_eq(t3_d->dims, dims3, 3);
     t3_h = tl_tensor_clone_d2h(t3_d);
     ck_assert_array_float_eq_tol((float*)t3_h->data, data3, t3_h->len, 0);
     tl_tensor_free_data_too(t3_h);
     tl_tensor_free_data_too_cuda(t3_d);

     tl_tensor_free(t_h);
     tl_tensor_free_data_too_cuda(t_d);
}
END_TEST

START_TEST(test_tl_tensor_repeat_d2h)
{
     int dims[] = {2, 3};
     float data[] = {1, 2, 3, 4, 5, 6};
     int dims1[] = {1, 2, 3};
     float data1[] = {1, 2, 3, 4, 5, 6};
     int dims2[] = {2, 2, 3};
     float data2[] = {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
     int dims3[] = {3, 2, 3};
     float data3[] = {1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6};
     tl_tensor *t_h, *t_d, *t1, *t2, *t3;

     t_h = tl_tensor_create(data, 2, dims, TL_FLOAT);
     t_d = tl_tensor_clone_h2d(t_h);

     t1 = tl_tensor_repeat_d2h(t_d, 1);
     ck_assert_array_int_eq(t1->dims, dims1, 3);
     ck_assert_array_float_eq_tol((float*)t1->data, data1, t1->len, 0);
     tl_tensor_free_data_too(t1);

     t2 = tl_tensor_repeat_d2h(t_d, 2);
     ck_assert_array_int_eq(t2->dims, dims2, 3);
     ck_assert_array_float_eq_tol((float*)t2->data, data2, t2->len, 0);
     tl_tensor_free_data_too(t2);

     t3 = tl_tensor_repeat_d2h(t_d, 3);
     ck_assert_array_int_eq(t3->dims, dims3, 3);
     ck_assert_array_float_eq_tol((float*)t3->data, data3, t3->len, 0);
     tl_tensor_free_data_too(t3);

     tl_tensor_free(t_h);
     tl_tensor_free_data_too_cuda(t_d);
}
END_TEST

START_TEST(test_tl_tensor_arange_cuda)
{
    tl_tensor *dst_d, *dst_h, *t;

    t = tl_tensor_create(ARR(int16_t,0,1,2), 1, ARR(int,3), TL_INT16);
    dst_d = tl_tensor_arange_cuda(0, 3, 1, TL_INT16);
    dst_h = tl_tensor_clone_d2h(dst_d);
    tl_assert_tensor_eq(dst_h, t);
    tl_tensor_free(t);
    tl_tensor_free_data_too(dst_h);
    tl_tensor_free_data_too_cuda(dst_d);

    t = tl_tensor_create(ARR(float,0.1,1.6,3.1), 1, ARR(int,3), TL_FLOAT);
    dst_d = tl_tensor_arange_cuda(0.1, 3.2, 1.5, TL_FLOAT);
    dst_h = tl_tensor_clone_d2h(dst_d);
    tl_assert_tensor_eq(dst_h, t);
    tl_tensor_free(t);
    tl_tensor_free_data_too(dst_h);
    tl_tensor_free_data_too_cuda(dst_d);

    t = tl_tensor_create(ARR(float,0.1,1.6), 1, ARR(int,2), TL_FLOAT);
    dst_d = tl_tensor_arange_cuda(0.1, 3.1, 1.5, TL_FLOAT);
    dst_h = tl_tensor_clone_d2h(dst_d);
    tl_assert_tensor_eq(dst_h, t);
    tl_tensor_free(t);
    tl_tensor_free_data_too(dst_h);
    tl_tensor_free_data_too_cuda(dst_d);
}
END_TEST

START_TEST(test_tl_tensor_rearange_cuda)
{
    tl_tensor *dst_d, *dst_h, *t;

    t = tl_tensor_create(ARR(int16_t,0,1,2), 1, ARR(int,3), TL_INT16);
    dst_d = tl_tensor_zeros_cuda(1, ARR(int, 3), TL_INT16);
    tl_tensor_rearange_cuda(dst_d, 0, 3, 1);
    dst_h = tl_tensor_clone_d2h(dst_d);
    tl_assert_tensor_eq(dst_h, t);
    tl_tensor_free(t);
    tl_tensor_free_data_too(dst_h);
    tl_tensor_free_data_too_cuda(dst_d);

    t = tl_tensor_create(ARR(float,0.1,1.6,3.1), 1, ARR(int,3), TL_FLOAT);
    dst_d = tl_tensor_zeros_cuda(1, ARR(int, 3), TL_FLOAT);
    tl_tensor_rearange_cuda(dst_d, 0.1, 3.2, 1.5);
    dst_h = tl_tensor_clone_d2h(dst_d);
    tl_assert_tensor_eq(dst_h, t);
    tl_tensor_free(t);
    tl_tensor_free_data_too(dst_h);
    tl_tensor_free_data_too_cuda(dst_d);

    t = tl_tensor_create(ARR(float,0.1,1.6), 1, ARR(int,2), TL_FLOAT);
    dst_d = tl_tensor_zeros_cuda(1, ARR(int, 2), TL_FLOAT);
    tl_tensor_rearange_cuda(dst_d, 0.1, 3.1, 1.5);
    dst_h = tl_tensor_clone_d2h(dst_d);
    tl_assert_tensor_eq(dst_h, t);
    tl_tensor_free(t);
    tl_tensor_free_data_too(dst_h);
    tl_tensor_free_data_too_cuda(dst_d);
}
END_TEST

START_TEST(test_tl_tensor_fprint_cuda)
{
    tl_tensor *t;
    FILE *fp;
    char s[BUFSIZ];

    t = tl_tensor_zeros_cuda(3, (int[]){1, 2, 3}, TL_FLOAT);

    fp = tmpfile();
    ck_assert_ptr_ne(fp, NULL);
    tl_tensor_fprint_cuda(fp, t, NULL);
    rewind(fp);
    ck_assert_ptr_ne(fgets(s, 100, fp), NULL);
    ck_assert_ptr_ne(fgets(s+strlen(s), 100, fp), NULL);
    ck_assert_str_eq(s, "[[[0.000 0.000 0.000]\n"
                     "  [0.000 0.000 0.000]]]\n");
    fclose(fp);

    fp = tmpfile();
    ck_assert_ptr_ne(fp, NULL);
    tl_tensor_fprint_cuda(fp, t, "%.4f");
    rewind(fp);
    ck_assert_ptr_ne(fgets(s, 100, fp), NULL);
    ck_assert_ptr_ne(fgets(s+strlen(s), 100, fp), NULL);
    ck_assert_str_eq(s, "[[[0.0000 0.0000 0.0000]\n"
                     "  [0.0000 0.0000 0.0000]]]\n");
    fclose(fp);

    tl_tensor_free_data_too_cuda(t);
}
END_TEST

START_TEST(test_tl_tensor_print_cuda)
{
}
END_TEST

START_TEST(test_tl_tensor_save_cuda)
{
    tl_tensor *t;
    FILE *fp;
    char s[BUFSIZ];

    t = tl_tensor_zeros_cuda(3, (int[]){1, 2, 3}, TL_FLOAT);

    tl_tensor_save_cuda("__test_tensor_save_tmp", t, NULL);
    fp = fopen("__test_tensor_save_tmp", "r");
    ck_assert_ptr_ne(fp, NULL);
    ck_assert_ptr_ne(fgets(s, 100, fp), NULL);
    ck_assert_ptr_ne(fgets(s+strlen(s), 100, fp), NULL);
    ck_assert_str_eq(s, "[[[0.000 0.000 0.000]\n"
                     "  [0.000 0.000 0.000]]]\n");
    fclose(fp);
    ck_assert_int_eq(remove("__test_tensor_save_tmp"), 0);

    /* ck_assert_int_lt(tl_tensor_save("__non_exist_dir/tmp", t, NULL), 0); */
    tl_tensor_free_data_too_cuda(t);
}
END_TEST

START_TEST(test_tl_tensor_zeros_slice_cuda)
{
    tl_tensor *t1, *t2;
    void *data_h;
    int i;

    t1 = tl_tensor_zeros_cuda(1, (int[]){1}, TL_INT8);
    t2 = tl_tensor_zeros_slice_cuda(t1, 0, 1, TL_INT8);
    data_h = tl_clone_d2h(t2->data, tl_size_of(t2->dtype)*t2->len);
    ck_assert_int_eq(t2->ndim, 1);
    ck_assert_int_eq(t2->dtype, TL_INT8);
    ck_assert_int_eq(t2->len, 1);
    ck_assert(t2->dims[0] == 1);
    for (i = 0; i < t2->len; i++)
        ck_assert(((int8_t *)data_h)[i] == 0);
    tl_tensor_free_data_too_cuda(t1);
    tl_tensor_free_data_too_cuda(t2);
    tl_free(data_h);

    t1 = tl_tensor_zeros_cuda(3, (int[]){1, 2, 3}, TL_INT16);
    t2 = tl_tensor_zeros_slice_cuda(t1, 2, 2, TL_UINT8);
    data_h = tl_clone_d2h(t2->data, tl_size_of(t2->dtype)*t2->len);
    ck_assert_int_eq(t2->ndim, 3);
    ck_assert_int_eq(t2->dtype, TL_UINT8);
    ck_assert_int_eq(t2->len, 4);
    ck_assert(t2->dims[0] == 1);
    ck_assert(t2->dims[1] == 2);
    ck_assert(t2->dims[2] == 2);
    for (i = 0; i < t2->len; i++)
        ck_assert(((uint8_t *)data_h)[i] == 0);
    tl_tensor_free_data_too_cuda(t1);
    tl_tensor_free_data_too_cuda(t2);
    tl_free(data_h);
}
END_TEST

START_TEST(test_tl_tensor_slice_cuda)
{
    tl_tensor *t1, *t2;
    int ndim = 3;
    int dims[3] = {1, 2, 3};
    uint16_t data[6] = {1, 2, 3, 4, 5, 6};
    uint16_t data_slice1[4] = {2, 3, 5, 6};
    uint16_t data_slice2[3] = {1, 2, 3};
    void *data_h, *data_d;
    int i;

    data_d = tl_clone_h2d(data, sizeof(uint16_t)*6);
    t1 = tl_tensor_create(data_d, ndim, dims, TL_UINT16);
    t2 = tl_tensor_slice_cuda(t1, NULL, 2, 1, 2);
    data_h = tl_clone_d2h(t2->data, tl_size_of(t2->dtype)*t2->len);
    ck_assert_int_eq(t2->ndim, 3);
    ck_assert_int_eq(t2->dtype, TL_UINT16);
    ck_assert_int_eq(t2->len, 4);
    ck_assert(t2->dims[0] == 1);
    ck_assert(t2->dims[1] == 2);
    ck_assert(t2->dims[2] == 2);
    for (i = 0; i < t2->len; i++)
        ck_assert(((uint16_t *)data_h)[i] == data_slice1[i]);
    tl_tensor_free(t1);
    tl_tensor_free_data_too_cuda(t2);
    tl_free_cuda(data_d);
    tl_free(data_h);

    data_d = tl_clone_h2d(data, sizeof(uint16_t)*6);
    t1 = tl_tensor_create(data_d, ndim, dims, TL_UINT16);
    t2 = tl_tensor_zeros_slice_cuda(t1, 1, 1, TL_UINT16);
    t2 = tl_tensor_slice_cuda(t1, t2, 1, 0, 1);
    data_h = tl_clone_d2h(t2->data, tl_size_of(t2->dtype)*t2->len);
    ck_assert_int_eq(t2->ndim, 3);
    ck_assert_int_eq(t2->dtype, TL_UINT16);
    ck_assert_int_eq(t2->len, 3);
    ck_assert(t2->dims[0] == 1);
    ck_assert(t2->dims[1] == 1);
    ck_assert(t2->dims[2] == 3);
    for (i = 0; i < t2->len; i++)
        ck_assert(((uint16_t *)data_h)[i] == data_slice2[i]);
    tl_tensor_free(t1);
    tl_tensor_free_data_too_cuda(t2);
    tl_free_cuda(data_d);
    tl_free(data_h);
}
END_TEST

START_TEST(test_tl_tensor_maxreduce_cuda)
{
    tl_tensor *src, *dst, *arg;
    int dims[3] = {2, 3, 2};
    int32_t data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    int32_t dst_data1[4] = {5, 6, 11, 12};
    int32_t dst_data2[6] = {7, 8, 9, 10, 11, 12};
    int32_t dst_data3[6] = {2, 4, 6, 8, 10, 12};
    int32_t arg_data[6] = {1, 1, 1, 1, 1, 1};
    int32_t *data_d, *data_h;
    int i;

    data_d = tl_clone_h2d(data, sizeof(int32_t)*12);
    src = tl_tensor_create(data_d, 3, dims, TL_INT32);
    dst = tl_tensor_maxreduce_cuda(src, NULL, NULL, 1);
    data_h = tl_clone_d2h(dst->data, tl_size_of(dst->dtype)*dst->len);
    ck_assert_int_eq(dst->ndim, 3);
    ck_assert_int_eq(dst->dtype, TL_INT32);
    ck_assert_int_eq(dst->len, 4);
    ck_assert(dst->dims[0] == 2);
    ck_assert(dst->dims[1] == 1);
    ck_assert(dst->dims[2] == 2);
    for (i = 0; i < dst->len; i++)
        ck_assert(((int32_t *)data_h)[i] == dst_data1[i]);
    tl_tensor_free_data_too_cuda(dst);
    tl_free(data_h);

    dst = tl_tensor_zeros_slice_cuda(src, 0, 1, TL_INT32);
    dst = tl_tensor_maxreduce_cuda(src, dst, NULL, 0);
    data_h = tl_clone_d2h(dst->data, tl_size_of(dst->dtype)*dst->len);
    ck_assert_int_eq(dst->ndim, 3);
    ck_assert_int_eq(dst->dtype, TL_INT32);
    ck_assert_int_eq(dst->len, 6);
    ck_assert(dst->dims[0] == 1);
    ck_assert(dst->dims[1] == 3);
    ck_assert(dst->dims[2] == 2);
    for (i = 0; i < dst->len; i++)
        ck_assert(((int32_t *)data_h)[i] == dst_data2[i]);
    tl_tensor_free_data_too_cuda(dst);
    tl_free(data_h);

    dst = tl_tensor_zeros_slice_cuda(src, 2, 1, TL_INT32);
    arg = tl_tensor_zeros_slice_cuda(src, 2, 1, TL_INT32);
    dst = tl_tensor_maxreduce_cuda(src, dst, arg, 2);
    ck_assert_int_eq(dst->ndim, 3);
    ck_assert_int_eq(dst->dtype, TL_INT32);
    ck_assert_int_eq(dst->len, 6);
    ck_assert(dst->dims[0] == 2);
    ck_assert(dst->dims[1] == 3);
    ck_assert(dst->dims[2] == 1);
    data_h = tl_clone_d2h(dst->data, tl_size_of(dst->dtype)*dst->len);
    for (i = 0; i < dst->len; i++)
        ck_assert(((int32_t *)data_h)[i] == dst_data3[i]);
    ck_assert_int_eq(arg->ndim, 3);
    ck_assert_int_eq(arg->dtype, TL_INT32);
    ck_assert_int_eq(arg->len, 6);
    ck_assert(arg->dims[0] == 2);
    ck_assert(arg->dims[1] == 3);
    ck_assert(arg->dims[2] == 1);
    data_h = tl_clone_d2h(arg->data, tl_size_of(arg->dtype)*arg->len);
    for (i = 0; i < arg->len; i++)
        ck_assert(((int32_t *)data_h)[i] == arg_data[i]);
    tl_free(data_h);
    tl_tensor_free_data_too_cuda(dst);
    tl_tensor_free_data_too_cuda(arg);

    tl_free_cuda(data_d);
    tl_tensor_free(src);
}
END_TEST

START_TEST(test_tl_tensor_elew_cuda)
{
    tl_tensor *src1, *src2, *dst;
    int8_t src1_data[6] = {1, 1, 2, 2, 3, 3};
    int8_t src2_data[6] = {1, 2, 3, 4, 5, 6};
    int8_t dst_data[6] = {1, 2, 6, 8, 15, 18};
    void *data_d1, *data_d2, *data_h;
    int dims[2] = {2, 3};
    int i;

    data_d1 = tl_clone_h2d(src1_data, sizeof(int8_t)*6);
    data_d2 = tl_clone_h2d(src2_data, sizeof(int8_t)*6);

    src1 = tl_tensor_create(data_d1, 2, dims, TL_INT8);
    src2 = tl_tensor_create(data_d2, 2, dims, TL_INT8);
    dst = tl_tensor_elew_cuda(src1, src2, NULL, TL_MUL);
    data_h = tl_clone_d2h(dst->data, tl_size_of(dst->dtype)*dst->len);
    ck_assert_int_eq(dst->ndim, 2);
    ck_assert_int_eq(dst->dtype, TL_INT8);
    ck_assert_int_eq(dst->len, 6);
    ck_assert(dst->dims[0] == 2);
    ck_assert(dst->dims[1] == 3);
    for (i = 0; i < dst->len; i++)
        ck_assert(((int8_t *)data_h)[i] == dst_data[i]);
    tl_tensor_free_data_too_cuda(dst);
    tl_free(data_h);

    src1 = tl_tensor_create(data_d1, 2, dims, TL_INT8);
    src2 = tl_tensor_create(data_d2, 2, dims, TL_INT8);
    dst = tl_tensor_zeros_cuda(2, dims, TL_INT8);
    dst = tl_tensor_elew_cuda(src1, src2, dst, TL_MUL);
    data_h = tl_clone_d2h(dst->data, tl_size_of(dst->dtype)*dst->len);
    ck_assert_int_eq(dst->ndim, 2);
    ck_assert_int_eq(dst->dtype, TL_INT8);
    ck_assert_int_eq(dst->len, 6);
    ck_assert(dst->dims[0] == 2);
    ck_assert(dst->dims[1] == 3);
    for (i = 0; i < dst->len; i++)
        ck_assert(((int8_t *)data_h)[i] == dst_data[i]);
    tl_tensor_free_data_too_cuda(dst);
    tl_free(data_h);

    tl_tensor_free(src1);
    tl_tensor_free(src2);
    tl_free_cuda(data_d1);
    tl_free_cuda(data_d2);
}
END_TEST

START_TEST(test_tl_tensor_dot_product_cuda)
{
    tl_tensor *src1, *src2, *dst;
    int8_t src1_data[6] = {1, 1, 2, 2, 3, 3};
    int8_t src2_data[6] = {1, 2, 3, 4, 5, 6};
    int8_t dst_data[1] = {50};
    void *data_d1, *data_d2, *data_h;
    int dims[2] = {2, 3};

    data_d1 = tl_clone_h2d(src1_data, sizeof(int8_t)*6);
    data_d2 = tl_clone_h2d(src2_data, sizeof(int8_t)*6);

    src1 = tl_tensor_create(data_d1, 2, dims, TL_INT8);
    src2 = tl_tensor_create(data_d2, 2, dims, TL_INT8);
    dst = tl_tensor_dot_product_cuda(src1, src2, NULL, NULL, NULL);
    data_h = tl_clone_d2h(dst->data, tl_size_of(dst->dtype)*dst->len);
    ck_assert_int_eq(dst->len, tl_tensor_dot_product_cuda_ws_len(src1));
    ck_assert_int_eq(dst->ndim, 1);
    ck_assert_int_eq(dst->dtype, TL_INT8);
    ck_assert_int_eq(dst->len, 1);
    ck_assert(dst->dims[0] == 1);
    ck_assert_array_int_eq((int8_t *)data_h, dst_data, dst->len);
    tl_tensor_free_data_too_cuda(dst);
    tl_free(data_h);

    /* src1 = tl_tensor_create(data_d1, 2, dims, TL_INT8); */
    /* src2 = tl_tensor_create(data_d2, 2, dims, TL_INT8); */
    /* dst = tl_tensor_zeros_cuda(2, dims, TL_INT8); */
    /* dst = tl_tensor_elew_cuda(src1, src2, dst, TL_MUL); */
    /* data_h = tl_clone_d2h(dst->data, tl_size_of(dst->dtype)*dst->len); */
    /* ck_assert_int_eq(dst->ndim, 2); */
    /* ck_assert_int_eq(dst->dtype, TL_INT8); */
    /* ck_assert_int_eq(dst->len, 6); */
    /* ck_assert(dst->dims[0] == 2); */
    /* ck_assert(dst->dims[1] == 3); */
    /* for (i = 0; i < dst->len; i++) */
    /*     ck_assert(((int8_t *)data_h)[i] == dst_data[i]); */
    /* tl_tensor_free_data_too_cuda(dst); */
    /* tl_free(data_h); */

    /* tl_tensor_free(src1); */
    /* tl_tensor_free(src2); */
    /* tl_free_cuda(data_d1); */
    /* tl_free_cuda(data_d2); */
}
END_TEST

START_TEST(test_tl_tensor_convert_cuda)
{
    float data_f[5] = {-1, 0, 1, 255, 256};
    uint8_t data_ui8[5] = {0, 0, 1, 255, 255};
    void *data_d, *data_h;
    tl_tensor *t1, *t2;

    data_d = tl_clone_h2d(data_f, sizeof(float)*5);
    t1 = tl_tensor_create(data_d, 1, (int[]){5}, TL_FLOAT);

    t2 = tl_tensor_convert_cuda(t1, NULL, TL_UINT8);
    ck_assert_int_eq(t2->ndim, 1);
    ck_assert_int_eq(t2->dtype, TL_UINT8);
    ck_assert_int_eq(t2->len, t1->len);
    ck_assert(t2->dims[0] == t1->dims[0]);
    data_h = tl_clone_d2h(t2->data, tl_size_of(t2->dtype)*t2->len);
    for (int i = 0; i < 5; i++)
        ck_assert_uint_eq(((uint8_t*)data_h)[i], data_ui8[i]);
    tl_tensor_free_data_too_cuda(t2);
    tl_free(data_h);

    t2 = tl_tensor_zeros_cuda(1, (int[]){5}, TL_UINT8);
    t2 = tl_tensor_convert_cuda(t1, t2, TL_UINT8);
    ck_assert_int_eq(t2->ndim, 1);
    ck_assert_int_eq(t2->dtype, TL_UINT8);
    ck_assert_int_eq(t2->len, t1->len);
    ck_assert(t2->dims[0] == t1->dims[0]);
    data_h = tl_clone_d2h(t2->data, tl_size_of(t2->dtype)*t2->len);
    for (int i = 0; i < 5; i++)
        ck_assert_uint_eq(((uint8_t*)data_h)[i], data_ui8[i]);
    tl_tensor_free_data_too_cuda(t2);
    tl_free(data_h);

    tl_tensor_free(t1);
}
END_TEST

START_TEST(test_tl_tensor_transpose_cuda)
{
    tl_tensor *src, *dst;
    int dims1[3] = {2, 3, 2};
    int dims2[3] = {3, 2, 2};
    uint8_t data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    int axes1[3] = {0, 2, 1};
    uint8_t dst_data1[12] = {1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12};
    int axes2[3] = {1, 2, 0};
    uint8_t dst_data2[12] = {1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12};
    void *data_d, *data_h;
    int i;

    data_d = tl_clone_h2d(data, sizeof(uint8_t)*12);
    src = tl_tensor_create(data_d, 3, dims1, TL_UINT8);

    dst = tl_tensor_transpose_cuda(src, NULL, axes1);
    ck_assert_int_eq(dst->ndim, 3);
    ck_assert_int_eq(dst->dtype, TL_UINT8);
    ck_assert_int_eq(dst->len, 12);
    ck_assert(dst->dims[0] == 2);
    ck_assert(dst->dims[1] == 2);
    ck_assert(dst->dims[2] == 3);
    data_h = tl_clone_d2h(dst->data, tl_size_of(dst->dtype)*dst->len);
    for (i = 0; i < dst->len; i++)
        ck_assert(((int8_t *)data_h)[i] == dst_data1[i]);
    tl_tensor_free_data_too_cuda(dst);
    tl_free(data_h);

    dst = tl_tensor_zeros_cuda(3, dims2, TL_UINT8);
    dst = tl_tensor_transpose_cuda(src, dst, axes2);
    ck_assert_int_eq(dst->ndim, 3);
    ck_assert_int_eq(dst->dtype, TL_UINT8);
    ck_assert_int_eq(dst->len, 12);
    ck_assert(dst->dims[0] == 3);
    ck_assert(dst->dims[1] == 2);
    ck_assert(dst->dims[2] == 2);
    data_h = tl_clone_d2h(dst->data, tl_size_of(dst->dtype)*dst->len);
    for (i = 0; i < dst->len; i++)
        ck_assert(((int8_t *)data_h)[i] == dst_data2[i]);
    tl_tensor_free_data_too_cuda(dst);
    tl_free(data_h);

    tl_tensor_free(src);
    tl_free_cuda(data_d);
}
END_TEST

START_TEST(test_tl_tensor_lrelu_cuda)
{
    float data_f[5] = {-1, 0, 1, 255, -256};
    float *data_f_d;
    float data_lrelu_f[5] = {-0.1, 0, 1, 255, -25.6};
    float *data_lrelu_f_h;
    float negslope = 0.1;
    tl_tensor *t1, *t2;

    data_f_d = tl_clone_h2d(data_f, sizeof(float) * 5);
    t1 = tl_tensor_create(data_f_d, 1, (int[]){5}, TL_FLOAT);

    t2 = tl_tensor_lrelu_cuda(t1, NULL, negslope);
    data_lrelu_f_h = tl_clone_d2h(t2->data, sizeof(float) * 5);
    ck_assert(tl_tensor_issameshape(t1, t2));
    ck_assert(t1->dtype == t2->dtype);
    ck_assert_array_float_eq_tol((float *)data_lrelu_f_h, data_lrelu_f, t2->len, 0);
    tl_tensor_free_data_too_cuda(t2);
    tl_free(data_lrelu_f_h);

    t2 = tl_tensor_zeros_cuda(1, (int[]){5}, TL_FLOAT);
    t2 = tl_tensor_lrelu_cuda(t1, t2, negslope);
    data_lrelu_f_h = tl_clone_d2h(t2->data, sizeof(float) * 5);
    ck_assert_array_float_eq_tol((float *)data_lrelu_f_h, data_lrelu_f, t2->len, 0);
    tl_tensor_free_data_too_cuda(t2);
    tl_free(data_lrelu_f_h);

    tl_free_cuda(data_f_d);
    tl_tensor_free(t1);
}
END_TEST

START_TEST(test_tl_tensor_resize_cuda)
{
    float src_data_h[] = {1, 2, 3,
                          4, 5, 6,
                          1, 2, 3,
                          4, 5, 6};
    int src_ndim = 3;
    int src_dims[] = {2, 2, 3};
    float true_data[] = {1, 1, 2, 2, 3, 3,
                         1, 1, 2, 2, 3, 3,
                         4, 4, 5, 5, 6, 6,
                         4, 4, 5, 5, 6, 6,
                         1, 1, 2, 2, 3, 3,
                         1, 1, 2, 2, 3, 3,
                         4, 4, 5, 5, 6, 6,
                         4, 4, 5, 5, 6, 6};
    float dst_data_h[48] = {0};
    int dst_ndim = 3;
    int dst_dims[] = {2, 4, 6};
    float *src_data_d, *dst_data_d;
    tl_tensor *src, *dst;

    src_data_d = tl_clone_h2d(src_data_h, sizeof(src_data_h));
    src = tl_tensor_create(src_data_d, src_ndim, src_dims, TL_FLOAT);

    dst = tl_tensor_resize_cuda(src, NULL, dst_dims, TL_NEAREST);
    ck_assert_int_eq(dst->ndim, dst_ndim);
    ck_assert_int_eq(dst->dtype, TL_FLOAT);
    tl_memcpy_d2h(dst_data_h, dst->data, tl_size_of(TL_FLOAT)*dst->len);
    for (int i = 0; i < dst->len; i++)
        ck_assert_float_eq_tol(((float *)dst_data_h)[i], true_data[i], 1e-5);
    tl_tensor_free_data_too_cuda(dst);

    dst_data_d = tl_clone_h2d(dst_data_h, sizeof(dst_data_h));
    dst = tl_tensor_create(dst_data_d, dst_ndim, dst_dims, TL_FLOAT);
    dst = tl_tensor_resize_cuda(src, dst, dst_dims, TL_NEAREST);
    tl_memcpy_d2h(dst_data_h, dst->data, tl_size_of(TL_FLOAT)*dst->len);
    for (int i = 0; i < dst->len; i++)
        ck_assert_float_eq_tol(((float *)dst_data_h)[i], true_data[i], 1e-5);
    tl_tensor_free(dst);

    tl_tensor_free(src);
    tl_free_cuda(src_data_d);
    tl_free_cuda(dst_data_d);
}
END_TEST

START_TEST(test_tl_tensor_transform_bboxSQD_cuda)
{
    int width = 1248;
    int height = 384;
    int img_width = 1242;
    int img_height = 375;
    int x_shift = 0;
    int y_shift = 0;
    const int N = 36;
    float h_dst_data_true[] = {
        1.228140e+03,    3.483211e+02,    1.241000e+03,    3.740000e+02,
        1.067210e+03,    2.356478e+02,    1.241000e+03,    3.740000e+02,
        1.188710e+03,    3.301930e+02,    1.241000e+03,    3.740000e+02,
        1.160467e+03,    3.155269e+02,    1.241000e+03,    3.740000e+02,
        1.220873e+03,    2.927379e+02,    1.241000e+03,    3.740000e+02,
        1.178875e+03,    2.479402e+02,    1.241000e+03,    3.740000e+02,
        1.146917e+03,    3.051918e+02,    1.241000e+03,    3.740000e+02,
        1.206713e+03,    2.496181e+02,    1.241000e+03,    3.740000e+02,
        1.195503e+03,    3.455338e+02,    1.241000e+03,    3.740000e+02,
    };
    float h_anchor_data[] = {
        1.232203e+03,    3.686400e+02,    3.600000e+01,    3.700000e+01,
        1.232203e+03,    3.686400e+02,    3.660000e+02,    1.740000e+02,
        1.232203e+03,    3.686400e+02,    1.150000e+02,    5.900000e+01,
        1.232203e+03,    3.686400e+02,    1.620000e+02,    8.700000e+01,
        1.232203e+03,    3.686400e+02,    3.800000e+01,    9.000000e+01,
        1.232203e+03,    3.686400e+02,    2.580000e+02,    1.730000e+02,
        1.232203e+03,    3.686400e+02,    2.240000e+02,    1.080000e+02,
        1.232203e+03,    3.686400e+02,    7.800000e+01,    1.700000e+02,
        1.232203e+03,    3.686400e+02,    7.200000e+01,    4.300000e+01,
    };
    float h_delta_data[] = {
        2.700084e-01,    2.615839e-01,   -8.299431e-01,    1.566731e-01,
        6.803536e-02,    2.853769e-03,    9.450632e-03,    3.848182e-01,
        1.548163e-01,    9.279537e-02,   -3.444237e-02,    1.990532e-01,
        7.056128e-02,    7.749645e-02,   -4.338923e-02,    1.839040e-01,
        1.729851e-01,   -5.256999e-04,   -4.591562e-01,    4.249639e-01,
        1.180988e-01,   -2.344110e-02,   -5.017939e-01,    2.466247e-01,
        7.285635e-02,    6.465448e-02,   -1.534719e-01,    1.558371e-01,
        1.453436e-01,   -1.644313e-02,   -2.296714e-01,    2.599697e-01,
        6.938356e-02,    1.221377e-01,   -2.235517e-03,   -6.906012e-02,
    };
    float h_dst_data[N];
    float *d_delta_data;
    float *d_anchor_data;
    float *d_dst_data;
    int ndim = 5;
    int dims[5] = {1, 1, 3, 3, 4};
    tl_tensor *delta, *anchor, *dst;
    size_t size = sizeof(float) * N;

    d_delta_data = (float *)tl_alloc_cuda(size);
    d_anchor_data = (float *)tl_alloc_cuda(size);
    d_dst_data = (float *)tl_alloc_cuda(size);
    tl_memcpy_h2d(d_delta_data, h_delta_data, size);
    tl_memcpy_h2d(d_anchor_data, h_anchor_data, size);
    delta = tl_tensor_create(d_delta_data, ndim, dims, TL_FLOAT);
    anchor = tl_tensor_create(d_anchor_data, ndim, dims, TL_FLOAT);
    dst = tl_tensor_create(d_dst_data, ndim, dims, TL_FLOAT);

    tl_tensor_transform_bboxSQD_cuda(delta, anchor, dst, width, height,
                                     img_width, img_height, x_shift, y_shift);

    tl_memcpy_d2h(h_dst_data, d_dst_data, size);

    for (int i = 0; i < N; i++)
        ck_assert_float_eq_tol(h_dst_data_true[i], h_dst_data[i], 1e-3);

    tl_tensor_free_data_too_cuda(delta);
    tl_tensor_free_data_too_cuda(anchor);
    tl_tensor_free_data_too_cuda(dst);
}
END_TEST

START_TEST(test_tl_tensor_detect_yolov3_cuda)
{
    tl_tensor *feature;
    tl_tensor *anchors;
    tl_tensor *box_centers;
    tl_tensor *box_sizes;
    tl_tensor *boxes;
    tl_tensor *confs;
    tl_tensor *probs;
    tl_tensor *feature_d;
    tl_tensor *anchors_d;
    tl_tensor *box_centers_d;
    tl_tensor *box_sizes_d;
    tl_tensor *boxes_d;
    tl_tensor *confs_d;
    tl_tensor *probs_d;
    tl_tensor *box_centers_true;
    tl_tensor *box_sizes_true;
    tl_tensor *boxes_true;
    tl_tensor *confs_true;
    tl_tensor *probs_true;

    int class_num = 3;
    int anchor_num = 3;
    int H = 5;
    int W = 5;
    int C = anchor_num * (5 + class_num);
    int img_h = H * 32;
    int img_w = W * 32;
    int count;

    feature = tl_tensor_zeros(4, ARR(int, 1, C, H, W), TL_FLOAT);
    anchors = tl_tensor_zeros(2, ARR(int, anchor_num, 2), TL_FLOAT);
    count = tl_read_floats("data/feature.txt", feature->len, feature->data);
    ck_assert_int_eq(count, feature->len);
    count = tl_read_floats("data/anchors.txt", anchors->len, anchors->data);
    ck_assert_int_eq(count, anchors->len);

    feature_d = tl_tensor_clone_h2d(feature);
    anchors_d = tl_tensor_clone_h2d(anchors);
    box_centers_d = tl_tensor_zeros_cuda(5, ARR(int, 1, anchor_num, 2, H, W),
                                         TL_FLOAT);
    box_sizes_d = tl_tensor_zeros_cuda(5, ARR(int, 1, anchor_num, 2, H, W),
                                       TL_FLOAT);
    boxes_d = tl_tensor_zeros_cuda(5, ARR(int, 1, anchor_num, 4, H, W),
                                   TL_FLOAT);
    confs_d = tl_tensor_zeros_cuda(5, ARR(int, 1, anchor_num, 1, H, W),
                                   TL_FLOAT);
    probs_d = tl_tensor_zeros_cuda(5, ARR(int, 1, anchor_num, class_num, H, W),
                                   TL_FLOAT);

    tl_tensor_detect_yolov3_cuda(feature_d, anchors_d, box_centers_d,
                                 box_sizes_d, boxes_d, confs_d, probs_d,
                                 img_h, img_w);

    box_centers = tl_tensor_clone_d2h(box_centers_d);
    box_sizes = tl_tensor_clone_d2h(box_sizes_d);
    boxes = tl_tensor_clone_d2h(boxes_d);
    confs = tl_tensor_clone_d2h(confs_d);
    probs = tl_tensor_clone_d2h(probs_d);

    /* FILE *fp; */
    /* fp = fopen("data/confs_out.txt", "w"); */
    /* tl_tensor_fprint(fp, confs, "%.8f"); */
    /* fclose(fp); */
    /* fp = fopen("data/probs_out.txt", "w"); */
    /* tl_tensor_fprint(fp, probs, "%.8f"); */
    /* fclose(fp); */
    /* fp = fopen("data/boxes_out.txt", "w"); */
    /* tl_tensor_fprint(fp, boxes, "%.8f"); */
    /* fclose(fp); */
    /* fp = fopen("data/box_centers_out.txt", "w"); */
    /* tl_tensor_fprint(fp, box_centers, "%.8f"); */
    /* fclose(fp); */
    /* fp = fopen("data/box_sizes_out.txt", "w"); */
    /* tl_tensor_fprint(fp, box_sizes, "%.8f"); */
    /* fclose(fp); */

    confs_true = tl_tensor_zeros(confs->ndim, confs->dims, confs->dtype);
    probs_true = tl_tensor_zeros(probs->ndim, probs->dims, probs->dtype);
    boxes_true = tl_tensor_zeros(boxes->ndim, boxes->dims, boxes->dtype);
    box_centers_true = tl_tensor_zeros(box_centers->ndim, box_centers->dims,
                                       box_centers->dtype);
    box_sizes_true = tl_tensor_zeros(box_sizes->ndim, box_sizes->dims,
                                       box_sizes->dtype);
    count = tl_read_floats("data/confs.txt", confs_true->len, confs_true->data);
    ck_assert_int_eq(count, confs->len);
    count = tl_read_floats("data/probs.txt", probs_true->len, probs_true->data);
    ck_assert_int_eq(count, probs->len);
    count = tl_read_floats("data/boxes.txt", boxes_true->len, boxes_true->data);
    ck_assert_int_eq(count, boxes->len);
    count = tl_read_floats("data/box_centers.txt", box_centers_true->len,
                           box_centers_true->data);
    ck_assert_int_eq(count, box_centers->len);
    count = tl_read_floats("data/box_sizes.txt", box_sizes_true->len,
                           box_sizes_true->data);
    ck_assert_int_eq(count, box_sizes->len);

    tl_assert_tensor_eq_tol(confs_true, confs, 1e-2);
    tl_assert_tensor_eq_tol(probs_true, probs, 1e-2);
    tl_assert_tensor_eq_tol(box_centers_true, box_centers, 1e-2);
    tl_assert_tensor_eq_tol(box_sizes_true, box_sizes, 1e-2);
    tl_assert_tensor_eq_tol(boxes_true, boxes, 1e-2);

    tl_tensor_free_data_too(feature);
    tl_tensor_free_data_too(anchors);
    tl_tensor_free_data_too(box_centers);
    tl_tensor_free_data_too(box_sizes);
    tl_tensor_free_data_too(boxes);
    tl_tensor_free_data_too(confs);
    tl_tensor_free_data_too(probs);
    tl_tensor_free_data_too_cuda(feature_d);
    tl_tensor_free_data_too_cuda(anchors_d);
    tl_tensor_free_data_too_cuda(box_centers_d);
    tl_tensor_free_data_too_cuda(box_sizes_d);
    tl_tensor_free_data_too_cuda(boxes_d);
    tl_tensor_free_data_too_cuda(confs_d);
    tl_tensor_free_data_too_cuda(probs_d);
    tl_tensor_free_data_too(box_centers_true);
    tl_tensor_free_data_too(box_sizes_true);
    tl_tensor_free_data_too(boxes_true);
    tl_tensor_free_data_too(confs_true);
    tl_tensor_free_data_too(probs_true);
}
END_TEST

static void check_sorted_data(int *src, int *res, int N, tl_sort_dir dir)
{
    int *src_hist;
    int *res_hist;
    size_t size = sizeof(int) * N;

    src_hist = (int *)tl_alloc(size);
    res_hist = (int *)tl_alloc(size);
    memset(src_hist, 0, size);
    memset(res_hist, 0, size);

    for (int i = 0; i < N; i++) {
        ck_assert_int_lt(src[i], N);
        src_hist[src[i]]++;
        ck_assert_int_lt(res[i], N);
        res_hist[res[i]]++;
    }

    for (int i = 0; i < N; i++)
        ck_assert_int_eq(src_hist[i], res_hist[i]);

    if (dir == TL_SORT_DIR_ASCENDING) {
        for (int i = 0; i < N - 1; i++)
            ck_assert_int_le(res[i], res[i+1]);
    } else {
        for (int i = 0; i < N - 1; i++)
            ck_assert_int_ge(res[i], res[i+1]);
    }

    tl_free(src_hist);
    tl_free(res_hist);
}

static void check_sorted_index(int *src, int *res, int *res_index, int N)
{
    for (int i = 0; i < N; i++)
        ck_assert_int_eq(src[res_index[i]], res[i]);
}

START_TEST(test_tl_tensor_sort1d_cuda)
{
    const int N = 65536;
    int h_input[N];
    int h_output[N];
    int *d_output;
    int dims[1] = {N};
    size_t size = sizeof(int) * N;
    tl_tensor *src;
    tl_sort_dir dir;

    srand(time(NULL));
    dir = rand() % 2;
    for (int i = 0; i < N; i++) {
        h_input[i] = rand() % N;
    }

    d_output = (int *)tl_alloc_cuda(size);
    tl_memcpy_h2d(d_output, h_input, size);
    src = tl_tensor_create(d_output, 1, dims, TL_INT32);

    tl_tensor_sort1d_cuda(src, dir);

    tl_memcpy_d2h(h_output, d_output, size);
    check_sorted_data(h_input, h_output, N, dir);

    tl_tensor_free_data_too_cuda(src);
}
END_TEST

START_TEST(test_tl_tensor_sort1d_by_key_cuda)
{
    const int N = 65536;
    int h_input[N];
    int h_input_index[N];
    int h_output[N];
    int h_output_index[N];
    int *d_output;
    int *d_output_index;
    int dims[1] = {N};
    size_t size = sizeof(int) * N;
    tl_tensor *src, *index;
    tl_sort_dir dir;

    srand(time(NULL));
    dir = rand() % 2;
    for (int i = 0; i < N; i++) {
        h_input[i] = rand() % N;
        h_input_index[i] = i;
    }

    d_output = (int *)tl_alloc_cuda(size);
    d_output_index = (int *)tl_alloc_cuda(size);
    tl_memcpy_h2d(d_output, h_input, size);
    tl_memcpy_h2d(d_output_index, h_input_index, size);
    src = tl_tensor_create(d_output, 1, dims, TL_INT32);
    index = tl_tensor_create(d_output_index, 1, dims, TL_INT32);

    tl_tensor_sort1d_by_key_cuda(src, index, dir);

    tl_memcpy_d2h(h_output, d_output, size);
    tl_memcpy_d2h(h_output_index, d_output_index, size);
    check_sorted_data(h_input, h_output, N, dir);
    check_sorted_index(h_input, h_output, h_output_index, N);

    tl_tensor_free_data_too_cuda(src);
    tl_tensor_free_data_too_cuda(index);
}
END_TEST

START_TEST(test_tl_tensor_pick1d_cuda)
{
    const int N = 65536;
    int M;
    int h_input[N];
    int *h_selected_index;
    int *h_output;
    int *d_input;
    int *d_selected_index;
    int *d_output;
    int dims_N[1];
    int dims_M[1];
    size_t size_N, size_M;
    tl_tensor *src, *index, *dst;

    srand(time(NULL));
    M = rand() % N;
    dims_N[0] = N;
    dims_M[0] = M;
    size_N = sizeof(int) * N;
    size_M = sizeof(int) * M;
    h_selected_index = (int *)tl_alloc(size_M);
    h_output = (int *)tl_alloc(size_M);
    for (int i = 0; i < N; i++) {
        h_input[i] = rand() % N;
    }
    for (int i = 0; i < M; i++) {
        h_selected_index[i] = rand() % N;
    }

    d_input = (int *)tl_alloc_cuda(size_N);
    d_selected_index = (int *)tl_alloc_cuda(size_M);
    d_output = (int *)tl_alloc_cuda(size_M);
    tl_memcpy_h2d(d_input, h_input, size_N);
    tl_memcpy_h2d(d_selected_index, h_selected_index, size_M);
    src = tl_tensor_create(d_input, 1, dims_N, TL_INT32);
    index = tl_tensor_create(d_selected_index, 1, dims_M, TL_INT32);
    dst = tl_tensor_create(d_output, 1, dims_M, TL_INT32);

    tl_tensor_pick1d_cuda(src, index, dst, 1, M);

    tl_memcpy_d2h(h_output, d_output, size_M);
    check_sorted_index(h_input, h_output, h_selected_index, M);

    tl_tensor_free_data_too_cuda(src);
    tl_tensor_free_data_too_cuda(index);
    tl_tensor_free_data_too_cuda(dst);
    tl_free(h_selected_index);
    tl_free(h_output);
}
END_TEST

START_TEST(test_tl_tensor_submean_cuda)
{
    uint8_t src_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    uint8_t *src_data_d;
    float *dst_data_d;
    float true_data[] = {0, 3, 6, 9, 0, 3, 6, 9, 0, 3, 6, 9};
    double mean[] = {1, 2, 3};
    tl_tensor *dst, *true_tensor;
    tl_tensor *src_d, *dst_d;

    src_data_d = tl_clone_h2d(src_data, sizeof(uint8_t) * 12);
    dst_data_d = tl_alloc_cuda(sizeof(float) * 12);

    true_tensor = tl_tensor_create(true_data, 3, ARR(int,3,2,2), TL_FLOAT);
    src_d = tl_tensor_create(src_data_d, 3, ARR(int,2,2,3), TL_UINT8);
    dst_d = tl_tensor_submean_cuda(src_d, NULL, mean);
    dst = tl_tensor_clone_d2h(dst_d);
    ck_assert_int_eq(dst->dtype, TL_FLOAT);
    ck_assert_int_eq(dst->ndim, 3);
    ck_assert_array_int_eq(dst->dims, ARR(int,3,2,2), 3);
    tl_assert_tensor_eq(dst, true_tensor);
    tl_tensor_free_data_too_cuda(dst_d);
    tl_tensor_free_data_too(dst);

    dst_d = tl_tensor_create(dst_data_d, 3, ARR(int,3,2,2), TL_FLOAT);
    tl_tensor_submean_cuda(src_d, dst_d, mean);
    dst = tl_tensor_clone_d2h(dst_d);
    tl_assert_tensor_eq(dst, true_tensor);
    tl_tensor_free(dst_d);
    tl_tensor_free_data_too(dst);

    tl_free_cuda(src_data_d);
    tl_free_cuda(dst_data_d);
    tl_tensor_free(src_d);
    tl_tensor_free(true_tensor);
}
END_TEST
/* end of tests */

Suite *make_tensor_cuda_suite(void)
{
    Suite *s;
    TCase *tc_tensor_cuda;

    s = suite_create("tensor_cuda");
    tc_tensor_cuda = tcase_create("tensor_cuda");
    tcase_add_checked_fixture(tc_tensor_cuda, setup, teardown);
    tcase_set_timeout(tc_tensor_cuda, 3000);

    tcase_add_test(tc_tensor_cuda, test_tl_tensor_zeros_cuda);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_free_data_too_cuda);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_clone_h2d);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_clone_d2h);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_clone_d2d);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_repeat_h2d);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_repeat_d2d);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_repeat_d2h);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_arange_cuda);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_rearange_cuda);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_fprint_cuda);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_print_cuda);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_save_cuda);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_zeros_slice_cuda);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_slice_cuda);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_maxreduce_cuda);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_elew_cuda);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_dot_product_cuda);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_convert_cuda);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_transpose_cuda);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_lrelu_cuda);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_transform_bboxSQD_cuda);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_detect_yolov3_cuda);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_resize_cuda);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_sort1d_cuda);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_sort1d_by_key_cuda);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_pick1d_cuda);
    tcase_add_test(tc_tensor_cuda, test_tl_tensor_submean_cuda);
    /* end of adding tests */

    suite_add_tcase(s, tc_tensor_cuda);

    return s;
}

#endif /* TL_CUDA */

