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

#ifdef TL_CUDA

#include "lightnettest/ln_test.h"
#include "tl_util.h"

static void checked_setup(void)
{
}

static void checked_teardown(void)
{
}

LN_TEST_START(test_tl_is_device_mem)
{
     void *p_d, *p_h;

     p_d = tl_alloc_cuda(sizeof(float));
     p_h = tl_alloc(sizeof(float));

     ck_assert_int_eq(tl_is_device_mem(p_d), 1);
     ck_assert_int_eq(tl_is_device_mem(p_h), 0);

     tl_free(p_h);
     tl_free_cuda(p_d);
}
LN_TEST_END

LN_TEST_START(test_tl_alloc_cuda)
{
}
LN_TEST_END

LN_TEST_START(test_tl_memcpy_h2d)
{
     int array[] = {0, 1, 2};
     int *array_h = tl_alloc(sizeof(int)*3);
     int *array_d = tl_alloc_cuda(sizeof(int)*3);

     tl_memcpy_h2d(array_d, array, sizeof(int)*3);
     tl_memcpy_d2h(array_h, array_d, sizeof(int)*3);
     for (int i = 0; i < 3; i++)
          ck_assert_int_eq(array_h[i], array[i]);

     tl_free(array_h);
     tl_free_cuda(array_d);
}
LN_TEST_END

LN_TEST_START(test_tl_memcpy_d2h)
{
}
LN_TEST_END

LN_TEST_START(test_tl_memcpy_d2d)
{
     int array[] = {0, 1, 2};
     int *array_h = tl_alloc(sizeof(int)*3);
     int *array_d1 = tl_alloc_cuda(sizeof(int)*3);
     int *array_d2 = tl_alloc_cuda(sizeof(int)*3);

     tl_memcpy_h2d(array_d1, array, sizeof(int)*3);
     tl_memcpy_d2d(array_d2, array_d1, sizeof(int)*3);
     tl_memcpy_d2h(array_h, array_d2, sizeof(int)*3);
     for (int i = 0; i < 3; i++)
          ck_assert_int_eq(array_h[i], array[i]);

     tl_free(array_h);
     tl_free_cuda(array_d1);
     tl_free_cuda(array_d2);
}
LN_TEST_END

LN_TEST_START(test_tl_clone_h2d)
{
     int array[] = {0, 1, 2};
     int *array_d, *array_h;
     int i;

     array_d = tl_clone_h2d(array, sizeof(int)*3);
     array_h = tl_clone_d2h(array_d, sizeof(int)*3);
     for (i = 0; i < 3; i++)
          ck_assert_int_eq(array_h[i], i);

     tl_free(array_h);
     tl_free_cuda(array_d);
}
LN_TEST_END

LN_TEST_START(test_tl_clone_d2h)
{
}
LN_TEST_END

LN_TEST_START(test_tl_clone_d2d)
{
     int array[] = {0, 1, 2};
     int *array_d1, *array_d2, *array_h;
     int i;

     array_d1 = tl_clone_h2d(array, sizeof(int)*3);
     array_d2 = tl_clone_d2d(array_d1, sizeof(int)*3);
     array_h = tl_clone_d2h(array_d2, sizeof(int)*3);
     for (i = 0; i < 3; i++)
          ck_assert_int_eq(array_h[i], i);

     tl_free(array_h);
     tl_free_cuda(array_d1);
     tl_free_cuda(array_d2);
}
LN_TEST_END

LN_TEST_START(test_tl_repeat_h2d)
{
     int array[] = {0, 1, 2};
     int *array_d, *array_h;
     int i;

     array_d = tl_repeat_h2d(array, sizeof(int)*3, 3);
     array_h = tl_clone_d2h(array_d, sizeof(int)*3*3);
     for (i = 0; i < 3*3; i++)
          ck_assert_int_eq(array_h[i], i%3);

     tl_free(array_h);
     tl_free_cuda(array_d);
}
LN_TEST_END

LN_TEST_START(test_tl_repeat_d2h)
{
     int array[] = {0, 1, 2};
     int *array_d, *array_h;
     int i;

     array_d = tl_clone_h2d(array, sizeof(int)*3);
     array_h = tl_repeat_d2h(array_d, sizeof(int)*3, 3);
     for (i = 0; i < 3*3; i++)
          ck_assert_int_eq(array_h[i], i%3);

     tl_free(array_h);
     tl_free_cuda(array_d);
}
LN_TEST_END

LN_TEST_START(test_tl_repeat_d2d)
{
     int array[] = {0, 1, 2};
     int *array_d1, *array_d2, *array_h;
     int i;

     array_d1 = tl_clone_h2d(array, sizeof(int)*3);
     array_d2 = tl_repeat_d2d(array_d1, sizeof(int)*3, 3);
     array_h = tl_clone_d2h(array_d2, sizeof(int)*3*3);
     for (i = 0; i < 3*3; i++)
          ck_assert_int_eq(array_h[i], i%3);

     tl_free(array_h);
     tl_free_cuda(array_d1);
     tl_free_cuda(array_d2);
}
LN_TEST_END
/* end of tests */

LN_TEST_TCASE_START_TIMEOUT(util_cuda, checked_setup, checked_teardown, 30)
{
    LN_TEST_ADD_TEST(test_tl_is_device_mem);
    LN_TEST_ADD_TEST(test_tl_alloc_cuda);
    LN_TEST_ADD_TEST(test_tl_memcpy_h2d);
    LN_TEST_ADD_TEST(test_tl_memcpy_d2h);
    LN_TEST_ADD_TEST(test_tl_memcpy_d2d);
    LN_TEST_ADD_TEST(test_tl_clone_h2d);
    LN_TEST_ADD_TEST(test_tl_clone_d2h);
    LN_TEST_ADD_TEST(test_tl_clone_d2d);
    LN_TEST_ADD_TEST(test_tl_repeat_h2d);
    LN_TEST_ADD_TEST(test_tl_repeat_d2h);
    LN_TEST_ADD_TEST(test_tl_repeat_d2d);
}
LN_TEST_TCASE_END

LN_TEST_ADD_TCASE(util_cuda);

#endif /* TL_CUDA */

