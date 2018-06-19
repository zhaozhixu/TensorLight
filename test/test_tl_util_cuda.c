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

#include "test_tensorlight.h"

static void setup(void)
{
}

static void teardown(void)
{
}

START_TEST(test_tl_is_device_mem)
{
}
END_TEST

START_TEST(test_tl_alloc_cuda)
{
}
END_TEST

START_TEST(test_tl_clone_h2d)
{
}
END_TEST

START_TEST(test_tl_clone_d2h)
{
}
END_TEST

START_TEST(test_tl_clone_d2d)
{
}
END_TEST

START_TEST(test_tl_repeat_h2d)
{
}
END_TEST

START_TEST(test_tl_repeat_d2h)
{
}
END_TEST

START_TEST(test_tl_repeat_d2d)
{
}
END_TEST
/* end of tests */

Suite *make_util_cuda_suite(void)
{
     Suite *s;
     TCase *tc_util_cuda;

     s = suite_create("util_cuda");
     tc_util_cuda = tcase_create("util_cuda");
     tcase_add_checked_fixture(tc_util_cuda, setup, teardown);

     tcase_add_test(tc_util_cuda, test_tl_is_device_mem);
     tcase_add_test(tc_util_cuda, test_tl_alloc_cuda);
     tcase_add_test(tc_util_cuda, test_tl_clone_h2d);
     tcase_add_test(tc_util_cuda, test_tl_clone_d2h);
     tcase_add_test(tc_util_cuda, test_tl_clone_d2d);
     tcase_add_test(tc_util_cuda, test_tl_repeat_h2d);
     tcase_add_test(tc_util_cuda, test_tl_repeat_d2h);
     tcase_add_test(tc_util_cuda, test_tl_repeat_d2d);
     /* end of adding tests */

     suite_add_tcase(s, tc_util_cuda);

     return s;
}

#endif /* TL_CUDA */

