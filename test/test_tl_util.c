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

#include "lightnettest/ln_test.h"
#include "tl_check.h"
#include "tl_util.h"

static int *data;
static int data_len;
static size_t data_size;

static void checked_setup(void)
{
     data_len = 5;
     data_size = sizeof(int) * data_len;
     data = tl_alloc(data_size);
     int i;
     for (i = 0; i < data_len; i++)
          data[i] = i;
}

static void checked_teardown(void)
{
     tl_free(data);
}

LN_TEST_START(test_tl_alloc)
{
}
LN_TEST_END

LN_TEST_START(test_tl_memcpy)
{
     int *data_cpy = tl_alloc(data_size);
     tl_memcpy(data_cpy, data, data_size);
     for (int i = 0; i < data_len; i++)
          ck_assert_int_eq(data_cpy[i], data[i]);
     tl_free(data_cpy);
}
LN_TEST_END

LN_TEST_START(test_tl_clone)
{
     int *data_clone;
     int i;

     data_clone = tl_clone(data, data_size);
     for (i = 0; i < data_len; i++)
          ck_assert_int_eq(data_clone[i], data[i]);
     tl_free(data_clone);
}
LN_TEST_END

LN_TEST_START(test_tl_repeat)
{
     int *data_repeat;
     int i;

     data_repeat = tl_repeat(data, data_size, 5);
     for (i = 0; i < data_len*5; i++)
          ck_assert_int_eq(data_repeat[i], data[i%5]);
     tl_free(data_repeat);
}
LN_TEST_END

LN_TEST_START(test_tl_read_floats)
{
    int count;
    float data[3];
    float data_true[3] = {1.2, 3.4, 5.6};

    count = tl_read_floats("data/test_floats.txt", 3, data);
    ck_assert_int_eq(count, 3);
    ck_assert_array_float_eq_tol(data, data_true, 3, 0);
}
LN_TEST_END
/* end of tests */

LN_TEST_TCASE_START(util, checked_setup, checked_teardown)
{
    LN_TEST_ADD_TEST(test_tl_alloc);
    LN_TEST_ADD_TEST(test_tl_memcpy);
    LN_TEST_ADD_TEST(test_tl_clone);
    LN_TEST_ADD_TEST(test_tl_repeat);
    LN_TEST_ADD_TEST(test_tl_read_floats);

}
LN_TEST_TCASE_END

LN_TEST_ADD_TCASE(util);
