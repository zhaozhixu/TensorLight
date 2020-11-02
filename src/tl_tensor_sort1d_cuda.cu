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

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "tl_tensor.h"
#include "tl_util.h"

template<typename T>
static void thrust_sort(T *data, int len, tl_sort_dir dir)
{
    if (dir == TL_SORT_DIR_DESCENDING)
        thrust::sort(thrust::device, data, data + len, thrust::greater<T>());
    else
        thrust::sort(thrust::device, data, data + len, thrust::less<T>());
}

void tl_tensor_sort1d_cuda(tl_tensor *key, tl_sort_dir dir)
{
    assert(key);
    assert(tl_is_device_mem(key->data));
    assert(key->ndim == 1);

    switch (key->dtype) {
    case TL_DOUBLE:
        thrust_sort<double>((double *)key->data, key->len, dir);
        break;
    case TL_FLOAT:
        thrust_sort<float>((float *)key->data, key->len, dir);
        break;
    case TL_INT32:
        thrust_sort<int32_t>((int32_t *)key->data, key->len, dir);
        break;
    case TL_INT16:
        thrust_sort<int16_t>((int16_t *)key->data, key->len, dir);
        break;
    case TL_INT8:
        thrust_sort<int8_t>((int8_t *)key->data, key->len, dir);
        break;
    case TL_UINT32:
        thrust_sort<uint32_t>((uint32_t *)key->data, key->len, dir);
        break;
    case TL_UINT16:
        thrust_sort<uint16_t>((uint16_t *)key->data, key->len, dir);
        break;
    case TL_UINT8:
        thrust_sort<uint8_t>((uint8_t *)key->data, key->len, dir);
        break;
    case TL_BOOL:
        thrust_sort<int>((int *)key->data, key->len, dir);
        break;
    default:
        assert(0 && "unsupported tl_dtype");
        break;
    }
    tl_cuda_device_sync();
}
