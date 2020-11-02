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

template<typename T1, typename T2>
static void thrust_sort_by_key(T1 *data, T2 *index, int len, tl_sort_dir dir)
{
    if (dir == TL_SORT_DIR_DESCENDING)
        thrust::sort_by_key(thrust::device, data, data + len, index,
                            thrust::greater<T1>());
    else
        thrust::sort_by_key(thrust::device, data, data + len, index,
                            thrust::less<T1>());
}

void tl_tensor_sort1d_by_key_cuda(tl_tensor *key, tl_tensor *val,
                                  tl_sort_dir dir)
{
    assert(key);
    assert(tl_is_device_mem(key->data));
    assert(key->ndim == 1);
    assert(tl_is_device_mem(val->data));
    assert(val->ndim == 1);
    assert(val->len == key->len);

    switch (key->dtype) {
    case TL_DOUBLE:
        switch (val->dtype) {
        case TL_DOUBLE:
            thrust_sort_by_key<double, double>((double *)key->data,
                                               (double *)val->data,
                                               key->len, dir);
            break;
        case TL_FLOAT:
            thrust_sort_by_key<double, float>((double *)key->data,
                                              (float *)val->data,
                                              key->len, dir);
            break;
        case TL_INT32:
            thrust_sort_by_key<double, int32_t>((double *)key->data,
                                                (int32_t *)val->data,
                                                key->len, dir);
            break;
        case TL_INT16:
            thrust_sort_by_key<double, int16_t>((double *)key->data,
                                                (int16_t *)val->data,
                                                key->len, dir);
            break;
        case TL_INT8:
            thrust_sort_by_key<double, int8_t>((double *)key->data,
                                               (int8_t *)val->data,
                                               key->len, dir);
            break;
        case TL_UINT32:
            thrust_sort_by_key<double, uint32_t>((double *)key->data,
                                                 (uint32_t *)val->data,
                                                 key->len, dir);
            break;
        case TL_UINT16:
            thrust_sort_by_key<double, uint16_t>((double *)key->data,
                                                 (uint16_t *)val->data,
                                                 key->len, dir);
            break;
        case TL_UINT8:
            thrust_sort_by_key<double, uint8_t>((double *)key->data,
                                                (uint8_t *)val->data,
                                                key->len, dir);
            break;
        case TL_BOOL:
            thrust_sort_by_key<double, int>((double *)key->data,
                                            (int *)val->data,
                                            key->len, dir);
            break;
        default:
            assert(0 && "unsupported tl_dtype of 'val'");
            break;
        }
        break;
    case TL_FLOAT:
        switch (val->dtype) {
        case TL_DOUBLE:
            thrust_sort_by_key<float, double>((float *)key->data,
                                              (double *)val->data,
                                              key->len, dir);
            break;
        case TL_FLOAT:
            thrust_sort_by_key<float, float>((float *)key->data,
                                             (float *)val->data,
                                             key->len, dir);
            break;
        case TL_INT32:
            thrust_sort_by_key<float, int32_t>((float *)key->data,
                                               (int32_t *)val->data,
                                               key->len, dir);
            break;
        case TL_INT16:
            thrust_sort_by_key<float, int16_t>((float *)key->data,
                                               (int16_t *)val->data,
                                               key->len, dir);
            break;
        case TL_INT8:
            thrust_sort_by_key<float, int8_t>((float *)key->data,
                                              (int8_t *)val->data,
                                              key->len, dir);
            break;
        case TL_UINT32:
            thrust_sort_by_key<float, uint32_t>((float *)key->data,
                                                (uint32_t *)val->data,
                                                key->len, dir);
            break;
        case TL_UINT16:
            thrust_sort_by_key<float, uint16_t>((float *)key->data,
                                                (uint16_t *)val->data,
                                                key->len, dir);
            break;
        case TL_UINT8:
            thrust_sort_by_key<float, uint8_t>((float *)key->data,
                                               (uint8_t *)val->data,
                                               key->len, dir);
            break;
        case TL_BOOL:
            thrust_sort_by_key<float, int>((float *)key->data,
                                           (int *)val->data,
                                           key->len, dir);
            break;
        default:
            assert(0 && "unsupported tl_dtype of 'val'");
            break;
        }
        break;
    case TL_INT32:
        switch (val->dtype) {
        case TL_DOUBLE:
            thrust_sort_by_key<int32_t, double>((int32_t *)key->data,
                                                (double *)val->data,
                                                key->len, dir);
            break;
        case TL_FLOAT:
            thrust_sort_by_key<int32_t, float>((int32_t *)key->data,
                                               (float *)val->data,
                                               key->len, dir);
            break;
        case TL_INT32:
            thrust_sort_by_key<int32_t, int32_t>((int32_t *)key->data,
                                                 (int32_t *)val->data,
                                                 key->len, dir);
            break;
        case TL_INT16:
            thrust_sort_by_key<int32_t, int16_t>((int32_t *)key->data,
                                                 (int16_t *)val->data,
                                                 key->len, dir);
            break;
        case TL_INT8:
            thrust_sort_by_key<int32_t, int8_t>((int32_t *)key->data,
                                                (int8_t *)val->data,
                                                key->len, dir);
            break;
        case TL_UINT32:
            thrust_sort_by_key<int32_t, uint32_t>((int32_t *)key->data,
                                                  (uint32_t *)val->data,
                                                  key->len, dir);
            break;
        case TL_UINT16:
            thrust_sort_by_key<int32_t, uint16_t>((int32_t *)key->data,
                                                  (uint16_t *)val->data,
                                                  key->len, dir);
            break;
        case TL_UINT8:
            thrust_sort_by_key<int32_t, uint8_t>((int32_t *)key->data,
                                                 (uint8_t *)val->data,
                                                 key->len, dir);
            break;
        case TL_BOOL:
            thrust_sort_by_key<int32_t, int>((int32_t *)key->data,
                                             (int *)val->data,
                                             key->len, dir);
            break;
        default:
            assert(0 && "unsupported tl_dtype of 'val'");
            break;
        }
        break;
    case TL_INT16:
        switch (val->dtype) {
        case TL_DOUBLE:
            thrust_sort_by_key<int16_t, double>((int16_t *)key->data,
                                                (double *)val->data,
                                                key->len, dir);
            break;
        case TL_FLOAT:
            thrust_sort_by_key<int16_t, float>((int16_t *)key->data,
                                               (float *)val->data,
                                               key->len, dir);
            break;
        case TL_INT32:
            thrust_sort_by_key<int16_t, int32_t>((int16_t *)key->data,
                                                 (int32_t *)val->data,
                                                 key->len, dir);
            break;
        case TL_INT16:
            thrust_sort_by_key<int16_t, int16_t>((int16_t *)key->data,
                                                 (int16_t *)val->data,
                                                 key->len, dir);
            break;
        case TL_INT8:
            thrust_sort_by_key<int16_t, int8_t>((int16_t *)key->data,
                                                (int8_t *)val->data,
                                                key->len, dir);
            break;
        case TL_UINT32:
            thrust_sort_by_key<int16_t, uint32_t>((int16_t *)key->data,
                                                  (uint32_t *)val->data,
                                                  key->len, dir);
            break;
        case TL_UINT16:
            thrust_sort_by_key<int16_t, uint16_t>((int16_t *)key->data,
                                                  (uint16_t *)val->data,
                                                  key->len, dir);
            break;
        case TL_UINT8:
            thrust_sort_by_key<int16_t, uint8_t>((int16_t *)key->data,
                                                 (uint8_t *)val->data,
                                                 key->len, dir);
            break;
        case TL_BOOL:
            thrust_sort_by_key<int16_t, int>((int16_t *)key->data,
                                             (int *)val->data,
                                             key->len, dir);
            break;
        default:
            assert(0 && "unsupported tl_dtype of 'val'");
            break;
        }
        break;
    case TL_INT8:
        switch (val->dtype) {
        case TL_DOUBLE:
            thrust_sort_by_key<int8_t, double>((int8_t *)key->data,
                                               (double *)val->data,
                                               key->len, dir);
            break;
        case TL_FLOAT:
            thrust_sort_by_key<int8_t, float>((int8_t *)key->data,
                                              (float *)val->data,
                                              key->len, dir);
            break;
        case TL_INT32:
            thrust_sort_by_key<int8_t, int32_t>((int8_t *)key->data,
                                                (int32_t *)val->data,
                                                key->len, dir);
            break;
        case TL_INT16:
            thrust_sort_by_key<int8_t, int16_t>((int8_t *)key->data,
                                                (int16_t *)val->data,
                                                key->len, dir);
            break;
        case TL_INT8:
            thrust_sort_by_key<int8_t, int8_t>((int8_t *)key->data,
                                               (int8_t *)val->data,
                                               key->len, dir);
            break;
        case TL_UINT32:
            thrust_sort_by_key<int8_t, uint32_t>((int8_t *)key->data,
                                                 (uint32_t *)val->data,
                                                 key->len, dir);
            break;
        case TL_UINT16:
            thrust_sort_by_key<int8_t, uint16_t>((int8_t *)key->data,
                                                 (uint16_t *)val->data,
                                                 key->len, dir);
            break;
        case TL_UINT8:
            thrust_sort_by_key<int8_t, uint8_t>((int8_t *)key->data,
                                                (uint8_t *)val->data,
                                                key->len, dir);
            break;
        case TL_BOOL:
            thrust_sort_by_key<int8_t, int>((int8_t *)key->data,
                                            (int *)val->data,
                                            key->len, dir);
            break;
        default:
            assert(0 && "unsupported tl_dtype of 'val'");
            break;
        }
        break;
    case TL_UINT32:
        switch (val->dtype) {
        case TL_DOUBLE:
            thrust_sort_by_key<uint32_t, double>((uint32_t *)key->data,
                                                 (double *)val->data,
                                                 key->len, dir);
            break;
        case TL_FLOAT:
            thrust_sort_by_key<uint32_t, float>((uint32_t *)key->data,
                                                (float *)val->data,
                                                key->len, dir);
            break;
        case TL_INT32:
            thrust_sort_by_key<uint32_t, int32_t>((uint32_t *)key->data,
                                                  (int32_t *)val->data,
                                                  key->len, dir);
            break;
        case TL_INT16:
            thrust_sort_by_key<uint32_t, int16_t>((uint32_t *)key->data,
                                                  (int16_t *)val->data,
                                                  key->len, dir);
            break;
        case TL_INT8:
            thrust_sort_by_key<uint32_t, int8_t>((uint32_t *)key->data,
                                                 (int8_t *)val->data,
                                                 key->len, dir);
            break;
        case TL_UINT32:
            thrust_sort_by_key<uint32_t, uint32_t>((uint32_t *)key->data,
                                                   (uint32_t *)val->data,
                                                   key->len, dir);
            break;
        case TL_UINT16:
            thrust_sort_by_key<uint32_t, uint16_t>((uint32_t *)key->data,
                                                   (uint16_t *)val->data,
                                                   key->len, dir);
            break;
        case TL_UINT8:
            thrust_sort_by_key<uint32_t, uint8_t>((uint32_t *)key->data,
                                                  (uint8_t *)val->data,
                                                  key->len, dir);
            break;
        case TL_BOOL:
            thrust_sort_by_key<uint32_t, int>((uint32_t *)key->data,
                                              (int *)val->data,
                                              key->len, dir);
            break;
        default:
            assert(0 && "unsupported tl_dtype of 'val'");
            break;
        }
        break;
    case TL_UINT16:
        switch (val->dtype) {
        case TL_DOUBLE:
            thrust_sort_by_key<uint16_t, double>((uint16_t *)key->data,
                                                 (double *)val->data,
                                                 key->len, dir);
            break;
        case TL_FLOAT:
            thrust_sort_by_key<uint16_t, float>((uint16_t *)key->data,
                                                (float *)val->data,
                                                key->len, dir);
            break;
        case TL_INT32:
            thrust_sort_by_key<uint16_t, int32_t>((uint16_t *)key->data,
                                                  (int32_t *)val->data,
                                                  key->len, dir);
            break;
        case TL_INT16:
            thrust_sort_by_key<uint16_t, int16_t>((uint16_t *)key->data,
                                                  (int16_t *)val->data,
                                                  key->len, dir);
            break;
        case TL_INT8:
            thrust_sort_by_key<uint16_t, int8_t>((uint16_t *)key->data,
                                                 (int8_t *)val->data,
                                                 key->len, dir);
            break;
        case TL_UINT32:
            thrust_sort_by_key<uint16_t, uint32_t>((uint16_t *)key->data,
                                                   (uint32_t *)val->data,
                                                   key->len, dir);
            break;
        case TL_UINT16:
            thrust_sort_by_key<uint16_t, uint16_t>((uint16_t *)key->data,
                                                   (uint16_t *)val->data,
                                                   key->len, dir);
            break;
        case TL_UINT8:
            thrust_sort_by_key<uint16_t, uint8_t>((uint16_t *)key->data,
                                                  (uint8_t *)val->data,
                                                  key->len, dir);
            break;
        case TL_BOOL:
            thrust_sort_by_key<uint16_t, int>((uint16_t *)key->data,
                                              (int *)val->data,
                                              key->len, dir);
            break;
        default:
            assert(0 && "unsupported tl_dtype of 'val'");
            break;
        }
        break;
    case TL_UINT8:
        switch (val->dtype) {
        case TL_DOUBLE:
            thrust_sort_by_key<uint8_t, double>((uint8_t *)key->data,
                                                (double *)val->data,
                                                key->len, dir);
            break;
        case TL_FLOAT:
            thrust_sort_by_key<uint8_t, float>((uint8_t *)key->data,
                                               (float *)val->data,
                                               key->len, dir);
            break;
        case TL_INT32:
            thrust_sort_by_key<uint8_t, int32_t>((uint8_t *)key->data,
                                                 (int32_t *)val->data,
                                                 key->len, dir);
            break;
        case TL_INT16:
            thrust_sort_by_key<uint8_t, int16_t>((uint8_t *)key->data,
                                                 (int16_t *)val->data,
                                                 key->len, dir);
            break;
        case TL_INT8:
            thrust_sort_by_key<uint8_t, int8_t>((uint8_t *)key->data,
                                                (int8_t *)val->data,
                                                key->len, dir);
            break;
        case TL_UINT32:
            thrust_sort_by_key<uint8_t, uint32_t>((uint8_t *)key->data,
                                                  (uint32_t *)val->data,
                                                  key->len, dir);
            break;
        case TL_UINT16:
            thrust_sort_by_key<uint8_t, uint16_t>((uint8_t *)key->data,
                                                  (uint16_t *)val->data,
                                                  key->len, dir);
            break;
        case TL_UINT8:
            thrust_sort_by_key<uint8_t, uint8_t>((uint8_t *)key->data,
                                                 (uint8_t *)val->data,
                                                 key->len, dir);
            break;
        case TL_BOOL:
            thrust_sort_by_key<uint8_t, int>((uint8_t *)key->data,
                                             (int *)val->data,
                                             key->len, dir);
            break;
        default:
            assert(0 && "unsupported tl_dtype of 'val'");
            break;
        }
        break;
    case TL_BOOL:
        switch (val->dtype) {
        case TL_DOUBLE:
            thrust_sort_by_key<int, double>((int *)key->data,
                                            (double *)val->data,
                                            key->len, dir);
            break;
        case TL_FLOAT:
            thrust_sort_by_key<int, float>((int *)key->data,
                                           (float *)val->data,
                                           key->len, dir);
            break;
        case TL_INT32:
            thrust_sort_by_key<int, int32_t>((int *)key->data,
                                             (int32_t *)val->data,
                                             key->len, dir);
            break;
        case TL_INT16:
            thrust_sort_by_key<int, int16_t>((int *)key->data,
                                             (int16_t *)val->data,
                                             key->len, dir);
            break;
        case TL_INT8:
            thrust_sort_by_key<int, int8_t>((int *)key->data,
                                            (int8_t *)val->data,
                                            key->len, dir);
            break;
        case TL_UINT32:
            thrust_sort_by_key<int, uint32_t>((int *)key->data,
                                              (uint32_t *)val->data,
                                              key->len, dir);
            break;
        case TL_UINT16:
            thrust_sort_by_key<int, uint16_t>((int *)key->data,
                                              (uint16_t *)val->data,
                                              key->len, dir);
            break;
        case TL_UINT8:
            thrust_sort_by_key<int, uint8_t>((int *)key->data,
                                             (uint8_t *)val->data,
                                             key->len, dir);
            break;
        case TL_BOOL:
            thrust_sort_by_key<int, int>((int *)key->data,
                                         (int *)val->data,
                                         key->len, dir);
            break;
        default:
            assert(0 && "unsupported tl_dtype of 'val'");
            break;
        }
        break;
    default:
        assert(0 && "unsupported tl_dtype");
        break;
    }
    tl_cuda_device_sync();
}
