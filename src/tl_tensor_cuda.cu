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

#include "tl_tensor_internal_cuda.h"

TL_EXPORT void tl_tensor_free_data_too_cuda(tl_tensor *t)
{
    if (!t)
        return;
    tl_free_cuda(t->data);
    tl_tensor_free(t);
}

TL_EXPORT tl_tensor *tl_tensor_zeros_cuda(int ndim, const int *dims, tl_dtype dtype)
{
    tl_tensor *t;
    size_t size;

    t = tl_tensor_create(NULL, ndim, dims, dtype);
    t->owner = t;
    size = t->len * tl_size_of(dtype);
    t->data = tl_alloc_cuda(size);
    tl_memset_cuda(t->data, 0, size);
    return t;
}

TL_EXPORT tl_tensor *tl_tensor_clone_h2d(const tl_tensor *src)
{
    void *data;
    tl_tensor *dst;

    assert(src);
    data = tl_clone_h2d(src->data, src->len * tl_size_of(src->dtype));
    dst = tl_tensor_create(data, src->ndim, src->dims, src->dtype);
    dst->owner = dst;
    return dst;
}

TL_EXPORT tl_tensor *tl_tensor_clone_d2h(const tl_tensor *src)
{
    void *data;
    tl_tensor *dst;

    assert(src);
    data = tl_clone_d2h(src->data, src->len * tl_size_of(src->dtype));
    dst = tl_tensor_create(data, src->ndim, src->dims, src->dtype);
    dst->owner = dst;
    return dst;
}

TL_EXPORT tl_tensor *tl_tensor_clone_d2d(const tl_tensor *src)
{
    void *data;
    tl_tensor *dst;

    assert(src);
    data = tl_clone_d2d(src->data, src->len * tl_size_of(src->dtype));
    dst = tl_tensor_create(data, src->ndim, src->dims, src->dtype);
    dst->owner = dst;
    return dst;
}

TL_EXPORT tl_tensor *tl_tensor_repeat_h2d(const tl_tensor *src, int times)
{
    void *data;
    int *dims;
    tl_tensor *dst;

    assert(src);
    data = tl_repeat_h2d(src->data, src->len * tl_size_of(src->dtype), times);
    dims = (int *)tl_alloc(sizeof(int) * (src->ndim + 1));
    memmove(dims + 1, src->dims, sizeof(int) * (src->ndim));
    dims[0] = times;
    dst = tl_tensor_create(data, src->ndim + 1, dims, src->dtype);
    dst->owner = dst;
    tl_free(dims);
    return dst;
}

TL_EXPORT tl_tensor *tl_tensor_repeat_d2h(const tl_tensor *src, int times)
{
    void *data;
    int *dims;
    tl_tensor *dst;

    assert(src);
    data = tl_repeat_d2h(src->data, src->len * tl_size_of(src->dtype), times);
    dims = (int *)tl_alloc(sizeof(int) * (src->ndim + 1));
    memmove(dims + 1, src->dims, sizeof(int) * (src->ndim));
    dims[0] = times;
    dst = tl_tensor_create(data, src->ndim + 1, dims, src->dtype);
    dst->owner = dst;
    tl_free(dims);
    return dst;
}

TL_EXPORT tl_tensor *tl_tensor_repeat_d2d(const tl_tensor *src, int times)
{
    void *data;
    int *dims;
    tl_tensor *dst;

    assert(src);
    data = tl_repeat_d2d(src->data, src->len * tl_size_of(src->dtype), times);
    dims = (int *)tl_alloc(sizeof(int) * (src->ndim + 1));
    memmove(dims + 1, src->dims, sizeof(int) * (src->ndim));
    dims[0] = times;
    dst = tl_tensor_create(data, src->ndim + 1, dims, src->dtype);
    dst->owner = dst;
    tl_free(dims);
    return dst;
}

/* arrange at host, copy to device */
TL_EXPORT tl_tensor *tl_tensor_arange_cuda(double start, double stop, double step, tl_dtype dtype)
{
    int dims[1];
    void *data;
    tl_tensor *dst;
    double len, elem;
    size_t dsize;

    dsize = tl_size_of(dtype);
#ifdef TL_DEBUG
    double max_d, min_d;
    max_d = tl_dtype_max_double(dtype);
    min_d = tl_dtype_min_double(dtype);
    assert(start >= min_d && start <= max_d);
    assert(stop >= min_d && stop <= max_d);
    assert(step >= min_d && step <= max_d);
    assert(step != 0);
    assert(stop > start); /* TODO: expand to all possibilities */
#endif

    len = ceil((stop - start) / step);
    if (len > INT32_MAX)
        return NULL;

    dims[0] = (int)len;
    dst = tl_tensor_zeros_cuda(1, dims, dtype);
    data = tl_tensor_zeros(1, dims, dtype);
    for (int i = 0; i < dims[0]; i++) {
        elem = start + step * i;
        tl_convert(tl_padd(data, i, dsize), dtype, &elem, TL_DOUBLE);
    }
    tl_memcpy_h2d(dst->data, data, tl_size_of(dst->dtype) * dst->len);

    return dst;
}

template <typename T>
__global__ void rearange_kernel(T *dst, int len, double start, double step, int block_size,
                                int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;

    dst[di] = (T)(start + step * di);
}

TL_EXPORT void tl_tensor_rearange_cuda(tl_tensor *src, double start, double stop, double step)
{
    double len;

#ifdef TL_DEBUG
    double max_d, min_d;
    max_d = tl_dtype_max_double(dtype);
    min_d = tl_dtype_min_double(dtype);
    assert(start >= min_d && start <= max_d);
    assert(stop >= min_d && stop <= max_d);
    assert(step >= min_d && step <= max_d);
    assert(step != 0);
    assert(stop > start); /* TODO: expand to all possibilities */
#endif

    len = ceil((stop - start) / step);

    assert(len <= INT32_MAX);
    assert(src->ndim == 1);
    assert(src->len == (int)len);
    assert(src->data);

    int thread_num, block_num;
    thread_num = src->len;
    block_num = BLOCK_NUM(BLOCK_SIZE, thread_num);

    switch (src->dtype) {
    case TL_DOUBLE:
        rearange_kernel<<<block_num, BLOCK_SIZE>>>((double *)src->data, src->len, start, step,
                                                     BLOCK_SIZE, thread_num);
        break;
    case TL_FLOAT:
        rearange_kernel<<<block_num, BLOCK_SIZE>>>((float *)src->data, src->len, start, step,
                                                     BLOCK_SIZE, thread_num);
        break;
    case TL_INT32:
        rearange_kernel<<<block_num, BLOCK_SIZE>>>((int32_t *)src->data, src->len, start, step,
                                                     BLOCK_SIZE, thread_num);
        break;
    case TL_INT16:
        rearange_kernel<<<block_num, BLOCK_SIZE>>>((int16_t *)src->data, src->len, start, step,
                                                     BLOCK_SIZE, thread_num);
        break;
    case TL_INT8:
        rearange_kernel<<<block_num, BLOCK_SIZE>>>((int8_t *)src->data, src->len, start, step,
                                                     BLOCK_SIZE, thread_num);
        break;
    case TL_UINT32:
        rearange_kernel<<<block_num, BLOCK_SIZE>>>((uint32_t *)src->data, src->len, start, step,
                                                     BLOCK_SIZE, thread_num);
        break;
    case TL_UINT16:
        rearange_kernel<<<block_num, BLOCK_SIZE>>>((uint16_t *)src->data, src->len, start, step,
                                                     BLOCK_SIZE, thread_num);
        break;
    case TL_UINT8:
        rearange_kernel<<<block_num, BLOCK_SIZE>>>((uint8_t *)src->data, src->len, start, step,
                                                     BLOCK_SIZE, thread_num);
        break;
    case TL_BOOL:
        rearange_kernel<<<block_num, BLOCK_SIZE>>>((int *)src->data, src->len, start, step,
                                                     BLOCK_SIZE, thread_num);
        break;
    default:
        assert(0 && "unsupported tl_dtype");
        break;
    }
    tl_cuda_device_sync();
}

TL_EXPORT void tl_tensor_fprint_cuda(FILE *stream, const tl_tensor *t, const char *fmt)
{
    tl_tensor *t_host;

    t_host = tl_tensor_clone_d2h(t);
    tl_tensor_fprint(stream, t_host, fmt);
    tl_tensor_free_data_too(t_host);
}

TL_EXPORT void tl_tensor_print_cuda(const tl_tensor *t, const char *fmt)
{
    tl_tensor_fprint_cuda(stdout, t, fmt);
}

TL_EXPORT int tl_tensor_save_cuda(const char *file_name, const tl_tensor *t, const char *fmt)
{
    tl_tensor *t_host;
    int ret;

    t_host = tl_tensor_clone_d2h(t);
    ret = tl_tensor_save(file_name, t_host, fmt);
    tl_tensor_free_data_too(t_host);
    return ret;
}
