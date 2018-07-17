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

#include <errno.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

#include "tl_util.h"

#define MAX_DEVICE_NUM 16

void tl_cuda_set_device(int n)
{
     TL_CUDA_CK(cudaSetDevice(n));
}

int tl_cuda_get_device()
{
     int n = 0;
     TL_CUDA_CK(cudaGetDevice(&n));
     return n;
}

int tl_is_device_mem(const void *ptr)
{
     assert(ptr);
     cudaPointerAttributes attributes;
     cudaError_t status;

     status = cudaPointerGetAttributes(&attributes, ptr);
     if (status == cudaErrorInvalidValue)
          return 0;
     TL_CUDA_CK(status);
     return attributes.memoryType == cudaMemoryTypeDevice;
}


void *tl_alloc_cuda(size_t size)
{
     void *p;

     assert(size > 0);
     TL_CUDA_CK(cudaMalloc(&p, size));
     assert(p);

     return p;
}

void tl_memcpy_h2d(void *dst, const void *src, size_t size)
{
     assert(!tl_is_device_mem(src));
     assert(tl_is_device_mem(dst));
     TL_CUDA_CK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

void tl_memcpy_d2h(void *dst, const void *src, size_t size)
{
     assert(tl_is_device_mem(src));
     assert(!tl_is_device_mem(dst));
     TL_CUDA_CK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

void tl_memcpy_d2d(void *dst, const void *src, size_t size)
{
     assert(tl_is_device_mem(src));
     assert(tl_is_device_mem(dst));
     TL_CUDA_CK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
}

void tl_free_cuda(void *p)
{
     assert(tl_is_device_mem(p));
     TL_CUDA_CK(cudaFree(p));
}

void *tl_clone_h2d(const void *src, size_t size)
{
     void *p;

     assert(!tl_is_device_mem(src));
     p = tl_alloc_cuda(size);
     TL_CUDA_CK(cudaMemcpy(p, src, size, cudaMemcpyHostToDevice));
     return p;
}

void *tl_clone_d2h(const void *src, size_t size)
{
     void *p;

     assert(tl_is_device_mem(src));
     p = tl_alloc(size);
     TL_CUDA_CK(cudaMemcpy(p, src, size, cudaMemcpyDeviceToHost));
     return p;
}

void *tl_clone_d2d(const void *src, size_t size)
{
     void *p;

     assert(tl_is_device_mem(src));
     p = tl_alloc_cuda(size);
     TL_CUDA_CK(cudaMemcpy(p, src, size, cudaMemcpyDeviceToDevice));
     return p;
}

void *tl_repeat_h2d(void *data, size_t size, int times)
{
     void *p, *dst;
     int i;

     assert(!tl_is_device_mem(data) && times > 0);
     dst = p = tl_alloc_cuda(size * times);
     for (i = 0; i < times; i++, p = (char *)p + size)
          TL_CUDA_CK(cudaMemcpy(p, data, size, cudaMemcpyHostToDevice));
     return dst;
}

void *tl_repeat_d2h(void *data, size_t size, int times)
{
     void *p, *dst;
     int i;

     assert(tl_is_device_mem(data) && times > 0);
     dst = p = tl_alloc(size * times);
     for (i = 0; i < times; i++, p = (char *)p + size)
          TL_CUDA_CK(cudaMemcpy(p, data, size, cudaMemcpyDeviceToHost));
     return dst;
}

void *tl_repeat_d2d(void *data, size_t size, int times)
{
     void *p, *dst;
     int i;

     assert(tl_is_device_mem(data) && times > 0);
     dst = p = tl_alloc_cuda(size * times);
     for (i = 0; i < times; i++, p = (char *)p + size)
          TL_CUDA_CK(cudaMemcpy(p, data, size, cudaMemcpyDeviceToDevice));
     return dst;
}

#ifdef TL_CUDNN

cudnnHandle_t tl_cudnn_handle()
{
     static int has_init[MAX_DEVICE_NUM] = {0};
     static cudnnHandle_t handles[MAX_DEVICE_NUM];
     int i = tl_cuda_get_device();

     if (i >= MAX_DEVICE_NUM) {
          tl_err_bt("ERROR: CUDA device number(%d) exceeds MAX_DEVICE_NUM(%d)",
                    i, MAX_DEVICE_NUM);
     }
     if (!has_init[i]) {
          cudnnCreate(&handles[i]);
          has_init[i] = 1;
     }
     return handles[i];
}

#endif  /* TL_CUDNN */

#endif  /* TL_CUDA */
