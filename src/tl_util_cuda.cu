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

#include <errno.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

#include "tl_util.h"

int tl_is_device_mem(const void *ptr)
{
     assert(ptr);
     cudaPointerAttributes attributes;
     TL_CUDA_CK(cudaPointerGetAttributes(&attributes, ptr));
     return attributes.memoryType == cudaMemoryTypeDevice;
}


void *tl_alloc_cuda(size_t size)
{
     void *p = NULL;

     assert(size > 0);
     TL_CUDA_CK(cudaMalloc(&p, size));
     assert(p);

     return p;
}

void *tl_clone_h2d(const void *src, size_t size)
{
     void *p;

     assert(src);
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

     assert(data && times > 0);
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
