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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>
#include <math.h>

#include "tl_tensor.h"
#include "tl_util.h"

#define BLOCK_SIZE 512
#define BLOCK_NUM(bs, tn) (((tn) + (bs) - 1) / (bs))
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

static inline __device__ int get_index(int *ids, int ndim, int *dims)
{
     int i, id;
     for (i = 0, id = ids[0]; i < ndim-1; i++)
          id = dims[i+1] * id + ids[i+1];
     return id;
}

static inline __device__ void get_coords(int id, int *ids, int ndim, int *dims)
{
     for (int i = ndim-1; i >=0; i--) {
          ids[i] = id % dims[i];
          id /= dims[i];
     }
}

tl_tensor *tl_tensor_create_cuda(void *data, int ndim, const int *dims,
                                 tl_dtype dtype)
{
     tl_tensor *t;
     size_t size;

     t = (tl_tensor *)tl_alloc(sizeof(tl_tensor));
     t->len = tl_compute_length(ndim, dims);
     t->ndim = ndim;
     t->dims = (int *)tl_clone(dims, sizeof(int) * ndim);
     t->dtype = dtype;
     size = t->len * tl_size_of(dtype);
     if (!data) {
          t->data = tl_alloc_cuda(size);
          TL_CUDA_CK(cudaMemset(t->data, 0, size));
     } else {
          t->data = data;
     }

     return t;
}

void tl_tensor_free_data_too_cuda(tl_tensor *t)
{
     TL_CUDA_CK(cudaFree(t->data));
     tl_tensor_free(t);
}

tl_tensor *tl_tensor_clone_h2d(const tl_tensor *src)
{
     void *data;
     tl_tensor *dst;

     assert(src);
     data = tl_clone_h2d(src->data, src->len*tl_size_of(src->dtype));
     dst = tl_tensor_create_cuda(data, src->ndim, src->dims, src->dtype);
     return dst;
}

tl_tensor *tl_tensor_clone_d2h(const tl_tensor *src)
{
     void *data;
     tl_tensor *dst;

     assert(src);
     data = tl_clone_d2h(src->data, src->len*tl_size_of(src->dtype));
     dst = tl_tensor_create(data, src->ndim, src->dims, src->dtype);
     return dst;
}

tl_tensor *tl_tensor_clone_d2d(const tl_tensor *src)
{
     void *data;
     tl_tensor *dst;

     assert(src);
     data = tl_clone_d2d(src->data, src->len*tl_size_of(src->dtype));
     dst = tl_tensor_create_cuda(data, src->ndim, src->dims, src->dtype);
     return dst;
}

void tl_tensor_fprint_cuda(FILE *stream, const tl_tensor *t, const char *fmt)
{
     tl_tensor *t_host;

     t_host = tl_tensor_clone_d2h(t);
     tl_tensor_fprint(stream, t_host, fmt);
     tl_tensor_free_data_too(t_host);
}

void tl_tensor_print_cuda(const tl_tensor *t, const char *fmt)
{
     tl_tensor_fprint_cuda(stdout, t, fmt);
}

int tl_tensor_save_cuda(const char *file_name, const tl_tensor *t,
                        const char *fmt)
{
     tl_tensor *t_host;
     int ret;

     t_host = tl_tensor_clone_d2h(t);
     ret = tl_tensor_save(file_name, t_host, fmt);
     tl_tensor_free_data_too(t_host);
     return ret;
}

tl_tensor *tl_tensor_create_slice_cuda(const tl_tensor *src, int axis, int len,
                                       tl_dtype dtype)
{
     tl_tensor *dst;
     int *dims;

     assert(src);
     assert(axis < src->ndim && axis >= 0);
     assert(len <= src->dims[axis] && len > 0);

     dims = (int *)tl_clone(src->dims, sizeof(int) * src->ndim);
     dims[axis] = len;
     dst = tl_tensor_create_cuda(NULL, src->ndim, dims, dtype);
     tl_free(dims);

     return dst;
}

template <typename T>
__global__ void slice_kernel(T *src, T *dst, int start, int s_vol,
                             int d_vol, int vol, int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;
     int si = di / d_vol * s_vol + di % d_vol + start * vol;
     dst[di] = src[si];
}

tl_tensor *tl_tensor_slice_cuda(const tl_tensor *src, tl_tensor *dst, int axis,
                                int start, int len)
{
     int i;
     int d_vol, s_vol, vol;
     int thread_num, block_num;

     assert(src && tl_is_device_mem(src->data));
     assert(axis < src->ndim && axis >= 0);
     assert(len <= src->dims[axis] && len > 0);
     assert(start < src->dims[axis] && start >= 0);
     assert(len + start <= src->dims[axis]);
     if (dst) {
#ifndef NDEBUG
          assert(tl_is_device_mem(dst->data));
          assert(src->dtype == dst->dtype);
          assert(dst->ndim == src->ndim);
          for (i = 0; i < src->ndim; i++)
               assert(i == axis ? dst->dims[i] == len :
                      dst->dims[i] == src->dims[i]);
#endif
     } else {
          dst = tl_tensor_create_slice_cuda(src, axis, len, src->dtype);
     }

     for (i = axis+1, vol = 1; i < dst->ndim; i++)
          vol *= dst->dims[i];
     d_vol = vol * dst->dims[axis];
     s_vol = vol * src->dims[axis];
     thread_num = dst->len;
     block_num = BLOCK_NUM(BLOCK_SIZE, thread_num);

     switch (src->dtype) {
     case TL_DOUBLE:
          slice_kernel<double><<<block_num, BLOCK_SIZE>>>((double *)src->data,
                                                          (double *)dst->data,
                                                          start, s_vol, d_vol, vol,
                                                          BLOCK_SIZE, thread_num);
          break;
     case TL_FLOAT:
          slice_kernel<float><<<block_num, BLOCK_SIZE>>>((float *)src->data,
                                                         (float *)dst->data,
                                                         start, s_vol, d_vol, vol,
                                                         BLOCK_SIZE, thread_num);
          break;
     case TL_INT32:
          slice_kernel<int32_t><<<block_num, BLOCK_SIZE>>>((int32_t *)src->data,
                                                           (int32_t *)dst->data,
                                                           start, s_vol, d_vol, vol,
                                                           BLOCK_SIZE, thread_num);
          break;
     case TL_INT16:
          slice_kernel<int16_t><<<block_num, BLOCK_SIZE>>>((int16_t *)src->data,
                                                           (int16_t *)dst->data,
                                                           start, s_vol, d_vol, vol,
                                                           BLOCK_SIZE, thread_num);
          break;
     case TL_INT8:
          slice_kernel<int8_t><<<block_num, BLOCK_SIZE>>>((int8_t *)src->data,
                                                          (int8_t *)dst->data,
                                                          start, s_vol, d_vol, vol,
                                                          BLOCK_SIZE, thread_num);
          break;
     case TL_UINT32:
          slice_kernel<uint32_t><<<block_num, BLOCK_SIZE>>>((uint32_t *)src->data,
                                                            (uint32_t *)dst->data,
                                                            start, s_vol, d_vol, vol,
                                                            BLOCK_SIZE, thread_num);
          break;
     case TL_UINT16:
          slice_kernel<uint16_t><<<block_num, BLOCK_SIZE>>>((uint16_t *)src->data,
                                                            (uint16_t *)dst->data,
                                                            start, s_vol, d_vol, vol,
                                                            BLOCK_SIZE, thread_num);
          break;
     case TL_UINT8:
          slice_kernel<uint8_t><<<block_num, BLOCK_SIZE>>>((uint8_t *)src->data,
                                                           (uint8_t *)dst->data,
                                                           start, s_vol, d_vol, vol,
                                                           BLOCK_SIZE, thread_num);
          break;
     case TL_BOOL:
          slice_kernel<tl_bool_t><<<block_num, BLOCK_SIZE>>>((tl_bool_t *)src->data,
                                                             (tl_bool_t *)dst->data,
                                                             start, s_vol, d_vol, vol,
                                                             BLOCK_SIZE, thread_num);
          break;
     default:
          assert(0 && "unsupported tl_dtype");
          break;
     }
     TL_CUDA_CK(cudaDeviceSynchronize());
     return dst;
}

template <typename T>
__global__ void maxreduce_kernel(T *src, T *dst, int32_t *arg, int dim_size,
                                 int reduce_vol, int batch_vol,
                                 int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;

     /* src[si] is the first element in this thread to be compared, then
        si = batch_vol * batch + (di - reduce_vol * batch),
        where batch = di / reduce_vol,
        which is the same as the following code: */
     int si = (batch_vol - reduce_vol) * (di / reduce_vol) + di;
     T now = src[si], max = now;
     int maxi = 0;
     for (int i = 1; i < dim_size; i++) {
          now = src[si+i*reduce_vol];
          if (now > max) {
               max = now;
               maxi = i;
          }
     }
     dst[di] = max;
     if (arg)
          arg[di] = maxi;
}


tl_tensor *tl_tensor_maxreduce_cuda(const tl_tensor *src, tl_tensor *dst,
                                    tl_tensor *arg, int axis)
{
     /* suppose the shape of src is [N, C, H, W], dim = 1, then thread_num is N x H x W
        reduce_vol is H x W, index_vol is C x H x W */
     int thread_num, block_num, reduce_vol, index_vol;
     void *arg_data;
     int i;

     tl_check_dtype(src->dtype);
     assert(src && tl_is_device_mem(src->data));
     assert(axis < src->ndim && axis >= 0);
     if (dst) {
#ifndef NDEBUG
          assert(tl_is_device_mem(dst->data));
          assert(src->dtype == dst->dtype);
          for (i = 0; i < dst->ndim; i++)
               assert(i == axis ? dst->dims[i] == 1 :
                      dst->dims[i] == src->dims[i]);
#endif
     } else {
          dst = tl_tensor_create_slice_cuda(src, axis, 1, src->dtype);
     }
     if (arg) {
#ifndef NDEBUG
          assert(tl_is_device_mem(arg->data));
          assert(arg->dtype == TL_INT32);
          for (i = 0; i < arg->ndim; i++)
               assert(i == axis ? arg->dims[i] == 1 :
                      arg->dims[i] == src->dims[i]);
#endif
          arg_data = arg->data;
     } else {
          arg_data = NULL;
     }

     for (i = axis+1, thread_num = 1; i < dst->ndim; i++)
          thread_num *= dst->dims[i];
     reduce_vol = thread_num;
     index_vol = thread_num * src->dims[axis];
     for (i = 0; i < axis; i++)
          thread_num *= dst->dims[i];
     block_num = BLOCK_NUM(BLOCK_SIZE, thread_num);

     switch (src->dtype) {
     case TL_DOUBLE:
          maxreduce_kernel<double><<<block_num, BLOCK_SIZE>>>((double *)src->data,
                                                              (double *)dst->data,
                                                              (int32_t *)arg_data,
                                                              src->dims[axis],
                                                              reduce_vol,
                                                              index_vol,
                                                              BLOCK_SIZE,
                                                              thread_num);
          break;
     case TL_FLOAT:
          maxreduce_kernel<float><<<block_num, BLOCK_SIZE>>>((float *)src->data,
                                                             (float *)dst->data,
                                                             (int32_t *)arg_data,
                                                             src->dims[axis],
                                                             reduce_vol,
                                                             index_vol,
                                                             BLOCK_SIZE,
                                                             thread_num);
          break;
     case TL_INT32:
          maxreduce_kernel<int32_t><<<block_num, BLOCK_SIZE>>>((int32_t *)src->data,
                                                               (int32_t *)dst->data,
                                                               (int32_t *)arg_data,
                                                               src->dims[axis],
                                                               reduce_vol,
                                                               index_vol,
                                                               BLOCK_SIZE,
                                                               thread_num);
          break;
     case TL_INT16:
          maxreduce_kernel<int16_t><<<block_num, BLOCK_SIZE>>>((int16_t *)src->data,
                                                               (int16_t *)dst->data,
                                                               (int32_t *)arg_data,
                                                               src->dims[axis],
                                                               reduce_vol,
                                                               index_vol,
                                                               BLOCK_SIZE,
                                                               thread_num);
          break;
     case TL_INT8:
          maxreduce_kernel<int8_t><<<block_num, BLOCK_SIZE>>>((int8_t *)src->data,
                                                              (int8_t *)dst->data,
                                                              (int32_t *)arg_data,
                                                              src->dims[axis],
                                                              reduce_vol,
                                                              index_vol,
                                                              BLOCK_SIZE,
                                                              thread_num);
          break;
     case TL_UINT32:
          maxreduce_kernel<uint32_t><<<block_num, BLOCK_SIZE>>>((uint32_t *)src->data,
                                                                (uint32_t *)dst->data,
                                                                (int32_t *)arg_data,
                                                                src->dims[axis],
                                                                reduce_vol,
                                                                index_vol,
                                                                BLOCK_SIZE,
                                                                thread_num);
          break;
     case TL_UINT16:
          maxreduce_kernel<uint16_t><<<block_num, BLOCK_SIZE>>>((uint16_t *)src->data,
                                                                (uint16_t *)dst->data,
                                                                (int32_t *)arg_data,
                                                                src->dims[axis],
                                                                reduce_vol,
                                                                index_vol,
                                                                BLOCK_SIZE,
                                                                thread_num);
          break;
     case TL_UINT8:
          maxreduce_kernel<uint8_t><<<block_num, BLOCK_SIZE>>>((uint8_t *)src->data,
                                                               (uint8_t *)dst->data,
                                                               (int32_t *)arg_data,
                                                               src->dims[axis],
                                                               reduce_vol,
                                                               index_vol,
                                                               BLOCK_SIZE,
                                                               thread_num);
          break;
     case TL_BOOL:
          maxreduce_kernel<tl_bool_t><<<block_num, BLOCK_SIZE>>>((tl_bool_t *)src->data,
                                                                 (tl_bool_t *)dst->data,
                                                                 (int32_t *)arg_data,
                                                                 src->dims[axis],
                                                                 reduce_vol,
                                                                 index_vol,
                                                                 BLOCK_SIZE,
                                                                 thread_num);
          break;
     default:
          assert(0 && "unsupported tl_dtype");
          break;
     }
     TL_CUDA_CK(cudaDeviceSynchronize());
     return dst;
}

template <typename T>
__global__ void mul_kernel(T *src1, T *src2, T *dst, int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;
     dst[di] = src1[di] * src2[di];
}

__global__ void mul_bool_kernel(tl_bool_t *src1, tl_bool_t *src2,
                                tl_bool_t *dst, int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;
     int res = src1[di] * src2[di];
     if (res)
          dst[di] = TL_TRUE;
     else
          dst[di] = TL_FALSE;
}

template <typename T>
__global__ void div_kernel(T *src1, T *src2, T *dst, int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;
     // assert(src2[di] && "divided by zero");
     dst[di] = src1[di] / src2[di];
}

__global__ void div_bool_kernel(tl_bool_t *src1, tl_bool_t *src2,
                                tl_bool_t *dst, int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;
     int res = src1[di] / src2[di];
     if (res)
          dst[di] = TL_TRUE;
     else
          dst[di] = TL_FALSE;
}

template <typename T>
__global__ void sum_kernel(T *src1, T *src2, T *dst, int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;
     dst[di] = src1[di] + src2[di];
}

__global__ void sum_bool_kernel(tl_bool_t *src1, tl_bool_t *src2,
                                tl_bool_t *dst, int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;
     int res = src1[di] + src2[di];
     if (res)
          dst[di] = TL_TRUE;
     else
          dst[di] = TL_FALSE;
}

template <typename T>
__global__ void sub_kernel(T *src1, T *src2, T *dst, int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;
     dst[di] = src1[di] - src2[di];
}

__global__ void sub_bool_kernel(tl_bool_t *src1, tl_bool_t *src2,
                                tl_bool_t *dst, int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;
     int res = src1[di] - src2[di];
     if (res)
          dst[di] = TL_TRUE;
     else
          dst[di] = TL_FALSE;
}

template <typename T>
__global__ void max_kernel(T *src1, T *src2, T *dst, int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;
     dst[di] = max(src1[di], src2[di]);
}

template <typename T>
__global__ void min_kernel(T *src1, T *src2, T *dst, int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;
     dst[di] = min(src1[di], src2[di]);
}

template <typename T>
__global__ void pow_int_kernel(T *src1, T *src2, T *dst, T type_max, T type_min,
                               int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;

     float f1, f2, fd;

     f1 = (float)src1[di];
     f2 = (float)src2[di];
     fd = powf(f1, f2);
     if (fd >= type_max)
          dst[di] = type_max;
     else if (fd <= type_min)
          dst[di] = type_min;
     else
          dst[di] = (T)fd;
}

__global__ void pow_double_kernel(double *src1, double *src2, double *dst,
                                  int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;

     dst[di] = pow(src1[di], src2[di]);
}

__global__ void pow_float_kernel(float *src1, float *src2, float *dst,
                                 int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;

     dst[di] = powf(src1[di], src2[di]);
}

__global__ void pow_bool_kernel(tl_bool_t *src1, tl_bool_t *src2, tl_bool_t *dst,
                                int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;

     float f1, f2, fd;

     f1 = (float)src1[di];
     f2 = (float)src2[di];
     fd = powf(f1, f2);
     if (fd > 0 || fd < 0)
          dst[di] = TL_TRUE;
     else
          dst[di] = TL_FALSE;
}

tl_tensor *tl_tensor_elew_cuda(const tl_tensor *src1, const tl_tensor *src2,
                               tl_tensor *dst, tl_elew_op elew_op)
{
     assert(tl_tensor_issameshape(src1, src2));
     assert(tl_is_device_mem(src1->data) && tl_is_device_mem(src2->data));
     assert(src1->dtype == src2->dtype);
     if (dst) {
          assert(tl_is_device_mem(dst->data));
          assert(tl_tensor_issameshape(src1, dst));
          assert(src1->dtype == dst->dtype);
     } else {
          dst = tl_tensor_create_cuda(NULL, src1->ndim, src2->dims, src1->dtype);
     }

     int thread_num = dst->len;
     int block_num = BLOCK_NUM(BLOCK_SIZE, thread_num);

     switch (src1->dtype) {
     case TL_DOUBLE:
          switch (elew_op) {
          case TL_MUL:
               mul_kernel<double><<<block_num, BLOCK_SIZE>>>((double *)src1->data,
                                                             (double *)src2->data,
                                                             (double *)dst->data,
                                                             BLOCK_SIZE,
                                                             thread_num);
               break;
          case TL_DIV:
               div_kernel<double><<<block_num, BLOCK_SIZE>>>((double *)src1->data,
                                                             (double *)src2->data,
                                                             (double *)dst->data,
                                                             BLOCK_SIZE,
                                                             thread_num);
               break;
          case TL_SUM:
               sum_kernel<double><<<block_num, BLOCK_SIZE>>>((double *)src1->data,
                                                             (double *)src2->data,
                                                             (double *)dst->data,
                                                             BLOCK_SIZE,
                                                             thread_num);
               break;
          case TL_SUB:
               sub_kernel<double><<<block_num, BLOCK_SIZE>>>((double *)src1->data,
                                                             (double *)src2->data,
                                                             (double *)dst->data,
                                                             BLOCK_SIZE,
                                                             thread_num);
               break;
          case TL_MAX:
               max_kernel<double><<<block_num, BLOCK_SIZE>>>((double *)src1->data,
                                                             (double *)src2->data,
                                                             (double *)dst->data,
                                                             BLOCK_SIZE,
                                                             thread_num);
               break;
          case TL_MIN:
               min_kernel<double><<<block_num, BLOCK_SIZE>>>((double *)src1->data,
                                                             (double *)src2->data,
                                                             (double *)dst->data,
                                                             BLOCK_SIZE,
                                                             thread_num);
               break;
          case TL_POW:
               pow_double_kernel<<<block_num, BLOCK_SIZE>>>((double *)src1->data,
                                                            (double *)src2->data,
                                                            (double *)dst->data,
                                                            BLOCK_SIZE,
                                                            thread_num);
               break;
          default:
               assert(0 && "unsopported tl_elew_op");
               break;
          }
          break;
     case TL_FLOAT:
          switch (elew_op) {
          case TL_MUL:
               mul_kernel<float><<<block_num, BLOCK_SIZE>>>((float *)src1->data,
                                                            (float *)src2->data,
                                                            (float *)dst->data,
                                                            BLOCK_SIZE,
                                                            thread_num);
               break;
          case TL_DIV:
               div_kernel<float><<<block_num, BLOCK_SIZE>>>((float *)src1->data,
                                                            (float *)src2->data,
                                                            (float *)dst->data,
                                                            BLOCK_SIZE,
                                                            thread_num);
               break;
          case TL_SUM:
               sum_kernel<float><<<block_num, BLOCK_SIZE>>>((float *)src1->data,
                                                            (float *)src2->data,
                                                            (float *)dst->data,
                                                            BLOCK_SIZE,
                                                            thread_num);
               break;
          case TL_SUB:
               sub_kernel<float><<<block_num, BLOCK_SIZE>>>((float *)src1->data,
                                                            (float *)src2->data,
                                                            (float *)dst->data,
                                                            BLOCK_SIZE,
                                                            thread_num);
               break;
          case TL_MAX:
               max_kernel<float><<<block_num, BLOCK_SIZE>>>((float *)src1->data,
                                                            (float *)src2->data,
                                                            (float *)dst->data,
                                                            BLOCK_SIZE,
                                                            thread_num);
               break;
          case TL_MIN:
               min_kernel<float><<<block_num, BLOCK_SIZE>>>((float *)src1->data,
                                                            (float *)src2->data,
                                                            (float *)dst->data,
                                                            BLOCK_SIZE,
                                                            thread_num);
               break;
          case TL_POW:
               pow_float_kernel<<<block_num, BLOCK_SIZE>>>((float *)src1->data,
                                                           (float *)src2->data,
                                                           (float *)dst->data,
                                                           BLOCK_SIZE,
                                                           thread_num);
               break;
          default:
               assert(0 && "unsopported tl_elew_op");
               break;
          }
          break;
     case TL_INT32:
          switch (elew_op) {
          case TL_MUL:
               mul_kernel<int32_t><<<block_num, BLOCK_SIZE>>>((int32_t *)src1->data,
                                                              (int32_t *)src2->data,
                                                              (int32_t *)dst->data,
                                                              BLOCK_SIZE,
                                                              thread_num);
               break;
          case TL_DIV:
               div_kernel<int32_t><<<block_num, BLOCK_SIZE>>>((int32_t *)src1->data,
                                                              (int32_t *)src2->data,
                                                              (int32_t *)dst->data,
                                                              BLOCK_SIZE,
                                                              thread_num);
               break;
          case TL_SUM:
               sum_kernel<int32_t><<<block_num, BLOCK_SIZE>>>((int32_t *)src1->data,
                                                              (int32_t *)src2->data,
                                                              (int32_t *)dst->data,
                                                              BLOCK_SIZE,
                                                              thread_num);
               break;
          case TL_SUB:
               sub_kernel<int32_t><<<block_num, BLOCK_SIZE>>>((int32_t *)src1->data,
                                                              (int32_t *)src2->data,
                                                              (int32_t *)dst->data,
                                                              BLOCK_SIZE,
                                                              thread_num);
               break;
          case TL_MAX:
               max_kernel<int32_t><<<block_num, BLOCK_SIZE>>>((int32_t *)src1->data,
                                                              (int32_t *)src2->data,
                                                              (int32_t *)dst->data,
                                                              BLOCK_SIZE,
                                                              thread_num);
               break;
          case TL_MIN:
               min_kernel<int32_t><<<block_num, BLOCK_SIZE>>>((int32_t *)src1->data,
                                                              (int32_t *)src2->data,
                                                              (int32_t *)dst->data,
                                                              BLOCK_SIZE,
                                                              thread_num);
               break;
          case TL_POW:
               pow_int_kernel<int32_t><<<block_num, BLOCK_SIZE>>>((int32_t *)src1->data,
                                                                  (int32_t *)src2->data,
                                                                  (int32_t *)dst->data,
                                                                  INT32_MAX,
                                                                  INT32_MIN,
                                                                  BLOCK_SIZE,
                                                                  thread_num);
               break;
          default:
               assert(0 && "unsopported tl_elew_op");
               break;
          }
          break;
     case TL_INT16:
          switch (elew_op) {
          case TL_MUL:
               mul_kernel<int16_t><<<block_num, BLOCK_SIZE>>>((int16_t *)src1->data,
                                                              (int16_t *)src2->data,
                                                              (int16_t *)dst->data,
                                                              BLOCK_SIZE,
                                                              thread_num);
               break;
          case TL_DIV:
               div_kernel<int16_t><<<block_num, BLOCK_SIZE>>>((int16_t *)src1->data,
                                                              (int16_t *)src2->data,
                                                              (int16_t *)dst->data,
                                                              BLOCK_SIZE,
                                                              thread_num);
               break;
          case TL_SUM:
               sum_kernel<int16_t><<<block_num, BLOCK_SIZE>>>((int16_t *)src1->data,
                                                              (int16_t *)src2->data,
                                                              (int16_t *)dst->data,
                                                              BLOCK_SIZE,
                                                              thread_num);
               break;
          case TL_SUB:
               sub_kernel<int16_t><<<block_num, BLOCK_SIZE>>>((int16_t *)src1->data,
                                                              (int16_t *)src2->data,
                                                              (int16_t *)dst->data,
                                                              BLOCK_SIZE,
                                                              thread_num);
               break;
          case TL_MAX:
               max_kernel<int16_t><<<block_num, BLOCK_SIZE>>>((int16_t *)src1->data,
                                                              (int16_t *)src2->data,
                                                              (int16_t *)dst->data,
                                                              BLOCK_SIZE,
                                                              thread_num);
               break;
          case TL_MIN:
               min_kernel<int16_t><<<block_num, BLOCK_SIZE>>>((int16_t *)src1->data,
                                                              (int16_t *)src2->data,
                                                              (int16_t *)dst->data,
                                                              BLOCK_SIZE,
                                                              thread_num);
               break;
          case TL_POW:
               pow_int_kernel<int16_t><<<block_num, BLOCK_SIZE>>>((int16_t *)src1->data,
                                                                  (int16_t *)src2->data,
                                                                  (int16_t *)dst->data,
                                                                  INT16_MAX,
                                                                  INT16_MIN,
                                                                  BLOCK_SIZE,
                                                                  thread_num);
               break;
          default:
               assert(0 && "unsopported tl_elew_op");
               break;
          }
          break;
     case TL_INT8:
          switch (elew_op) {
          case TL_MUL:
               mul_kernel<int8_t><<<block_num, BLOCK_SIZE>>>((int8_t *)src1->data,
                                                             (int8_t *)src2->data,
                                                             (int8_t *)dst->data,
                                                             BLOCK_SIZE,
                                                             thread_num);
               break;
          case TL_DIV:
               div_kernel<int8_t><<<block_num, BLOCK_SIZE>>>((int8_t *)src1->data,
                                                             (int8_t *)src2->data,
                                                             (int8_t *)dst->data,
                                                             BLOCK_SIZE,
                                                             thread_num);
               break;
          case TL_SUM:
               sum_kernel<int8_t><<<block_num, BLOCK_SIZE>>>((int8_t *)src1->data,
                                                             (int8_t *)src2->data,
                                                             (int8_t *)dst->data,
                                                             BLOCK_SIZE,
                                                             thread_num);
               break;
          case TL_SUB:
               sub_kernel<int8_t><<<block_num, BLOCK_SIZE>>>((int8_t *)src1->data,
                                                             (int8_t *)src2->data,
                                                             (int8_t *)dst->data,
                                                             BLOCK_SIZE,
                                                             thread_num);
               break;
          case TL_MAX:
               max_kernel<int8_t><<<block_num, BLOCK_SIZE>>>((int8_t *)src1->data,
                                                             (int8_t *)src2->data,
                                                             (int8_t *)dst->data,
                                                             BLOCK_SIZE,
                                                             thread_num);
               break;
          case TL_MIN:
               min_kernel<int8_t><<<block_num, BLOCK_SIZE>>>((int8_t *)src1->data,
                                                             (int8_t *)src2->data,
                                                             (int8_t *)dst->data,
                                                             BLOCK_SIZE,
                                                             thread_num);
               break;
          case TL_POW:
               pow_int_kernel<int8_t><<<block_num, BLOCK_SIZE>>>((int8_t *)src1->data,
                                                                 (int8_t *)src2->data,
                                                                 (int8_t *)dst->data,
                                                                 INT8_MAX,
                                                                 INT8_MIN,
                                                                 BLOCK_SIZE,
                                                                 thread_num);
               break;
          default:
               assert(0 && "unsopported tl_elew_op");
               break;
          }
          break;
     case TL_UINT32:
          switch (elew_op) {
          case TL_MUL:
               mul_kernel<uint32_t><<<block_num, BLOCK_SIZE>>>((uint32_t *)src1->data,
                                                               (uint32_t *)src2->data,
                                                               (uint32_t *)dst->data,
                                                               BLOCK_SIZE,
                                                               thread_num);
               break;
          case TL_DIV:
               div_kernel<uint32_t><<<block_num, BLOCK_SIZE>>>((uint32_t *)src1->data,
                                                               (uint32_t *)src2->data,
                                                               (uint32_t *)dst->data,
                                                               BLOCK_SIZE,
                                                               thread_num);
               break;
          case TL_SUM:
               sum_kernel<uint32_t><<<block_num, BLOCK_SIZE>>>((uint32_t *)src1->data,
                                                               (uint32_t *)src2->data,
                                                               (uint32_t *)dst->data,
                                                               BLOCK_SIZE,
                                                               thread_num);
               break;
          case TL_SUB:
               sub_kernel<uint32_t><<<block_num, BLOCK_SIZE>>>((uint32_t *)src1->data,
                                                               (uint32_t *)src2->data,
                                                               (uint32_t *)dst->data,
                                                               BLOCK_SIZE,
                                                               thread_num);
               break;
          case TL_MAX:
               max_kernel<uint32_t><<<block_num, BLOCK_SIZE>>>((uint32_t *)src1->data,
                                                               (uint32_t *)src2->data,
                                                               (uint32_t *)dst->data,
                                                               BLOCK_SIZE,
                                                               thread_num);
               break;
          case TL_MIN:
               min_kernel<uint32_t><<<block_num, BLOCK_SIZE>>>((uint32_t *)src1->data,
                                                               (uint32_t *)src2->data,
                                                               (uint32_t *)dst->data,
                                                               BLOCK_SIZE,
                                                               thread_num);
               break;
          case TL_POW:
               pow_int_kernel<uint32_t><<<block_num, BLOCK_SIZE>>>((uint32_t *)src1->data,
                                                                   (uint32_t *)src2->data,
                                                                   (uint32_t *)dst->data,
                                                                   UINT32_MAX,
                                                                   0,
                                                                   BLOCK_SIZE,
                                                                   thread_num);
               break;
          default:
               assert(0 && "unsopported tl_elew_op");
               break;
          }
          break;
     case TL_UINT16:
          switch (elew_op) {
          case TL_MUL:
               mul_kernel<uint16_t><<<block_num, BLOCK_SIZE>>>((uint16_t *)src1->data,
                                                               (uint16_t *)src2->data,
                                                               (uint16_t *)dst->data,
                                                               BLOCK_SIZE,
                                                               thread_num);
               break;
          case TL_DIV:
               div_kernel<uint16_t><<<block_num, BLOCK_SIZE>>>((uint16_t *)src1->data,
                                                               (uint16_t *)src2->data,
                                                               (uint16_t *)dst->data,
                                                               BLOCK_SIZE,
                                                               thread_num);
               break;
          case TL_SUM:
               sum_kernel<uint16_t><<<block_num, BLOCK_SIZE>>>((uint16_t *)src1->data,
                                                               (uint16_t *)src2->data,
                                                               (uint16_t *)dst->data,
                                                               BLOCK_SIZE,
                                                               thread_num);
               break;
          case TL_SUB:
               sub_kernel<uint16_t><<<block_num, BLOCK_SIZE>>>((uint16_t *)src1->data,
                                                               (uint16_t *)src2->data,
                                                               (uint16_t *)dst->data,
                                                               BLOCK_SIZE,
                                                               thread_num);
               break;
          case TL_MAX:
               max_kernel<uint16_t><<<block_num, BLOCK_SIZE>>>((uint16_t *)src1->data,
                                                               (uint16_t *)src2->data,
                                                               (uint16_t *)dst->data,
                                                               BLOCK_SIZE,
                                                               thread_num);
               break;
          case TL_MIN:
               min_kernel<uint16_t><<<block_num, BLOCK_SIZE>>>((uint16_t *)src1->data,
                                                               (uint16_t *)src2->data,
                                                               (uint16_t *)dst->data,
                                                               BLOCK_SIZE,
                                                               thread_num);
               break;
          case TL_POW:
               pow_int_kernel<uint16_t><<<block_num, BLOCK_SIZE>>>((uint16_t *)src1->data,
                                                                   (uint16_t *)src2->data,
                                                                   (uint16_t *)dst->data,
                                                                   UINT16_MAX,
                                                                   0,
                                                                   BLOCK_SIZE,
                                                                   thread_num);
               break;
          default:
               assert(0 && "unsopported tl_elew_op");
               break;
          }
          break;
     case TL_UINT8:
          switch (elew_op) {
          case TL_MUL:
               mul_kernel<uint8_t><<<block_num, BLOCK_SIZE>>>((uint8_t *)src1->data,
                                                              (uint8_t *)src2->data,
                                                              (uint8_t *)dst->data,
                                                              BLOCK_SIZE,
                                                              thread_num);
               break;
          case TL_DIV:
               div_kernel<uint8_t><<<block_num, BLOCK_SIZE>>>((uint8_t *)src1->data,
                                                              (uint8_t *)src2->data,
                                                              (uint8_t *)dst->data,
                                                              BLOCK_SIZE,
                                                              thread_num);
               break;
          case TL_SUM:
               sum_kernel<uint8_t><<<block_num, BLOCK_SIZE>>>((uint8_t *)src1->data,
                                                              (uint8_t *)src2->data,
                                                              (uint8_t *)dst->data,
                                                              BLOCK_SIZE,
                                                              thread_num);
               break;
          case TL_SUB:
               sub_kernel<uint8_t><<<block_num, BLOCK_SIZE>>>((uint8_t *)src1->data,
                                                              (uint8_t *)src2->data,
                                                              (uint8_t *)dst->data,
                                                              BLOCK_SIZE,
                                                              thread_num);
               break;
          case TL_MAX:
               max_kernel<uint8_t><<<block_num, BLOCK_SIZE>>>((uint8_t *)src1->data,
                                                              (uint8_t *)src2->data,
                                                              (uint8_t *)dst->data,
                                                              BLOCK_SIZE,
                                                              thread_num);
               break;
          case TL_MIN:
               min_kernel<uint8_t><<<block_num, BLOCK_SIZE>>>((uint8_t *)src1->data,
                                                              (uint8_t *)src2->data,
                                                              (uint8_t *)dst->data,
                                                              BLOCK_SIZE,
                                                              thread_num);
               break;
          case TL_POW:
               pow_int_kernel<uint8_t><<<block_num, BLOCK_SIZE>>>((uint8_t *)src1->data,
                                                                  (uint8_t *)src2->data,
                                                                  (uint8_t *)dst->data,
                                                                  UINT8_MAX,
                                                                  0,
                                                                  BLOCK_SIZE,
                                                                  thread_num);
               break;
          default:
               assert(0 && "unsopported tl_elew_op");
               break;
          }
          break;
     case TL_BOOL:
          switch (elew_op) {
          case TL_MUL:
               mul_bool_kernel<<<block_num, BLOCK_SIZE>>>((tl_bool_t *)src1->data,
                                                          (tl_bool_t *)src2->data,
                                                          (tl_bool_t *)dst->data,
                                                          BLOCK_SIZE,
                                                          thread_num);
               break;
          case TL_DIV:
               div_bool_kernel<<<block_num, BLOCK_SIZE>>>((tl_bool_t *)src1->data,
                                                          (tl_bool_t *)src2->data,
                                                          (tl_bool_t *)dst->data,
                                                          BLOCK_SIZE,
                                                          thread_num);
               break;
          case TL_SUM:
               sum_bool_kernel<<<block_num, BLOCK_SIZE>>>((tl_bool_t *)src1->data,
                                                          (tl_bool_t *)src2->data,
                                                          (tl_bool_t *)dst->data,
                                                          BLOCK_SIZE,
                                                          thread_num);
               break;
          case TL_SUB:
               sub_bool_kernel<<<block_num, BLOCK_SIZE>>>((tl_bool_t *)src1->data,
                                                          (tl_bool_t *)src2->data,
                                                          (tl_bool_t *)dst->data,
                                                          BLOCK_SIZE,
                                                          thread_num);
               break;
          case TL_MAX:
               max_kernel<tl_bool_t><<<block_num, BLOCK_SIZE>>>((tl_bool_t *)src1->data,
                                                                (tl_bool_t *)src2->data,
                                                                (tl_bool_t *)dst->data,
                                                                BLOCK_SIZE,
                                                                thread_num);
               break;
          case TL_MIN:
               min_kernel<tl_bool_t><<<block_num, BLOCK_SIZE>>>((tl_bool_t *)src1->data,
                                                                (tl_bool_t *)src2->data,
                                                                (tl_bool_t *)dst->data,
                                                                BLOCK_SIZE,
                                                                thread_num);
               break;
          case TL_POW:
               pow_bool_kernel<<<block_num, BLOCK_SIZE>>>((tl_bool_t *)src1->data,
                                                          (tl_bool_t *)src2->data,
                                                          (tl_bool_t *)dst->data,
                                                          BLOCK_SIZE,
                                                          thread_num);
               break;
          default:
               assert(0 && "unsopported tl_elew_op");
               break;
          }
          break;
     default:
          assert(0 && "unsupported tl_dtype");
          break;
     }
     TL_CUDA_CK(cudaDeviceSynchronize());

     return dst;
}

__global__ void convert_kernel(void *src, void *dst,
                               tl_dtype dtype_s, tl_dtype dtype_d,
                               int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;

     double val_d;
     float val_f;
     int32_t val_i32;
     uint32_t val_u32;
     int16_t val_i16;
     uint16_t val_u16;
     int8_t val_i8;
     uint8_t val_u8;

     switch (dtype_d) {
     case TL_DOUBLE:
          switch (dtype_s) {
          case TL_DOUBLE:
               ((double *)dst)[di] = ((double *)src)[di];
               break;
          case TL_FLOAT:
               ((double *)dst)[di] = (double)((float *)src)[di];
               break;
          case TL_INT32:
               ((double *)dst)[di] = (double)((int32_t *)src)[di];
               break;
          case TL_INT16:
               ((double *)dst)[di] = (double)((int16_t *)src)[di];
               break;
          case TL_INT8:
               ((double *)dst)[di] = (double)((int8_t *)src)[di];
               break;
          case TL_UINT32:
               ((double *)dst)[di] = (double)((uint32_t *)src)[di];
               break;
          case TL_UINT16:
               ((double *)dst)[di] = (double)((uint16_t *)src)[di];
               break;
          case TL_UINT8:
               ((double *)dst)[di] = (double)((uint8_t *)src)[di];
               break;
          case TL_BOOL:
               ((double *)dst)[di] = (double)((tl_bool_t *)src)[di];
               break;
          default:
               assert(0 && "unsupported tl_dtype");
               break;
          }
          break;
     case TL_FLOAT:
          switch (dtype_s) {
          case TL_DOUBLE:
               val_d = ((double *)src)[di];
               if (val_d >= FLT_MAX)
                    ((float *)dst)[di] = FLT_MAX;
               else if (val_d <= -FLT_MAX)
                    ((float *)dst)[di] = -FLT_MAX;
               else
                    ((float *)dst)[di] = (float)val_d;
               break;
          case TL_FLOAT:
               ((float *)dst)[di] = ((float *)src)[di];
               break;
          case TL_INT32:
               ((float *)dst)[di] = (float)((int32_t *)src)[di];
               break;
          case TL_INT16:
               ((float *)dst)[di] = (float)((int16_t *)src)[di];
               break;
          case TL_INT8:
               ((float *)dst)[di] = (float)((int8_t *)src)[di];
               break;
          case TL_UINT32:
               ((float *)dst)[di] = (float)((uint32_t *)src)[di];
               break;
          case TL_UINT16:
               ((float *)dst)[di] = (float)((uint16_t *)src)[di];
               break;
          case TL_UINT8:
               ((float *)dst)[di] = (float)((uint8_t *)src)[di];
               break;
          case TL_BOOL:
               ((float *)dst)[di] = (float)((tl_bool_t *)src)[di];
               break;
          default:
               assert(0 && "unsupported tl_dtype");
               break;
          }
          break;
     case TL_INT32:
          switch (dtype_s) {
          case TL_DOUBLE:
               val_d = ((double *)src)[di];
               if (val_d >= INT32_MAX)
                    ((int32_t *)dst)[di] = INT32_MAX;
               else if (val_d <= INT32_MIN)
                    ((int32_t *)dst)[di] = INT32_MIN;
               else
                    ((int32_t *)dst)[di] = (int32_t)val_d;
               break;
          case TL_FLOAT:
               val_f = ((float *)src)[di];
               if (val_f >= INT32_MAX)
                    ((int32_t *)dst)[di] = INT32_MAX;
               else if (val_f <= INT32_MIN)
                    ((int32_t *)dst)[di] = INT32_MIN;
               else
                    ((int32_t *)dst)[di] = (int32_t)val_f;
               break;
          case TL_INT32:
               ((int32_t *)dst)[di] = ((int32_t *)src)[di];
               break;
          case TL_INT16:
               ((int32_t *)dst)[di] = (int32_t)((int16_t *)src)[di];
               break;
          case TL_INT8:
               ((int32_t *)dst)[di] = (int32_t)((int8_t *)src)[di];
               break;
          case TL_UINT32:
               val_u32 = ((uint32_t *)src)[di];
               if (val_u32 >= INT32_MAX)
                    ((int32_t *)dst)[di] = INT32_MAX;
               else
                    ((int32_t *)dst)[di] = (int32_t)val_u32;
               break;
          case TL_UINT16:
               ((int32_t *)dst)[di] = (int32_t)((uint16_t *)src)[di];
               break;
          case TL_UINT8:
               ((int32_t *)dst)[di] = (int32_t)((uint8_t *)src)[di];
               break;
          case TL_BOOL:
               ((int32_t *)dst)[di] = (int32_t)((tl_bool_t *)src)[di];
               break;
          default:
               assert(0 && "unsupported tl_dtype");
               break;
          }
          break;
     case TL_INT16:
          switch (dtype_s) {
          case TL_DOUBLE:
               val_d = ((double *)src)[di];
               if (val_d >= INT16_MAX)
                    ((int16_t *)dst)[di] = INT16_MAX;
               else if (val_d <= INT16_MIN)
                    ((int16_t *)dst)[di] = INT16_MIN;
               else
                    ((int16_t *)dst)[di] = (int16_t)val_d;
               break;
          case TL_FLOAT:
               val_f = ((float *)src)[di];
               if (val_f >= INT16_MAX)
                    ((int16_t *)dst)[di] = INT16_MAX;
               else if (val_f <= INT16_MIN)
                    ((int16_t *)dst)[di] = INT16_MIN;
               else
                    ((int16_t *)dst)[di] = (int16_t)val_f;
               break;
          case TL_INT32:
               val_i32 = ((int32_t *)src)[di];
               if (val_i32 >= INT16_MAX)
                    ((int16_t *)dst)[di] = INT16_MAX;
               else if (val_i32 <= INT16_MIN)
                    ((int16_t *)dst)[di] = INT16_MIN;
               else
                    ((int16_t *)dst)[di] = (int16_t)val_i32;
               break;
          case TL_INT16:
               ((int16_t *)dst)[di] = ((int16_t *)src)[di];
               break;
          case TL_INT8:
               ((int16_t *)dst)[di] = (int16_t)((int8_t *)src)[di];
               break;
          case TL_UINT32:
               val_u32 = ((uint32_t *)src)[di];
               if (val_u32 >= INT16_MAX)
                    ((int16_t *)dst)[di] = INT16_MAX;
               else
                    ((int16_t *)dst)[di] = (int16_t)val_u32;
               break;
          case TL_UINT16:
               val_u16 = ((uint16_t *)src)[di];
               if (val_u16 >= INT16_MAX)
                    ((int16_t *)dst)[di] = INT16_MAX;
               else
                    ((int16_t *)dst)[di] = (int16_t)val_u16;
               break;
          case TL_UINT8:
               ((int16_t *)dst)[di] = (int16_t)((uint8_t *)src)[di];
               break;
          case TL_BOOL:
               ((int16_t *)dst)[di] = (int16_t)((tl_bool_t *)src)[di];
               break;
          default:
               assert(0 && "unsupported tl_dtype");
               break;
          }
          break;
     case TL_INT8:
          switch (dtype_s) {
          case TL_DOUBLE:
               val_d = ((double *)src)[di];
               if (val_d >= INT8_MAX)
                    ((int8_t *)dst)[di] = INT8_MAX;
               else if (val_d <= INT8_MIN)
                    ((int8_t *)dst)[di] = INT8_MIN;
               else
                    ((int8_t *)dst)[di] = (int8_t)val_d;
               break;
          case TL_FLOAT:
               val_f = ((float *)src)[di];
               if (val_f >= INT8_MAX)
                    ((int8_t *)dst)[di] = INT8_MAX;
               else if (val_f <= INT8_MIN)
                    ((int8_t *)dst)[di] = INT8_MIN;
               else
                    ((int8_t *)dst)[di] = (int8_t)val_f;
               break;
          case TL_INT32:
               val_i32 = ((int32_t *)src)[di];
               if (val_i32 >= INT8_MAX)
                    ((int8_t *)dst)[di] = INT8_MAX;
               else if (val_i32 <= INT8_MIN)
                    ((int8_t *)dst)[di] = INT8_MIN;
               else
                    ((int8_t *)dst)[di] = (int8_t)val_i32;
               break;
          case TL_INT16:
               val_i16 = ((int16_t *)src)[di];
               if (val_i16 >= INT8_MAX)
                    ((int8_t *)dst)[di] = INT8_MAX;
               else if (val_i16 <= INT8_MIN)
                    ((int8_t *)dst)[di] = INT8_MIN;
               else
                    ((int8_t *)dst)[di] = (int8_t)val_i16;
               break;
          case TL_INT8:
               ((int8_t *)dst)[di] = ((int8_t *)src)[di];
               break;
          case TL_UINT32:
               val_u32 = ((uint32_t *)src)[di];
               if (val_u32 >= INT8_MAX)
                    ((int8_t *)dst)[di] = INT8_MAX;
               else
                    ((int8_t *)dst)[di] = (int8_t)val_u32;
               break;
          case TL_UINT16:
               val_u16 = ((uint16_t *)src)[di];
               if (val_u16 >= INT8_MAX)
                    ((int8_t *)dst)[di] = INT8_MAX;
               else
                    ((int8_t *)dst)[di] = (int8_t)val_u16;
               break;
          case TL_UINT8:
               val_u8 = ((uint8_t *)src)[di];
               if (val_u8 >= INT8_MAX)
                    ((int8_t *)dst)[di] = INT8_MAX;
               else
                    ((int8_t *)dst)[di] = (int8_t)val_u8;
               break;
          case TL_BOOL:
               ((int8_t *)dst)[di] = (int8_t)((tl_bool_t *)src)[di];
               break;
          default:
               assert(0 && "unsupported tl_dtype");
               break;
          }
          break;
     case TL_UINT32:
          switch (dtype_s) {
          case TL_DOUBLE:
               val_d = ((double *)src)[di];
               if (val_d >= UINT32_MAX)
                    ((uint32_t *)dst)[di] = UINT32_MAX;
               else if (val_d < 0)
                    ((uint32_t *)dst)[di] = 0;
               else
                    ((uint32_t *)dst)[di] = (uint32_t)val_d;
               break;
          case TL_FLOAT:
               val_f = ((float *)src)[di];
               if (val_f >= UINT32_MAX)
                    ((uint32_t *)dst)[di] = UINT32_MAX;
               else if (val_f < 0)
                    ((uint32_t *)dst)[di] = 0;
               else
                    ((uint32_t *)dst)[di] = (uint32_t)val_f;
               break;
          case TL_INT32:
               val_i32 = ((int32_t *)src)[di];
               if (val_i32 >= 0)
                    ((uint32_t *)dst)[di] = (uint32_t)val_i32;
               else
                    ((uint32_t *)dst)[di] = 0;
               break;
          case TL_INT16:
               val_i16 = ((int16_t *)src)[di];
               if (val_i16 >= 0)
                    ((uint32_t *)dst)[di] = (uint32_t)val_i16;
               else
                    ((uint32_t *)dst)[di] = 0;
               break;
          case TL_INT8:
               val_i8 = ((int8_t *)src)[di];
               if (val_i8 >= 0)
                    ((uint32_t *)dst)[di] = (uint32_t)val_i8;
               else
                    ((uint32_t *)dst)[di] = 0;
               break;
          case TL_UINT32:
               ((uint32_t *)dst)[di] = ((uint32_t *)src)[di];
               break;
          case TL_UINT16:
               ((uint32_t *)dst)[di] = (uint32_t)((uint16_t *)src)[di];
               break;
          case TL_UINT8:
               ((uint32_t *)dst)[di] = (uint32_t)((uint8_t *)src)[di];
               break;
          case TL_BOOL:
               ((uint32_t *)dst)[di] = (uint32_t)((tl_bool_t *)src)[di];
               break;
          default:
               assert(0 && "unsupported tl_dtype");
               break;
          }
          break;
     case TL_UINT16:
          switch (dtype_s) {
          case TL_DOUBLE:
               val_d = ((double *)src)[di];
               if (val_d >= UINT16_MAX)
                    ((uint16_t *)dst)[di] = UINT16_MAX;
               else if (val_d < 0)
                    ((uint16_t *)dst)[di] = 0;
               else
                    ((uint16_t *)dst)[di] = (uint16_t)val_d;
               break;
          case TL_FLOAT:
               val_f = ((float *)src)[di];
               if (val_f >= UINT16_MAX)
                    ((uint16_t *)dst)[di] = UINT16_MAX;
               else if (val_f < 0)
                    ((uint16_t *)dst)[di] = 0;
               else
                    ((uint16_t *)dst)[di] = (uint16_t)val_f;
               break;
          case TL_INT32:
               val_i32 = ((int32_t *)src)[di];
               if (val_i32 >= UINT16_MAX)
                    ((uint16_t *)dst)[di] = UINT16_MAX;
               else if (val_i32 < 0)
                    ((uint16_t *)dst)[di] = 0;
               else
                    ((uint16_t *)dst)[di] = (uint16_t)val_i32;
               break;
          case TL_INT16:
               val_i16 = ((int16_t *)src)[di];
               if (val_i16 >= 0)
                    ((uint16_t *)dst)[di] = (uint16_t)val_i16;
               else
                    ((uint16_t *)dst)[di] = 0;
               break;
          case TL_INT8:
               val_i8 = ((int8_t *)src)[di];
               if (val_i8 >= 0)
                    ((uint16_t *)dst)[di] = (uint16_t)val_i8;
               else
                    ((uint16_t *)dst)[di] = 0;
               break;
          case TL_UINT32:
               val_u32 = ((uint32_t *)src)[di];
               if (val_u32 >= UINT16_MAX)
                    ((uint16_t *)dst)[di] = UINT16_MAX;
               else
                    ((uint16_t *)dst)[di] = (uint16_t)val_u32;
               break;
          case TL_UINT16:
               ((uint16_t *)dst)[di] = ((uint16_t *)src)[di];
               break;
          case TL_UINT8:
               ((uint16_t *)dst)[di] = (uint16_t)((uint8_t *)src)[di];
               break;
          case TL_BOOL:
               ((uint16_t *)dst)[di] = (uint16_t)((tl_bool_t *)src)[di];
               break;
          default:
               assert(0 && "unsupported tl_dtype");
               break;
          }
          break;
     case TL_UINT8:
          switch (dtype_s) {
          case TL_DOUBLE:
               val_d = ((double *)src)[di];
               if (val_d >= UINT8_MAX)
                    ((uint8_t *)dst)[di] = UINT8_MAX;
               else if (val_d < 0)
                    ((uint8_t *)dst)[di] = 0;
               else
                    ((uint8_t *)dst)[di] = (uint8_t)val_d;
               break;
          case TL_FLOAT:
               val_f = ((float *)src)[di];
               if (val_f >= UINT8_MAX)
                    ((uint8_t *)dst)[di] = UINT8_MAX;
               else if (val_f < 0)
                    ((uint8_t *)dst)[di] = 0;
               else
                    ((uint8_t *)dst)[di] = (uint8_t)val_f;
               break;
          case TL_INT32:
               val_i32 = ((int32_t *)src)[di];
               if (val_i32 >= UINT8_MAX)
                    ((uint8_t *)dst)[di] = UINT8_MAX;
               else if (val_i32 < 0)
                    ((uint8_t *)dst)[di] = 0;
               else
                    ((uint8_t *)dst)[di] = (uint8_t)val_i32;
               break;
          case TL_INT16:
               val_i16 = ((int16_t *)src)[di];
               if (val_i16 >= UINT8_MAX)
                    ((uint8_t *)dst)[di] = UINT8_MAX;
               else if (val_i16 < 0)
                    ((uint8_t *)dst)[di] = 0;
               else
                    ((uint8_t *)dst)[di] = (uint8_t)val_i16;
               break;
          case TL_INT8:
               val_i8 = ((int8_t *)src)[di];
               if (val_i8 >= 0)
                    ((uint8_t *)dst)[di] = (uint8_t)val_i8;
               else
                    ((uint8_t *)dst)[di] = 0;
               break;
          case TL_UINT32:
               val_u32 = ((uint32_t *)src)[di];
               if (val_u32 >= UINT8_MAX)
                    ((uint8_t *)dst)[di] = UINT8_MAX;
               else
                    ((uint8_t *)dst)[di] = (uint8_t)val_u32;
               break;
          case TL_UINT16:
               val_u16 = ((uint16_t *)src)[di];
               if (val_u16 >= UINT8_MAX)
                    ((uint8_t *)dst)[di] = UINT8_MAX;
               else
                    ((uint8_t *)dst)[di] = (uint8_t)val_u16;
               break;
          case TL_UINT8:
               ((uint8_t *)dst)[di] = ((uint8_t *)src)[di];
               break;
          case TL_BOOL:
               ((uint8_t *)dst)[di] = (uint8_t)((tl_bool_t *)src)[di];
               break;
          default:
               assert(0 && "unsupported tl_dtype");
               break;
          }
          break;
     case TL_BOOL:
          switch (dtype_s) {
          case TL_DOUBLE:
               val_d = ((double *)src)[di];
               if (val_d > 0 || val_d < 0)
                    ((tl_bool_t *)dst)[di] = TL_TRUE;
               else
                    ((tl_bool_t *)dst)[di] = TL_FALSE;
               break;
          case TL_FLOAT:
               val_f = ((float *)src)[di];
               if (val_f > 0 || val_f < 0)
                    ((tl_bool_t *)dst)[di] = TL_TRUE;
               else
                    ((tl_bool_t *)dst)[di] = TL_FALSE;
               break;
          case TL_INT32:
               val_i32 = ((int32_t *)src)[di];
               if (val_i32)
                    ((tl_bool_t *)dst)[di] = TL_TRUE;
               else
                    ((tl_bool_t *)dst)[di] = TL_FALSE;
               break;
          case TL_INT16:
               val_i16 = ((int16_t *)src)[di];
               if (val_i16)
                    ((tl_bool_t *)dst)[di] = TL_TRUE;
               else
                    ((tl_bool_t *)dst)[di] = TL_FALSE;
               break;
          case TL_INT8:
               val_i8 = ((int8_t *)src)[di];
               if (val_i8)
                    ((tl_bool_t *)dst)[di] = TL_TRUE;
               else
                    ((tl_bool_t *)dst)[di] = TL_FALSE;
               break;
          case TL_UINT32:
               val_u32 = ((uint32_t *)src)[di];
               if (val_u32)
                    ((tl_bool_t *)dst)[di] = TL_TRUE;
               else
                    ((tl_bool_t *)dst)[di] = TL_FALSE;
               break;
          case TL_UINT16:
               val_u16 = ((uint16_t *)src)[di];
               if (val_u16)
                    ((tl_bool_t *)dst)[di] = TL_TRUE;
               else
                    ((tl_bool_t *)dst)[di] = TL_FALSE;
               break;
          case TL_UINT8:
               val_u8 = ((uint8_t *)src)[di];
               if (val_u8)
                    ((tl_bool_t *)dst)[di] = TL_TRUE;
               else
                    ((tl_bool_t *)dst)[di] = TL_FALSE;
               break;
          case TL_BOOL:
               ((tl_bool_t *)dst)[di] = ((tl_bool_t *)src)[di];
               break;
          default:
               assert(0 && "unsupported tl_dtype");
               break;
          }
          break;
     default:
          assert(0 && "unsupported tl_dtype");
          break;
     }
}

tl_tensor *tl_tensor_convert_cuda(const tl_tensor *src, tl_tensor *dst,
                                  tl_dtype dtype_d)
{
     tl_dtype dtype_s;
     int thread_num, block_num;

     assert(src && tl_is_device_mem(src->data));
     if (dst) {
          assert(tl_is_device_mem(dst->data));
          assert(tl_tensor_issameshape(src, dst));
          assert(dst->dtype == dtype_d);
     } else {
          dst = tl_tensor_create_cuda(NULL, src->ndim, src->dims, dtype_d);
     }

     dtype_s = src->dtype;
     thread_num = dst->len;
     block_num = BLOCK_NUM(BLOCK_SIZE, thread_num);
     convert_kernel<<<block_num, BLOCK_SIZE>>>(src->data, dst->data,
                                               dtype_s, dtype_d,
                                               BLOCK_SIZE, thread_num);
     TL_CUDA_CK(cudaDeviceSynchronize());

     return dst;
}

template <typename T>
__global__ void transpose_kernel(T *src, T *dst, int ndim,
                                 int *s_dims, int *d_dims,
                                 int *s_ids, int *d_ids,
                                 int *axes, int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;

     int *t_s_ids = s_ids + di * ndim;
     int *t_d_ids = d_ids + di * ndim;
     get_coords(di, t_d_ids, ndim, d_dims);
     for (int i = 0; i < ndim; i++)
          t_s_ids[axes[i]] = t_d_ids[i];
     int si = get_index(t_s_ids, ndim, s_dims);

     dst[di] = src[si];
}

tl_tensor *tl_tensor_transpose_cuda(const tl_tensor *src, tl_tensor *dst,
                                    const int *axes, tl_tensor *workspace)
{
     int *s_ids, *d_ids, *s_dims, *d_dims, *axes_device;
     int thread_num, block_num;
     int i;

#ifndef NDEBUG
     int *tmp = (int *)tl_alloc(src->ndim * sizeof(int));
     memset(tmp, 0, src->ndim * sizeof(int));
     for (i = 0; i < src->ndim; i++)
          tmp[axes[i]] = 1;
     for (i = 0; i < src->ndim; i++)
          assert(tmp[i] && "axes don't match src tensor's shape");
     tl_free(tmp);
     assert(src && tl_is_device_mem(src->data));
#endif
     if (dst) {
#ifndef NDEBUG
          assert(tl_is_device_mem(dst->data));
          assert(src->dtype == dst->dtype);
          assert(src->len == dst->len);
          assert(src->ndim == dst->ndim);
          for (i = 0; i < dst->ndim; i++)
               assert(src->dims[axes[i]] = dst->dims[i]);
#endif
     } else {
          d_dims = (int *)tl_alloc(src->ndim * sizeof(int));
          for (i = 0; i < src->ndim; i++)
               d_dims[i] = src->dims[axes[i]];
          dst = tl_tensor_create_cuda(NULL, src->ndim, d_dims, src->dtype);
          tl_free(d_dims);
     }

     thread_num = dst->len;
     block_num = BLOCK_NUM(BLOCK_SIZE, thread_num);
     s_dims = (int *)tl_clone_h2d(src->dims, sizeof(int) * src->ndim);
     d_dims = (int *)tl_clone_h2d(dst->dims, sizeof(int) * dst->ndim);
     axes_device = (int *)tl_clone_h2d(axes, sizeof(int) * src->ndim);
     if (!workspace) {
          s_ids = (int *)tl_alloc_cuda(sizeof(int) * dst->ndim * thread_num);
          d_ids = (int *)tl_alloc_cuda(sizeof(int) * dst->ndim * thread_num);
     } else {
          assert(tl_is_device_mem(workspace->data));
          assert(workspace->dtype == TL_INT32);
          assert(workspace->len == dst->ndim * dst->len * 2);
          s_ids = (int32_t *)workspace->data;
          d_ids = &((int32_t *)workspace->data)[workspace->len / 2];
     }

     switch (src->dtype) {
     case TL_DOUBLE:
          transpose_kernel<double><<<block_num, BLOCK_SIZE>>>((double *)src->data,
                                                              (double *)dst->data,
                                                              dst->ndim,
                                                              s_dims, d_dims,
                                                              s_ids, d_ids,
                                                              axes_device,
                                                              BLOCK_SIZE,
                                                              thread_num);
          break;
     case TL_FLOAT:
          transpose_kernel<float><<<block_num, BLOCK_SIZE>>>((float *)src->data,
                                                             (float *)dst->data,
                                                             dst->ndim,
                                                             s_dims, d_dims,
                                                             s_ids, d_ids,
                                                             axes_device,
                                                             BLOCK_SIZE,
                                                             thread_num);
          break;
     case TL_INT32:
          transpose_kernel<int32_t><<<block_num, BLOCK_SIZE>>>((int32_t *)src->data,
                                                               (int32_t *)dst->data,
                                                               dst->ndim,
                                                               s_dims, d_dims,
                                                               s_ids, d_ids,
                                                               axes_device,
                                                               BLOCK_SIZE,
                                                               thread_num);
          break;
     case TL_INT16:
          transpose_kernel<int16_t><<<block_num, BLOCK_SIZE>>>((int16_t *)src->data,
                                                               (int16_t *)dst->data,
                                                               dst->ndim,
                                                               s_dims, d_dims,
                                                               s_ids, d_ids,
                                                               axes_device,
                                                               BLOCK_SIZE,
                                                               thread_num);
          break;
     case TL_INT8:
          transpose_kernel<int8_t><<<block_num, BLOCK_SIZE>>>((int8_t *)src->data,
                                                              (int8_t *)dst->data,
                                                              dst->ndim,
                                                              s_dims, d_dims,
                                                              s_ids, d_ids,
                                                              axes_device,
                                                              BLOCK_SIZE,
                                                              thread_num);
          break;
     case TL_UINT32:
          transpose_kernel<uint32_t><<<block_num, BLOCK_SIZE>>>((uint32_t *)src->data,
                                                                (uint32_t *)dst->data,
                                                                dst->ndim,
                                                                s_dims, d_dims,
                                                                s_ids, d_ids,
                                                                axes_device,
                                                                BLOCK_SIZE,
                                                                thread_num);
          break;
     case TL_UINT16:
          transpose_kernel<uint16_t><<<block_num, BLOCK_SIZE>>>((uint16_t *)src->data,
                                                                (uint16_t *)dst->data,
                                                                dst->ndim,
                                                                s_dims, d_dims,
                                                                s_ids, d_ids,
                                                                axes_device,
                                                                BLOCK_SIZE,
                                                                thread_num);
          break;
     case TL_UINT8:
          transpose_kernel<uint8_t><<<block_num, BLOCK_SIZE>>>((uint8_t *)src->data,
                                                               (uint8_t *)dst->data,
                                                               dst->ndim,
                                                               s_dims, d_dims,
                                                               s_ids, d_ids,
                                                               axes_device,
                                                               BLOCK_SIZE,
                                                               thread_num);
          break;
     case TL_BOOL:
          transpose_kernel<tl_bool_t><<<block_num, BLOCK_SIZE>>>((tl_bool_t *)src->data,
                                                                 (tl_bool_t *)dst->data,
                                                                 dst->ndim,
                                                                 s_dims, d_dims,
                                                                 s_ids, d_ids,
                                                                 axes_device,
                                                                 BLOCK_SIZE,
                                                                 thread_num);
          break;
     default:
          assert(0 && "unsupported tl_dtype");
          break;
     }
     TL_CUDA_CK(cudaDeviceSynchronize());

     if (!workspace) {
          tl_free_cuda(s_ids);
          tl_free_cuda(d_ids);
     }
     tl_free_cuda(s_dims);
     tl_free_cuda(d_dims);
     tl_free_cuda(axes_device);

     return dst;
}

tl_tensor *tl_tensor_vtranspose_cuda(const tl_tensor *src, tl_tensor *dst,
                                     tl_tensor *workspace, ...)
{
     int *axes;
     va_list ap;
     int i;

     assert(src && src->ndim > 0);
     axes = (int *)tl_alloc(sizeof(int) * src->ndim);
     va_start(ap, workspace);
     for (i = 0; i < src->ndim; i++) {
          axes[i] = va_arg(ap, int);
          assert(axes[i] > 0);
     }
     va_end(ap);
     dst = tl_tensor_transpose_cuda(src, dst, axes, workspace);
     tl_free(axes);
     return dst;
}

#endif  /* TL_CUDA */
