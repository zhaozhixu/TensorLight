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

#ifdef TL_CUDNN

#include "tl_tensor.h"
#include "tl_util.h"

static int compute_out_dim(int in_dim, int pad, int filter_size, int stride)
{
     return (in_dim + pad*2 - filter_size) / stride + 1;
}

/* NCHW tensor only */
tl_tensor *tl_tensor_conv2d_cudnn(tl_tensor *src, tl_tensor *filter,
                                  tl_tensor *dst, int *strides, int *pads)
{
     int batch, in_h, in_w, in_c, out_h, out_w, out_c;
     int filter_h, filter_w, stride_h, stride_w, pad_h, pad_w;
     int ndim = 4; /* TODO: add more ndim support */

     assert(src && tl_is_device_mem(src->data));
     assert(src->ndim == ndim);
     assert(filter && tl_is_device_mem(filter->data));
     assert(src->dtype == TL_FLOAT); /* TODO: add more type support */
     assert(src->dtype == filter->dtype);
     assert(filter->ndim == ndim);
     assert(filter->dims[1] == src->dims[1]);

     batch = src->dims[0];
     in_c = src->dims[1];
     in_h = src->dims[2];
     in_w = src->dims[3];
     filter_h = filter->dims[2];
     filter_w = filter->dims[3];
     stride_h = stride[0];
     stride_w = stride[1];
     pad_h = pads[0];
     pad_w = pads[1];
     out_c = filter->dims[0];
     out_h = compute_out_dim(in_h, pad_h, filter_h, stride_h);
     out_w = compute_out_dim(in_w, pad_w, filter_w, stride_w);

     if (dst) {
          assert(tl_is_device_mem(dst->data));
          assert(dst->dtype == src->dtype);
          assert(dst->ndim == ndim);
          assert(dst->dims[0] == batch);
          assert(dst->dims[1] == out_c);
          assert(dst->dims[2] == out_h);
          assert(dst->dims[3] == out_w);
     } else {
          dst = tl_tensor_create(NULL, ndim, (int[]){batch, out_c,
                         out_h, out_w}, src->dtype);
     }
}

#endif  /* TL_CUDNN */
