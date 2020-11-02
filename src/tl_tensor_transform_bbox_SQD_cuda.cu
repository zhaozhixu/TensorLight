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

static __global__ void transform_bboxSQD_kernel(float *delta, float *anchor, float *res, int width,
                                                int height, int img_width, int img_height,
                                                int x_shift, int y_shift, int block_size, int total)
{
    int di = blockIdx.x * block_size + threadIdx.x;
    if (di >= total)
        return;

    /* TODO: FIXME: only support batch_size = 1 */
    float x_scale = 1.0 * img_width / width;
    float y_scale = 1.0 * img_height / height;
    float E = 2.718281828;

    /* take 4 elements from each of delta and anchor */
    int si = di * 4;
    float d[4] = { delta[si], delta[si + 1], delta[si + 2], delta[si + 3] };
    float a[4] = { anchor[si], anchor[si + 1], anchor[si + 2], anchor[si + 3] };
    /* compute and put 4 result elements to res,
       according to SqueezeDet's source code */

    /* TODO: don't know why (maybe the resize),
       always has some shift compared to groundtruth*/
    float cx = (a[0] + d[0] * a[2]) * x_scale + x_shift;
    float cy = (a[1] + d[1] * a[3]) * y_scale + y_shift;
    float w = (a[2] * (d[2] < 1 ? expf(d[2]) : d[2] * E)) * x_scale;
    float h = (a[3] * (d[3] < 1 ? expf(d[3]) : d[3] * E)) * y_scale;
    res[si] = min(max(cx - w * 0.5, 0), img_width - 1);
    res[si + 1] = min(max(cy - h * 0.5, 0), img_height - 1);
    res[si + 2] = max(min(cx + w * 0.5, img_width - 1), 0);
    res[si + 3] = max(min(cy + h * 0.5, img_height - 1), 0);
}

TL_EXPORT tl_tensor *tl_tensor_transform_bboxSQD_cuda(const tl_tensor *delta,
                                                      const tl_tensor *anchor, tl_tensor *dst,
                                                      int width, int height, int img_width,
                                                      int img_height, int x_shift, int y_shift)
{
    assert(delta && anchor);
    assert(tl_is_device_mem(delta->data));
    assert(tl_tensor_issameshape(delta, anchor));
    assert(delta->dtype == TL_FLOAT);
    assert(delta->dtype == anchor->dtype);
    assert(delta->ndim == 5);
    assert(delta->dims[4] == 4);
    assert(width > 0 && height > 0 && img_width > 0 && img_height > 0);
    if (dst) {
        assert(dst->data);
        assert(tl_is_device_mem(dst->data));
        assert(tl_tensor_issameshape(delta, dst));
        assert(dst->dtype == delta->dtype);
    } else {
        dst = tl_tensor_zeros_cuda(delta->ndim, delta->dims, delta->dtype);
    }

    int i, thread_num, block_num;
    for (i = 0, thread_num = 1; i < dst->ndim - 1; i++)
        thread_num *= dst->dims[i];
    block_num = BLOCK_NUM(BLOCK_SIZE, thread_num);

    transform_bboxSQD_kernel<<<block_num, BLOCK_SIZE>>>((float *)delta->data,
                                                          (float *)anchor->data, (float *)dst->data,
                                                          width, height, img_width, img_height,
                                                          x_shift, y_shift, BLOCK_SIZE, thread_num);
    tl_cuda_device_sync();
    return dst;
}
