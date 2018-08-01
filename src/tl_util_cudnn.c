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

#include "tl_util.h"

cudnnHandle_t tl_cudnn_handle(void)
{
     static int has_init[TL_MAX_CUDA_DEVICE] = {0};
     static cudnnHandle_t handles[TL_MAX_CUDA_DEVICE];
     int i = tl_cuda_get_device();

     if (i >= TL_MAX_CUDA_DEVICE) {
          tl_err_bt("ERROR: CUDA device index(%d) exceeds TL_MAX_CUDA_DEVICE(%d)",
                    i, TL_MAX_CUDA_DEVICE);
     }
     if (!has_init[i]) {
          TL_CUDNN_CK(cudnnCreate(&handles[i]));
          has_init[i] = 1;
     }
     return handles[i];
}

#endif  /* TL_CUDNN */
