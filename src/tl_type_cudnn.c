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

#include "tl_type.h"

cudnnDataType_t tl_dtype_to_cudnn_dtype(tl_dtype dtype)
{
     switch (dtype) {
     case TL_DOUBLE:
          return CUDNN_DATA_DOUBLE;
     case TL_FLOAT:
          return CUDNN_DATA_FLOAT;
     case TL_INT32:
          return CUDNN_DATA_INT32;
     case TL_INT8:
          return CUDNN_DATA_INT8;
     default:
          tl_err_bt("ERROR: not supported tl_dtype-to-cudnnDataType_t conversion\n");
          return -1;
     }
}

#endif  /* TL_CUDNN */
