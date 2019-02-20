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

#ifndef _TL_TENSOR_H_
#define _TL_TENSOR_H_

/* end of config defines */

#include "tl_type.h"

#define TL_MAJOR_VERSION (0)
#define TL_MINOR_VERSION (1)
#define TL_MICRO_VERSION (0)

#define TL_MAXDIM 8

struct tl_tensor {
     tl_dtype          dtype;
     int               len;
     int               ndim;
     int              *dims;
     void             *data;
     struct tl_tensor *owner;          /* data owner, NULL if it's itself */
     void             *backend_data;   /* for other backend dependent data */
};
typedef struct tl_tensor tl_tensor;

#ifdef __cplusplus
TL_CPPSTART
#endif

int tl_tensor_index(const tl_tensor *t, int *coords);
void tl_tensor_coords(const tl_tensor *t, int index, int *coords);
int tl_tensor_issameshape(const tl_tensor *t1, const tl_tensor *t2);
tl_tensor *tl_tensor_create(void *data, int ndim, const int *dims,
                            tl_dtype dtype);
void tl_tensor_free(tl_tensor *t);
void tl_tensor_free_data_too(tl_tensor *t);
size_t tl_tensor_size(tl_tensor *t);
tl_tensor *tl_tensor_zeros(int ndim, const int *dims, tl_dtype dtype);
tl_tensor *tl_tensor_clone(const tl_tensor *src);
tl_tensor *tl_tensor_repeat(const tl_tensor *src, int times);
tl_tensor *tl_tensor_arange(double start, double stop, double step,
                            tl_dtype dtype);
void tl_tensor_rearange(tl_tensor *src, double start, double stop,
                        double step);
void tl_tensor_fprint(FILE *stream, const tl_tensor *t, const char *fmt);
void tl_tensor_print(const tl_tensor *t, const char *fmt);
int tl_tensor_save(const char *file_name, const tl_tensor *t, const char *fmt);
tl_tensor *tl_tensor_create_slice(void *data, const tl_tensor *src, int axis,
                                  int len, tl_dtype dtype);
tl_tensor *tl_tensor_zeros_slice(const tl_tensor *src, int axis, int len,
                                 tl_dtype dtype);
tl_tensor *tl_tensor_slice(const tl_tensor *src, tl_tensor *dst, int axis,
                           int start, int len);
tl_tensor *tl_tensor_slice_nocopy(tl_tensor *src, tl_tensor *dst,
                                  int axis, int start, int len);
tl_tensor *tl_tensor_concat(const tl_tensor *src1, const tl_tensor *src2,
                            tl_tensor *dst, int axis);
tl_tensor *tl_tensor_reshape(tl_tensor *src, int ndim, const int *dims);
void tl_tensor_reshape_src(tl_tensor *src, int ndim, const int *dims);
tl_tensor *tl_tensor_maxreduce(const tl_tensor *src, tl_tensor *dst,
                               tl_tensor *arg, int axis);
tl_tensor *tl_tensor_elew(const tl_tensor *src1, const tl_tensor *src2,
                          tl_tensor *dst, tl_elew_op elew_op);
tl_tensor *tl_tensor_elew_param(const tl_tensor *src, double param,
                                tl_tensor *dst, tl_elew_op elew_op);
tl_tensor *tl_tensor_transpose(const tl_tensor *src, tl_tensor *dst,
                               const int *axes);
tl_tensor *tl_tensor_lrelu(const tl_tensor *src, tl_tensor *dst,
                           float negslope);
tl_tensor *tl_tensor_convert(const tl_tensor *src, tl_tensor *dst,
                             tl_dtype dtype_d);
tl_tensor *tl_tensor_resize(const tl_tensor *src, tl_tensor *dst,
                            const int *new_dims, tl_resize_type rtype);

#ifdef TL_CUDA

tl_tensor *tl_tensor_zeros_cuda(int ndim, const int *dims, tl_dtype dtype);
void tl_tensor_free_data_too_cuda(tl_tensor *t);
tl_tensor *tl_tensor_clone_h2d(const tl_tensor *src);
tl_tensor *tl_tensor_clone_d2h(const tl_tensor *src);
tl_tensor *tl_tensor_clone_d2d(const tl_tensor *src);
tl_tensor *tl_tensor_repeat_h2d(const tl_tensor *src, int times);
tl_tensor *tl_tensor_repeat_d2h(const tl_tensor *src, int times);
tl_tensor *tl_tensor_repeat_d2d(const tl_tensor *src, int times);
tl_tensor *tl_tensor_arange_cuda(double start, double stop, double step,
                                 tl_dtype dtype);
void tl_tensor_rearange_cuda(tl_tensor *src, double start, double stop,
                             double step);
void tl_tensor_fprint_cuda(FILE *stream, const tl_tensor *t, const char *fmt);
void tl_tensor_print_cuda(const tl_tensor *t, const char *fmt);
int tl_tensor_save_cuda(const char *file_name, const tl_tensor *t,
                        const char *fmt);
tl_tensor *tl_tensor_zeros_slice_cuda(const tl_tensor *src, int axis, int len,
                                       tl_dtype dtype);
tl_tensor *tl_tensor_slice_cuda(const tl_tensor *src, tl_tensor *dst, int axis,
                                int start, int len);
tl_tensor *tl_tensor_maxreduce_cuda(const tl_tensor *src, tl_tensor *dst,
                                    tl_tensor *arg, int axis);
tl_tensor *tl_tensor_elew_cuda(const tl_tensor *src1, const tl_tensor *src2,
                               tl_tensor *dst, tl_elew_op elew_op);
tl_tensor *tl_tensor_convert_cuda(const tl_tensor *src, tl_tensor *dst,
                                  tl_dtype dtype_d);
tl_tensor *tl_tensor_transpose_cuda(const tl_tensor *src, tl_tensor *dst,
                                    const int *axes);
tl_tensor *tl_tensor_lrelu_cuda(const tl_tensor *src, tl_tensor *dst,
                                float negslope);
tl_tensor *tl_tensor_resize_cuda(const tl_tensor *src, tl_tensor *dst,
                                 const int *new_dims, tl_resize_type rtype);
tl_tensor *tl_tensor_transform_bboxSQD_cuda(const tl_tensor *delta,
                                            const tl_tensor *anchor,
                                            tl_tensor *dst,
                                            int width, int height,
                                            int img_width, int img_height,
                                            int x_shift, int y_shift);
void tl_tensor_sort1d_cuda(tl_tensor *key, tl_sort_dir dir);
void tl_tensor_sort1d_by_key_cuda(tl_tensor *key, tl_tensor *val,
                                  tl_sort_dir dir);
tl_tensor *tl_tensor_pick1d_cuda(const tl_tensor *src, const tl_tensor *index,
                                 tl_tensor *dst, int stride, int len);
void tl_tensor_detect_yolov3_cuda(const tl_tensor *feature,
                                  const tl_tensor *anchors,
                                  tl_tensor *box_centers,
                                  tl_tensor *box_sizes,
                                  tl_tensor *boxes,
                                  tl_tensor *confs, tl_tensor *probs,
                                  int img_h, int img_w);

#endif  /* TL_CUDA */

#ifdef __cplusplus
TL_CPPEND
#endif

#endif  /* _TL_TENSOR_H_ */
