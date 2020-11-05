/*
 * Copyright (c) 2018-2020 Zhixu Zhao
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

#define YOLO_MAX_ANCHOR_NUM 64

static __global__ void detect_yolov3_kernel(const float *feature, const float *anchors,
                                            float *box_centers, float *box_sizes, float *boxes,
                                            float *confs, float *probs, int grid_h, int grid_w,
                                            int img_h, int img_w, int class_num, int anchor_num,
                                            int block_size, int feature_len)
{
    assert(anchor_num <= YOLO_MAX_ANCHOR_NUM);

    __shared__ float anchors_cache[YOLO_MAX_ANCHOR_NUM * 2];

    if (threadIdx.x < anchor_num * 2) {
        anchors_cache[threadIdx.x] = anchors[threadIdx.x];
    }
    __syncthreads();

    int ti = blockIdx.x * block_size + threadIdx.x;
    if (ti >= feature_len) /* ti is now the output index in a feature map */
        return;

    int hw = grid_h * grid_w;
    int anchor_volumn = feature_len / anchor_num;
    int ai = ti % anchor_volumn;
    float exp_f = expf(feature[ti]);
    float sigmoided = 1 / (1 + 1 / exp_f);

    if (ai < hw * 2) { /* box_centers */
        float center;
        if (ai < hw) /* x */
            center = (sigmoided + ai % grid_w) * ((float)img_w / grid_w);
        else /* y */
            center = (sigmoided + (ai - hw) / grid_w) * ((float)img_h / grid_h);
        box_centers[ai + ti / anchor_volumn * hw * 2] = center;
    }

    if (ai >= hw * 2 && ai < hw * 4) { /* box_sizes */
        float size;
        if (ai < hw * 3) /* w */
            size = anchors_cache[ti / anchor_volumn * 2] * max(min(exp_f, 50), 1e-9);
        else /* h */
            size = anchors_cache[ti / anchor_volumn * 2 + 1] * max(min(exp_f, 50), 1e-9);
        box_sizes[ai - hw * 2 + ti / anchor_volumn * hw * 2] = size;
    }

    if (ai >= hw * 4 && ai < hw * 5) /* conf */
        confs[ai - hw * 4 + ti / anchor_volumn * hw] = sigmoided;

    if (ai >= hw * 5) /* probs */
        probs[ai - hw * 5 + ti / anchor_volumn * hw * class_num] = sigmoided;

    __syncthreads();

    if (ti >= anchor_num * grid_h * grid_w * 2)
        return;

    float center = box_centers[ti];
    float size = box_sizes[ti];
    int outer_index = ti / (hw * 2) * hw * 4;
    int inner_index = ti % (hw * 2);
    boxes[outer_index + inner_index] = center - size / 2;
    boxes[outer_index + hw * 2 + inner_index] = center + size / 2;
}

// feature in [N, C, H, W] order, where N = 1, C = anchor_num * (5 + class_num)
// anchors in [anchor_num, 2] order, where the 2nd dim is (w, h)
// box_centers in [N, anchor_num, 2, H, W] order, the 3rd dim is (x, y)
// box_sizes in [N, anchor_num, 2, H, W] order, the 3rd dim is (w, h)
// boxes in [N, anchor_num, 4, H, W] order, the 3rd dim is (x_min, y_min,
// x_max, y_max)
// confs in [N, anchor_num, 1, H, W] order
// probs in [N, anchor_num, class_num, H, W] order
TL_EXPORT void tl_tensor_detect_yolov3_cuda(const tl_tensor *feature, const tl_tensor *anchors,
                                            tl_tensor *box_centers, tl_tensor *box_sizes,
                                            tl_tensor *boxes, tl_tensor *confs, tl_tensor *probs,
                                            int img_h, int img_w)
{
    assert(feature && tl_is_device_mem(feature->data));
    assert(feature->dtype == TL_FLOAT);
    assert(feature->ndim == 4);
    assert(feature->dims[0] == 1);
    assert(anchors && tl_is_device_mem(anchors->data));
    assert(anchors->dtype == TL_FLOAT);
    assert(anchors->ndim == 2);

    int anchor_num = anchors->dims[0];
    int class_num = feature->dims[1] / anchor_num - 5;
    int H = feature->dims[2];
    int W = feature->dims[3];

    assert(box_centers && tl_is_device_mem(box_centers->data));
    assert(box_centers->dtype == TL_FLOAT);
    assert(box_centers->ndim == 5);
    assert(box_centers->dims[0] == 1);
    assert(box_centers->dims[1] == anchor_num);
    assert(box_centers->dims[2] == 2);
    assert(box_centers->dims[3] == H);
    assert(box_centers->dims[4] == W);

    assert(box_sizes && tl_is_device_mem(box_sizes->data));
    assert(box_sizes->dtype == TL_FLOAT);
    assert(box_sizes->ndim == 5);
    assert(box_sizes->dims[0] == 1);
    assert(box_sizes->dims[1] == anchor_num);
    assert(box_sizes->dims[2] == 2);
    assert(box_sizes->dims[3] == H);
    assert(box_sizes->dims[4] == W);

    assert(boxes && tl_is_device_mem(boxes->data));
    assert(boxes->dtype == TL_FLOAT);
    assert(boxes->ndim == 5);
    assert(boxes->dims[0] == 1);
    assert(boxes->dims[1] == anchor_num);
    assert(boxes->dims[2] == 4);
    assert(boxes->dims[3] == H);
    assert(boxes->dims[4] == W);

    assert(confs && tl_is_device_mem(confs->data));
    assert(confs->dtype == TL_FLOAT);
    assert(confs->ndim == 5);
    assert(confs->dims[0] == 1);
    assert(confs->dims[1] == anchor_num);
    assert(confs->dims[2] == 1);
    assert(confs->dims[3] == H);
    assert(confs->dims[4] == W);

    assert(probs && tl_is_device_mem(probs->data));
    assert(probs->dtype == TL_FLOAT);
    assert(probs->ndim == 5);
    assert(probs->dims[0] == 1);
    assert(probs->dims[1] == anchor_num);
    assert(probs->dims[2] == class_num);
    assert(probs->dims[3] == H);
    assert(probs->dims[4] == W);

    int block_num = BLOCK_NUM(BLOCK_SIZE, feature->len);
    detect_yolov3_kernel<<<block_num, BLOCK_SIZE>>>(
        (float *)feature->data, (float *)anchors->data, (float *)box_centers->data,
        (float *)box_sizes->data, (float *)boxes->data, (float *)confs->data, (float *)probs->data,
        H, W, img_h, img_w, class_num, anchor_num, BLOCK_SIZE, feature->len);
    tl_cuda_device_sync();
}
