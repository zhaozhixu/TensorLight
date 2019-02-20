#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>
#include <math.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "tl_tensor.h"
#include "tl_util.h"

#define BLOCK_SIZE 1024
#define BLOCK_NUM(bs, tn) (((tn) + (bs) - 1) / (bs))
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

#define MAX_ANCHOR_NUM 64
__global__ void detect_yolov3_kernel(const float *feature,
                                     const float *anchors,
                                     float *box_centers, float *box_sizes,
                                     float *boxes, float *confs, float *probs,
                                     int grid_h, int grid_w,
                                     int img_h, int img_w,
                                     int class_num, int anchor_num,
                                     int block_size, int feature_len)
{

    assert(anchor_num <= MAX_ANCHOR_NUM);

    float ratio_h = (float)img_h / (float)grid_h;
    float ratio_w = (float)img_w / (float)grid_w;
    __shared__ float scaled_anchors[MAX_ANCHOR_NUM * 2];

    if (threadIdx.x < anchor_num * 2) {
        scaled_anchors[threadIdx.x] = anchors[threadIdx.x];
        /* scaled_anchors[threadIdx.x] = threadIdx.x % 2 == 0 ? /\* necessary? *\/ */
        /*     scaled_anchors[threadIdx.x] / ratio_w : */
        /*     scaled_anchors[threadIdx.x] / ratio_h; */
    }
    __syncthreads();

    int ti = blockIdx.x * block_size + threadIdx.x;
    if (ti >= feature_len)  /* ti is now index in a feature map */
        return;

    int hw = grid_h * grid_w;
    int anchor_volumn = feature_len / anchor_num;
    int ai = ti % anchor_volumn;
    float f = feature[ti];
    float sigmoided = 1 / (1 + expf(-f));

    if (ai < hw * 2) {                  /* box_centers */
        float center;
        if (ai < hw)            /* x */
            center = (sigmoided + ai % grid_w) * ratio_w;
        else                    /* y */
            center = (sigmoided + (ai - hw) / grid_w) * ratio_h;
        box_centers[ai + ti / anchor_volumn * hw * 2] = center;
    }

    if (ai >= hw * 2 && ai < hw * 4) { /* box_sizes */
        float size;
        if (ai < hw * 3)        /* w */
            size = scaled_anchors[ti / anchor_volumn * 2]
                * max(min(expf(f), 50), 1e-9);
        else                    /* h */
            size = scaled_anchors[ti / anchor_volumn * 2 + 1]
                * max(min(expf(f), 50), 1e-9);
        /* if (ai < hw * 3)        /\* w *\/ */
        /*     size = scaled_anchors[ti / anchor_volumn * 2] */
        /*         * max(min(expf(f), 50), 1e-9) * ratio_w; */
        /* else                    /\* h *\/ */
        /*     size = scaled_anchors[ti / anchor_volumn * 2 + 1] */
        /*         * max(min(expf(f), 50), 1e-9) * ratio_h; */
        /* printf("size_index = %d, %f\n", ai - hw * 2 + ti / anchor_volumn * hw * 2, size); */
        box_sizes[ai - hw * 2 + ti / anchor_volumn * hw * 2] = size;
        /* int i = ai - hw * 2 + ti / anchor_volumn * hw * 2; */
        /* printf("size_index = %d, %f\n", i, box_sizes[i]); */
    }

    if (ai >= hw * 4 && ai < hw * 5)   /* conf */
        confs[ai - hw * 4 + ti / anchor_volumn * hw] = sigmoided;
    if (ai >= hw * 5)                  /* probs */
        probs[ai - hw * 5 + ti / anchor_volumn * hw * class_num] = sigmoided;

    if (ti >= anchor_num * grid_h * grid_w * 2)
        return;

    __shared__ float centers[BLOCK_SIZE];
    __shared__ float sizes[BLOCK_SIZE];
    centers[threadIdx.x] = box_centers[ti];
    sizes[threadIdx.x] = box_sizes[ti];
    /* float center = box_centers[ti]; */
    /* float size = box_sizes[ti]; */
    int outer_index = ti / (hw * 2) * hw * 4;
    int inner_index = ti % (hw * 2);
    boxes[outer_index + inner_index] = center - box_sizes[ti] / 2;
    /* printf("x_index = %d, data = %f, center = %f, size = %f\n", outer_index + inner_index, center - box_sizes[ti] / 2, center, box_sizes[ti]); */
    boxes[outer_index + hw * 2 + inner_index] = center + box_sizes[ti] / 2;
    /* printf("y_index = %d, data = %f, center = %f, size = %f\n", outer_index + hw * 2 + inner_index, center + box_sizes[ti] / 2, center, box_sizes[ti]); */
    /* boxes[outer_index + inner_index] = center - size / 2; */
    /* printf("x_index = %d, data = %f, center = %f, size = %f\n", outer_index + inner_index, center - size / 2, center, size); */
    /* boxes[outer_index + hw * 2 + inner_index] = center + size / 2; */
    /* printf("y_index = %d, data = %f, center = %f, size = %f\n", outer_index + hw * 2 + inner_index, center + size / 2, center, size); */
}

// feature in [N, C, H, W] order, where N = 1, C = anchor_num * (5 + class_num)
// anchors in [anchor_num, 2] order, where the 2nd dim is (w, h)
// box_centers in [N, anchor_num, 2, H, W] order, the 3rd dim is (x, y)
// box_sizes in [N, anchor_num, 2, H, W] order, the 3rd dim is (w, h)
// boxes in [N, anchor_num, 4, H, W] order, the 3rd dim is (x_min, y_min,
// x_max, y_max)
// confs in [N, anchor_num, 1, H, W] order
// probs in [N, anchor_num, class_num, H, W] order
void tl_tensor_detect_yolov3_cuda(const tl_tensor *feature,
                                  const tl_tensor *anchors,
                                  tl_tensor *box_centers,
                                  tl_tensor *box_sizes,
                                  tl_tensor *boxes,
                                  tl_tensor *confs, tl_tensor *probs,
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
    printf("boxes->len = %d\n", boxes->len);
    int block_num = BLOCK_NUM(BLOCK_SIZE, feature->len);
    detect_yolov3_kernel<<<block_num, BLOCK_SIZE>>>((float *)feature->data,
                                                    (float *)anchors->data,
                                                    (float *)box_centers->data,
                                                    (float *)box_sizes->data,
                                                    (float *)boxes->data,
                                                    (float *)confs->data,
                                                    (float *)probs->data,
                                                    H, W, img_h, img_w,
                                                    class_num, anchor_num,
                                                    BLOCK_SIZE,
                                                    feature->len);
    tl_cuda_device_sync();
}
