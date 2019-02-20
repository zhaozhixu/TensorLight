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
__global__ void detection_yolov3_kernel(const float *feature,
                                        const float *anchors,
                                        float *box_centers, float *box_sizes,
                                        float *confs, float *probs,
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
        scaled_anchors[threadIdx.x] = threadIdx.x % 2 == 0 ? /* necessary? */
            scaled_anchors[threadIdx.x] / ratio_w :
            scaled_anchors[threadIdx.x] / ratio_h;
    }
    __syncthreads();

    int fi = blockIdx.x * block_size + threadIdx.x; /* index in a feature map */
    if (fi >= feature_len)
        return;

    int hw = grid_h * grid_w;
    int anchor_volumn = feature_len / anchor_num;
    int ai = fi % anchor_volumn;
    float f = feature[fi];
    float sigmoided = 1 / (1 + expf(-f));

    if (ai < hw * 2) {                  /* box_centers */
        float center;
        if (ai < hw)            /* x */
            center = (sigmoided + ai % grid_w) * ratio_w;
        else                    /* y */
            center = (sigmoided + ai / grid_w) * ratio_h;
        box_centers[ai + fi / anchor_volumn * hw * 2] = center;
        /* printf("fi = %d, ai = %d, box_centers index: %d\n", fi, ai, ai + fi / anchor_volumn * hw * 2); */
    }
    /* box_centers[0] = 1; */

    if (ai >= hw * 2 && ai < hw * 4) { /* box_sizes */
        float size;
        if (ai < hw * 3)        /* w */
            size = scaled_anchors[fi / anchor_volumn * 2]
                * min(max(f, 50), 1e-9) * ratio_w;
        else                    /* h */
            size = scaled_anchors[fi / anchor_volumn * 2 + 1]
                * min(max(f, 50), 1e-9) * ratio_h;
        box_sizes[ai - hw * 2 + fi / anchor_volumn * hw * 2] = size;
        /* printf("size = %f\n", size); */
        /* printf("fi = %d, ai = %d, box_sizes index: %d, size = %f\n", fi, ai, ai - hw * 2 + fi / anchor_volumn * hw * 2, size); */
    }

    if (ai >= hw * 4 && ai < hw * 5)   /* conf */
        confs[ai % hw + fi / anchor_volumn * hw] = sigmoided;
    if (ai >= hw * 5)                  /* probs */
        probs[ai - hw * 5 + fi / anchor_volumn * hw * class_num] = sigmoided;

}

// feature in [N, C, H, W] order, where N = 1, C = anchor_num * (5 + class_num)
// anchors in [anchor_num, 2] order, where the 2nd dim is (w, h)
// box_centers in [N, anchor_num, 2, H, W] order, the 3rd dim is (x, y)
// box_sizes in [N, anchor_num, 2, H, W] order, the 3rd dim is (w, h)
// confs in [N, anchor_num, 1, H, W] order
// probs in [N, anchor_num, class_num, H, W] order
void tl_tensor_detection_yolov3_cuda(const tl_tensor *feature,
                                     const tl_tensor *anchors,
                                     tl_tensor *box_centers,
                                     tl_tensor *box_sizes,
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

    if (box_centers) {
        assert(tl_is_device_mem(box_centers->data));
        assert(box_centers->dtype == TL_FLOAT);
        assert(box_centers->ndim == 5);
        assert(box_centers->dims[0] == 1);
        assert(box_centers->dims[1] == anchor_num);
        assert(box_centers->dims[2] == 2);
        assert(box_centers->dims[3] == H);
        assert(box_centers->dims[4] == W);
    } else {
        int dims[5] = {1, anchor_num, 2, H, W};
        box_centers = tl_tensor_zeros_cuda(5, dims, TL_FLOAT);
    }

    if (box_sizes) {
        assert(tl_is_device_mem(box_sizes->data));
        assert(box_sizes->dtype == TL_FLOAT);
        assert(box_sizes->ndim == 5);
        assert(box_sizes->dims[0] == 1);
        assert(box_sizes->dims[1] == anchor_num);
        assert(box_sizes->dims[2] == 2);
        assert(box_sizes->dims[3] == H);
        assert(box_sizes->dims[4] == W);
    } else {
        int dims[5] = {1, anchor_num, 2, H, W};
        box_sizes = tl_tensor_zeros_cuda(5, dims, TL_FLOAT);
    }

    if (confs) {
        assert(tl_is_device_mem(confs->data));
        assert(confs->dtype == TL_FLOAT);
        assert(confs->ndim == 5);
        assert(confs->dims[0] == 1);
        assert(confs->dims[1] == anchor_num);
        assert(confs->dims[2] == 1);
        assert(confs->dims[3] == H);
        assert(confs->dims[4] == W);
    } else {
        int dims[5] = {1, anchor_num, 1, H, W};
        confs = tl_tensor_zeros_cuda(5, dims, TL_FLOAT);
    }

    if (probs) {
        assert(tl_is_device_mem(probs->data));
        assert(probs->dtype == TL_FLOAT);
        assert(probs->ndim == 5);
        assert(probs->dims[0] == 1);
        assert(probs->dims[1] == anchor_num);
        assert(probs->dims[2] == class_num);
        assert(probs->dims[3] == H);
        assert(probs->dims[4] == W);
    } else {
        int dims[5] = {1, anchor_num, class_num, H, W};
        probs = tl_tensor_zeros_cuda(5, dims, TL_FLOAT);
    }

    printf("feature->len = %d\n", feature->len);
    printf("feature shape: [%d %d %d %d]\n", feature->dims[0], feature->dims[1], feature->dims[2], feature->dims[3]);
    printf("box_sizes->len = %d\n", box_sizes->len);
    printf("box_centers->len = %d\n", box_centers->len);
    int block_num = BLOCK_NUM(BLOCK_SIZE, feature->len);
    detection_yolov3_kernel<<<block_num, BLOCK_SIZE>>>((float *)feature->data,
                                                       (float *)anchors->data,
                                                       (float *)box_centers->data,
                                                       (float *)box_sizes->data,
                                                       (float *)confs->data,
                                                       (float *)probs->data,
                                                       H, W, img_h, img_w,
                                                       class_num, anchor_num,
                                                       BLOCK_SIZE,
                                                       feature->len);
    tl_cuda_device_sync();
}
