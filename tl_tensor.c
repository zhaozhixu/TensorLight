#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "tl_tensor.h"

#define MAXDIM 8
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define MAX_THREADS_PER_BLOCK 1024
#define BLOCK_SIZE MAX_THREADS_PER_BLOCK

static float EPSILON = 1e-16;

/* static __device__ float E = 2.718281828; */

static int get_index(int *ids, int ndim, int *dims)
{
     int i, id;
     for (i = 0, id = ids[0]; i < ndim-1; i++)
          id = dims[i+1] * id + ids[i+1];
     return id;
}

static void get_indexes(int id, int *ids, int ndim, int *dims)
{
     for (int i = ndim-1; i >=0; i--) {
          ids[i] = id % dims[i];
          id = id / dims[i];
     }
}

/* __global__ void sliceTensorKernel(uint8_t *src, uint8_t *dst, int sdim, int ddim, int start, int block_size) */
/* { */
/*      int di = blockIdx.x * block_size + threadIdx.x; */
/*      /\* si is the index of src elements to be copied. */
/*         The "block index" of src[si] is (blockIdx.x / ddim * sdim + blockIdx.x % ddim + start) *\/ */
/*      int si = (blockIdx.x / ddim * sdim + blockIdx.x % ddim + start) * block_size + threadIdx.x; */
/*      dst[di] = src[si]; */
/* } */

__global__ void sliceTensorKernel(uint8_t *src, uint8_t *dst, int start, int s_vol, int d_vol, int vol, int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;
     int si = di / d_vol * s_vol + di % d_vol + start * vol;
     dst[di] = src[si];
}

__global__ void reduceArgMaxKernel(uint8_t *src, uint8_t *dst, uint8_t *arg, int dim_size, int reduce_vol, int batch_vol, int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;

     /* src[si] is the first element in this thread to be compared, then
        si = batch_vol * batch + (di - reduce_vol * batch),
        where batch = di / reduce_vol,
        which is the same as the following code: */
     int si = (batch_vol - reduce_vol) * (di / reduce_vol) + di;
     uint8_t now = src[si], max = now;
     int maxi = 0;
     for (int i = 1; i < dim_size; i++) {
          now = src[si+i*reduce_vol];
          if (now > max) {
               max = now;
               maxi = i;
          }
     }
     dst[di] = max;
     arg[di] = maxi;
}

__global__ void multiplyElementKernel(uint8_t *src1, uint8_t *src2, uint8_t *dst, int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;
     dst[di] = src1[di] * src2[di];
}

__global__ void transposeTensorKernel(uint8_t *src, uint8_t *dst, int ndim, int *s_dims, int *d_dims, int *s_ids, int *d_ids, int *axes, int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;

     int *t_s_ids = s_ids + di * ndim;
     int *t_d_ids = d_ids + di * ndim;
     get_indexes(di, t_d_ids, ndim, d_dims);
     for (int i = 0; i < ndim; i++)
          t_s_ids[axes[i]] = t_d_ids[i];
     int si = get_index(t_s_ids, ndim, s_dims);

     dst[di] = src[si];
}

__global__ void transformBboxSQDKernel(uint8_t *delta, uint8_t *anchor, uint8_t *res, float width, float height, float img_width, float img_height, int x_shift, int y_shift, int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;

     /* int batch_idx = di / anchor_num; */
     /* now only support batch_size = 1 */
     float x_scale = 1.0 * img_width / width;
     float y_scale = 1.0 * img_height / height;

     /* (not used) si is the index of the first elements to be computed in the thread, then
        si = 4 * anchor_num * batch_idx + (di - anchor_num * batch_idx),
        which is the same as the following code: */
     /* int si = 3 * anchor_num * batch_idx  + di; */
     /* take 4 elements from each of delta and anchor */
     int si = di * 4;
     uint8_t d[4] = {delta[si], delta[si+1], delta[si+2], delta[si+3]};
     uint8_t a[4] = {anchor[si], anchor[si+1], anchor[si+2], anchor[si+3]};
     /* compute and put 4 result elements to res, according to SqueezeDet's source code */

     /* TODO: don't know why (maybe the resize), always has some shift compared to groundtruth*/
     uint8_t cx = (a[0] + d[0] * a[2]) * x_scale + x_shift;
     uint8_t cy = (a[1] + d[1] * a[3]) * y_scale + y_shift;
     uint8_t w = (a[2] * (d[2] < 1 ? expf(d[2]) : d[2] * E)) * x_scale;
     uint8_t h = (a[3] * (d[3] < 1 ? expf(d[3]) : d[3] * E)) * y_scale;
     res[si] = min(max(cx - w * 0.5, 0), img_width - 1);
     res[si+1] = min(max(cy - h * 0.5, 0), img_height - 1);
     res[si+2] = max(min(cx + w * 0.5, img_width - 1), 0);
     res[si+3] = max(min(cy + h * 0.5, img_height - 1), 0);
}

__global__ void pickElementsKernel(uint8_t *src, uint8_t *dst, int *idx, int stride, int block_size, int total)
{
     int di = blockIdx.x * block_size + threadIdx.x;
     if (di >= total)
          return;
     int si = idx[di];
     for (int i = 0; i < stride; i++)
          dst[di*stride+i] = src[si*stride+i];
}

int tl_tensor_isvalid(const tl_tensor *t)
{
     return (t && t->data &&
             t->ndim < MAXDIM && t->ndim > 0 &&
             t->len == tl_compute_length(t->ndim, t->dims));
}

int tl_tensor_issameshape(const tl_tensor *t1, const tl_tensor *t2)
{
     assert(tl_tensor_isvalid(t1) && tl_tensor_isvalid(t2));
     int ndim;

     if (t1->ndim == t2->ndim) {
          ndim = t1->ndim;
          while (--ndim >= 0)
               if (t1->dims[ndim] != t2->dims[ndim])
                    return 0;
          return 1;
     }
     return 0;
}

tl_tensor *tl_tensor_create(void *data, int ndim, const int *dims,
                            tl_dtype dtype)
{
     int i;
     tl_tensor *t;
     size_t size;

     t = (tl_tensor *)tl_alloc(sizeof(tl_tensor));
     t->ndim = ndim;
     t->dims = (int *)tl_clone(dims, sizeof(int) * ndim);
     t->len = tl_compute_length(ndim, dims);
     t->dtype = dtype;
     size = t->len * tl_size_of(dtype);
     if (!data) {
          t->data = tl_alloc(size);
          memset(t->data, 0, size);
     } else {
          t->data = data;
     }

     return t;
}

void tl_tensor_free(tl_tensor *t, int do_free_data)
{
     assert(tl_tensor_isvalid(t));
     tl_free(t->dims);
     if (do_free_data) {
          tl_free(t->data);
     }
     tl_free(t);
}

/* TODO: va_list length not checked */
tl_tensor *tl_tensor_zeros(tl_dtype dtype, int ndim, ...)
{
     tl_tensor *t;
     int i;
     int *dims;
     va_list ap;

     dims = (int *)tl_alloc(sizeof(int) * ndim);
     va_start(ap, ndim);
     for (i = 0; i < ndim; i++) {
          dims[i] = va_arg(ap, int);
          if (dims[i] <= 0)
               tl_err_bt("ERROR: tl_tensor_zeros: dims[%d] = %d <= 0, incorrect tensor shape\n",
                         i, dims[i]);
     }
     va_end(ap);

     t = tl_tensor_create(NULL, ndim, dims, dtype);
     return t;
}

tl_tensor *tl_tensor_clone(const tl_tensor *src)
{
     assert(tl_tensor_isvalid(src));
     void *data;
     tl_tensor *dst;

     data = tl_clone(src->data, src->len*tl_size_of(src->dtype));
     dst = tl_tensor_create(data, src->ndim, src->dims, src->dtype);
     return dst;
}

void tl_tensor_fprint(FILE *stream, const tl_tensor *t, const char *fmt)
{
     assert(tl_tensor_isvalid(tensor));
     /* dimision size and how deep current chars go */
     int dim_sizes[MAXDIM], dim_levels[MAXDIM];
     int ndim, len, *dims; /* pointer short cut */
     void *data;
     /* buffer for brackets */
     char left_buf[MAXDIM+1], right_buf[MAXDIM+1];
     char *lp, *rp;
     char *fmt_use;
     size_t right_len;
     int i, j, k;

     ndim = tensor->ndim;
     len = tensor->len;
     dims = tensor->dims;
     data = tensor->data;
     lp = left_buf;
     rp = right_buf;
     dim_sizes[ndim-1] = tensor->dims[ndim-1];
     dim_levels[ndim-1] = 0;
     if (fmt) {
          fmt_use = (char *)tl_alloc(strlen(fmt) + 1);
          strcpy(fmt_use, fmt);
     } else {
          fmt_use = tl_fmt(t->dtype);
     }

     for (i = ndim-2; i >= 0; i--) {
          dim_sizes[i] = dims[i] * dim_sizes[i+1];
          dim_levels[i] = 0;
     }
     for (i = 0; i < len; i++) {
          for (j = 0; j < ndim; j++) {
               if (i % dim_sizes[j] == 0)
                    dim_levels[j]++;
               if (dim_levels[j] == 1) {
                    *lp++ = '[';
                    dim_levels[j]++;
               }
               if (dim_levels[j] == 3) {
                    *rp++ = ']';
                    if (j != 0 && dim_levels[j] > dim_levels[j-1]) {
                         *lp++ = '[';
                         dim_levels[j] = 2;
                    } else
                         dim_levels[j] = 0;
               }
          }
          *lp = *rp = '\0';
          fprintf(stream, "%s", right_buf);
          if (*right_buf != '\0') {
               fprintf(stream, "\n");
               right_len = strlen(right_buf);
               for (k = ndim-right_len; k > 0; k--)
                    fprintf(stream, " ");
          }
          fprintf(stream, "%s", left_buf);
          if (*left_buf == '\0')
               fprintf(stream, " ");
          fprintf(stream, fmt_use, data[i]);
          lp = left_buf, rp = right_buf;
     }
     for (j = 0; j < ndim; j++)
          fprintf(stream, "]");
     fprintf(stream, "\n");
     tl_free(fmt_use);
}

void tl_tensor_print(const tl_tensor *tensor, const char *fmt)
{
     tl_tensor_fprint(stdout, tensor, fmt);
}

void tl_tensor_save(const char *file_name, const tl_tensor *tensor,
                    const char *fmt)
{
     FILE *fp = fopen(file_name, "w");
     tl_tensor_fprint(fp, tensor, fmt);
     fclose(fp);
}

tl_tensor *tl_tensor_create_slice(const tl_tensor *src, int dim, int len,
                                  tl_dtype dtype)
{
     assert(tl_tensor_isvalid(src));
     assert(dim < src->ndim && dim >= 0);
     assert(len+start <= src->dims[dim]);
     tl_tensor *dst;
     int *dims;

     dims = (int *)tl_clone(src->dims, sizeof(int) * src->ndim);
     dims[dim] = len;
     dst = tl_tensor_create(NULL, src->ndim, dims, dtype);
     tl_free(dims);

     return dst;
}

tl_tensor *tl_tensor_slice(const tl_tensor *src, tl_tensor *dst, int dim,
                           int start, int len)
{
     int i;
     assert(tl_tensor_isvalid(src));
     if (dst) {
          assert(tl_tensor_isvalid(dst));
          assert(src->dtype == dst->dtype);
          assert(dst->ndim == src->ndim);
          for (i = 0; i < dst->ndim; i++)
               assert(i == dim ? dst->dims[i] == len :
                      dst->dims[i] == src->dims[i]);
     }

     int d_vol, s_vol, vol;
     /* block size and number of cuda threads */
     int thread_num, block_size, block_num;
     int si, di;
     size_t dsize;

     if (!dst)
          dst = tl_tensor_create_slice(src, dim, len, src->dtype);
     for (i = dim+1, vol = 1; i < dst->ndim; i++)
          vol *= dst->dims[i];
     d_vol = vol * dst->dims[dim];
     s_vol = vol * src->dims[dim];
     thread_num = dst->len;
     block_size = MAX_THREADS_PER_BLOCK;
     block_num = thread_num / block_size + 1;

     dsize = tl_size_of(src->dtype);
     for (di = 0; di < thread_num; di++) {
          si = di / d_vol * s_vol + di % d_vol + start * vol;
          tl_passign(dst->data, di, src->data, si, dsize);
     }

     return dst;
}

/* in-place reshape tensor */
tl_tensor *tl_tensor_reshape(const tl_tensor *src, int ndim, const int *dims)
{
     assert(tl_tensor_isvalid(src));
     assert(dims);
     assert(src->len == tl_compute_length(ndim, dims));
     tl_tensor *dst;

     dst = tl_tensor_create(src->data, ndim, dims);
     return dst;
}

tl_tensor *tl_tensor_maxreduce(const tl_tensor *src, tl_tensor *dst,
                                   tl_tensor *arg, int dim)
{
     int i;
     assert(tl_tensor_isvalid(src));
     assert(dim < src->ndim && dim >= 0);
     if (dst) {
          assert(tl_tensor_isvalid(dst));
          assert(src->dtype == dst->dtype);
          for (i = 0; i < dst->ndim; i++)
               assert(i == dim ? dst->dims[i] == 1 :
                      dst->dims[i] == src->dims[i]);
     }
     if (arg) {
          assert(tl_tensor_isvalid(arg));
          assert(arg->dtype == TL_INT32);
          for (i = 0; i < arg->ndim; i++)
               assert(i == dim ? arg->dims[i] == 1 :
                      arg->dims[i] == src->dims[i]);
     }

     /* suppose the shape of src is [N, C, H, W], dim = 1, then thread_num is N x H x W
        reduce_vol is H x W, index_vol is C x H x W */
     int i, thread_num, block_size, block_num, reduce_vol, index_vol;
     int di, si, maxi;
     void *data_s, *data_d, *data_a, *nowp, *maxp;
     size_t dsize;
     tl_dtype dtype;

     if (!dst)
          dst = tl_tensor_create_slice(src, dim, 1, src->dtype);
     for (i = dim+1, thread_num = 1; i < dst->ndim; i++)
          thread_num *= dst->dims[i];
     reduce_vol = thread_num;
     index_vol = thread_num * src->dims[dim];
     for (i = 0; i < dim; i++)
          thread_num *= dst->dims[i];
     block_size = MAX_THREADS_PER_BLOCK;
     block_num = thread_num / block_size + 1;

     /* reduceArgMaxKernel<<<block_num, block_size>>>(src->data, dst->data, arg->data, src->dims[dim], reduce_vol, index_vol, block_size, thread_num); */

     dtype = src->dtype;
     dsize = tl_size_of(dtype);
     nowp = tl_alloc(dsize);
     maxp = tl_alloc(dsize);
     data_s = src->data;
     data_d = dst->data;
     if (arg)
          data_a = arg->data;
     for (di = 0; di < thread_num; di++) {
          /* src[si] is the first element in this thread to be compared, then
             si = batch_vol * batch + (di - reduce_vol * batch),
             where batch = di / reduce_vol,
             which is the same as the following code: */
          si = (index_vol - reduce_vol) * (di / reduce_vol) + di;
          tl_passign(nowp, 0, data_s, si, dsize);
          tl_passign(maxp, 0, nowp, 0, dsize);
          for (i = 1, maxi = 0; i < dim_size; i++) {
               tl_passign(nowp, 0, data_s, si+i*reduce_vol);
               if (tl_pointer_cmp(nowp, maxp, dtype) > 0) {
                    tl_passign(maxp, 0, nowp, 0, dsize);
                    maxi = i;
               }
          }
          tl_passign(data_d, di, maxp, 0, dsize);
          if (arg)
               ((int32_t *)data_a)[di] = maxi;
     }
     tl_free(nowp);
     tl_free(maxp);

     return dst;
}

tl_tensor *tl_tensor_mul(const tl_tensor *src1, const tl_tensor *src2, tl_tensor *dst)
{
     assert(tl_tensor_issameshape(src1, src2));
     assert(src1->dtype == src2->dtype);
     if (dst) {
          assert(tl_tensor_issameshape(src1, dst));
          assert(src1->dtype == dst->dtype);
     }

     int thread_num, block_size, block_num;
     int di, si;

     if (!dst)
          dst = tl_tensor_create(NULL, src1->ndim, src2->dims, src1->dtype);
     thread_num = dst->len;
     block_size = MAX_THREADS_PER_BLOCK;
     block_num = thread_num / block_size + 1;

     for (di = 0; di < thread_num; di++) {

          dst[di] = src1[di] * src2[di];
     }
     return dst;
}

/* (optional) workspace size equals (sizeof(int) * dst->ndim * dst->len), two of them */
tl_tensor *transposeTensor(const tl_tensor *src, tl_tensor *dst, int *axes, int **workspace)
{
     assert(tl_tensor_isvalid(src) && tl_tensor_isvalid(dst));
     assert(src->len == dst->len);
     assert(src->ndim == dst->ndim);

     int *s_ids, *d_ids, *s_dims, *d_dims;
     int thread_num, block_size, block_num;
     thread_num = dst->len;
     block_size = MAX_THREADS_PER_BLOCK;
     block_num = thread_num / block_size + 1;
     s_dims = (int *)tl_clone(src->dims, sizeof(int) * src->ndim);
     d_dims = (int *)tl_clone(dst->dims, sizeof(int) * dst->ndim);
     if (!workspace) {
          s_ids = (int *)tl_alloc(sizeof(int) * dst->ndim * thread_num);
          d_ids = (int *)tl_alloc(sizeof(int) * dst->ndim * thread_num);
          /* checkError(cudaMalloc(&s_ids, sizeof(int) * dst->ndim * thread_num)); */
          /* checkError(cudaMalloc(&d_ids, sizeof(int) * dst->ndim * thread_num)); */
     } else {
          s_ids = workspace[0];
          d_ids = workspace[1];
     }

     /* transposeTensorKernel<<<block_num, block_size>>>(src->data, dst->data, dst->ndim, s_dims, d_dims, s_ids, d_ids, axes, block_size, thread_num); */

     int di, si;
     for (di = 0; di < thread_num; di++) {
          int *t_s_ids = s_ids + di * ndim;
          int *t_d_ids = d_ids + di * ndim;
          get_indexes(di, t_d_ids, ndim, d_dims);
          for (i = 0; i < ndim; i++)
               t_s_ids[axes[i]] = t_d_ids[i];
          int si = get_index(t_s_ids, ndim, s_dims);

          dst[di] = src[si];
     }

     if (!workspace) {
          tl_free(s_ids);
          tl_free(d_ids);
          /* checkError(cudaFree(s_ids)); */
          /* checkError(cudaFree(d_ids)); */
     }
     tl_free(s_dims);
     tl_free(d_dims);
     /* checkError(cudaFree(s_dims)); */
     /* checkError(cudaFree(d_dims)); */
     return dst;
}

/* TODO: multiple type tensor */
/* transform from bbox delta to bbox coordinates, using hyper param EXP_THRESH = 1.0.
   delta, anchor, res are all of the same shape [..., 4]
   width and height are resized image width and height.
   x_scales and y_scales are (temporary) pointers to width/original_width and height/original_height. */
tl_tensor *transformBboxSQD(const tl_tensor *delta, const tl_tensor *anchor, tl_tensor *res, int width, int height, int img_width, int img_height)
{
     assert(tl_tensor_issameshape(delta, anchor));
     assert(tl_tensor_issameshape(delta, res));
     assert(delta->ndim == 5);
     assert(delta->dims[4] == 4);
     /* assert(isDeviceMem(delta->data) && isDeviceMem(anchor->data) && isDeviceMem(res->data)); */

     /* take 4 elements from each of delta and anchor,
        and put 4 result elements to res in one thread */
     int i, thread_num, block_size, block_num;
     for (i = 0, thread_num = 1; i < res->ndim-1; i++)
          thread_num *= res->dims[i];
     block_size = MAX_THREADS_PER_BLOCK;
     block_num = thread_num / block_size + 1;

     /* transformBboxSQDKernel<<<block_num, block_size>>>(delta->data, anchor->data, res->data, width, height, img_width, img_height, x_shift, y_shift, block_size, thread_num); */

     int di, si;
     for (di = 0; di < thread_num; di++) {
          /* int batch_idx = di / anchor_num; */
          /* now only support batch_size = 1 */
          float x_scale = 1.0 * img_width / width;
          float y_scale = 1.0 * img_height / height;

          /* (not used) si is the index of the first elements to be computed in the thread, then
             si = 4 * anchor_num * batch_idx + (di - anchor_num * batch_idx),
             which is the same as the following code: */
          /* int si = 3 * anchor_num * batch_idx  + di; */
          /* take 4 elements from each of delta and anchor */
          int si = di * 4;
          uint8_t d[4] = {delta[si], delta[si+1], delta[si+2], delta[si+3]};
          uint8_t a[4] = {anchor[si], anchor[si+1], anchor[si+2], anchor[si+3]};
          /* compute and put 4 result elements to res, according to SqueezeDet's source code */

          /* TODO: don't know why (maybe the resize), always has some shift compared to groundtruth*/
          uint8_t cx = (a[0] + d[0] * a[2]) * x_scale;
          uint8_t cy = (a[1] + d[1] * a[3]) * y_scale;
          uint8_t w = (a[2] * (d[2] < 1 ? expf(d[2]) : d[2] * E)) * x_scale;
          uint8_t h = (a[3] * (d[3] < 1 ? expf(d[3]) : d[3] * E)) * y_scale;
          res[si] = min(max(cx - w * 0.5, 0), img_width - 1);
          res[si+1] = min(max(cy - h * 0.5, 0), img_height - 1);
          res[si+2] = max(min(cx + w * 0.5, img_width - 1), 0);
          res[si+3] = max(min(cy + h * 0.5, img_height - 1), 0);
     }
     return res;
}

void tensorIndexSort(tl_tensor *src, int *idx)
{
     assert(tl_tensor_isvalid(src));
     assert(idx);
     assert(isDeviceMem(src->data) && isDeviceMem(idx));

     /* the thrust call below can be unreliable, sometimes produces error */
     /* now it works with compilation flag -arch=sm_35 */
     /* TODO: replace thrust call by our own kernel */
     /* thrust::sort_by_key(thrust::device, src->data, src->data + src->len, idx, thrust::greater<uint8_t>()); */
}

void pickElements(uint8_t *src, uint8_t *dst, int stride, int *idx, int len)
{
     assert(src && dst && idx);
     assert(isDeviceMem(src) && isDeviceMem(dst) && isDeviceMem(idx));

     int thread_num, block_size, block_num;
     thread_num = len;
     block_size = MAX_THREADS_PER_BLOCK;
     block_num = thread_num / block_size + 1;

     pickElementsKernel<<<block_num, block_size>>>(src, dst, idx, stride, block_size, thread_num);
}

/* void pickElements(uint8_t* src,uint8_t* dst,int stride,int* idx,int len) */
/* { */
/*      assert(src && dst && idx); */

/*      for (int i = 0; i < len; i++) { */
/*           for (int j = 0; j < stride; j++) { */
/*                fprintf(stderr, "i: %d j: %d idx[i]: %d src[idx[i]]: %.2f", */
/*                        i, j, idx[i], src[idx[i]]); */
/*                fprintf(stderr, "\n"); */
/*                dst[i*stride+j] = src[idx[i]*stride+j]; */
/*           } */
/*      } */
/* } */

/* compute the iou of two bboxes whose elements are {top_left_x, top_left_y, bottom_right_x, bottom_right_y} */
float computeIou(float *bbox0, float *bbox1)
{
     assert(bbox0 && bbox1);

     float lr, tb;              /* left-right, top-bottom for intersection*/
     float intersection, total;
     lr = min(bbox0[2], bbox1[2]) - max(bbox0[0], bbox1[0]);
     if (lr >= 0) {
          tb = min(bbox0[3], bbox1[3]) - max(bbox0[1], bbox1[1]);
          if (tb >= 0) {
               intersection = tb * lr + EPSILON;
               total = (bbox0[2] - bbox0[0]) * (bbox0[3] - bbox0[1]) +
                    (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) - intersection;
               return intersection / (total + EPSILON);
          }
     }
     return 0;
}
