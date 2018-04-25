#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>

#include "tl_tensor.h"
#include "tl_util.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

static inline int get_index(int *ids, int ndim, int *dims)
{
     int i, id;
     for (i = 0, id = ids[0]; i < ndim-1; i++)
          id = dims[i+1] * id + ids[i+1];
     return id;
}

static inline void get_indexes(int id, int *ids, int ndim, int *dims)
{
     for (int i = ndim-1; i >=0; i--) {
          ids[i] = id % dims[i];
          id = id / dims[i];
     }
}

static inline int compute_length(int ndim, const int *dims)
{
     int i, len;

     assert(ndim > 0);
     assert(dims);
     for (i = 0, len = 1; i < ndim; i++) {
          assert(dims[i] > 0);
          len *= dims[i];
     }
     return len;
}

static inline void check_dim(int ndim, const int *dims)
{
     int i;

     assert(ndim > 0);
     assert(dims);
     for (i = 0; i < ndim; i++)
          assert(dims[i] > 0);
}

static inline void check_tensor(const tl_tensor *t)
{
     assert(t && t->data);
     assert(t->dtype >= 0 && t->dtype < TL_DTYPE_SIZE);
     assert(t->len == compute_length(t->ndim, t->dims));
}

int tl_tensor_issameshape(const tl_tensor *t1, const tl_tensor *t2)
{
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
     tl_tensor *t;
     size_t size;

     t = (tl_tensor *)tl_alloc(sizeof(tl_tensor));
     t->len = compute_length(ndim, dims);
     t->ndim = ndim;
     t->dims = (int *)tl_clone(dims, sizeof(int) * ndim);
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

     assert(ndim > 0);
     dims = (int *)tl_alloc(sizeof(int) * ndim);
     va_start(ap, ndim);
     for (i = 0; i < ndim; i++) {
          dims[i] = va_arg(ap, int);
          assert(dims[i] > 0);
     }
     va_end(ap);

     t = tl_tensor_create(NULL, ndim, dims, dtype);
     tl_free(dims);
     return t;
}

tl_tensor *tl_tensor_clone(const tl_tensor *src)
{
     void *data;
     tl_tensor *dst;

     data = tl_clone(src->data, src->len*tl_size_of(src->dtype));
     dst = tl_tensor_create(data, src->ndim, src->dims, src->dtype);
     return dst;
}

void tl_tensor_fprint(FILE *stream, const tl_tensor *t, const char *fmt)
{
     int ndim, len, *dims; /* pointer short cut */
     void *data;
     size_t dsize;

     /* dimision size and how deep current chars go */
     int *dim_sizes, *dim_levels;
     /* buffer for brackets */
     char *left_buf, *right_buf;
     char *lp, *rp;
     size_t right_len;
     int i, j, k;

     ndim = t->ndim;
     len = t->len;
     dims = t->dims;
     data = t->data;
     dsize = tl_size_of(t->dtype);

     dim_sizes = (int *)tl_alloc(sizeof(int) * ndim);
     dim_levels = (int *)tl_alloc(sizeof(int) * ndim);
     dim_sizes[ndim-1] = dims[ndim-1];
     dim_levels[ndim-1] = 0;
     left_buf = (char *)tl_alloc(sizeof(char) * (ndim+1));
     right_buf = (char *)tl_alloc(sizeof(char) * (ndim+1));
     lp = left_buf;
     rp = right_buf;

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
          tl_gfprintf(stream,fmt, tl_padd(data, i, dsize), t->dtype);
          lp = left_buf, rp = right_buf;
     }
     for (j = 0; j < ndim; j++)
          fprintf(stream, "]");
     fprintf(stream, "\n");

     tl_free(dim_sizes);
     tl_free(dim_levels);
     tl_free(left_buf);
     tl_free(right_buf);
}

void tl_tensor_print(const tl_tensor *tensor, const char *fmt)
{
     tl_tensor_fprint(stdout, tensor, fmt);
}

int tl_tensor_save(const char *file_name, const tl_tensor *tensor,
                    const char *fmt)
{
     FILE *fp;

     fp = fopen(file_name, "w");
     if (!fp) {
          tl_err_ret("ERROR: cannot open %s", file_name);
          return -1;
     }
     tl_tensor_fprint(fp, tensor, fmt);
     fclose(fp);
     return 0;
}

tl_tensor *tl_tensor_create_slice(const tl_tensor *src, int dim, int len,
                              tl_dtype dtype)
{
     tl_tensor *dst;
     int *dims;

     assert(dim < src->ndim && dim >= 0);
     assert(len <= src->dims[dim]);

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
     int d_vol, s_vol, vol;
     int thread_num;
     int si, di;
     size_t dsize;

     if (dst) {
          assert(src->dtype == dst->dtype);
          assert(dst->ndim == src->ndim);
          assert(dim < src->ndim && dim >= 0);
          assert(len+start <= src->dims[dim]);
#ifndef NDEBUG
          for (i = 0; i < src->ndim; i++)
               assert(i == dim ? dst->dims[i] == len :
                    dst->dims[i] == src->dims[i]);
#endif
     } else {
          dst = tl_tensor_create_slice(src, dim, len, src->dtype);
     }

     for (i = dim+1, vol = 1; i < dst->ndim; i++)
          vol *= dst->dims[i];
     d_vol = vol * dst->dims[dim];
     s_vol = vol * src->dims[dim];
     thread_num = dst->len;

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
     tl_tensor *dst;

     assert(src->len == compute_length(ndim, dims));
     dst = tl_tensor_create(src->data, ndim, dims, src->dtype);
     return dst;
}

tl_tensor *tl_tensor_maxreduce(const tl_tensor *src, tl_tensor *dst,
                              tl_tensor *arg, int dim)
{
     /* suppose the shape of src is [N, C, H, W], dim = 1, then thread_num is N x H x W
        reduce_vol is H x W, index_vol is C x H x W */
     int thread_num, reduce_vol, index_vol;
     int i, di, si, maxi;
     int dim_size;
     void *data_s, *data_d, *data_a, *nowp, *maxp;
     size_t dsize;
     tl_dtype dtype;
     tl_gcmp_func cmp;

     assert(dim < src->ndim && dim >= 0);
     if (dst) {
          assert(src->dtype == dst->dtype);
#ifndef NDEBUG
          for (i = 0; i < dst->ndim; i++)
               assert(i == dim ? dst->dims[i] == 1 :
                    dst->dims[i] == src->dims[i]);
#endif
     } else {
          dst = tl_tensor_create_slice(src, dim, 1, src->dtype);
     }
     if (arg) {
          assert(arg->dtype == TL_INT32);
          for (i = 0; i < arg->ndim; i++)
               assert(i == dim ? arg->dims[i] == 1 :
                    arg->dims[i] == src->dims[i]);
     }

     for (i = dim+1, thread_num = 1; i < dst->ndim; i++)
          thread_num *= dst->dims[i];
     reduce_vol = thread_num;
     index_vol = thread_num * src->dims[dim];
     for (i = 0; i < dim; i++)
          thread_num *= dst->dims[i];

     dtype = src->dtype;
     cmp = tl_gcmp_getfunc(dtype);
     dsize = tl_size_of(dtype);
     dim_size = src->dims[dim];
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
               tl_passign(nowp, 0, data_s, si+i*reduce_vol, dsize);
               if (cmp(nowp, maxp) > 0) {
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

tl_tensor *tl_tensor_mul(const tl_tensor *src1, const tl_tensor *src2,
                         tl_tensor *dst)
{
     int thread_num;
     int di;
     size_t dsize;
     tl_dtype dtype;
     void *s1_data, *s2_data, *d_data;
     void *mul_res;
     tl_gmul_func mul;

     assert(tl_tensor_issameshape(src1, src2));
     assert(src1->dtype == src2->dtype);
     if (dst) {
          assert(tl_tensor_issameshape(src1, dst));
          assert(src1->dtype == dst->dtype);
     } else {
          dst = tl_tensor_create(NULL, src1->ndim, src2->dims, src1->dtype);
     }

     thread_num = dst->len;
     s1_data = src1->data;
     s2_data = src2->data;
     d_data = dst->data;
     dtype = src1->dtype;
     dsize = tl_size_of(dtype);
     mul = tl_gmul_getfunc(dtype);
     mul_res = tl_alloc(dsize);
     for (di = 0; di < thread_num; di++) {
          mul(tl_padd(s1_data, di, dsize),
               tl_padd(s2_data, di, dsize), mul_res);
          tl_passign(d_data, di, mul_res, 0, dsize);
     }
     tl_free(mul_res);

     return dst;
}

/* (optional) workspace size equals (sizeof(int) * dst->ndim * dst->len), two of them */
tl_tensor *tl_tensor_transpose(const tl_tensor *src, tl_tensor *dst,
                              const int *dims, int **workspace)
{
     int i, j, found;
     int *s_ids, *d_ids, *s_dims, *d_dims;
     int thread_num;
     int di, si;
     int ndim;
     int *t_s_ids;
     int *t_d_ids;
     size_t dsize;
     void *s_data, *d_data;

     if (dst) {
          assert(src->dtype == dst->dtype);
          assert(src->len == dst->len);
          assert(src->ndim == dst->ndim);
#ifndef NDEBUG
          for (i = 0; i < dst->ndim; i++) {
               for (j = 0, found = 0; j < src->ndim; j++) {
                    if (dst->dims[i] == src->dims[j]) {
                         found = 1;
                         break;
                    }
               }
               assert(found && "tl_tensor_transpose: unmatched tensor shape of src and dst");
          }
#endif
     } else {
          dst = tl_tensor_create(NULL, src->ndim, dims, src->dtype);
     }

     thread_num = dst->len;
     s_dims = (int *)tl_clone(src->dims, sizeof(int) * src->ndim);
     d_dims = (int *)tl_clone(dst->dims, sizeof(int) * dst->ndim);
     if (!workspace) {
          s_ids = (int *)tl_alloc(sizeof(int) * dst->ndim * thread_num);
          d_ids = (int *)tl_alloc(sizeof(int) * dst->ndim * thread_num);
     } else {
          s_ids = workspace[0];
          d_ids = workspace[1];
     }

     dsize = tl_size_of(src->dtype);
     s_data = src->data;
     d_data = dst->data;
     ndim = dst->ndim;
     for (di = 0; di < thread_num; di++) {
          t_s_ids = s_ids + di * ndim;
          t_d_ids = d_ids + di * ndim;
          get_indexes(di, t_d_ids, ndim, d_dims);
          for (i = 0; i < ndim; i++)
               t_s_ids[dims[i]] = t_d_ids[i];
          si = get_index(t_s_ids, ndim, s_dims);

          tl_passign(d_data, di, s_data, si, dsize);
     }

     if (!workspace) {
          tl_free(s_ids);
          tl_free(d_ids);
     }
     tl_free(s_dims);
     tl_free(d_dims);

     return dst;
}
