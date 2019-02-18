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

#include <math.h>
#include <limits.h>
#include <float.h>
#include <stdint.h>
#include <assert.h>
#include "tl_util.h"
#include "tl_type.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

static const size_t dtype_size[TL_DTYPE_SIZE] = {
    sizeof(double),
    sizeof(float),
    sizeof(int32_t),
    sizeof(int16_t),
    sizeof(int8_t),
    sizeof(uint32_t),
    sizeof(uint16_t),
    sizeof(uint8_t),
    sizeof(tl_bool_t)
};

static const char *dtype_fmt[TL_DTYPE_SIZE] = {
    "%.3f", "%.3f", "%d", "%d", "%d", "%u", "%u", "%u", "%d"
};

static const char *dtype_name[TL_DTYPE_SIZE] = {
    "TL_DOUBLE", "TL_FLOAT", "TL_INT32", "TL_INT16", "TL_INT8",
    "TL_UINT32", "TL_UINT16", "TL_UINT8", "TL_BOOL"
};

void tl_dtype_max(tl_dtype dtype, void *ret)
{
    assert(ret);
    switch (dtype) {
    case TL_DOUBLE:
        *(double *)ret = DBL_MAX;
        break;
    case TL_FLOAT:
        *(float *)ret = FLT_MAX;
        break;
    case TL_INT32:
        *(int32_t *)ret = INT32_MAX;
        break;
    case TL_INT16:
        *(int16_t *)ret = INT16_MAX;
        break;
    case TL_INT8:
        *(int8_t *)ret = INT8_MAX;
        break;
    case TL_UINT32:
        *(uint32_t *)ret = UINT32_MAX;
        break;
    case TL_UINT16:
        *(uint16_t *)ret = UINT16_MAX;
        break;
    case TL_UINT8:
        *(uint8_t *)ret = UINT8_MAX;
        break;
    case TL_BOOL:
        *(tl_bool_t *)ret = 1;
        break;
    default:
        assert(0 && "unsupported tl_dtype");
        break;
    }
}

void tl_dtype_min(tl_dtype dtype, void *ret)
{
    assert(ret);
    switch (dtype) {
    case TL_DOUBLE:
        *(double *)ret = -DBL_MAX;
        break;
    case TL_FLOAT:
        *(float *)ret = -FLT_MAX;
        break;
    case TL_INT32:
        *(int32_t *)ret = INT32_MIN;
        break;
    case TL_INT16:
        *(int16_t *)ret = INT16_MIN;
        break;
    case TL_INT8:
        *(int8_t *)ret = INT8_MIN;
        break;
    case TL_UINT32:
        *(uint32_t *)ret = 0;
        break;
    case TL_UINT16:
        *(uint16_t *)ret = 0;
        break;
    case TL_UINT8:
        *(uint8_t *)ret = 0;
        break;
    case TL_BOOL:
        *(tl_bool_t *)ret = 0;
        break;
    default:
        assert(0 && "unsupported tl_dtype");
        break;
    }
}

double tl_dtype_max_double(tl_dtype dtype)
{
    void *max;
    double max_d;
    max = tl_alloc(tl_size_of(dtype));
    tl_dtype_max(dtype, max);
    tl_convert(&max_d, TL_DOUBLE, max, dtype);
    tl_free(max);

    return max_d;
}

double tl_dtype_min_double(tl_dtype dtype)
{
    void *min;
    double min_d;
    min = tl_alloc(tl_size_of(dtype));
    tl_dtype_min(dtype, min);
    tl_convert(&min_d, TL_DOUBLE, min, dtype);
    tl_free(min);

    return min_d;
}

size_t tl_size_of(tl_dtype dtype)
{
    tl_check_dtype(dtype);
    return dtype_size[dtype];
}

const char *tl_dtype_fmt(tl_dtype dtype)
{
    tl_check_dtype(dtype);
    return dtype_fmt[dtype];
}

const char *tl_dtype_name(tl_dtype dtype)
{
    tl_check_dtype(dtype);
    return dtype_name[dtype];
}

tl_dtype tl_dtype_from_str(const char *str)
{
    if (!strcmp(str, "TL_DOUBLE"))
        return TL_DOUBLE;
    if (!strcmp(str, "TL_FLOAT"))
        return TL_FLOAT;
    if (!strcmp(str, "TL_INT32"))
        return TL_INT32;
    if (!strcmp(str, "TL_INT16"))
        return TL_INT16;
    if (!strcmp(str, "TL_INT8"))
        return TL_INT8;
    if (!strcmp(str, "TL_UINT32"))
        return TL_UINT32;
    if (!strcmp(str, "TL_UINT16"))
        return TL_UINT16;
    if (!strcmp(str, "TL_UINT8"))
        return TL_UINT8;
    if (!strcmp(str, "TL_BOOL"))
        return TL_BOOL;
    return -1;
}

/* tl_fprintf_func */
static int fprintf_double(FILE *fp, const char *fmt, void *p)
{
    if (!fmt)
        return fprintf(fp, dtype_fmt[TL_DOUBLE], *(double *)p);
    else
        return fprintf(fp, fmt, *(double *)p);
}

static int fprintf_float(FILE *fp, const char *fmt, void *p)
{
    if (!fmt)
        return fprintf(fp, dtype_fmt[TL_FLOAT], *(float *)p);
    else
        return fprintf(fp, fmt, *(float *)p);
}

static int fprintf_int32(FILE *fp, const char *fmt, void *p)
{
    if (!fmt)
        return fprintf(fp, dtype_fmt[TL_INT32], *(int32_t *)p);
    else
        return fprintf(fp, fmt, *(int32_t *)p);
}

static int fprintf_int16(FILE *fp, const char *fmt, void *p)
{
    if (!fmt)
        return fprintf(fp, dtype_fmt[TL_INT16], *(int16_t *)p);
    else
        return fprintf(fp, fmt, *(int16_t *)p);
}

static int fprintf_int8(FILE *fp, const char *fmt, void *p)
{
    if (!fmt)
        return fprintf(fp, dtype_fmt[TL_INT8], *(int8_t *)p);
    else
        return fprintf(fp, fmt, *(int8_t *)p);
}

static int fprintf_uint32(FILE *fp, const char *fmt, void *p)
{
    if (!fmt)
        return fprintf(fp, dtype_fmt[TL_UINT32], *(uint32_t *)p);
    else
        return fprintf(fp, fmt, *(uint16_t *)p);
}

static int fprintf_uint16(FILE *fp, const char *fmt, void *p)
{
    if (!fmt)
        return fprintf(fp, dtype_fmt[TL_UINT16], *(uint16_t *)p);
    else
        return fprintf(fp, fmt, *(uint16_t *)p);
}

static int fprintf_uint8(FILE *fp, const char *fmt, void *p)
{
    if (!fmt)
        return fprintf(fp, dtype_fmt[TL_UINT8], *(uint8_t *)p);
    else
        return fprintf(fp, fmt, *(uint8_t *)p);
}

static int fprintf_bool(FILE *fp, const char *fmt, void *p)
{
    if (!fmt)
        return fprintf(fp, dtype_fmt[TL_BOOL], *(tl_bool_t *)p);
    else
        return fprintf(fp, fmt, *(tl_bool_t *)p);
}

static tl_fprintf_func fprintf_func[TL_DTYPE_SIZE] = {
    fprintf_double,
    fprintf_float,
    fprintf_int32,
    fprintf_int16,
    fprintf_int8,
    fprintf_uint32,
    fprintf_uint16,
    fprintf_uint8,
    fprintf_bool
};

int tl_fprintf(FILE* fp, const char* fmt, void* p, tl_dtype dtype)
{
    tl_check_dtype(dtype);
    return fprintf_func[dtype](fp, fmt, p);
}

tl_fprintf_func tl_fprintf_getfunc(tl_dtype dtype)
{
    tl_check_dtype(dtype);
    return fprintf_func[dtype];
}

/* tl_cmp_func */
static int cmp_double(void *p1, void *p2)
{
    return *(double *)p1 - *(double *)p2;
}

static int cmp_float(void *p1, void *p2)
{
    return *(float *)p1 - *(float *)p2;
}

static int cmp_int32(void *p1, void *p2)
{
    return *(int32_t *)p1 - *(int32_t *)p2;
}

static int cmp_int16(void *p1, void *p2)
{
    return *(int16_t *)p1 - *(int16_t *)p2;
}

static int cmp_int8(void *p1, void *p2)
{
    return *(int8_t *)p1 - *(int8_t *)p2;
}

static int cmp_uint32(void *p1, void *p2)
{
    return *(uint32_t *)p1 - *(uint32_t *)p2;
}

static int cmp_uint16(void *p1, void *p2)
{
    return *(uint16_t *)p1 - *(uint16_t *)p2;
}

static int cmp_uint8(void *p1, void *p2)
{
    return *(uint8_t *)p1 - *(uint8_t *)p2;
}

static int cmp_bool(void *p1, void *p2)
{
    return *(tl_bool_t *)p1 - *(tl_bool_t *)p2;
}

static tl_cmp_func cmp_func[TL_DTYPE_SIZE] = {
    cmp_double,
    cmp_float,
    cmp_int32,
    cmp_int16,
    cmp_int8,
    cmp_uint32,
    cmp_uint16,
    cmp_uint8,
    cmp_bool
};

int tl_cmp(void *p1, void *p2, tl_dtype dtype)
{
    tl_check_dtype(dtype);
    return cmp_func[dtype](p1, p2);
}

tl_cmp_func tl_cmp_getfunc(tl_dtype dtype)
{
    tl_check_dtype(dtype);
    return cmp_func[dtype];
}

/* tl_elew_func */
typedef void (*elew_op_func) (void *p1, void *p2, void *res);

static const char *elew_op_name[TL_ELEW_OP_SIZE] = {
    "TL_MUL", "TL_DIV", "TL_SUM", "TL_SUB", "TL_MAX",
    "TL_MIN", "TL_POW"
};

tl_elew_op tl_elew_op_from_str(char *str)
{
    if (!strcmp(str, "TL_MUL"))
        return TL_MUL;
    if (!strcmp(str, "TL_DIV"))
        return TL_DIV;
    if (!strcmp(str, "TL_SUM"))
        return TL_SUM;
    if (!strcmp(str, "TL_SUB"))
        return TL_SUB;
    if (!strcmp(str, "TL_MAX"))
        return TL_MAX;
    if (!strcmp(str, "TL_MIN"))
        return TL_MIN;
    if (!strcmp(str, "TL_POW"))
        return TL_POW;
    return -1;
}

const char *tl_elew_op_name(tl_elew_op op)
{
    tl_check_elew_op(op);
    return elew_op_name[op];
}

static void mul_double(void *p1, void *p2, void *res)
{
    *(double *)res = *(double *)p1 * *(double *)p2;
}

static void div_double(void *p1, void *p2, void *res)
{
    assert(*(double *)p2);
    *(double *)res = *(double *)p1 / *(double *)p2;
}

static void sum_double(void *p1, void *p2, void *res)
{
    *(double *)res = *(double *)p1 + *(double *)p2;
}

static void sub_double(void *p1, void *p2, void *res)
{
    *(double *)res = *(double *)p1 - *(double *)p2;
}

static void max_double(void *p1, void *p2, void *res)
{
    *(double *)res = max(*(double *)p1, *(double *)p2);
}

static void min_double(void *p1, void *p2, void *res)
{
    *(double *)res = min(*(double *)p1, *(double *)p2);
}

static void pow_double(void *p1, void *p2, void *res)
{
    *(double *)res = pow(*(double *)p1, *(double *)p2);
}

static elew_op_func elew_op_double[TL_ELEW_OP_SIZE] = {
    mul_double,
    div_double,
    sum_double,
    sub_double,
    max_double,
    min_double,
    pow_double
};

static void elew_double(void *p1, void *p2, void *res, tl_elew_op elew_op)
{
    tl_check_elew_op(elew_op);
    elew_op_double[elew_op](p1, p2, res);
}

static void mul_float(void *p1, void *p2, void *res)
{
    *(float *)res = *(float *)p1 * *(float *)p2;
}

static void div_float(void *p1, void *p2, void *res)
{
    assert(*(float *)p2);
    *(float *)res = *(float *)p1 / *(float *)p2;
}

static void sum_float(void *p1, void *p2, void *res)
{
    *(float *)res = *(float *)p1 + *(float *)p2;
}

static void sub_float(void *p1, void *p2, void *res)
{
    *(float *)res = *(float *)p1 - *(float *)p2;
}

static void max_float(void *p1, void *p2, void *res)
{
    *(float *)res = max(*(float *)p1, *(float *)p2);
}

static void min_float(void *p1, void *p2, void *res)
{
    *(float *)res = min(*(float *)p1, *(float *)p2);
}

static void pow_float(void *p1, void *p2, void *res)
{
    *(float *)res = powf(*(float *)p1, *(float *)p2);
}

static elew_op_func elew_op_float[TL_ELEW_OP_SIZE] = {
    mul_float,
    div_float,
    sum_float,
    sub_float,
    max_float,
    min_float,
    pow_float
};

static void elew_float(void *p1, void *p2, void *res, tl_elew_op elew_op)
{
    tl_check_elew_op(elew_op);
    elew_op_float[elew_op](p1, p2, res);
}

static void mul_int32(void *p1, void *p2, void *res)
{
    *(int32_t *)res = *(int32_t *)p1 * *(int32_t *)p2;
}

static void div_int32(void *p1, void *p2, void *res)
{
    assert(*(int32_t *)p2);
    *(int32_t *)res = *(int32_t *)p1 / *(int32_t *)p2;
}

static void sum_int32(void *p1, void *p2, void *res)
{
    *(int32_t *)res = *(int32_t *)p1 + *(int32_t *)p2;
}

static void sub_int32(void *p1, void *p2, void *res)
{
    *(int32_t *)res = *(int32_t *)p1 - *(int32_t *)p2;
}

static void max_int32(void *p1, void *p2, void *res)
{
    *(int32_t *)res = max(*(int32_t *)p1, *(int32_t *)p2);
}

static void min_int32(void *p1, void *p2, void *res)
{
    *(int32_t *)res = min(*(int32_t *)p1, *(int32_t *)p2);
}

static void pow_int32(void *p1, void *p2, void *res)
{
    double d1, d2, dr;

    d1 = (double)*(int32_t *)p1;
    d2 = (double)*(int32_t *)p2;
    dr = pow(d1, d2);
    if (dr >= INT32_MAX)
        *(int32_t *)res = INT32_MAX;
    else if (dr <= INT32_MIN)
        *(int32_t *)res = INT32_MIN;
    else
        *(int32_t *)res = (int32_t)dr;
}

static elew_op_func elew_op_int32[TL_ELEW_OP_SIZE] = {
    mul_int32,
    div_int32,
    sum_int32,
    sub_int32,
    max_int32,
    min_int32,
    pow_int32
};

static void elew_int32(void *p1, void *p2, void *res, tl_elew_op elew_op)
{
    tl_check_elew_op(elew_op);
    elew_op_int32[elew_op](p1, p2, res);
}

static void mul_int16(void *p1, void *p2, void *res)
{
    *(int16_t *)res = *(int16_t *)p1 * *(int16_t *)p2;
}

static void div_int16(void *p1, void *p2, void *res)
{
    assert(*(int16_t *)p2);
    *(int16_t *)res = *(int16_t *)p1 / *(int16_t *)p2;
}

static void sum_int16(void *p1, void *p2, void *res)
{
    *(int16_t *)res = *(int16_t *)p1 + *(int16_t *)p2;
}

static void sub_int16(void *p1, void *p2, void *res)
{
    *(int16_t *)res = *(int16_t *)p1 - *(int16_t *)p2;
}

static void max_int16(void *p1, void *p2, void *res)
{
    *(int16_t *)res = max(*(int16_t *)p1, *(int16_t *)p2);
}

static void min_int16(void *p1, void *p2, void *res)
{
    *(int16_t *)res = min(*(int16_t *)p1, *(int16_t *)p2);
}

static void pow_int16(void *p1, void *p2, void *res)
{
    double d1, d2, dr;

    d1 = (double)*(int16_t *)p1;
    d2 = (double)*(int16_t *)p2;
    dr = pow(d1, d2);
    if (dr >= INT16_MAX)
        *(int16_t *)res = INT16_MAX;
    else if (dr <= INT16_MIN)
        *(int16_t *)res = INT16_MIN;
    else
        *(int16_t *)res = (int16_t)dr;
}

static elew_op_func elew_op_int16[TL_ELEW_OP_SIZE] = {
    mul_int16,
    div_int16,
    sum_int16,
    sub_int16,
    max_int16,
    min_int16,
    pow_int16
};

static void elew_int16(void *p1, void *p2, void *res, tl_elew_op elew_op)
{
    tl_check_elew_op(elew_op);
    elew_op_int16[elew_op](p1, p2, res);
}

static void mul_int8(void *p1, void *p2, void *res)
{
    *(int8_t *)res = *(int8_t *)p1 * *(int8_t *)p2;
}

static void div_int8(void *p1, void *p2, void *res)
{
    assert(*(int8_t *)p2);
    *(int8_t *)res = *(int8_t *)p1 / *(int8_t *)p2;
}

static void sum_int8(void *p1, void *p2, void *res)
{
    *(int8_t *)res = *(int8_t *)p1 + *(int8_t *)p2;
}

static void sub_int8(void *p1, void *p2, void *res)
{
    *(int8_t *)res = *(int8_t *)p1 - *(int8_t *)p2;
}

static void max_int8(void *p1, void *p2, void *res)
{
    *(int8_t *)res = max(*(int8_t *)p1, *(int8_t *)p2);
}

static void min_int8(void *p1, void *p2, void *res)
{
    *(int8_t *)res = min(*(int8_t *)p1, *(int8_t *)p2);
}

static void pow_int8(void *p1, void *p2, void *res)
{
    double d1, d2, dr;

    d1 = (double)*(int8_t *)p1;
    d2 = (double)*(int8_t *)p2;
    dr = pow(d1, d2);
    if (dr >= INT8_MAX)
        *(int8_t *)res = INT8_MAX;
    else if (dr <= INT8_MIN)
        *(int8_t *)res = INT8_MIN;
    else
        *(int8_t *)res = (int8_t)dr;
}

static elew_op_func elew_op_int8[TL_ELEW_OP_SIZE] = {
    mul_int8,
    div_int8,
    sum_int8,
    sub_int8,
    max_int8,
    min_int8,
    pow_int8
};

static void elew_int8(void *p1, void *p2, void *res, tl_elew_op elew_op)
{
    tl_check_elew_op(elew_op);
    elew_op_int8[elew_op](p1, p2, res);
}

static void mul_uint32(void *p1, void *p2, void *res)
{
    *(uint32_t *)res = *(uint32_t *)p1 * *(uint32_t *)p2;
}

static void div_uint32(void *p1, void *p2, void *res)
{
    assert(*(uint32_t *)p2);
    *(uint32_t *)res = *(uint32_t *)p1 / *(uint32_t *)p2;
}

static void sum_uint32(void *p1, void *p2, void *res)
{
    *(uint32_t *)res = *(uint32_t *)p1 + *(uint32_t *)p2;
}

static void sub_uint32(void *p1, void *p2, void *res)
{
    *(uint32_t *)res = *(uint32_t *)p1 - *(uint32_t *)p2;
}

static void max_uint32(void *p1, void *p2, void *res)
{
    *(uint32_t *)res = max(*(uint32_t *)p1, *(uint32_t *)p2);
}

static void min_uint32(void *p1, void *p2, void *res)
{
    *(uint32_t *)res = min(*(uint32_t *)p1, *(uint32_t *)p2);
}

static void pow_uint32(void *p1, void *p2, void *res)
{
    double d1, d2, dr;

    d1 = (double)*(uint32_t *)p1;
    d2 = (double)*(uint32_t *)p2;
    dr = pow(d1, d2);
    if (dr >= UINT32_MAX)
        *(uint32_t *)res = UINT32_MAX;
    else
        *(uint32_t *)res = (uint32_t)dr;
}

static elew_op_func elew_op_uint32[TL_ELEW_OP_SIZE] = {
    mul_uint32,
    div_uint32,
    sum_uint32,
    sub_uint32,
    max_uint32,
    min_uint32,
    pow_uint32
};

static void elew_uint32(void *p1, void *p2, void *res, tl_elew_op elew_op)
{
    tl_check_elew_op(elew_op);
    elew_op_uint32[elew_op](p1, p2, res);
}

static void mul_uint16(void *p1, void *p2, void *res)
{
    *(uint16_t *)res = *(uint16_t *)p1 * *(uint16_t *)p2;
}

static void div_uint16(void *p1, void *p2, void *res)
{
    assert(*(uint16_t *)p2);
    *(uint16_t *)res = *(uint16_t *)p1 / *(uint16_t *)p2;
}

static void sum_uint16(void *p1, void *p2, void *res)
{
    *(uint16_t *)res = *(uint16_t *)p1 + *(uint16_t *)p2;
}

static void sub_uint16(void *p1, void *p2, void *res)
{
    *(uint16_t *)res = *(uint16_t *)p1 - *(uint16_t *)p2;
}

static void max_uint16(void *p1, void *p2, void *res)
{
    *(uint16_t *)res = max(*(uint16_t *)p1, *(uint16_t *)p2);
}

static void min_uint16(void *p1, void *p2, void *res)
{
    *(uint16_t *)res = min(*(uint16_t *)p1, *(uint16_t *)p2);
}

static void pow_uint16(void *p1, void *p2, void *res)
{
    double d1, d2, dr;

    d1 = (double)*(uint16_t *)p1;
    d2 = (double)*(uint16_t *)p2;
    dr = pow(d1, d2);
    if (dr >= UINT16_MAX)
        *(uint16_t *)res = UINT16_MAX;
    else
        *(uint16_t *)res = (uint16_t)dr;
}

static elew_op_func elew_op_uint16[TL_ELEW_OP_SIZE] = {
    mul_uint16,
    div_uint16,
    sum_uint16,
    sub_uint16,
    max_uint16,
    min_uint16,
    pow_uint16
};

static void elew_uint16(void *p1, void *p2, void *res, tl_elew_op elew_op)
{
    tl_check_elew_op(elew_op);
    elew_op_uint16[elew_op](p1, p2, res);
}

static void mul_uint8(void *p1, void *p2, void *res)
{
    *(uint8_t *)res = *(uint8_t *)p1 * *(uint8_t *)p2;
}

static void div_uint8(void *p1, void *p2, void *res)
{
    assert(*(uint8_t *)p2);
    *(uint8_t *)res = *(uint8_t *)p1 / *(uint8_t *)p2;
}

static void sum_uint8(void *p1, void *p2, void *res)
{
    *(uint8_t *)res = *(uint8_t *)p1 + *(uint8_t *)p2;
}

static void sub_uint8(void *p1, void *p2, void *res)
{
    *(uint8_t *)res = *(uint8_t *)p1 - *(uint8_t *)p2;
}

static void max_uint8(void *p1, void *p2, void *res)
{
    *(uint8_t *)res = max(*(uint8_t *)p1, *(uint8_t *)p2);
}

static void min_uint8(void *p1, void *p2, void *res)
{
    *(uint8_t *)res = min(*(uint8_t *)p1, *(uint8_t *)p2);
}

static void pow_uint8(void *p1, void *p2, void *res)
{
    double d1, d2, dr;

    d1 = (double)*(uint8_t *)p1;
    d2 = (double)*(uint8_t *)p2;
    dr = pow(d1, d2);
    if (dr >= UINT8_MAX)
        *(uint8_t *)res = UINT8_MAX;
    else
        *(uint8_t *)res = (uint8_t)dr;
}

static elew_op_func elew_op_uint8[TL_ELEW_OP_SIZE] = {
    mul_uint8,
    div_uint8,
    sum_uint8,
    sub_uint8,
    max_uint8,
    min_uint8,
    pow_uint8
};

static void elew_uint8(void *p1, void *p2, void *res, tl_elew_op elew_op)
{
    tl_check_elew_op(elew_op);
    elew_op_uint8[elew_op](p1, p2, res);
}


static void mul_bool(void *p1, void *p2, void *res)
{
    int r = *(tl_bool_t *)p1 * *(tl_bool_t *)p2;
    if (r)
        *(tl_bool_t *)res = TL_TRUE;
    else
        *(tl_bool_t *)res = TL_FALSE;
}

static void div_bool(void *p1, void *p2, void *res)
{
    assert(*(tl_bool_t *)p2);
    int r = *(tl_bool_t *)p1 / *(tl_bool_t *)p2;
    if (r)
        *(tl_bool_t *)res = TL_TRUE;
    else
        *(tl_bool_t *)res = TL_FALSE;
}

static void sum_bool(void *p1, void *p2, void *res)
{
    int r = *(tl_bool_t *)p1 + *(tl_bool_t *)p2;
    if (r)
        *(tl_bool_t *)res = TL_TRUE;
    else
        *(tl_bool_t *)res = TL_FALSE;
}

static void sub_bool(void *p1, void *p2, void *res)
{
    int r = *(tl_bool_t *)p1 - *(tl_bool_t *)p2;
    if (r)
        *(tl_bool_t *)res = TL_TRUE;
    else
        *(tl_bool_t *)res = TL_FALSE;
}

static void max_bool(void *p1, void *p2, void *res)
{
    *(tl_bool_t *)res = max(*(tl_bool_t *)p1, *(tl_bool_t *)p2);
}

static void min_bool(void *p1, void *p2, void *res)
{
    *(tl_bool_t *)res = min(*(tl_bool_t *)p1, *(tl_bool_t *)p2);
}

static void pow_bool(void *p1, void *p2, void *res)
{
    double d1, d2, dr;

    d1 = (double)*(tl_bool_t *)p1;
    d2 = (double)*(tl_bool_t *)p2;
    dr = pow(d1, d2);
    if (dr > 0 || dr < 0)
        *(tl_bool_t *)res = TL_TRUE;
    else
        *(tl_bool_t *)res = TL_FALSE;
}

static elew_op_func elew_op_bool[TL_ELEW_OP_SIZE] = {
    mul_bool,
    div_bool,
    sum_bool,
    sub_bool,
    max_bool,
    min_bool,
    pow_bool
};

static void elew_bool(void *p1, void *p2, void *res, tl_elew_op elew_op)
{
    tl_check_elew_op(elew_op);
    elew_op_bool[elew_op](p1, p2, res);
}

static tl_elew_func elew_func[TL_DTYPE_SIZE] = {
    elew_double,
    elew_float,
    elew_int32,
    elew_int16,
    elew_int8,
    elew_uint32,
    elew_uint16,
    elew_uint8,
    elew_bool
};

void tl_elew(void *p1, void *p2, void *res, tl_elew_op elew_op, tl_dtype dtype)
{
    tl_check_dtype(dtype);
    elew_func[dtype](p1, p2, res, elew_op);
}

tl_elew_func tl_elew_getfunc(tl_dtype dtype)
{
    tl_check_dtype(dtype);
    return elew_func[dtype];
}

/* tl_lrelu */
#define LRELU(pd, ps, ns, type)                         \
    do {                                                \
        type _s = *(type *)(ps);                        \
        *(type *)(pd) = _s >= 0 ? _s : _s * (type)(ns); \
    } while (0)

void tl_lrelu(void *pd, const void *ps, float negslope, tl_dtype dtype)
{
    tl_check_dtype(dtype);

    switch (dtype) {
    case TL_DOUBLE:
        LRELU(pd, ps, negslope, double);
        break;
    case TL_FLOAT:
        LRELU(pd, ps, negslope, float);
        break;
    case TL_INT32:
        LRELU(pd, ps, negslope, int32_t);
        break;
    case TL_INT16:
        LRELU(pd, ps, negslope, int16_t);
        break;
    case TL_INT8:
        LRELU(pd, ps, negslope, int8_t);
        break;
    case TL_UINT32:
        LRELU(pd, ps, negslope, uint32_t);
        break;
    case TL_UINT16:
        LRELU(pd, ps, negslope, uint16_t);
        break;
    case TL_UINT8:
        LRELU(pd, ps, negslope, uint8_t);
        break;
    case TL_BOOL:
        LRELU(pd, ps, negslope, tl_bool_t);
        break;
    default:
        assert(0 && "unsupported tl_dtype");
        break;
    }
}
#undef LRELU

/* tl_convert */
void tl_convert(void *pd, tl_dtype dtype_d, const void *ps, tl_dtype dtype_s)
{
    tl_check_dtype(dtype_d);
    tl_check_dtype(dtype_s);

    double val_d;
    float val_f;
    int32_t val_i32;
    uint32_t val_u32;
    int16_t val_i16;
    uint16_t val_u16;
    int8_t val_i8;
    uint8_t val_u8;

    switch (dtype_d) {
    case TL_DOUBLE:
        switch (dtype_s) {
        case TL_DOUBLE:
            *(double *)pd = *(double *)ps;
            break;
        case TL_FLOAT:
            *(double *)pd = (double)*(float *)ps;
            break;
        case TL_INT32:
            *(double *)pd = (double)*(int32_t *)ps;
            break;
        case TL_INT16:
            *(double *)pd = (double)*(int16_t *)ps;
            break;
        case TL_INT8:
            *(double *)pd = (double)*(int8_t *)ps;
            break;
        case TL_UINT32:
            *(double *)pd = (double)*(uint32_t *)ps;
            break;
        case TL_UINT16:
            *(double *)pd = (double)*(uint16_t *)ps;
            break;
        case TL_UINT8:
            *(double *)pd = (double)*(uint8_t *)ps;
            break;
        case TL_BOOL:
            *(double *)pd = (double)*(tl_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_FLOAT:
        switch (dtype_s) {
        case TL_DOUBLE:
            val_d = *(double *)ps;
            if (val_d >= FLT_MAX)
                *(float *)pd = FLT_MAX;
            else if (val_d <= -FLT_MAX)
                *(float *)pd = -FLT_MAX;
            else
                *(float *)pd = (float)val_d;
            break;
        case TL_FLOAT:
            *(float *)pd = *(float *)ps;
            break;
        case TL_INT32:
            *(float *)pd = (float)*(int32_t *)ps;
            break;
        case TL_INT16:
            *(float *)pd = (float)*(int16_t *)ps;
            break;
        case TL_INT8:
            *(float *)pd = (float)*(int8_t *)ps;
            break;
        case TL_UINT32:
            *(float *)pd = (float)*(uint32_t *)ps;
            break;
        case TL_UINT16:
            *(float *)pd = (float)*(uint16_t *)ps;
            break;
        case TL_UINT8:
            *(float *)pd = (float)*(uint8_t *)ps;
            break;
        case TL_BOOL:
            *(float *)pd = (float)*(tl_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_INT32:
        switch (dtype_s) {
        case TL_DOUBLE:
            val_d = *(double *)ps;
            if (val_d >= INT32_MAX)
                *(int32_t *)pd = INT32_MAX;
            else if (val_d <= INT32_MIN)
                *(int32_t *)pd = INT32_MIN;
            else
                *(int32_t *)pd = (int32_t)val_d;
            break;
        case TL_FLOAT:
            val_f = *(float *)ps;
            if (val_f >= INT32_MAX)
                *(int32_t *)pd = INT32_MAX;
            else if (val_f <= INT32_MIN)
                *(int32_t *)pd = INT32_MIN;
            else
                *(int32_t *)pd = (int32_t)val_f;
            break;
        case TL_INT32:
            *(int32_t *)pd = *(int32_t *)ps;
            break;
        case TL_INT16:
            *(int32_t *)pd = (int32_t)*(int16_t *)ps;
            break;
        case TL_INT8:
            *(int32_t *)pd = (int32_t)*(int8_t *)ps;
            break;
        case TL_UINT32:
            val_u32 = *(uint32_t *)ps;
            if (val_u32 >= INT32_MAX)
                *(int32_t *)pd = INT32_MAX;
            else
                *(int32_t *)pd = (int32_t)val_u32;
            break;
        case TL_UINT16:
            /* printf("*ps = %d\n", *(uint16_t *)ps); */
            *(int32_t *)pd = (int32_t)*(uint16_t *)ps;
            /* printf("*pd = %d\n", *(int32_t *)pd); */
            break;
        case TL_UINT8:
            *(int32_t *)pd = (int32_t)*(uint8_t *)ps;
            break;
        case TL_BOOL:
            *(int32_t *)pd = (int32_t)*(tl_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_INT16:
        switch (dtype_s) {
        case TL_DOUBLE:
            val_d = *(double *)ps;
            if (val_d >= INT16_MAX)
                *(int16_t *)pd = INT16_MAX;
            else if (val_d <= INT16_MIN)
                *(int16_t *)pd = INT16_MIN;
            else
                *(int16_t *)pd = (int16_t)val_d;
            break;
        case TL_FLOAT:
            val_f = *(float *)ps;
            if (val_f >= INT16_MAX)
                *(int16_t *)pd = INT16_MAX;
            else if (val_f <= INT16_MIN)
                *(int16_t *)pd = INT16_MIN;
            else
                *(int16_t *)pd = (int16_t)val_f;
            break;
        case TL_INT32:
            val_i32 = *(int32_t *)ps;
            if (val_i32 >= INT16_MAX)
                *(int16_t *)pd = INT16_MAX;
            else if (val_i32 <= INT16_MIN)
                *(int16_t *)pd = INT16_MIN;
            else
                *(int16_t *)pd = (int16_t)val_i32;
            break;
        case TL_INT16:
            *(int16_t *)pd = *(int16_t *)ps;
            break;
        case TL_INT8:
            *(int16_t *)pd = (int16_t)*(int8_t *)ps;
            break;
        case TL_UINT32:
            val_u32 = *(uint32_t *)ps;
            if (val_u32 >= INT16_MAX)
                *(int16_t *)pd = INT16_MAX;
            else
                *(int16_t *)pd = (int16_t)val_u32;
            break;
        case TL_UINT16:
            val_u16 = *(uint16_t *)ps;
            if (val_u16 >= INT16_MAX)
                *(int16_t *)pd = INT16_MAX;
            else
                *(int16_t *)pd = (int16_t)val_u16;
            break;
        case TL_UINT8:
            *(int16_t *)pd = (int16_t)*(uint8_t *)ps;
            break;
        case TL_BOOL:
            *(int16_t *)pd = (int16_t)*(tl_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_INT8:
        switch (dtype_s) {
        case TL_DOUBLE:
            val_d = *(double *)ps;
            if (val_d >= INT8_MAX)
                *(int8_t *)pd = INT8_MAX;
            else if (val_d <= INT8_MIN)
                *(int8_t *)pd = INT8_MIN;
            else
                *(int8_t *)pd = (int8_t)val_d;
            break;
        case TL_FLOAT:
            val_f = *(float *)ps;
            if (val_f >= INT8_MAX)
                *(int8_t *)pd = INT8_MAX;
            else if (val_f <= INT8_MIN)
                *(int8_t *)pd = INT8_MIN;
            else
                *(int8_t *)pd = (int8_t)val_f;
            break;
        case TL_INT32:
            val_i32 = *(int32_t *)ps;
            if (val_i32 >= INT8_MAX)
                *(int8_t *)pd = INT8_MAX;
            else if (val_i32 <= INT8_MIN)
                *(int8_t *)pd = INT8_MIN;
            else
                *(int8_t *)pd = (int8_t)val_i32;
            break;
        case TL_INT16:
            val_i16 = *(int16_t *)ps;
            if (val_i16 >= INT8_MAX)
                *(int8_t *)pd = INT8_MAX;
            else if (val_i16 <= INT8_MIN)
                *(int8_t *)pd = INT8_MIN;
            else
                *(int8_t *)pd = (int8_t)val_i16;
            break;
        case TL_INT8:
            *(int8_t *)pd = *(int8_t *)ps;
            break;
        case TL_UINT32:
            val_u32 = *(uint32_t *)ps;
            if (val_u32 >= INT8_MAX)
                *(int8_t *)pd = INT8_MAX;
            else
                *(int8_t *)pd = (int8_t)val_u32;
            break;
        case TL_UINT16:
            val_u16 = *(uint16_t *)ps;
            if (val_u16 >= INT8_MAX)
                *(int8_t *)pd = INT8_MAX;
            else
                *(int8_t *)pd = (int8_t)val_u16;
            break;
        case TL_UINT8:
            val_u8 = *(uint8_t *)ps;
            if (val_u8 >= INT8_MAX)
                *(int8_t *)pd = INT8_MAX;
            else
                *(int8_t *)pd = (int8_t)val_u8;
            break;
        case TL_BOOL:
            *(int8_t *)pd = (int8_t)*(tl_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_UINT32:
        switch (dtype_s) {
        case TL_DOUBLE:
            val_d = *(double *)ps;
            if (val_d >= UINT32_MAX)
                *(uint32_t *)pd = UINT32_MAX;
            else if (val_d < 0)
                *(uint32_t *)pd = 0;
            else
                *(uint32_t *)pd = (uint32_t)val_d;
            break;
        case TL_FLOAT:
            val_f = *(float *)ps;
            if (val_f >= UINT32_MAX)
                *(uint32_t *)pd = UINT32_MAX;
            else if (val_f < 0)
                *(uint32_t *)pd = 0;
            else
                *(uint32_t *)pd = (uint32_t)val_f;
            break;
        case TL_INT32:
            val_i32 = *(int32_t *)ps;
            if (val_i32 >= 0)
                *(uint32_t *)pd = (uint32_t)val_i32;
            else
                *(uint32_t *)pd = 0;
            break;
        case TL_INT16:
            val_i16 = *(int16_t *)ps;
            if (val_i16 >= 0)
                *(uint32_t *)pd = (uint32_t)val_i16;
            else
                *(uint32_t *)pd = 0;
            break;
        case TL_INT8:
            val_i8 = *(int8_t *)ps;
            if (val_i8 >= 0)
                *(uint32_t *)pd = (uint32_t)val_i8;
            else
                *(uint32_t *)pd = 0;
            break;
        case TL_UINT32:
            *(uint32_t *)pd = *(uint32_t *)ps;
            break;
        case TL_UINT16:
            *(uint32_t *)pd = (uint32_t)*(uint16_t *)ps;
            break;
        case TL_UINT8:
            *(uint32_t *)pd = (uint32_t)*(uint8_t *)ps;
            break;
        case TL_BOOL:
            *(uint32_t *)pd = (uint32_t)*(tl_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_UINT16:
        switch (dtype_s) {
        case TL_DOUBLE:
            val_d = *(double *)ps;
            if (val_d >= UINT16_MAX)
                *(uint16_t *)pd = UINT16_MAX;
            else if (val_d < 0)
                *(uint16_t *)pd = 0;
            else
                *(uint16_t *)pd = (uint16_t)val_d;
            break;
        case TL_FLOAT:
            val_f = *(float *)ps;
            if (val_f >= UINT16_MAX)
                *(uint16_t *)pd = UINT16_MAX;
            else if (val_f < 0)
                *(uint16_t *)pd = 0;
            else
                *(uint16_t *)pd = (uint16_t)val_f;
            break;
        case TL_INT32:
            val_i32 = *(int32_t *)ps;
            if (val_i32 >= UINT16_MAX)
                *(uint16_t *)pd = UINT16_MAX;
            else if (val_i32 < 0)
                *(uint16_t *)pd = 0;
            else
                *(uint16_t *)pd = (uint16_t)val_i32;
            break;
        case TL_INT16:
            val_i16 = *(int16_t *)ps;
            if (val_i16 >= 0)
                *(uint16_t *)pd = (uint16_t)val_i16;
            else
                *(uint16_t *)pd = 0;
            break;
        case TL_INT8:
            val_i8 = *(int8_t *)ps;
            if (val_i8 >= 0)
                *(uint16_t *)pd = (uint16_t)val_i8;
            else
                *(uint16_t *)pd = 0;
            break;
        case TL_UINT32:
            val_u32 = *(uint32_t *)ps;
            if (val_u32 >= UINT16_MAX)
                *(uint16_t *)pd = UINT16_MAX;
            else
                *(uint16_t *)pd = (uint16_t)val_u32;
            break;
        case TL_UINT16:
            *(uint16_t *)pd = *(uint16_t *)ps;
            break;
        case TL_UINT8:
            *(uint16_t *)pd = (uint16_t)*(uint8_t *)ps;
            break;
        case TL_BOOL:
            *(uint16_t *)pd = (uint16_t)*(tl_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_UINT8:
        switch (dtype_s) {
        case TL_DOUBLE:
            val_d = *(double *)ps;
            if (val_d >= UINT8_MAX)
                *(uint8_t *)pd = UINT8_MAX;
            else if (val_d < 0)
                *(uint8_t *)pd = 0;
            else
                *(uint8_t *)pd = (uint8_t)val_d;
            break;
        case TL_FLOAT:
            val_f = *(float *)ps;
            if (val_f >= UINT8_MAX)
                *(uint8_t *)pd = UINT8_MAX;
            else if (val_f < 0)
                *(uint8_t *)pd = 0;
            else
                *(uint8_t *)pd = (uint8_t)val_f;
            break;
        case TL_INT32:
            val_i32 = *(int32_t *)ps;
            if (val_i32 >= UINT8_MAX)
                *(uint8_t *)pd = UINT8_MAX;
            else if (val_i32 < 0)
                *(uint8_t *)pd = 0;
            else
                *(uint8_t *)pd = (uint8_t)val_i32;
            break;
        case TL_INT16:
            val_i16 = *(int16_t *)ps;
            if (val_i16 >= UINT8_MAX)
                *(uint8_t *)pd = UINT8_MAX;
            else if (val_i16 < 0)
                *(uint8_t *)pd = 0;
            else
                *(uint8_t *)pd = (uint8_t)val_i16;
            break;
        case TL_INT8:
            val_i8 = *(int8_t *)ps;
            if (val_i8 >= 0)
                *(uint8_t *)pd = (uint8_t)val_i8;
            else
                *(uint8_t *)pd = 0;
            break;
        case TL_UINT32:
            val_u32 = *(uint32_t *)ps;
            if (val_u32 >= UINT8_MAX)
                *(uint8_t *)pd = UINT8_MAX;
            else
                *(uint8_t *)pd = (uint8_t)val_u32;
            break;
        case TL_UINT16:
            val_u16 = *(uint16_t *)ps;
            if (val_u16 >= UINT8_MAX)
                *(uint8_t *)pd = UINT8_MAX;
            else
                *(uint8_t *)pd = (uint8_t)val_u16;
            break;
        case TL_UINT8:
            *(uint8_t *)pd = *(uint8_t *)ps;
            break;
        case TL_BOOL:
            *(uint8_t *)pd = (uint8_t)*(tl_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    case TL_BOOL:
        switch (dtype_s) {
        case TL_DOUBLE:
            val_d = *(double *)ps;
            if (val_d > 0 || val_d < 0)
                *(tl_bool_t *)pd = TL_TRUE;
            else
                *(tl_bool_t *)pd = TL_FALSE;
            break;
        case TL_FLOAT:
            val_f = *(float *)ps;
            if (val_f > 0 || val_f < 0)
                *(tl_bool_t *)pd = TL_TRUE;
            else
                *(tl_bool_t *)pd = TL_FALSE;
            break;
        case TL_INT32:
            val_i32 = *(int32_t *)ps;
            if (val_i32)
                *(tl_bool_t *)pd = TL_TRUE;
            else
                *(tl_bool_t *)pd = TL_FALSE;
            break;
        case TL_INT16:
            val_i16 = *(int16_t *)ps;
            if (val_i16)
                *(tl_bool_t *)pd = TL_TRUE;
            else
                *(tl_bool_t *)pd = TL_FALSE;
            break;
        case TL_INT8:
            val_i8 = *(int8_t *)ps;
            if (val_i8)
                *(tl_bool_t *)pd = TL_TRUE;
            else
                *(tl_bool_t *)pd = TL_FALSE;
            break;
        case TL_UINT32:
            val_u32 = *(uint32_t *)ps;
            if (val_u32)
                *(tl_bool_t *)pd = TL_TRUE;
            else
                *(tl_bool_t *)pd = TL_FALSE;
            break;
        case TL_UINT16:
            val_u16 = *(uint16_t *)ps;
            if (val_u16)
                *(tl_bool_t *)pd = TL_TRUE;
            else
                *(tl_bool_t *)pd = TL_FALSE;
            break;
        case TL_UINT8:
            val_u8 = *(uint8_t *)ps;
            if (val_u8)
                *(tl_bool_t *)pd = TL_TRUE;
            else
                *(tl_bool_t *)pd = TL_FALSE;
            break;
        case TL_BOOL:
            *(tl_bool_t *)pd = *(tl_bool_t *)ps;
            break;
        default:
            assert(0 && "unsupported tl_dtype");
            break;
        }
        break;
    default:
        assert(0 && "unsupported tl_dtype");
        break;
    }
}

static const char *resize_type_name[TL_RESIZE_TYPE_SIZE] = {
    "TL_NEAREST", "TL_LINEAR"
};

const char *tl_resize_type_name(tl_resize_type rtype)
{
    tl_check_resize_type(rtype);
    return resize_type_name[rtype];
}

tl_resize_type tl_resize_type_from_str(const char *str)
{
    if (!strcmp(str, "TL_NEAREST"))
        return TL_NEAREST;
    if (!strcmp(str, "TL_LINEAR"))
        return TL_LINEAR;
    return -1;
}

static const char *sort_dir_name[TL_SORT_DIR_SIZE] = {
    "TL_SORT_DIR_ASCENDING", "TL_SORT_DIR_DESCENDING"
};

const char *tl_sort_dir_name(tl_sort_dir dir)
{
    tl_check_sort_dir(dir);
    return sort_dir_name[dir];
}

tl_sort_dir tl_sort_dir_from_str(const char *str)
{
    if (!strcmp(str, "TL_SORT_DIR_ASCENDING"))
        return TL_SORT_DIR_ASCENDING;
    if (!strcmp(str, "TL_SORT_DIR_DESCENDING"))
        return TL_SORT_DIR_DESCENDING;
    return -1;
}
