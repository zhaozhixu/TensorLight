#include "tl_util.h"
#include "tl_type.h"

/* TODO: maybe platform dependent */
static const size_t dtype_size[TL_DTYPE_SIZE] = {
     32, 32, 16, 8, 32, 16, 8, 32
};

static const char *dtype_fmt[TL_DTYPE_SIZE] = {
     "%.3f", "%d", "%d", "%d", "%u", "%u", "%u", "%d"
};


size_t tl_size_of(tl_dtype dtype)
{
     if (dtype < 0 || dtype >= TL_DTYPE_SIZE)
          tl_err_bt("ERROR: tl_size_of: unknown tl_dtype %d\n", dtype);
     return dtype_size[dtype];
}

char *tl_fmt(tl_dtype dtype)
{
     char *ret;

     if (dtype < 0 || dtype >= TL_DTYPE_SIZE)
          tl_err_bt("ERROR: tl_fmt: unknown tl_dtype %d\n", dtype);
     ret = (char *)tl_alloc(strlen(dtype_fmt[dtype]) + 1);
     strcpy(ret, dtype_fmt[dtype]);

     return ret;
}

static void gfprintf_float(FILE *fp, const char *fmt, void *p)
{
     fprintf(fp, fmt, *(float *)p);
}

static void gfprintf_int32(FILE *fp, const char *fmt, void *p)
{
     fprintf(fp, fmt, *(int32_t *)p);
}

static void gfprintf_int16(FILE *fp, const char *fmt, void *p)
{
     fprintf(fp, fmt, *(int16_t *)p);
}

static void gfprintf_int8(FILE *fp, const char *fmt, void *p)
{
     fprintf(fp, fmt, *(int8_t *)p);
}

static void gfprintf_uint32(FILE *fp, const char *fmt, void *p)
{
     fprintf(fp, fmt, *(uint16_t *)p);
}

static void gfprintf_uint16(FILE *fp, const char *fmt, void *p)
{
     fprintf(fp, fmt, *(uint16_t *)p);
}

static void gfprintf_uint8(FILE *fp, const char *fmt, void *p)
{
     fprintf(fp, fmt, *(uint8_t *)p);
}

static void gfprintf_bool(FILE *fp, const char *fmt, void *p)
{
     fprintf(fp, fmt, *(tl_bool_t *)p);
}

static tl_gfprintf_func gfprintf_func[TL_DTYPE_SIZE] = {
     gfprintf_float,
     gfprintf_int32,
     gfprintf_int16,
     gfprintf_int8,
     gfprintf_uint32,
     gfprintf_uint16,
     gfprintf_uint8,
     gfprintf_bool
};

void tl_gfprintf(FILE* fp, const char* fmt, void* p, tl_dtype dtype)
{
     if (dtype < 0 || dtype >= TL_DTYPE_SIZE)
          tl_err_bt("ERROR: tl_gmul: unknown tl_dtype %d\n", dtype);
     if (!fmt)
          (gfprintf_func[dtype])(fp, dtype_fmt[dtype], p);
     else
          (gfprintf_func[dtype])(fp, fmt, p);
}

tl_gfprintf_func tl_gfprintf_getfunc(tl_dtype dtype)
{
     return gfprintf_func[dtype];
}

static int gcmp_float(void *p1, void *p2)
{
     return *(float *)p1 - *(float *)p2;
}

static int gcmp_int32(void *p1, void *p2)
{
     return *(int32_t *)p1 - *(int32_t *)p2;
}

static int gcmp_int16(void *p1, void *p2)
{
     return *(int16_t *)p1 - *(int16_t *)p2;
}

static int gcmp_int8(void *p1, void *p2)
{
     return *(int8_t *)p1 - *(int8_t *)p2;
}

static int gcmp_uint32(void *p1, void *p2)
{
     return *(uint32_t *)p1 - *(uint32_t *)p2;
}

static int gcmp_uint16(void *p1, void *p2)
{
     return *(uint16_t *)p1 - *(uint16_t *)p2;
}

static int gcmp_uint8(void *p1, void *p2)
{
     return *(uint8_t *)p1 - *(uint8_t *)p2;
}

static int gcmp_bool(void *p1, void *p2)
{
     return *(tl_bool_t *)p1 - *(tl_bool_t *)p2;
}

static tl_gcmp_func gcmp_func[TL_DTYPE_SIZE] = {
     gcmp_float,
     gcmp_int32,
     gcmp_int16,
     gcmp_int8,
     gcmp_uint32,
     gcmp_uint16,
     gcmp_uint8,
     gcmp_bool
};

int tl_gcmp(void *p1, void *p2, tl_dtype dtype)
{
     if (dtype < 0 || dtype >= TL_DTYPE_SIZE)
          tl_err_bt("ERROR: tl_pointer_cmp: unknown tl_dtype %d\n", dtype);
     return (gcmp_func[dtype])(p1, p2);
}

tl_gcmp_func tl_gcmp_getfunc(tl_dtype dtype)
{
     return gcmp_func[dtype];
}

static void gmul_float(void *p1, void *p2, void *r)
{
     *(float *)r = *(float *)p1 * *(float *)p2;
}

static void gmul_int32(void *p1, void *p2, void *r)
{
     *(int32_t *)r = *(int32_t *)p1 * *(int32_t *)p2;
}

static void gmul_int16(void *p1, void *p2, void *r)
{
     *(int16_t *)r = *(int16_t *)p1 * *(int16_t *)p2;
}

static void gmul_int8(void *p1, void *p2, void *r)
{
     *(int8_t *)r = *(int8_t *)p1 * *(int8_t *)p2;
}

static void gmul_uint32(void *p1, void *p2, void *r)
{
     *(uint32_t *)r = *(uint32_t *)p1 * *(uint32_t *)p2;
}

static void gmul_uint16(void *p1, void *p2, void *r)
{
     *(uint16_t *)r = *(uint16_t *)p1 * *(uint16_t *)p2;
}

static void gmul_uint8(void *p1, void *p2, void *r)
{
     *(uint8_t *)r = *(uint8_t *)p1 * *(uint8_t *)p2;
}

static void gmul_bool(void *p1, void *p2, void *r)
{
     *(tl_bool_t *)r = *(tl_bool_t *)p1 * *(tl_bool_t *)p2;
}

static tl_gmul_func gmul_func[TL_DTYPE_SIZE] = {
     gmul_float,
     gmul_int32,
     gmul_int16,
     gmul_int8,
     gmul_uint32,
     gmul_uint16,
     gmul_uint8,
     gmul_bool
};

void tl_gmul(void *p1, void *p2, void *r, tl_dtype dtype)
{
     if (dtype < 0 || dtype >= TL_DTYPE_SIZE)
          tl_err_bt("ERROR: tl_gmul: unknown tl_dtype %d\n", dtype);
     (gmul_func[dtype])(p1, p2, r);
}

tl_gmul_func tl_gmul_getfunc(tl_dtype dtype)
{
     return gmul_func[dtype];
}
