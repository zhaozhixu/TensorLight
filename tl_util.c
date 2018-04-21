#include <errno.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

#include "tl_util.h"

void *tl_alloc(size_t size)
{
     void *p;

     p = malloc(size);
     if (p == NULL)
          tl_err_sys("malloc(%luz) failed", size);

     return p;
}

void *tl_clone(const void *src, size_t size)
{
     assert(src);
     void *p;
     p = tl_alloc(size);
     memmove(p, src, size);
     return p;
}

void *tl_repeat(void *data, size_t size, int times)
{
     assert(data && times > 0);
     void *p, *dst;
     int i;
     dst = p = tl_alloc(size * times);
     for (i = 0; i < times; i++, p = (char *)p + size * times)
          memmove(p, data, size);
     return dst;
}

int tl_compute_length(int ndim, int *dims)
{
     if (dims) {
          int i, len = 1;
          for (i = 0; i < ndim; i++)
               len *= dims[i];
          return len;
     }
     tl_err_msg("Warning: tl_compute_length: null dims\n");
     return 0;
}

size_t tl_size_of(tl_dtype dtype)
{
     size_t size;

     switch(dtype) {
     case TL_BOOL:
          size = sizeof(tl_bool_t);
          break;
     case TL_FLOAT:
          size = sizeof(float);
          break;
     case TL_INT32:
          size = sizeof(int32_t);
          break;
     case TL_INT16:
          size = sizeof(int16_t);
          break;
     case TL_INT8:
          size = sizeof(int8_t);
          break;
     case TL_UINT32:
          size = sizeof(uint32_t);
          break;
     case TL_UINT16:
          size = sizeof(uint16_t);
          break;
     case TL_UINT8:
          size = sizeof(uint8_t);
          break;
     default:
          tl_err_quit("ERROR: tl_size_of: unknown tl_dtype %d\n", dtype);
     }
     return size;
}

int tl_pointer_sub(void *p1, void *p2, tl_dtype dtype)
{
     switch (dtype) {
     case TL_BOOL:
          return (tl_bool_t *)p1 - (tl_bool_t *)p2;
     case TL_FLOAT:
          return (float *)p1 - (float *)p2;
     case TL_INT32:
          return (int32_t *)p1 - (int32_t *)p2;
     case TL_INT16:
          return (int16_t *)p1 - (int16_t *)p2;
     case TL_INT8:
          return (int8_t *)p1 - (int8_t *)p2;
     case TL_UINT32:
          return (uint32_t *)p1 - (uint32_t *)p2;
     case TL_UINT16:
          return (uint16_t *)p1 - (uint16_t *)p2;
     case TL_UINT8:
          return (uint8_t *)p1 - (uint8_t *)p2;
     default:
          tl_err_quit("ERROR: tl_pointer_sub: unknown tl_dtype %d\n", dtype);
     }
}

void *tl_pointer_add(void *p, int offset, tl_dtype dtype)
{
     switch (dtype) {
     case TL_BOOL:
          return (tl_bool_t *)p + offset;
     case TL_FLOAT:
          return (float *)p + offset;
     case TL_INT32:
          return (int32_t *)p + offset;
     case TL_INT16:
          return (int16_t *)p + offset;
     case TL_INT8:
          return (int8_t *)p + offset;
     case TL_UINT32:
          return (uint32_t *)p + offset;
     case TL_UINT16:
          return (uint16_t *)p + offset;
     case TL_UINT8:
          return (uint8_t *)p + offset;
     default:
          tl_err_quit("ERROR: tl_pointer_add: unknown tl_dtype %d\n",p dtype);
     }
}

static void err_doit(int errnoflag, int error, const char *fmt, va_list ap)
{
     char buf[TL_MAXLINE];

     vsnprintf(buf, TL_MAXLINE-1, fmt, ap);
     if (errnoflag)
          snprintf(buf+strlen(buf), TL_MAXLINE-strlen(buf)-1, ": %s",
               strerror(error));
     strcat(buf, "\n");
     fflush(stdout);
     fputs(buf, stderr);
     fflush(NULL);
}

/*
 * Nonfatal error unrelated to a system call.
 * Print a message and return.
 */
void tl_err_msg(const char *fmt, ...)
{
     va_list ap;
     va_start(ap, fmt);
     err_doit(0, 0, fmt, ap);
     va_end(ap);
}

/*
 * Nonfatal error unrelated to a system call.
 * Error code passed as explict parameter.
 * Print a message and return.
 */
void tl_err_cont(int error, const char *fmt, ...)
{
     va_list ap;
     va_start(ap, fmt);
     err_doit(1, error, fmt, ap);
     va_end(ap);
}

/*
 * Nonfatal error related to a system call.
 * Print a message and return.
 */
void tl_err_ret(const char *fmt, ...)
{
     va_list ap;
     va_start(ap, fmt);
     err_doit(1, errno, fmt, ap);
     va_end(ap);
}

/*
 * Fatal error unrelated to a system call.
 * Print a message and terminate.
 */
void tl_err_quit(const char *fmt, ...)
{
     va_list ap;
     va_start(ap, fmt);
     err_doit(0, 0, fmt, ap);
     va_end(ap);
     exit(1);
}

/*
 * Fatal error unrelated to a system call.
 * Error code passed as explict parameter.
 * Print a message and terminate.
 */
void tl_err_exit(int error, const char *fmt, ...)
{
     va_list
          ap;
     va_start(ap, fmt);
     err_doit(1, error, fmt, ap);
     va_end(ap);
     exit(1);
}

/*
 * Fatal error related to a system call.
 * Print a message and terminate.
 */
void tl_err_sys(const char *fmt, ...)
{
     va_list ap;
     va_start(ap, fmt);
     err_doit(1, errno, fmt, ap);
     va_end(ap);
     exit(1);
}

/*
 * Fatal error related to a system call.
 * Print a message, dump core, and terminate.
 */
void tl_err_dump(const char *fmt, ...)
{
     va_list ap;
     va_start(ap, fmt);
     err_doit(1, errno, fmt, ap);
     va_end(ap);
     abort();
/* dump core and terminate */
     exit(1);
/* shouldn’t get here */
}
