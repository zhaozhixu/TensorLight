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

int tl_compute_length(int ndim, const int *dims)
{
     int i, len;

     if (dims) {
          for (i = 0, len = 1; i < ndim; i++) {
               if (dims[i] <= 0)
                    tl_err_bt("ERROR: tl_compute_length: dims[%d] = %d <= 0\n",
                              i, dims[i]);
               len *= dims[i];
          }
          return len;
     }
     tl_err_bt("ERROR: tl_compute_length: null dims\n");
     return -1;                 /* should not reach here */
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
 * Print a message, dump core, and terminate.
 */
void tl_err_bt(const char *fmt, ...)
{
     va_list ap;
     va_start(ap, fmt);
     err_doit(0, 0, fmt, ap);
     va_end(ap);
     abort();
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
