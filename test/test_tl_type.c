#include "test_tsl.h"
#include "../src/tl_type.h"
#include "../src/tl_util.h"

static void setup(void)
{
}

static void teardown(void)
{
}

START_TEST(test_tl_size_of)
{
     ck_assert_int_eq(tl_size_of(TL_FLOAT), 4);
     ck_assert_int_eq(tl_size_of(TL_INT32), 4);
     ck_assert_int_eq(tl_size_of(TL_INT16), 2);
     ck_assert_int_eq(tl_size_of(TL_INT8), 1);
     ck_assert_int_eq(tl_size_of(TL_UINT32), 4);
     ck_assert_int_eq(tl_size_of(TL_UINT16), 2);
     ck_assert_int_eq(tl_size_of(TL_UINT8), 1);
     ck_assert_int_eq(tl_size_of(TL_BOOL), 4);
}
END_TEST

START_TEST(test_tl_psub)
{
     float p_float[2];
     int32_t p_int32[2];
     int16_t p_int16[2];
     int8_t p_int8[2];
     uint32_t p_uint32[2];
     uint16_t p_uint16[2];
     uint8_t p_uint8[2];
     tl_bool_t p_bool[2];

     ck_assert_int_eq(tl_psub(&p_float[1], &p_float[0], tl_size_of(TL_FLOAT)), 1);
     ck_assert_int_eq(tl_psub(&p_float[0], &p_float[1], tl_size_of(TL_FLOAT)), -1);
     ck_assert_int_eq(tl_psub(&p_float[0], &p_float[0], tl_size_of(TL_FLOAT)), 0);

     ck_assert_int_eq(tl_psub(&p_int32[1], &p_int32[0], tl_size_of(TL_INT32)), 1);
     ck_assert_int_eq(tl_psub(&p_int32[0], &p_int32[1], tl_size_of(TL_INT32)), -1);
     ck_assert_int_eq(tl_psub(&p_int32[0], &p_int32[0], tl_size_of(TL_INT32)), 0);

     ck_assert_int_eq(tl_psub(&p_int16[1], &p_int16[0], tl_size_of(TL_INT16)), 1);
     ck_assert_int_eq(tl_psub(&p_int16[0], &p_int16[1], tl_size_of(TL_INT16)), -1);
     ck_assert_int_eq(tl_psub(&p_int16[0], &p_int16[0], tl_size_of(TL_INT16)), 0);

     ck_assert_int_eq(tl_psub(&p_int8[1], &p_int8[0], tl_size_of(TL_INT8)), 1);
     ck_assert_int_eq(tl_psub(&p_int8[0], &p_int8[1], tl_size_of(TL_INT8)), -1);
     ck_assert_int_eq(tl_psub(&p_int8[0], &p_int8[0], tl_size_of(TL_INT8)), 0);

     ck_assert_int_eq(tl_psub(&p_uint32[1], &p_uint32[0], tl_size_of(TL_UINT32)), 1);
     ck_assert_int_eq(tl_psub(&p_uint32[0], &p_uint32[1], tl_size_of(TL_UINT32)), -1);
     ck_assert_int_eq(tl_psub(&p_uint32[0], &p_uint32[0], tl_size_of(TL_UINT32)), 0);

     ck_assert_int_eq(tl_psub(&p_uint16[1], &p_uint16[0], tl_size_of(TL_UINT16)), 1);
     ck_assert_int_eq(tl_psub(&p_uint16[0], &p_uint16[1], tl_size_of(TL_UINT16)), -1);
     ck_assert_int_eq(tl_psub(&p_uint16[0], &p_uint16[0], tl_size_of(TL_UINT16)), 0);

     ck_assert_int_eq(tl_psub(&p_uint8[1], &p_uint8[0], tl_size_of(TL_UINT8)), 1);
     ck_assert_int_eq(tl_psub(&p_uint8[0], &p_uint8[1], tl_size_of(TL_UINT8)), -1);
     ck_assert_int_eq(tl_psub(&p_uint8[0], &p_uint8[0], tl_size_of(TL_UINT8)), 0);

     ck_assert_int_eq(tl_psub(&p_bool[1], &p_bool[0], tl_size_of(TL_BOOL)), 1);
     ck_assert_int_eq(tl_psub(&p_bool[0], &p_bool[1], tl_size_of(TL_BOOL)), -1);
     ck_assert_int_eq(tl_psub(&p_bool[0], &p_bool[0], tl_size_of(TL_BOOL)), 0);
}
END_TEST

START_TEST(test_tl_padd)
{
     float p_float[3];
     int32_t p_int32[3];
     int16_t p_int16[3];
     int8_t p_int8[3];
     uint32_t p_uint32[3];
     uint16_t p_uint16[3];
     uint8_t p_uint8[3];
     tl_bool_t p_bool[3];

     ck_assert_ptr_eq(tl_padd(&p_float[1], 1, tl_size_of(TL_FLOAT)), &p_float[2]);
     ck_assert_ptr_eq(tl_padd(&p_float[1], -1, tl_size_of(TL_FLOAT)), &p_float[0]);
     ck_assert_ptr_eq(tl_padd(&p_float[1], 0, tl_size_of(TL_FLOAT)), &p_float[1]);

     ck_assert_ptr_eq(tl_padd(&p_int32[1], 1, tl_size_of(TL_INT32)), &p_int32[2]);
     ck_assert_ptr_eq(tl_padd(&p_int32[1], -1, tl_size_of(TL_INT32)), &p_int32[0]);
     ck_assert_ptr_eq(tl_padd(&p_int32[1], 0, tl_size_of(TL_INT32)), &p_int32[1]);

     ck_assert_ptr_eq(tl_padd(&p_int16[1], 1, tl_size_of(TL_INT16)), &p_int16[2]);
     ck_assert_ptr_eq(tl_padd(&p_int16[1], -1, tl_size_of(TL_INT16)), &p_int16[0]);
     ck_assert_ptr_eq(tl_padd(&p_int16[1], 0, tl_size_of(TL_INT16)), &p_int16[1]);

     ck_assert_ptr_eq(tl_padd(&p_int8[1], 1, tl_size_of(TL_INT8)), &p_int8[2]);
     ck_assert_ptr_eq(tl_padd(&p_int8[1], -1, tl_size_of(TL_INT8)), &p_int8[0]);
     ck_assert_ptr_eq(tl_padd(&p_int8[1], 0, tl_size_of(TL_INT8)), &p_int8[1]);

     ck_assert_ptr_eq(tl_padd(&p_uint32[1], 1, tl_size_of(TL_UINT32)), &p_uint32[2]);
     ck_assert_ptr_eq(tl_padd(&p_uint32[1], -1, tl_size_of(TL_UINT32)), &p_uint32[0]);
     ck_assert_ptr_eq(tl_padd(&p_uint32[1], 0, tl_size_of(TL_UINT32)), &p_uint32[1]);

     ck_assert_ptr_eq(tl_padd(&p_uint16[1], 1, tl_size_of(TL_UINT16)), &p_uint16[2]);
     ck_assert_ptr_eq(tl_padd(&p_uint16[1], -1, tl_size_of(TL_UINT16)), &p_uint16[0]);
     ck_assert_ptr_eq(tl_padd(&p_uint16[1], 0, tl_size_of(TL_UINT16)), &p_uint16[1]);

     ck_assert_ptr_eq(tl_padd(&p_uint8[1], 1, tl_size_of(TL_UINT8)), &p_uint8[2]);
     ck_assert_ptr_eq(tl_padd(&p_uint8[1], -1, tl_size_of(TL_UINT8)), &p_uint8[0]);
     ck_assert_ptr_eq(tl_padd(&p_uint8[1], 0, tl_size_of(TL_UINT8)), &p_uint8[1]);

     ck_assert_ptr_eq(tl_padd(&p_bool[1], 1, tl_size_of(TL_BOOL)), &p_bool[2]);
     ck_assert_ptr_eq(tl_padd(&p_bool[1], -1, tl_size_of(TL_BOOL)), &p_bool[0]);
     ck_assert_ptr_eq(tl_padd(&p_bool[1], 0, tl_size_of(TL_BOOL)), &p_bool[1]);
}
END_TEST

START_TEST(test_tl_passign)
{
     float p_float[2] = {0, 1};
     int32_t p_int32[2] = {0, 1};
     int16_t p_int16[2] = {0, 1};
     int8_t p_int8[2] = {0, 1};
     uint32_t p_uint32[2] = {0, 1};
     uint16_t p_uint16[2] = {0, 1};
     uint8_t p_uint8[2] = {0, 1};
     tl_bool_t p_bool[2] = {0, 1};

     tl_passign(p_float, 0, p_float, 1, tl_size_of(TL_FLOAT));
     ck_assert(p_float[0] == p_float[1]);

     tl_passign(p_int32, 0, p_int32, 1, tl_size_of(TL_INT32));
     ck_assert(p_int32[0] == p_int32[1]);

     tl_passign(p_int16, 0, p_int16, 1, tl_size_of(TL_INT16));
     ck_assert(p_int16[0] == p_int16[1]);

     tl_passign(p_int8, 0, p_int8, 1, tl_size_of(TL_INT8));
     ck_assert(p_int8[0] == p_int8[1]);

     tl_passign(p_uint32, 0, p_uint32, 1, tl_size_of(TL_UINT32));
     ck_assert(p_uint32[0] == p_uint32[1]);

     tl_passign(p_uint16, 0, p_uint16, 1, tl_size_of(TL_UINT16));
     ck_assert(p_uint16[0] == p_uint16[1]);

     tl_passign(p_uint8, 0, p_uint8, 1, tl_size_of(TL_UINT8));
     ck_assert(p_uint8[0] == p_uint8[1]);

     tl_passign(p_bool, 0, p_bool, 1, tl_size_of(TL_BOOL));
     ck_assert(p_bool[0] == p_bool[1]);
}
END_TEST

START_TEST(test_tl_fmt)
{
     char *fmt;

     fmt = tl_fmt(TL_FLOAT);
     ck_assert_str_eq(fmt, "%.3f");
     tl_free(fmt);

     fmt = tl_fmt(TL_INT32);
     ck_assert_str_eq(fmt, "%d");
     tl_free(fmt);

     fmt = tl_fmt(TL_INT16);
     ck_assert_str_eq(fmt, "%d");
     tl_free(fmt);

     fmt = tl_fmt(TL_INT8);
     ck_assert_str_eq(fmt, "%d");
     tl_free(fmt);

     fmt = tl_fmt(TL_UINT32);
     ck_assert_str_eq(fmt, "%u");
     tl_free(fmt);

     fmt = tl_fmt(TL_UINT16);
     ck_assert_str_eq(fmt, "%u");
     tl_free(fmt);

     fmt = tl_fmt(TL_UINT8);
     ck_assert_str_eq(fmt, "%u");
     tl_free(fmt);

     fmt = tl_fmt(TL_BOOL);
     ck_assert_str_eq(fmt, "%d");
     tl_free(fmt);
}
END_TEST

START_TEST(test_tl_pointer_sub)
{
     float p_float[2];
     int32_t p_int32[2];
     int16_t p_int16[2];
     int8_t p_int8[2];
     uint32_t p_uint32[2];
     uint16_t p_uint16[2];
     uint8_t p_uint8[2];
     tl_bool_t p_bool[2];

     ck_assert_int_eq(tl_pointer_sub(&p_float[1], &p_float[0], (TL_FLOAT)), 1);
     ck_assert_int_eq(tl_pointer_sub(&p_float[0], &p_float[1], (TL_FLOAT)), -1);
     ck_assert_int_eq(tl_pointer_sub(&p_float[0], &p_float[0], (TL_FLOAT)), 0);

     ck_assert_int_eq(tl_pointer_sub(&p_int32[1], &p_int32[0], (TL_INT32)), 1);
     ck_assert_int_eq(tl_pointer_sub(&p_int32[0], &p_int32[1], (TL_INT32)), -1);
     ck_assert_int_eq(tl_pointer_sub(&p_int32[0], &p_int32[0], (TL_INT32)), 0);

     ck_assert_int_eq(tl_pointer_sub(&p_int16[1], &p_int16[0], (TL_INT16)), 1);
     ck_assert_int_eq(tl_pointer_sub(&p_int16[0], &p_int16[1], (TL_INT16)), -1);
     ck_assert_int_eq(tl_pointer_sub(&p_int16[0], &p_int16[0], (TL_INT16)), 0);

     ck_assert_int_eq(tl_pointer_sub(&p_int8[1], &p_int8[0], (TL_INT8)), 1);
     ck_assert_int_eq(tl_pointer_sub(&p_int8[0], &p_int8[1], (TL_INT8)), -1);
     ck_assert_int_eq(tl_pointer_sub(&p_int8[0], &p_int8[0], (TL_INT8)), 0);

     ck_assert_int_eq(tl_pointer_sub(&p_uint32[1], &p_uint32[0], (TL_UINT32)), 1);
     ck_assert_int_eq(tl_pointer_sub(&p_uint32[0], &p_uint32[1], (TL_UINT32)), -1);
     ck_assert_int_eq(tl_pointer_sub(&p_uint32[0], &p_uint32[0], (TL_UINT32)), 0);

     ck_assert_int_eq(tl_pointer_sub(&p_uint16[1], &p_uint16[0], (TL_UINT16)), 1);
     ck_assert_int_eq(tl_pointer_sub(&p_uint16[0], &p_uint16[1], (TL_UINT16)), -1);
     ck_assert_int_eq(tl_pointer_sub(&p_uint16[0], &p_uint16[0], (TL_UINT16)), 0);

     ck_assert_int_eq(tl_pointer_sub(&p_uint8[1], &p_uint8[0], (TL_UINT8)), 1);
     ck_assert_int_eq(tl_pointer_sub(&p_uint8[0], &p_uint8[1], (TL_UINT8)), -1);
     ck_assert_int_eq(tl_pointer_sub(&p_uint8[0], &p_uint8[0], (TL_UINT8)), 0);

     ck_assert_int_eq(tl_pointer_sub(&p_bool[1], &p_bool[0], (TL_BOOL)), 1);
     ck_assert_int_eq(tl_pointer_sub(&p_bool[0], &p_bool[1], (TL_BOOL)), -1);
     ck_assert_int_eq(tl_pointer_sub(&p_bool[0], &p_bool[0], (TL_BOOL)), 0);
}
END_TEST

START_TEST(test_tl_pointer_add)
{
     float p_float[3];
     int32_t p_int32[3];
     int16_t p_int16[3];
     int8_t p_int8[3];
     uint32_t p_uint32[3];
     uint16_t p_uint16[3];
     uint8_t p_uint8[3];
     tl_bool_t p_bool[3];

     ck_assert_ptr_eq(tl_pointer_add(&p_float[1], 1, (TL_FLOAT)), &p_float[2]);
     ck_assert_ptr_eq(tl_pointer_add(&p_float[1], -1, (TL_FLOAT)), &p_float[0]);
     ck_assert_ptr_eq(tl_pointer_add(&p_float[1], 0, (TL_FLOAT)), &p_float[1]);

     ck_assert_ptr_eq(tl_pointer_add(&p_int32[1], 1, (TL_INT32)), &p_int32[2]);
     ck_assert_ptr_eq(tl_pointer_add(&p_int32[1], -1, (TL_INT32)), &p_int32[0]);
     ck_assert_ptr_eq(tl_pointer_add(&p_int32[1], 0, (TL_INT32)), &p_int32[1]);

     ck_assert_ptr_eq(tl_pointer_add(&p_int16[1], 1, (TL_INT16)), &p_int16[2]);
     ck_assert_ptr_eq(tl_pointer_add(&p_int16[1], -1, (TL_INT16)), &p_int16[0]);
     ck_assert_ptr_eq(tl_pointer_add(&p_int16[1], 0, (TL_INT16)), &p_int16[1]);

     ck_assert_ptr_eq(tl_pointer_add(&p_int8[1], 1, (TL_INT8)), &p_int8[2]);
     ck_assert_ptr_eq(tl_pointer_add(&p_int8[1], -1, (TL_INT8)), &p_int8[0]);
     ck_assert_ptr_eq(tl_pointer_add(&p_int8[1], 0, (TL_INT8)), &p_int8[1]);

     ck_assert_ptr_eq(tl_pointer_add(&p_uint32[1], 1, (TL_UINT32)), &p_uint32[2]);
     ck_assert_ptr_eq(tl_pointer_add(&p_uint32[1], -1, (TL_UINT32)), &p_uint32[0]);
     ck_assert_ptr_eq(tl_pointer_add(&p_uint32[1], 0, (TL_UINT32)), &p_uint32[1]);

     ck_assert_ptr_eq(tl_pointer_add(&p_uint16[1], 1, (TL_UINT16)), &p_uint16[2]);
     ck_assert_ptr_eq(tl_pointer_add(&p_uint16[1], -1, (TL_UINT16)), &p_uint16[0]);
     ck_assert_ptr_eq(tl_pointer_add(&p_uint16[1], 0, (TL_UINT16)), &p_uint16[1]);

     ck_assert_ptr_eq(tl_pointer_add(&p_uint8[1], 1, (TL_UINT8)), &p_uint8[2]);
     ck_assert_ptr_eq(tl_pointer_add(&p_uint8[1], -1, (TL_UINT8)), &p_uint8[0]);
     ck_assert_ptr_eq(tl_pointer_add(&p_uint8[1], 0, (TL_UINT8)), &p_uint8[1]);

     ck_assert_ptr_eq(tl_pointer_add(&p_bool[1], 1, (TL_BOOL)), &p_bool[2]);
     ck_assert_ptr_eq(tl_pointer_add(&p_bool[1], -1, (TL_BOOL)), &p_bool[0]);
     ck_assert_ptr_eq(tl_pointer_add(&p_bool[1], 0, (TL_BOOL)), &p_bool[1]);
}
END_TEST

START_TEST(test_tl_pointer_assign)
{
     float p_float[2] = {0, 1};
     int32_t p_int32[2] = {0, 1};
     int16_t p_int16[2] = {0, 1};
     int8_t p_int8[2] = {0, 1};
     uint32_t p_uint32[2] = {0, 1};
     uint16_t p_uint16[2] = {0, 1};
     uint8_t p_uint8[2] = {0, 1};
     tl_bool_t p_bool[2] = {0, 1};

     tl_pointer_assign(p_float, 0, p_float, 1, (TL_FLOAT));
     ck_assert(p_float[0] == p_float[1]);

     tl_pointer_assign(p_int32, 0, p_int32, 1, (TL_INT32));
     ck_assert(p_int32[0] == p_int32[1]);

     tl_pointer_assign(p_int16, 0, p_int16, 1, (TL_INT16));
     ck_assert(p_int16[0] == p_int16[1]);

     tl_pointer_assign(p_int8, 0, p_int8, 1, (TL_INT8));
     ck_assert(p_int8[0] == p_int8[1]);

     tl_pointer_assign(p_uint32, 0, p_uint32, 1, (TL_UINT32));
     ck_assert(p_uint32[0] == p_uint32[1]);

     tl_pointer_assign(p_uint16, 0, p_uint16, 1, (TL_UINT16));
     ck_assert(p_uint16[0] == p_uint16[1]);

     tl_pointer_assign(p_uint8, 0, p_uint8, 1, (TL_UINT8));
     ck_assert(p_uint8[0] == p_uint8[1]);

     tl_pointer_assign(p_bool, 0, p_bool, 1, (TL_BOOL));
     ck_assert(p_bool[0] == p_bool[1]);
}
END_TEST

START_TEST(test_tl_gfprintf)
{
     FILE *fp;
     float val_float = 1.2345;
     int32_t val_int32 = -1;
     int16_t val_int16 = -1;
     int8_t val_int8 = -1;
     uint32_t val_uint32 = 1;
     uint16_t val_uint16 = 1;
     uint8_t val_uint8 = 1;
     tl_bool_t val_bool = TL_TRUE;
     char s[10];

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     ck_assert_int_ge(tl_gfprintf(fp, NULL, &val_float, TL_FLOAT), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "1.235");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     ck_assert_int_ge(tl_gfprintf(fp, "%.1f", &val_float, TL_FLOAT), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "1.2");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     ck_assert_int_ge(tl_gfprintf(fp, NULL, &val_int32, TL_INT32), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "-1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     ck_assert_int_ge(tl_gfprintf(fp, NULL, &val_int16, TL_INT16), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "-1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     ck_assert_int_ge(tl_gfprintf(fp, NULL, &val_int8, TL_INT8), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "-1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     ck_assert_int_ge(tl_gfprintf(fp, NULL, &val_uint32, TL_UINT32), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     ck_assert_int_ge(tl_gfprintf(fp, NULL, &val_uint16, TL_UINT16), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     ck_assert_int_ge(tl_gfprintf(fp, NULL, &val_uint8, TL_UINT8), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     ck_assert_int_ge(tl_gfprintf(fp, NULL, &val_bool, TL_BOOL), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "1");
     fclose(fp);
}
END_TEST

START_TEST(test_tl_gfprintf_getfunc)
{
     FILE *fp;
     float val_float = 1.2345;
     int32_t val_int32 = -1;
     int16_t val_int16 = -1;
     int8_t val_int8 = -1;
     uint32_t val_uint32 = 1;
     uint16_t val_uint16 = 1;
     uint8_t val_uint8 = 1;
     tl_bool_t val_bool = TL_TRUE;
     tl_gfprintf_func gfprintf_func;
     char s[10];

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     gfprintf_func = tl_gfprintf_getfunc(TL_FLOAT);
     ck_assert_int_ge(gfprintf_func(fp, NULL, &val_float), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "1.235");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     gfprintf_func = tl_gfprintf_getfunc(TL_FLOAT);
     ck_assert_int_ge(gfprintf_func(fp, "%.1f", &val_float), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "1.2");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     gfprintf_func = tl_gfprintf_getfunc(TL_INT32);
     ck_assert_int_ge(gfprintf_func(fp, NULL, &val_int32), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "-1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     gfprintf_func = tl_gfprintf_getfunc(TL_INT16);
     ck_assert_int_ge(gfprintf_func(fp, NULL, &val_int16), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "-1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     gfprintf_func = tl_gfprintf_getfunc(TL_INT8);
     ck_assert_int_ge(gfprintf_func(fp, NULL, &val_int8), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "-1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     gfprintf_func = tl_gfprintf_getfunc(TL_UINT32);
     ck_assert_int_ge(gfprintf_func(fp, NULL, &val_uint32), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     gfprintf_func = tl_gfprintf_getfunc(TL_UINT16);
     ck_assert_int_ge(gfprintf_func(fp, NULL, &val_uint16), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     gfprintf_func = tl_gfprintf_getfunc(TL_UINT8);
     ck_assert_int_ge(gfprintf_func(fp, NULL, &val_uint8), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "1");
     fclose(fp);

     fp = tmpfile();
     ck_assert_ptr_ne(fp, NULL);
     gfprintf_func = tl_gfprintf_getfunc(TL_BOOL);
     ck_assert_int_ge(gfprintf_func(fp, NULL, &val_bool), 0);
     rewind(fp);
     ck_assert_ptr_ne(fgets(s, 10, fp), NULL);
     ck_assert_str_eq(s, "1");
     fclose(fp);
}
END_TEST

START_TEST(test_tl_gcmp)
{
     float val1_float = 1, val2_float = 2;
     int32_t val1_int32 = 1, val2_int32 = 2;
     int16_t val1_int16 = 1, val2_int16 = 2;
     int8_t val1_int8 = 1, val2_int8 = 2;
     uint32_t val1_uint32 = 1, val2_uint32 = 2;
     uint16_t val1_uint16 = 1, val2_uint16 = 2;
     uint8_t val1_uint8 = 1, val2_uint8 = 2;
     tl_bool_t val1_bool = TL_FALSE, val2_bool = TL_TRUE;

     ck_assert(tl_gcmp(&val1_float, &val2_float, TL_FLOAT) < 0);
     ck_assert(tl_gcmp(&val2_float, &val1_float, TL_FLOAT) > 0);
     ck_assert(tl_gcmp(&val1_float, &val1_float, TL_FLOAT) == 0);

     ck_assert(tl_gcmp(&val1_int32, &val2_int32, TL_INT32) < 0);
     ck_assert(tl_gcmp(&val2_int32, &val1_int32, TL_INT32) > 0);
     ck_assert(tl_gcmp(&val1_int32, &val1_int32, TL_INT32) == 0);

     ck_assert(tl_gcmp(&val1_int16, &val2_int16, TL_INT16) < 0);
     ck_assert(tl_gcmp(&val2_int16, &val1_int16, TL_INT16) > 0);
     ck_assert(tl_gcmp(&val1_int16, &val1_int16, TL_INT16) == 0);

     ck_assert(tl_gcmp(&val1_int8, &val2_int8, TL_INT8) < 0);
     ck_assert(tl_gcmp(&val2_int8, &val1_int8, TL_INT8) > 0);
     ck_assert(tl_gcmp(&val1_int8, &val1_int8, TL_INT8) == 0);

     ck_assert(tl_gcmp(&val1_uint32, &val2_uint32, TL_UINT32) < 0);
     ck_assert(tl_gcmp(&val2_uint32, &val1_uint32, TL_UINT32) > 0);
     ck_assert(tl_gcmp(&val1_uint32, &val1_uint32, TL_UINT32) == 0);

     ck_assert(tl_gcmp(&val1_uint16, &val2_uint16, TL_UINT16) < 0);
     ck_assert(tl_gcmp(&val2_uint16, &val1_uint16, TL_UINT16) > 0);
     ck_assert(tl_gcmp(&val1_uint16, &val1_uint16, TL_UINT16) == 0);

     ck_assert(tl_gcmp(&val1_uint8, &val2_uint8, TL_UINT8) < 0);
     ck_assert(tl_gcmp(&val2_uint8, &val1_uint8, TL_UINT8) > 0);
     ck_assert(tl_gcmp(&val1_uint8, &val1_uint8, TL_UINT8) == 0);

     ck_assert(tl_gcmp(&val1_bool, &val2_bool, TL_BOOL) < 0);
     ck_assert(tl_gcmp(&val2_bool, &val1_bool, TL_BOOL) > 0);
     ck_assert(tl_gcmp(&val1_bool, &val1_bool, TL_BOOL) == 0);
}
END_TEST

START_TEST(test_tl_gcmp_getfunc)
{
     float val1_float = 1, val2_float = 2;
     int32_t val1_int32 = 1, val2_int32 = 2;
     int16_t val1_int16 = 1, val2_int16 = 2;
     int8_t val1_int8 = 1, val2_int8 = 2;
     uint32_t val1_uint32 = 1, val2_uint32 = 2;
     uint16_t val1_uint16 = 1, val2_uint16 = 2;
     uint8_t val1_uint8 = 1, val2_uint8 = 2;
     tl_bool_t val1_bool = TL_FALSE, val2_bool = TL_TRUE;
     tl_gcmp_func gcmp_func;

     gcmp_func = tl_gcmp_getfunc(TL_FLOAT);
     ck_assert(gcmp_func(&val1_float, &val2_float) < 0);
     ck_assert(gcmp_func(&val2_float, &val1_float) > 0);
     ck_assert(gcmp_func(&val1_float, &val1_float) == 0);

     gcmp_func = tl_gcmp_getfunc(TL_INT32);
     ck_assert(gcmp_func(&val1_int32, &val2_int32) < 0);
     ck_assert(gcmp_func(&val2_int32, &val1_int32) > 0);
     ck_assert(gcmp_func(&val1_int32, &val1_int32) == 0);

     gcmp_func = tl_gcmp_getfunc(TL_INT16);
     ck_assert(gcmp_func(&val1_int16, &val2_int16) < 0);
     ck_assert(gcmp_func(&val2_int16, &val1_int16) > 0);
     ck_assert(gcmp_func(&val1_int16, &val1_int16) == 0);

     gcmp_func = tl_gcmp_getfunc(TL_INT8);
     ck_assert(gcmp_func(&val1_int8, &val2_int8) < 0);
     ck_assert(gcmp_func(&val2_int8, &val1_int8) > 0);
     ck_assert(gcmp_func(&val1_int8, &val1_int8) == 0);

     gcmp_func = tl_gcmp_getfunc(TL_UINT32);
     ck_assert(gcmp_func(&val1_uint32, &val2_uint32) < 0);
     ck_assert(gcmp_func(&val2_uint32, &val1_uint32) > 0);
     ck_assert(gcmp_func(&val1_uint32, &val1_uint32) == 0);

     gcmp_func = tl_gcmp_getfunc(TL_UINT16);
     ck_assert(gcmp_func(&val1_uint16, &val2_uint16) < 0);
     ck_assert(gcmp_func(&val2_uint16, &val1_uint16) > 0);
     ck_assert(gcmp_func(&val1_uint16, &val1_uint16) == 0);

     gcmp_func = tl_gcmp_getfunc(TL_UINT8);
     ck_assert(gcmp_func(&val1_uint8, &val2_uint8) < 0);
     ck_assert(gcmp_func(&val2_uint8, &val1_uint8) > 0);
     ck_assert(gcmp_func(&val1_uint8, &val1_uint8) == 0);

     gcmp_func = tl_gcmp_getfunc(TL_BOOL);
     ck_assert(gcmp_func(&val1_bool, &val2_bool) < 0);
     ck_assert(gcmp_func(&val2_bool, &val1_bool) > 0);
     ck_assert(gcmp_func(&val1_bool, &val1_bool) == 0);
}
END_TEST

START_TEST(test_tl_gmul)
{
     float val1_float = 1, val2_float = 2, val3_float;
     int32_t val1_int32 = 1, val2_int32 = 2, val3_int32;
     int16_t val1_int16 = 1, val2_int16 = 2, val3_int16;
     int8_t val1_int8 = 1, val2_int8 = 2, val3_int8;
     uint32_t val1_uint32 = 1, val2_uint32 = 2, val3_uint32;
     uint16_t val1_uint16 = 1, val2_uint16 = 2, val3_uint16;
     uint8_t val1_uint8 = 1, val2_uint8 = 2, val3_uint8;
     tl_bool_t val1_bool = TL_FALSE, val2_bool = TL_TRUE, val3_bool;

     tl_gmul(&val1_float, &val2_float, &val3_float, TL_FLOAT);
     ck_assert(val3_float == 2);

     tl_gmul(&val1_int32, &val2_int32, &val3_int32, TL_INT32);
     ck_assert(val3_int32 == 2);

     tl_gmul(&val1_int16, &val2_int16, &val3_int16, TL_INT16);
     ck_assert(val3_int16 == 2);

     tl_gmul(&val1_int8, &val2_int8, &val3_int8, TL_INT8);
     ck_assert(val3_int8 == 2);

     tl_gmul(&val1_uint32, &val2_uint32, &val3_uint32, TL_UINT32);
     ck_assert(val3_uint32 == 2);

     tl_gmul(&val1_uint16, &val2_uint16, &val3_uint16, TL_UINT16);
     ck_assert(val3_uint16 == 2);

     tl_gmul(&val1_uint8, &val2_uint8, &val3_uint8, TL_UINT8);
     ck_assert(val3_uint8 == 2);

     tl_gmul(&val1_bool, &val2_bool, &val3_bool, TL_BOOL);
     ck_assert(val3_bool == TL_FALSE);
}
END_TEST

START_TEST(test_tl_gmul_getfunc)
{
     float val1_float = 1, val2_float = 2, val3_float;
     int32_t val1_int32 = 1, val2_int32 = 2, val3_int32;
     int16_t val1_int16 = 1, val2_int16 = 2, val3_int16;
     int8_t val1_int8 = 1, val2_int8 = 2, val3_int8;
     uint32_t val1_uint32 = 1, val2_uint32 = 2, val3_uint32;
     uint16_t val1_uint16 = 1, val2_uint16 = 2, val3_uint16;
     uint8_t val1_uint8 = 1, val2_uint8 = 2, val3_uint8;
     tl_bool_t val1_bool = TL_FALSE, val2_bool = TL_TRUE, val3_bool;
     tl_gmul_func mul_func;

     mul_func = tl_gmul_getfunc(TL_FLOAT);
     mul_func(&val1_float, &val2_float, &val3_float);
     ck_assert(val3_float == 2);

     mul_func = tl_gmul_getfunc(TL_INT32);
     mul_func(&val1_int32, &val2_int32, &val3_int32);
     ck_assert(val3_int32 == 2);

     mul_func = tl_gmul_getfunc(TL_INT16);
     mul_func(&val1_int16, &val2_int16, &val3_int16);
     ck_assert(val3_int16 == 2);

     mul_func = tl_gmul_getfunc(TL_INT8);
     mul_func(&val1_int8, &val2_int8, &val3_int8);
     ck_assert(val3_int8 == 2);

     mul_func = tl_gmul_getfunc(TL_UINT32);
     mul_func(&val1_uint32, &val2_uint32, &val3_uint32);
     ck_assert(val3_uint32 == 2);

     mul_func = tl_gmul_getfunc(TL_UINT16);
     mul_func(&val1_uint16, &val2_uint16, &val3_uint16);
     ck_assert(val3_uint16 == 2);

     mul_func = tl_gmul_getfunc(TL_UINT8);
     mul_func(&val1_uint8, &val2_uint8, &val3_uint8);
     ck_assert(val3_uint8 == 2);

     mul_func = tl_gmul_getfunc(TL_BOOL);
     mul_func(&val1_bool, &val2_bool, &val3_bool);
     ck_assert(val3_bool == TL_FALSE);
}
END_TEST
/* end of tests */

Suite *make_type_suite(void)
{
     Suite *s;
     TCase *tc_type;

     s = suite_create("type");
     tc_type = tcase_create("type");
     tcase_add_checked_fixture(tc_type, setup, teardown);

     tcase_add_test(tc_type, test_tl_size_of);
     tcase_add_test(tc_type, test_tl_psub);
     tcase_add_test(tc_type, test_tl_padd);
     tcase_add_test(tc_type, test_tl_passign);
     tcase_add_test(tc_type, test_tl_fmt);
     tcase_add_test(tc_type, test_tl_pointer_sub);
     tcase_add_test(tc_type, test_tl_pointer_add);
     tcase_add_test(tc_type, test_tl_pointer_assign);
     tcase_add_test(tc_type, test_tl_gfprintf);
     tcase_add_test(tc_type, test_tl_gfprintf_getfunc);
     tcase_add_test(tc_type, test_tl_gcmp);
     tcase_add_test(tc_type, test_tl_gcmp_getfunc);
     tcase_add_test(tc_type, test_tl_gmul);
     tcase_add_test(tc_type, test_tl_gmul_getfunc);
     /* end of adding tests */

     suite_add_tcase(s, tc_type);

     return s;
}
