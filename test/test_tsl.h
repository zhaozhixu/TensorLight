#ifndef _TEST_TSL_H_
#define _TEST_TSL_H_

#include <stdio.h>
#include <check.h>

#ifdef __cplusplus
extern "C" {
#endif

Suite *make_master_suite(void);
Suite *make_type_suite(void);
Suite *make_tensor_suite(void);
/* end of declarations */

#ifdef __cplusplus
}
#endif

#endif /* _TEST_TSL_H_ */
