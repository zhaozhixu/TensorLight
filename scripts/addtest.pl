#! /usr/bin/env perl

use warnings;
use strict;
use File::Copy;
use Cwd 'abs_path';

my $usage = <<EOF;
Usage: $0 [-c <CONDITION>] ROOT MOD_NAME TEST_NAME(s)
Generate test templates for a module.
CONDITION is an optional condition macro.
ROOT is the path of the project root.
MOD_NAME is the module name.
TEST_NAME is the test name, usually the name of function to be tested.

Example:
	scripts/addtest.pl -c TL_CUDA . mod mod_func1 mod_func2

	Executing this example from project root will generate test templates
	test_tl_mod_func1 and test_tl_mod_func2 for module tl_mod in file
 	ROOT/test/test_tl_mod.c, and will be compiled only if TL_CUDA has been
    defined.
EOF
if (@ARGV < 1 or $ARGV[0] eq "-h" or $ARGV[0] eq "--help") {
  print $usage;
  exit;
}

my $condition;
my $condition_start = "";
my $condition_end = "";
if ($ARGV[0] eq "-c") {
  shift @ARGV;
  $condition = shift @ARGV;
  $condition_start = "\n#ifdef $condition\n";
  $condition_end = "\n#endif /* $condition */\n"
}

my $root = abs_path(shift @ARGV);
my $suite_name = shift @ARGV;
my @test_names = @ARGV;

my $tests_str = "";
my $test_add_str = "";
foreach my $test_name (@test_names) {
  $tests_str = $tests_str.<<EOF;

START_TEST(test_tl_${test_name})
{
}
END_TEST
EOF

  $test_add_str = $test_add_str.<<EOF;
     tcase_add_test(tc_$suite_name, test_tl_${test_name});
EOF
}
chomp $tests_str;
chomp $test_add_str;

my $suite_tpl = <<EOF;
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
$condition_start
#include "test_tensorlight.h"

static void setup(void)
{
}

static void teardown(void)
{
}
$tests_str
/* end of tests */

Suite *make_${suite_name}_suite(void)
{
     Suite *s;
     TCase *tc_${suite_name};

     s = suite_create("${suite_name}");
     tc_${suite_name} = tcase_create("${suite_name}");
     tcase_add_checked_fixture(tc_${suite_name}, setup, teardown);

$test_add_str
     /* end of adding tests */

     suite_add_tcase(s, tc_${suite_name});

     return s;
}
$condition_end
EOF

my $test_file = "$root/test/test_tl_${suite_name}.c";
if (-e $test_file) {
  copy($test_file, "$test_file.bak")
    or die "Cannot backup file $test_file: $!";
  open TEST_BAK, '<', "$test_file.bak"
    or die "Cannot open $test_file.bak: $!";
  open TEST, '>', $test_file
    or die "Cannot open $test_file: $!";
  while (<TEST_BAK>) {
    s|/\* end of tests \*/|$tests_str\n/* end of tests */|;
    s|     /\* end of adding tests \*/|$test_add_str\n     /* end of adding tests */|;
    print TEST;
  }
  close TEST;
  close TEST_BAK;
  exit 0;
}
open TEST, '>', $test_file
  or die "Cannot open $test_file: $!";
print TEST $suite_tpl;
close TEST;

my $declare = "Suite *make_${suite_name}_suite(void);";
my $header_file = "$root/test/test_tensorlight.h";
copy($header_file, "$header_file.bak")
  or die "Cannot backup file $header_file: $!";
open HEADER_BAK, '<', "$header_file.bak"
  or die "Cannot open $header_file.bak: $!";
open HEADER, '>', $header_file
  or die "Cannot open $header_file: $!";
while (<HEADER_BAK>) {
  if (defined $condition) {
    s|#endif /\* TL_CUDA \*/|$declare\n#endif /* TL_CUDA */|;
  } else {
    s|/\* end of normal declarations \*/|$declare\n/* end of normal declarations */|;
  }
  print HEADER;
}
close HEADER;
close HEADER_BAK;

my $adding_suite = "srunner_add_suite(sr, make_${suite_name}_suite());";
my $main_file = "$root/test/test_tensorlight.c";
copy($main_file, "$main_file.bak")
  or die "Cannot backup file $main_file: $!";
open MAIN_BAK, '<', "$main_file.bak"
  or die "Cannot open $main_file.bak: $!";
open MAIN, '>', $main_file
  or die "Cannot open $main_file: $!";
while (<MAIN_BAK>) {
    if (defined $condition) {
      s|#endif /\* TL_CUDA \*/|     $adding_suite\n#endif /* TL_CUDA */|;
    } else {
      s|/\* end of adding normal suites \*/|$adding_suite\n     /* end of adding normal suites */|;
    }
  print MAIN;
}
close MAIN;
close MAIN_BAK;
