#! /usr/bin/perl

use warnings;
use strict;

my $usage = <<EOF;
Usage: $0 PREFIX VERSION OUTDIR
Generate package information for pkg-config.
PREFIX is the directory where the library will be installed.
VERSION is the library version.
OUTDIR is the directory where the config file will be generated.
EOF

if (@ARGV < 3) {
  print $usage;
  exit;
}

my $prefix = $ARGV[0];
my $version = $ARGV[1];
my $outdir = $ARGV[2];
my $outfile = "${outdir}/tensorlight.pc";

my $config_template = <<EOF;
# Package Information for pkg-config

libdir=${prefix}/lib
includedir=${prefix}/include

Name: TensorLight
Description: Light-weight Tensor Operation Library
Version: ${version}
Libs: -L\$\{libdir\} -ltsl -lm
Cflags: -I\$\{includedir\}
EOF

open OUTFILE, '>', $outfile or die "Cannot open $outfile: $!";
print OUTFILE $config_template;
close OUTFILE;
