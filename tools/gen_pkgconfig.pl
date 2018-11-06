#! /usr/bin/env perl

use warnings;
use strict;

my $usage = <<EOF;
Usage: $0 TARGET PREFIX VERSION OUTDIR
Generate package information for pkg-config.
TARGET is the library name.
PREFIX is the directory where the library will be installed.
VERSION is the library version.
OUTDIR is the directory where the config file will be generated.
EOF

if (@ARGV < 4) {
  print $usage;
  exit;
}

my $target = $ARGV[0];
my $prefix = $ARGV[1];
my $version = $ARGV[2];
my $outdir = $ARGV[3];
my $outfile = "${outdir}/${target}.pc";

my $config_template = <<EOF;
# Package Information for pkg-config

libdir=${prefix}/lib
includedir=${prefix}/include

Name: ${target}
Description: Light-weight Tensor Operation Library
Version: ${version}
Libs: -L\$\{libdir\} -ltensorlight -lm
Cflags: -I\$\{includedir\}
EOF

open OUTFILE, '>', $outfile or die "Cannot open $outfile: $!";
print OUTFILE $config_template;
close OUTFILE;
