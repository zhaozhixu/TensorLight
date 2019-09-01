#! /usr/bin/env perl

use warnings;
use strict;
use Getopt::Long;
use 5.014;

my $usage = <<EOF;
Usage: $0 [options] INFILE
Generate generic-typed C code based on enum value.
INFILE is the source file contained source code to be translated.
If INFILE is ommited, read from stdin.

Options:
  -h, --help                 print this message
  -o, --outfile=OUTFILE      specify outfile; print to stdout if ommited
  -s, --showtypes            print all default usable enum-type mappings

Commands:
  \$enum2type(ENUM, TYPE)     map enum value ENUM to type TYPE; if not used,
                              use all default usable types
  \$switchtype(SWITCH, GTYPE) define switch argument as SWITCH, whose generic
                             type argument is GTYPE
  \$typeset(GTYPE, ENUM1, ENUM2...)
                             generic type argument GTYPE may be specilized to
                             types mapped by enum value ENUM1, ENUM2...; if
                             \$typeset and \$typenoset are both not used, GTYPE
                             may be specilized to any usable types
  \$typenoset(GTYPE, ENUM1, ENUM2...)
                             generic type argument GTYPE may not be specilized
                             to types mapped by enum value ENUM1, ENUM2...

Example:
  example.txt:
    \$enum2type(TL_FLOAT, float)
    \$enum2type(TL_DOUBLE, double)
    \$enum2type(TL_INT8, int8_t)
    \$switchtype(src->dtype, T1)
    \$switchtype(dst->dtype, T2)
    \$typeset(T1, TL_FLOAT, TL_INT8)
    \$typenoset(T2, TL_FLOAT)
    foo((T1 *)src->data, (T2 *)dst->data);

  Try this!
    \$ $0 example.txt
EOF

my %enum2type_default = (
                         "TL_DOUBLE" => "double",
                         "TL_FLOAT" => "float",
                         "TL_INT32" => "int32_t",
                         "TL_INT16" => "int16_t",
                         "TL_INT8" => "int8_t",
                         "TL_UINT32" => "uint32_t",
                         "TL_UINT16" => "uint16_t",
                         "TL_UINT8" => "uint8_t",
                         "TL_BOOL" => "tl_bool_t",
                        );

my $outfile = "";
my $showtypes = "";
GetOptions(
           'help' => sub {&exit_msg(0, $usage)},
           'outfile=s' => \$outfile,
           'showtypes' => \$showtypes,
          ) or &exit_msg(1, $usage);

if ($showtypes) {
    map {say "$_ => $enum2type_default{$_}"} sort keys %enum2type_default;
    exit 0;
}

my @inlines;
if (@ARGV != 0) {
    my $infile = shift @ARGV;
    open FILE, "<", $infile or die "Cannot open $infile: $!";
    @inlines = <FILE>;
    close FILE;
} else {
    @inlines = <STDIN>;
}

my %enum2type;
my %switchtype;
my %typeswitch;
my %typeset;
my %typenoset;
my @codelines = ();
foreach (@inlines) {
    if (/\$switchtype\((.+)\)/) {
        my ($switch_arg, $type_arg) = split /\s*,\s*/, $1;
        $switchtype{$switch_arg} = $type_arg;
        if (not exists $typeswitch{$type_arg}) {
            $typeswitch{$type_arg} = $switch_arg;
        } else {
            die "type_arg $type_arg has been defined by $switch_arg: $_";
        }
    } elsif (/\$enum2type\((.+)\)/) {
        my ($enum, $type) = split /\s*,\s*/, $1;
        $enum2type{$enum} = $type;
    } elsif (/\$typeset\((.+)\)/) {
        my ($type, @set) = split /\s*,\s*/, $1;
        die "Generic type $type not defined: $_"
            unless exists $typeswitch{$type};
        if (exists $typenoset{$type}) {
            my @interset = &intersection($typenoset{$type}, \@set);
            if (@interset) {
                my $str = "[".(join ", ", @interset)."]";
                die "Conflict enum value $str with previous \$typenoset: $_";
            }
        }
        $typeset{$type} = \@set;
    } elsif (/\$typenoset\((.+)\)/) {
        my ($type, @noset) = split /\s*,\s*/, $1;
        die "Generic type $type not defined: $_"
            unless exists $typeswitch{$type};
        if (exists $typeset{$type}) {
            my @interset = &intersection($typeset{$type}, \@noset);
            if (@interset) {
                my $str = "[".(join ", ", @interset)."]";
                die "Conflict enum value $str with previous \$typeset: $_";
            }
        }
        $typenoset{$type} = \@noset;
    } else {
        push @codelines, $_;
    }
}

%enum2type = %enum2type_default if not %enum2type;
foreach (keys %typeswitch) {
    if (not exists $typeset{$_} and not exists $typenoset{$_}) {
        $typeset{$_} = [sort keys %enum2type];
        $typenoset{$_} = [];
    } elsif (not exists $typeset{$_}) {
        if (exists $typenoset{$_}) {
            my @comp = sort &complement($typenoset{$_}, [keys %enum2type]);
            $typeset{$_} = \@comp;
        } else {
            $typeset{$_} = [sort keys %enum2type];
        }
    }
}

my $codestr = join "", @codelines;
my $comment = &gen_comment(join "", @inlines);

my $indent_level = 1;
my $indent_spaces = 4;
foreach my $switch_arg (sort keys %switchtype) {
    my $type_arg = $switchtype{$switch_arg};
    chomp $codestr;
    $codestr = &gen_switch($codestr, $switch_arg, $type_arg,
                           \%enum2type, \%typeset, $indent_spaces);
}
my $outstr = $comment . $codestr;
&indent_block($indent_level * $indent_spaces, \$outstr);
$outstr .= "\n";

if ($outfile eq "") {
    print $outstr;
} else {
    open FILE, '>', $outfile;
    print $outstr;
    close FILE;
}

sub gen_comment {
    my $code_str = shift;
    &to_comment(\$code_str);
    my $comment = <<EOF;
/*
 * Generated by $0 with
$code_str
 */
EOF
}

sub gen_switch {
    my $codestr = shift;
    my $switch_arg = shift;
    my $type_arg = shift;
    my $enum2type = shift;
    my $typeset = shift;
    my $indent_spaces = shift;

    my @cases = ();
    foreach my $enum (@{$typeset->{$type_arg}}) {
        my $codestr_copy = $codestr;
        $codestr_copy =~ s/\b$type_arg\b/$enum2type{$enum}/g;
        my $case_body = <<EOF;
$codestr_copy
break;
EOF
        &indent_block($indent_spaces, \$case_body);
        my $case_code = <<EOF;
case $enum:
$case_body
EOF
        push @cases, $case_code;
    }

    my $default_body = <<EOF;
assert(0 && "unsupported dtype for $switch_arg");
break;
EOF
    &indent_block($indent_spaces, \$default_body);
    my $default_code = <<EOF;
default:
$default_body
EOF
    push @cases, $default_code;

    my $switch_body = join "", @cases;
    chomp $switch_body;
    my $switch_code = <<EOF;
switch ($switch_arg) {
$switch_body
}
EOF
}

sub union {
    my @sets = @_;
    my @result;
    my %hash;
    foreach my $set (@sets) {
        foreach (@$set) {
            next if exists $hash{$_};
            push @result, $_;
            $hash{$_} = 1;
        }
    }
    @result;
}

sub intersection {
    my @sets = @_;
    my @result;
    my %hash;
    my $first_set = 1;
    foreach my $set (@sets) {
        if ($first_set) {
            foreach (@$set) {
                next if exists $hash{$_};
                $hash{$_} = 1;
            }
            $first_set = 0;
        } else {
            foreach (@$set) {
                next unless exists $hash{$_};
                push @result, $_;
            }
        }
    }
    @result;
}

sub complement {
    my $set = shift;
    my $full_set = shift;
    my %hash;
    map { $hash{$_} = 1 } @$set;
    grep { not exists $hash{$_} } @$full_set;
}

sub to_comment {
    my $code = shift;
    my @lines = split "\n", $$code;
    $$code = join "\n", map { $_ = " * ".$_; } @lines;
}

sub indent_block {
    my $nspaces = shift;
    my $strp = shift;
    my @lines = split "\n", $$strp;
    $$strp = join "\n", &indent_lines($nspaces, \@lines);
}

sub indent_lines {
    my $nspaces = shift;
    my $states = shift;
    foreach (@$states) {
        $_ = " "x$nspaces.$_ unless /^\s*$/;
    }
    @$states;
}

sub exit_msg {
    my $status = shift;
    my $msg = shift;
    print $msg;
    exit $status;
}
