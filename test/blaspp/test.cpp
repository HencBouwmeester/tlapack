/// @file test.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <complex>

#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "test.hh"

// -----------------------------------------------------------------------------
using testsweeper::ParamType;
using testsweeper::DataType;
using testsweeper::char2datatype;
using testsweeper::datatype2char;
using testsweeper::datatype2str;
using testsweeper::ansi_bold;
using testsweeper::ansi_red;
using testsweeper::ansi_normal;

// -----------------------------------------------------------------------------
// each section must have a corresponding entry in section_names
enum Section {
    newline = 0,  // zero flag forces newline
    blas1,
    blas2,
    blas3,
    aux,
    num_sections,  // last
};

const char* section_names[] = {
   "",  // none
   "Level 1 BLAS",
   "Level 2 BLAS",
   "Level 3 BLAS",
   "auxiliary",
};

// { "", nullptr, Section::newline } entries force newline in help
std::vector< testsweeper::routines_t > routines = {
    // Level 1 BLAS
    { "asum",   test_asum,   Section::blas1   },
    { "axpy",   test_axpy,   Section::blas1   },
    { "copy",   test_copy,   Section::blas1   },
    { "dot",    test_dot,    Section::blas1   },
    { "dotu",   test_dotu,   Section::blas1   },
    { "iamax",  test_iamax,  Section::blas1   },
    { "nrm2",   test_nrm2,   Section::blas1   },
    { "rot",    test_rot,    Section::blas1   },
    { "rotg",   test_rotg,   Section::blas1   },
    { "rotm",   test_rotm,   Section::blas1   },
    { "rotmg",  test_rotmg,  Section::blas1   },
    { "scal",   test_scal,   Section::blas1   },
    { "swap",   test_swap,   Section::blas1   },
    { "",       nullptr,     Section::newline },

    // Level 2 BLAS
    { "gemv",   test_gemv,   Section::blas2   },
    { "ger",    test_ger,    Section::blas2   },
    { "geru",   test_geru,   Section::blas2   },
    { "",       nullptr,     Section::newline },

    { "hemv",   test_hemv,   Section::blas2   },
    { "her",    test_her,    Section::blas2   },
    { "her2",   test_her2,   Section::blas2   },
    { "",       nullptr,     Section::newline },

    { "symv",   test_symv,   Section::blas2   },
    { "syr",    test_syr,    Section::blas2   },
    { "syr2",   test_syr2,   Section::blas2   },
    { "",       nullptr,     Section::newline },

    { "trmv",   test_trmv,   Section::blas2   },
    { "trsv",   test_trsv,   Section::blas2   },
    { "",       nullptr,     Section::newline },

    // Level 3 BLAS
    { "gemm",   test_gemm,   Section::blas3   },
    { "",       nullptr,     Section::newline },

    { "hemm",   test_hemm,   Section::blas3   },
    { "herk",   test_herk,   Section::blas3   },
    { "her2k",  test_her2k,  Section::blas3   },
    { "",       nullptr,     Section::newline },

    { "symm",   test_symm,   Section::blas3   },
    { "syrk",   test_syrk,   Section::blas3   },
    { "syr2k",  test_syr2k,  Section::blas3   },
    { "",       nullptr,     Section::newline },

    { "trmm",   test_trmm,   Section::blas3   },
    { "trsm",   test_trsm,   Section::blas3   },
    { "",       nullptr,     Section::newline },

    // auxiliary
    { "max",    test_max,    Section::aux     },
};

// -----------------------------------------------------------------------------
// Params class
// List of parameters

Params::Params():
    ParamsBase(),

    // w = width
    // p = precision
    // def = default
    // ----- test framework parameters
    //         name,       w,    type,         default, valid, help
    check     ( "check",   0,    ParamType::Value, 'y', "ny",  "check the results" ),
    ref       ( "ref",     0,    ParamType::Value, 'n', "ny",  "run reference; sometimes check -> ref" ),

    //          name,      w, p, type,         default, min,  max, help
    repeat    ( "repeat",  0,    ParamType::Value,   1,   1, 1000, "times to repeat each test" ),
    verbose   ( "verbose", 0,    ParamType::Value,   0,   0,   10, "verbose level" ),
    cache     ( "cache",   0,    ParamType::Value,  20,   1, 1024, "total cache size, in MiB" ),

    // ----- routine parameters
    //          name,      w,    type,            def,                    char2enum,         enum2char,         enum2str,         help
    datatype  ( "type",    4,    ParamType::List, DataType::Double,       char2datatype,     datatype2char,     datatype2str,     "s=single (float), d=double, c=complex-single, z=complex-double" ),
    layout    ( "layout",  6,    ParamType::List, blas::Layout::ColMajor, blas::char2layout, blas::layout2char, blas::layout2str, "layout: r=row major, c=column major" ),
    format    ( "format",  6,    ParamType::List, blas::Format::LAPACK,   blas::char2format, blas::format2char, blas::format2str, "format: l=lapack, t=tile" ),
    side      ( "side",    6,    ParamType::List, blas::Side::Left,       blas::char2side,   blas::side2char,   blas::side2str,   "side: l=left, r=right" ),
    uplo      ( "uplo",    6,    ParamType::List, blas::Uplo::Lower,      blas::char2uplo,   blas::uplo2char,   blas::uplo2str,   "triangle: l=lower, u=upper" ),
    trans     ( "trans",   7,    ParamType::List, blas::Op::NoTrans,      blas::char2op,     blas::op2char,     blas::op2str,     "transpose: n=no-trans, t=trans, c=conj-trans" ),
    transA    ( "transA",  7,    ParamType::List, blas::Op::NoTrans,      blas::char2op,     blas::op2char,     blas::op2str,     "transpose of A: n=no-trans, t=trans, c=conj-trans" ),
    transB    ( "transB",  7,    ParamType::List, blas::Op::NoTrans,      blas::char2op,     blas::op2char,     blas::op2str,     "transpose of B: n=no-trans, t=trans, c=conj-trans" ),
    diag      ( "diag",    7,    ParamType::List, blas::Diag::NonUnit,    blas::char2diag,   blas::diag2char,   blas::diag2str,   "diagonal: n=non-unit, u=unit" ),

    //          name,      w, p, type,            def,   min,     max, help
    dim       ( "dim",     6,    ParamType::List,          0,     1e9, "m by n by k dimensions" ),
    alpha     ( "alpha",   9, 4, ParamType::List,  pi,  -inf,     inf, "scalar alpha" ),
    beta      ( "beta",    9, 4, ParamType::List,   e,  -inf,     inf, "scalar beta" ),
    incx      ( "incx",    4,    ParamType::List,   1, -1000,    1000, "stride of x vector" ),
    incy      ( "incy",    4,    ParamType::List,   1, -1000,    1000, "stride of y vector" ),
    align     ( "align",   0,    ParamType::List,   1,     1,    1024, "column alignment (sets lda, ldb, etc. to multiple of align)" ),
    batch     ( "batch",   6,    ParamType::List, 100,     0,     1e6, "batch size" ),
    device    ( "device",  6,    ParamType::List,   0,     0,     100, "device id" ),
    pointer_mode ( "pointer-mode",  3,    ParamType::List, 'h',  "hd",          "h == host, d == device" ),

    // ----- output parameters
    // min, max are ignored
    //          name,                w, p, type,              default,               min, max, help
    error     ( "<T>LAPACK\nerror",    11, 4, ParamType::Output, testsweeper::no_data_flag,   0,   0, "numerical error" ),
    error2    ( "<T>LAPACK\nerror2",   11, 4, ParamType::Output, testsweeper::no_data_flag,   0,   0, "numerical error 2" ),
    error3    ( "<T>LAPACK\nerror3",   11, 4, ParamType::Output, testsweeper::no_data_flag,   0,   0, "numerical error 3" ),
    time      ( "<T>LAPACK\ntime (s)", 11, 4, ParamType::Output, testsweeper::no_data_flag,   0,   0, "time to solution" ),
    gflops    ( "<T>LAPACK\nGflop/s",  11, 4, ParamType::Output, testsweeper::no_data_flag,   0,   0, "Gflop/s rate" ),
    gbytes    ( "<T>LAPACK\nGbyte/s",  11, 4, ParamType::Output, testsweeper::no_data_flag,   0,   0, "Gbyte/s rate" ),

    time2     ( "time2 (s)",        11, 4, ParamType::Output, testsweeper::no_data_flag,   0,   0, "time to solution (2)" ),
    gflops2   ( "Gflop2/s",         11, 4, ParamType::Output, testsweeper::no_data_flag,   0,   0, "Gflop/s rate (2)" ),
    gbytes2   ( "Gbyte2/s",         11, 4, ParamType::Output, testsweeper::no_data_flag,   0,   0, "Gbyte/s rate (2)" ),

    time3     ( "time3 (s)",        11, 4, ParamType::Output, testsweeper::no_data_flag,   0,   0, "time to solution (3)" ),
    gflops3   ( "Gflop3/s",         11, 4, ParamType::Output, testsweeper::no_data_flag,   0,   0, "Gflop/s rate (3)" ),
    gbytes3   ( "Gbyte3/s",         11, 4, ParamType::Output, testsweeper::no_data_flag,   0,   0, "Gbyte/s rate (3)" ),

    time4     ( "time4 (s)",        11, 4, ParamType::Output, testsweeper::no_data_flag,   0,   0, "time to solution (4)" ),
    gflops4   ( "Gflop4/s",         11, 4, ParamType::Output, testsweeper::no_data_flag,   0,   0, "Gflop/s rate (4)" ),
    gbytes4   ( "Gbyte4/s",         11, 4, ParamType::Output, testsweeper::no_data_flag,   0,   0, "Gbyte/s rate (4)" ),

    ref_time  ( "Ref.\ntime (s)",   11, 4, ParamType::Output, testsweeper::no_data_flag,   0,   0, "reference time to solution" ),
    ref_gflops( "Ref.\nGflop/s",    11, 4, ParamType::Output, testsweeper::no_data_flag,   0,   0, "reference Gflop/s rate" ),
    ref_gbytes( "Ref.\nGbyte/s",    11, 4, ParamType::Output, testsweeper::no_data_flag,   0,   0, "reference Gbyte/s rate" ),

    // default -1 means "no check"
    okay      ( "status",              6,    ParamType::Output,  -1,   0,   0, "success indicator" ),
    msg       ( "",       1, ParamType::Output,  "",           "error message" )
{
    // set header different than command line prefix 
    pointer_mode.name("ptr", "pointer-mode");

    // mark standard set of output fields as used
    okay();
    error();
    time();

    // mark framework parameters as used, so they will be accepted on the command line
    check();
    repeat();
    verbose();
    cache();

    // routine's parameters are marked by the test routine; see main
}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    using testsweeper::QuitException;

    // check that all sections have names
    require( sizeof(section_names)/sizeof(*section_names) == Section::num_sections );

    int status = 0;
    try {
        printf( "Running tests from https://bitbucket.org/weslleyspereira/blaspp/src/tlapack/test\n" );

        // print input so running `test [input] > out.txt` documents input
        printf( "input: %s", argv[0] );
        for (int i = 1; i < argc; ++i) {
            // quote arg if necessary
            std::string arg( argv[i] );
            const char* wordchars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-=";
            if (arg.find_first_not_of( wordchars ) != std::string::npos)
                printf( " '%s'", argv[i] );
            else
                printf( " %s", argv[i] );
        }
        printf( "\n" );

        // Usage: test [params] routine
        if (argc < 2
            || strcmp( argv[argc-1], "-h" ) == 0
            || strcmp( argv[argc-1], "--help" ) == 0)
        {
            usage( argc, argv, routines, section_names );
            throw QuitException();
        }

        // find routine to test
        const char* routine = argv[ argc-1 ];
        testsweeper::test_func_ptr test_routine = find_tester( routine, routines );
        if (test_routine == nullptr) {
            usage( argc, argv, routines, section_names );
            throw std::runtime_error(
                std::string("routine ") + routine + " not found" );
        }

        // mark fields that are used (run=false)
        Params params;
        params.routine = routine;
        test_routine( params, false );

        // Parse parameters up to routine name.
        try {
            params.parse( routine, argc-2, argv+1 );
        }
        catch (const std::exception& ex) {
            params.help( routine );
            throw;
        }

        // show align column if it has non-default values
        if (params.align.size() != 1 || params.align() != 1) {
            params.align.width( 5 );
        }

        // run tests
        int repeat = params.repeat();
        testsweeper::DataType last = params.datatype();
        params.header();
        do {
            if (params.datatype() != last) {
                last = params.datatype();
                printf( "\n" );
            }
            for (int iter = 0; iter < repeat; ++iter) {
                try {
                    test_routine( params, true );
                }
                catch (const std::exception& ex) {
                    fprintf( stderr, "%s%sError: %s%s\n",
                             ansi_bold, ansi_red, ex.what(), ansi_normal );
                    params.okay() = false;
                }

                params.print();
                fflush( stdout );
                status += ! params.okay();
                params.reset_output();
            }
            if (repeat > 1) {
                printf( "\n" );
            }
        } while(params.next());

        if (status) {
            printf( "%d tests FAILED for %s.\n", status, routine );
        }
        else {
            printf( "All tests passed for %s.\n", routine );
        }
    }
    catch (const QuitException& ex) {
        // pass: no error to print
    }
    catch (const std::exception& ex) {
        fprintf( stderr, "\n%s%sError: %s%s\n",
                 ansi_bold, ansi_red, ex.what(), ansi_normal );
        status = -1;
    }

    return status;
}
