// @file example_practice_work_file.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/legacyArray.hpp>

// <T>LAPACK
#include <tlapack/blas/gemm.hpp>
#include <tlapack/blas/syrk.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/geqr2.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/lansy.hpp>
#include <tlapack/lapack/larfg.hpp>
#include <tlapack/lapack/larft.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/ung2r.hpp>

#include "geqrt3.hpp"

// C++ headers
#include <chrono>  // for high_resolution_clock
#include <iostream>
#include <memory>
#include <vector>

//------------------------------------------------------------------------------
/// Print matrix A in the standard output
template <typename matrix_t>
void printMatrix(const matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << A(i, j) << " ";
    }
}

template <typename T,
          TLAPACK_MATRIX matrix_a,
          TLAPACK_MATRIX matrix_ao,
          TLAPACK_MATRIX matrix_r,
          TLAPACK_MATRIX matrix_h>
void normCheck(matrix_a& A, matrix_ao& A_orig, matrix_r& R, matrix_h& Tmatrix)
{
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = tlapack::pair<idx_t, idx_t>;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    std::vector<T> tau(n);

    auto normA = tlapack::lange(tlapack::FROB_NORM, A_orig);

    for (idx_t i = 0; i < n; ++i) {
        tau[i] = Tmatrix(i, i);
    }

    // 2) Compute ||Q'Q - I||_F   DOESN'T WORK DUE TO TAU

    tlapack::lacpy(tlapack::Uplo::Upper, A, R);

    tlapack::ung2r(A, tau);

    T norm_orth_1, norm_repres_1;

    std::vector<T> work_;
    auto work = new_matrix(work_, n, n);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            work(i, j) = T(static_cast<float>(0));

    // work receives the identity n*n
    tlapack::laset(tlapack::UPPER_TRIANGLE, static_cast<T>(0.0),
                   static_cast<T>(1.0), work);
    // work receives Q'Q - I
    tlapack::gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans,
                  static_cast<T>(1.0), A, A, static_cast<T>(-1.0), work);
    // tlapack::syrk(tlapack::Uplo::Upper, tlapack::Op::Trans,
    // static_cast<T>(1.0),
    //               A, static_cast<T>(-1.0), work);

    // Compute ||Q'Q - I||_F
    norm_orth_1 = tlapack::lange(tlapack::FROB_NORM, work);
    // norm_orth_1 =
    //     tlapack::lansy(tlapack::FROB_NORM, tlapack::UPPER_TRIANGLE, work);

    std::cout << std::endl << "Q'Q-I = ";
    printMatrix(work);

    // 3) Compute ||QR - A||_F / ||A||_F
    // A=q

    {
        std::vector<T> work_;
        auto work = new_matrix(work_, m, n);
        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < m; ++i)
                work(i, j) = static_cast<float>(0);
        // Copy Q to work
        tlapack::lacpy(tlapack::GENERAL, A, work);

        tlapack::trmm(tlapack::Side::Right, tlapack::Uplo::Upper,

                      tlapack::Op::NoTrans, tlapack::Diag::NonUnit,
                      static_cast<T>(1.0), R, work);

        for (size_t j = 0; j < n; ++j)
            for (size_t i = 0; i < m; ++i)
                work(i, j) -= A_orig(i, j);

        norm_repres_1 = tlapack::lange(tlapack::FROB_NORM, work) / normA;
    }

    // *) Output

    std::cout << std::endl;
    std::cout << "||QR - A||_F/||A||_F = " << norm_repres_1
              << ", ||Q'Q - I||_F = " << norm_orth_1;
    std::cout << std::endl;
}

//------------------------------------------------------------------------------
template <typename T>
void run(size_t m, size_t n)
{
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = tlapack::pair<idx_t, idx_t>;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // Turn it off if m or n are large
    bool verbose = false;

    // Matrices
    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> R_;
    auto R = new_matrix(R_, n, n);
    std::vector<T> A_orig_;
    auto A_orig = new_matrix(A_orig_, m, n);
    std::vector<T> T_;
    auto Tmatrix = new_matrix(T_, n, n);

    // Initialize arrays with junk
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            A(i, j) = T(static_cast<float>(0xDEADBEEF));
        }
        for (size_t i = 0; i < n; ++i)
            Tmatrix(i, j) = T(static_cast<float>(0XFEE1DEAD));
    }
    // Generate a random matrix in A
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < m; ++i)
            if constexpr (tlapack::is_complex<T>)
                A(i, j) = T(
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            else
                A(i, j) = T(static_cast<float>(rand()) /
                            static_cast<float>(RAND_MAX));

    // Copy A to A_orig
    tlapack::lacpy(tlapack::GENERAL, A, A_orig);

    tlapack::geqrt3<T>(A, Tmatrix);

    std::cout << std::endl
              << std::endl
              << std::endl
              << "====================end of computation==================='"
              << std::endl;

    if (verbose) {
        normCheck<T>(A, A_orig, R, Tmatrix);
    }
}

//================================================================================
//================================================================================
int main(int argc, char** argv)
{
    int m, n;

    // Default arguments
    m = (argc < 2) ? 1 : atoi(argv[1]);
    n = (argc < 3) ? 1 : atoi(argv[2]);

    srand(3);  // Init random seed

    m = 19;
    n = 20;

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< float >( %d, %d )", m, n);
    run<float>(m, n);
    printf("-----------------------\n");

    printf("run< double >( %d, %d )", m, n);
    run<double>(m, n);
    printf("-----------------------\n");

    printf("run< long double >( %d, %d )", m, n);
    run<long double>(m, n);
    printf("-----------------------\n");

    printf("run< complex<float> >( %d, %d )", m, n);
    run<std::complex<float>>(m, n);
    printf("-----------------------\n");

    return 0;
}