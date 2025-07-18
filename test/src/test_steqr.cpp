/// @file test_steqr.cpp
/// @author Thijs Steel, KU Leuven, Belgium
/// @brief Test implicit QR variation of symmetric tridiagonal eigenvalue solver
//
// Copyright (c) 2021-2023, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// tlapack routines
#include <tlapack/blas/copy.hpp>
#include <tlapack/blas/gemm.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/steqr.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE("steqr is backward stable",
                   "[symmetriceigenvalues][steqr]",
                   TLAPACK_TYPES_TO_TEST)
{
    srand(1);

    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    const real_t zero(0);
    const real_t one(1);

    idx_t n;

    n = GENERATE(15, 40);

    // MatrixMarket reader
    uint64_t seed = GENERATE(3, 5, 6);

    std::mt19937 gen;
    gen.seed(seed);

    DYNAMIC_SECTION(" n = " << n)
    {
        const real_t eps = ulp<real_t>();
        real_t tol = real_t(20. * n) * eps;
        // Use a slightly larger tolerance for half precision
        if (eps > real_t(1.0e-6)) tol = tol * real_t(5.);

        std::vector<T> Q_;
        auto Q = new_matrix(Q_, n, n);

        std::vector<real_t> d1(n);
        std::vector<real_t> d2(n);
        std::vector<real_t> e1(n - 1);
        std::vector<real_t> e2(n - 1);
        std::vector<real_t> d_copy(n);
        std::vector<real_t> e_copy(n - 1);

        // Generate random tridiagonal matrix
        for (idx_t j = 0; j < n; ++j)
            d1[j] = rand_helper<real_t>(gen);
        for (idx_t j = 0; j + 1 < n; ++j)
            e1[j] = rand_helper<real_t>(gen);

        copy(d1, d_copy);
        copy(d1, d2);
        copy(e1, e_copy);
        copy(e1, e2);

        laset(Uplo::General, zero, one, Q);
        int err = steqr(false, d1, e1, Q);
        REQUIRE(err == 0);
        err = steqr(true, d2, e2, Q);
        REQUIRE(err == 0);

        // Check that eigenvalues of steqr(false...) and steqr(true...) are
        // bit-by-bit identical
        for (idx_t i = 0; i < n; ++i) {
            CHECK(d1[i] == d2[i]);
        }

        // Check that eigenvalues are sorted in ascending
        // order
        for (idx_t i = 0; i < n - 1; ++i) {
            CHECK(d2[i] <= d2[i + 1]);
        }

        // Test for Q's orthogonality
        std::vector<T> Wq_;
        auto Wq = new_matrix(Wq_, n, n);
        auto orth_Q = check_orthogonality(Q, Wq);
        CHECK(orth_Q <= tol);

        // Test Q * B * Q^H = A
        std::vector<T> B_;
        auto B = new_matrix(B_, n, n);
        laset(Uplo::General, zero, zero, B);
        for (idx_t j = 0; j < n; ++j) {
            B(j, j) = d1[j];
        }
        std::vector<T> A_;
        auto A = new_matrix(A_, n, n);
        laset(Uplo::General, zero, zero, A);
        A(0, 0) = d_copy[0];
        for (idx_t j = 1; j < n; ++j) {
            A(j, j - 1) = e_copy[j - 1];
            A(j - 1, j) = e_copy[j - 1];
            A(j, j) = d_copy[j];
        }
        real_t normA = tlapack::lange(tlapack::Norm::Max, A);
        std::vector<T> K_;
        auto K = new_matrix(K_, n, n);
        laset(Uplo::General, zero, zero, K);
        gemm(Op::NoTrans, Op::ConjTrans, real_t(1.), B, Q, real_t(0), K);
        gemm(Op::NoTrans, Op::NoTrans, real_t(1.), Q, K, real_t(-1.), A);
        real_t repres = lange(Norm::Max, A);
        CHECK(repres <= tol * normA);
    }
}
