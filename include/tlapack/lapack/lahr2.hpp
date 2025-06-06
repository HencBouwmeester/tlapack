/// @file lahr2.hpp
/// @author Thijs Steel, KU Leuven, Belgium
/// Adapted from @see
/// https://github.com/Reference-LAPACK/lapack/tree/master/SRC/dlahr2.f
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LAHR2_HH
#define TLAPACK_LAHR2_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/axpy.hpp"
#include "tlapack/blas/copy.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/gemv.hpp"
#include "tlapack/blas/scal.hpp"
#include "tlapack/blas/trmm.hpp"
#include "tlapack/blas/trmv.hpp"
#include "tlapack/lapack/lacpy.hpp"
#include "tlapack/lapack/larfg.hpp"

namespace tlapack {

/** Reduces a general square matrix to upper Hessenberg form
 *
 * The matrix Q is represented as a product of elementary reflectors
 * \[
 *          Q = H_ilo H_ilo+1 ... H_ihi,
 * \]
 * Each H_i has the form
 * \[
 *          H_i = I - tau * v * v',
 * \]
 * where tau is a scalar, and v is a vector with
 * \[
 *          v[0] = v[1] = ... = v[i] = 0; v[i+1] = 1,
 * \]
 * with v[i+2] through v[ihi] stored on exit below the diagonal
 * in the ith column of A, and tau in tau[i].
 *
 * @return  0 if success
 *
 * @param[in] k integer
 * @param[in] nb integer
 * @param[in,out] A n-by-n matrix.
 * @param[out] tau Real vector of length n-1.
 *      The scalar factors of the elementary reflectors.
 * @param[out] T nb-by-nb matrix.
 * @param[out] Y n-by-nb matrix.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_SMATRIX matrix_t,
          TLAPACK_VECTOR vector_t,
          TLAPACK_SMATRIX matrixT_t,
          TLAPACK_SMATRIX matrixY_t>
int lahr2(size_type<matrix_t> k,
          size_type<matrix_t> nb,
          matrix_t& A,
          vector_t& tau,
          matrixT_t& T,
          matrixY_t& Y)
{
    using TA = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;
    using real_t = real_type<TA>;

    // constants
    const real_t one(1);
    const idx_t n = nrows(A);

    // quick return if possible
    if (n <= 1) return 0;

    TA ei(0);
    for (idx_t i = 0; i < nb; ++i) {
        if (i > 0) {
            //
            // Update A(K+1:N,I), this rest will be updated later via
            // level 3 BLAS.
            //

            //
            // Update I-th column of A - Y * V**T
            // (Application of the reflectors from the right)
            //
            auto Y2 = slice(Y, range{k + 1, n}, range{0, i});
            auto Vti = slice(A, k + i, range{0, i});
            auto b = slice(A, range{k + 1, n}, i);
            for (idx_t j = 0; j < i; ++j)
                Vti[j] = conj(Vti[j]);
            gemv(NO_TRANS, -one, Y2, Vti, one, b);
            for (idx_t j = 0; j < i; ++j)
                Vti[j] = conj(Vti[j]);
            //
            // Apply I - V * T**T * V**T to this column (call it b) from the
            // left, using the last column of T as workspace
            //
            // Let  V = ( V1 )   and   b = ( b1 )   (first i rows)
            //          ( V2 )             ( b2 )
            //
            // where V1 is unit lower triangular
            //
            auto b1 = slice(b, range{0, i});
            auto b2 = slice(b, range{i, size(b)});
            auto V = slice(A, range{k + 1, n}, range{0, i});
            auto V1 = slice(V, range{0, i}, range{0, i});
            auto V2 = slice(V, range{i, nrows(V)}, range{0, i});
            //
            // w := V1**T * b1
            //
            auto w = slice(T, range{0, i}, nb - 1);
            copy(b1, w);
            trmv(LOWER_TRIANGLE, CONJ_TRANS, UNIT_DIAG, V1, w);
            //
            // w := w + V2**T * b2
            //
            gemv(CONJ_TRANS, one, V2, b2, one, w);
            //
            // w := T**T * w
            //
            auto T2 = slice(T, range{0, i}, range{0, i});
            trmv(UPPER_TRIANGLE, CONJ_TRANS, NON_UNIT_DIAG, T2, w);
            //
            // b2 := b2 - V2*w
            //
            gemv(NO_TRANS, -one, V2, w, one, b2);
            //
            // b1 := b1 - V1*w
            //
            trmv(LOWER_TRIANGLE, NO_TRANS, UNIT_DIAG, V1, w);
            axpy(-one, w, b1);

            A(k + i, i - 1) = ei;
        }
        auto v = slice(A, range{k + i + 1, n}, i);
        larfg(FORWARD, COLUMNWISE_STORAGE, v, tau[i]);

        // larf has been edited to not require A(k+i,i) = one
        // this is for thread safety. Since we already modified
        // A(k+i,i) before, this is not required here
        ei = v[0];
        v[0] = one;
        //
        // Compute  Y(K+1:N,I)
        //
        auto A2 = slice(A, range{k + 1, n}, range{i + 1, n - k});
        auto y = slice(Y, range{k + 1, n}, i);
        gemv(NO_TRANS, one, A2, v, y);
        auto t = slice(T, range{0, i}, i);
        auto A3 = slice(A, range{k + i + 1, n}, range{0, i});
        gemv(CONJ_TRANS, one, A3, v, t);
        auto Y2 = slice(Y, range{k + 1, n}, range{0, i});
        gemv(NO_TRANS, -one, Y2, t, one, y);
        scal(tau[i], y);
        //
        // Compute T(0:I+1,I)
        //
        scal(-tau[i], t);
        auto T2 = slice(T, range{0, i}, range{0, i});
        trmv(UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG, T2, t);
        T(i, i) = tau[i];
    }
    A(k + nb, nb - 1) = ei;
    //
    // Compute Y(0:k+1,0:nb)
    //
    auto A4 = slice(A, range{0, k + 1}, range{1, nb + 1});
    auto Y3 = slice(Y, range{0, k + 1}, range{0, nb});
    lacpy(GENERAL, A4, Y3);
    auto V1 = slice(A, range{k + 1, k + nb + 1}, range{0, nb});
    auto Y1 = slice(Y, range{0, k + 1}, range{0, nb});
    trmm(RIGHT_SIDE, LOWER_TRIANGLE, NO_TRANS, UNIT_DIAG, one, V1, Y1);
    if (k + nb + 1 < n) {
        auto A5 = slice(A, range{0, k + 1}, range{nb + 1, n - k});
        auto V2 = slice(A, range{k + nb + 1, n}, range{0, nb});
        gemm(NO_TRANS, NO_TRANS, one, A5, V2, one, Y1);
    }
    trmm(RIGHT_SIDE, UPPER_TRIANGLE, NO_TRANS, NON_UNIT_DIAG, one, T, Y1);

    return 0;
}

}  // namespace tlapack

#endif  // TLAPACK_LAHR2_HH
