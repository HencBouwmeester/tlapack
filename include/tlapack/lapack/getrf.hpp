/// @file getrf.hpp
/// @author Ali Lotfi, University of Colorado Denver, USA
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GETRF_HH
#define TLAPACK_GETRF_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/getrf_level0.hpp"
#include "tlapack/lapack/getrf_recursive.hpp"

namespace tlapack {

/// @brief Variants of the algorithm to compute the LU factorization.
enum class GetrfVariant : char { Level0 = '0', Recursive = 'R' };

/// @brief Options struct for getrf()
struct GetrfOpts {
    GetrfVariant variant = GetrfVariant::Recursive;
};

/** getrf computes an LU factorization of a general m-by-n matrix A.
 *
 *  The factorization has the form
 * \[
 *   P A = L U
 * \]
 *  where P is a permutation matrix constructed from our piv vector, L is lower
 * triangular with unit diagonal elements (lower trapezoidal if m > n), and U is
 * upper triangular (upper trapezoidal if m < n).
 *
 * @return  0 if success
 * @return  i+1 if failed to compute the LU on iteration i
 *
 * @param[in,out] A m-by-n matrix.
 *      On exit, the factors L and U from the factorization A=PLU;
 *      the unit diagonal elements of L are not stored.
 *
 * @param[in,out] piv is a k-by-1 integer vector where k=min(m,n)
 * and piv[i]=j where i<=j<=k-1, which means in the i-th iteration of the
 * algorithm, the j-th row needs to be swapped with i
 *
 * @param[in] opts Options.
 *      - variant:
 *          - Recursive = 'R',
 *          - Level0 = '0'
 *
 * @note To construct L and U, one proceeds as in the following steps
 *      1. Set matrices L m-by-k, and U k-by-n be to matrices with all zeros,
 * where k=min(m,n)
 *      2. Set elements on the diagonal of L to 1
 *      3. below the diagonal of A will be copied to L
 *      4. On and above the diagonal of A will be copied to U
 *
 * @ingroup variant_interface
 */
template <TLAPACK_MATRIX matrix_t, TLAPACK_VECTOR piv_t>
int getrf(matrix_t& A, piv_t& piv, const GetrfOpts& opts = {})
{
    // Call variant
    if (opts.variant == GetrfVariant::Recursive)
        return getrf_recursive(A, piv);
    else
        return getrf_level0(A, piv);
}

}  // namespace tlapack

#endif  // TLAPACK_GETRF_HH
