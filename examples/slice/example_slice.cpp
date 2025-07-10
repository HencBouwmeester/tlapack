/// @file example_pbtrf.cpp
/// @author Kyle Cunningham, Henricus Bouwmeester, Ella Addison-Taylor
/// University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack/LegacyBandedMatrix.hpp>
#include <tlapack/blas/trsm.hpp>
#include <tlapack/lapack/potf2.hpp>
#include <tlapack/blas/herk.hpp>
#include <tlapack/blas/gemm.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/mult_llh.hpp>
#include <tlapack/lapack/mult_uhu.hpp>
#include <tlapack/lapack/lanhe.hpp>
#include <tlapack/lapack/lange.hpp>

// <T>LAPACK
#include <../test/include/MatrixMarket.hpp>

// local file

// C++ headers
#include <algorithm>
#include <iomanip>

using namespace tlapack;

    template <typename matrix_t>
void printMatrix(const matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << std::setprecision(5) << std::setw(10) << A(i, j) << " ";
    }
    std::cout << std::endl;
}

    template <typename matrix_t>
void printBandedMatrix(const matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t kl = lowerband(A);
    const idx_t ku = upperband(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << ((i <= kl + j && j <= ku + i) ? A(i, j) : 0) << " ";
    }
}

#define isSlice_new(SliceSpec) !std::is_convertible<SliceSpec, idx_t>::value
template <
typename T,
         class idx_t,
         Layout layout,
         class SliceSpecRow,
         class SliceSpecCol,
         typename std::enable_if<isSlice_new(SliceSpecRow) && isSlice_new(SliceSpecCol),
         int>::type = 0>
         constexpr auto slice_new(LegacyMatrix<T, idx_t, layout>& A,
                 SliceSpecRow&& rows,
                 SliceSpecCol&& cols) noexcept
{

    idx_t j = cols.first;
    idx_t jj = cols.second - 1;

    idx_t i = rows.first;
    idx_t ii = rows.second - 1;

    idx_t kd = nrows(A);

    idx_t ABrows_first = i % kd;
    idx_t ABrows_second = ii % kd+1; 
    idx_t ABcols_first = j + (i/kd);
    idx_t ABcols_second = jj + (ii/kd)+1;

    idx_t numrows;
    if (ABrows_first > ABrows_second) 
    {
        numrows = kd - ABrows_first + ABrows_second;
        ABcols_second -= 1;
    }
    else
        numrows = ABrows_second - ABrows_first;

    return LegacyMatrix<T, idx_t, layout>(
            numrows, ABcols_second - ABcols_first,
            (layout == Layout::ColMajor)
            ? &A.ptr[ABrows_first + ABcols_first * A.ldim]
            : &A.ptr[ABrows_first * A.ldim + ABcols_first],
            A.ldim);
}
#undef isSlice_new                    

#define isSlice_ABm1(SliceSpec) !std::is_convertible<SliceSpec, idx_t>::value
template <
typename T, typename uplo_t,
         class idx_t,
         Layout layout,
         class SliceSpecRow,
         class SliceSpecCol,
         typename std::enable_if<isSlice_ABm1(SliceSpecRow) && isSlice_ABm1(SliceSpecCol),
         int>::type = 0>
         constexpr auto slice_ABm1(uplo_t uplo, LegacyMatrix<T, idx_t, layout>& A,
                 SliceSpecRow&& rows,
                 SliceSpecCol&& cols) noexcept
{
    idx_t ptr_offset = (uplo == tlapack::Uplo::Upper) ? A.ldim - 1 : 0;

    idx_t kd = nrows(A) - 1;
    idx_t numcols = ((kd + 1) * ncols(A) - 1) / kd;

    return LegacyMatrix<T, idx_t, layout>(
            rows.second - rows.first, numcols,
            (layout == Layout::ColMajor) ? &A.ptr[ptr_offset + rows.first + cols.first * A.ldim]
            : &A.ptr[(ptr_offset + rows.first) * A.ldim + cols.first],
            A.ldim - 1);
}
#undef isSlice_ABm1

/// @brief Options struct for pbtrf()
struct BlockedBandedCholeskyOpts : public EcOpts {
    constexpr BlockedBandedCholeskyOpts(const EcOpts& opts = {})
        : EcOpts(opts) {};

    size_t nb = 32;  ///< Block size
};


template<typename uplo_t, typename matrix_t>
void pbtrf(uplo_t uplo, matrix_t& A, size_t n, const BlockedBandedCholeskyOpts& opts = {})
{
    using T = tlapack::type_t<matrix_t>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = tlapack::pair<idx_t, idx_t>;
    using real_t = tlapack::real_type<T>;

    tlapack::Create<matrix_t> new_matrix;

    const idx_t nb = opts.nb;

    idx_t kd = nrows(A);

    // WE NEED TO REMOVE THIS WORK ARRAY
    //
    // first we need an inplace TRSM_TRI which receives two triangular matrices 
    // A and B, and computes B = op(A) * B or B = B * op(A)
    //
    // then we need an out-of-place TRMM such that C = alpha * C + beta * op(A) * B
    // where A, B, and C are triangular matrices with op(A) and B both being upper 
    // triangular or both lower triangular
    std::vector<T> work_(nb * nb);
    for (idx_t ii = 0; ii < nb * nb; ++ii) {
        if constexpr (tlapack::is_complex<T>) {
            work_[ii] = T(0, 0);
        }
        else
            work_[ii] = 0;
    }
    auto work = new_matrix(work_, nb, nb);

    if (uplo == tlapack::Uplo::Upper) {
        for (idx_t i = 0; i < n; i += nb) {

            idx_t ib = (n < nb + i) ? n - i : nb;

            auto A00 = slice_new(A, range(i, min(ib + i, n)), range(i, std::min(i + ib, n)));

            potf2(tlapack::Uplo::Upper, A00);

            if (i + ib < n) {
                // i2 = min(kd - ib, n - i - ib)
                idx_t i2 = (kd + i < n) ? kd - ib : n - i - ib;
                // i3 = min(ib, n-i-kd)
                idx_t i3 = (n > i + kd) ? min(ib, n - i - kd) : 0;


                if (i2 > 0) {
                    auto A01 = slice_new(A, range(i, ib + i),
                            range(i + ib, std::min(i + ib + i2, n)));

                    trsm(tlapack::Side::Left, tlapack::Uplo::Upper,
                            tlapack::Op::ConjTrans, tlapack::Diag::NonUnit,
                            real_t(1), A00, A01);

                    auto A11 = slice_new(A, range(i + ib, std::min(i + kd, n)),
                            range(i + ib, std::min(i + kd, n)));            

                    herk(tlapack::Uplo::Upper, tlapack::Op::ConjTrans,
                            real_t(-1), A01, real_t(1), A11);
                }

                if (i3 > 0) {
                    auto A02 = slice_new(A, range(i, i + ib),
                            range(i + kd, i + kd + i3));              

                    // WITH THE NEW KERNELS, WE SHOULD BE ABLE TO REMOVE THIS WORK
                    auto work02 = slice_new(work, range(0, ib), range(0, i3));

                    for (idx_t jj = 0; jj < i3; jj++)
                        for (idx_t ii = jj; ii < ib; ++ii)
                            work02(ii, jj) = A02(ii, jj);

                    trsm(tlapack::Side::Left, tlapack::Uplo::Upper,
                            tlapack::Op::ConjTrans, tlapack::Diag::NonUnit,
                            real_t(1), A00, work02);

                    auto A12 = slice_new(A, range(i + ib, i + kd),
                            range(i + kd, std::min(i + kd + i3, n)));

                    auto A01 = slice_new(A, range(i, ib + i),
                            range(i + ib, std::min(i + ib + i2, n)));

                    // WE MIGHT NEED A NEW KERNEL HERE AS WELL
                    gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans,
                            real_t(-1), A01, work02, real_t(1), A12);

                    auto A22 = slice_new(A, range(i + kd, std::min(i + kd + i3, n)),
                            range(i + kd, std::min(i + kd + i3, n)));

                    // WE MIGHT NEED A NEW KERNEL HERE AS WELL
                    herk(tlapack::Uplo::Upper, tlapack::Op::ConjTrans,
                            real_t(-1), work02, real_t(1), A22);

                    for (idx_t jj = 0; jj < i3; ++jj) {
                        for (idx_t ii = jj; ii < ib; ++ii) {
                            A02(ii, jj) = work02(ii, jj);
                        }
                    }
                }
            }
        }
    }
    else {  // uplo == Lower

        for (idx_t i = 0; i < n; i += nb) {
            idx_t ib = (nb + i < n) ? ib = nb : n - i;

            auto A00 = slice_new(A, range(i, i + ib), range(i, std::min(ib + i, n)));

            potf2(tlapack::Uplo::Lower, A00);

            if (i + ib <= n) {
                // i2 = min(kd - ib, n - i - ib)
                idx_t i2 = (kd + i < n) ? kd - ib : n - i - ib;
                // i3 = min(ib, n-i-kd)
                idx_t i3 = (n > i + kd) ? min(ib, n - i - kd) : 0;

                if (i2 > 0) {
                    auto A10 = slice_new(A, range(ib + i, ib + i2 + i),
                            range(i, ib + i));

                    trsm(tlapack::Side::Right, tlapack::Uplo::Lower,
                            tlapack::Op::ConjTrans, tlapack::Diag::NonUnit,
                            real_t(1), A00, A10);

                    auto A11 = slice_new(A, range(ib + i, ib + i2 + i),
                            range(i + ib, i + ib + i2));

                    herk(uplo, tlapack::Op::NoTrans, real_t(-1), A10,
                            real_t(1), A11);
                }

                if (i3 > 0) {
                    auto A10 = slice_new(A, range(ib + i, ib + i2 + i),
                            range(i, ib + i));

                    auto A20 = slice_new(A, range(kd + i, min(kd + i3 + i, n)), range(i, i + ib));

                    auto work20 = slice(work, range(0, i3), range(0, ib));

                    for (idx_t jj = 0; jj < ib; jj++) {
                        idx_t iiend = min(jj + 1, i3);
                        for (idx_t ii = 0; ii < iiend; ++ii) {
                            work20(ii, jj) = A20(ii, jj);
                        }
                    }

                    trsm(tlapack::Side::Right, uplo, tlapack::Op::ConjTrans,
                         tlapack::Diag::NonUnit, real_t(1), A00, work20);

                    auto A21 = slice_new(A, range(kd + i, kd + i + i3),
                            range(i + ib, i + ib + i2));

                     gemm(tlapack::Op::NoTrans, tlapack::Op::ConjTrans,
                          real_t(-1), work20, A10, real_t(1), A21);

                    auto A22 = slice_new(A, range(kd + i, kd + i + i3),
                            range(kd + i, kd + i + i3));

                     herk(uplo, tlapack::Op::NoTrans, real_t(-1), work20,
                          real_t(1), A22);

                    for (idx_t jj = 0; jj < ib; jj++) {
                        idx_t iiend = min(jj + 1, i3);
                        for (idx_t ii = 0; ii < iiend; ++ii) {
                            A20(ii, jj) = work20(ii, jj);
                        }
                    }
                }
            }
        }
    }
}

/*
 * THIS DOES NOT WORK FOR SOME REASON
 *
template <typename uplo_t, typename matrix_t, typename idx_t>
matrix_t create_CD_Matrx(uplo_t uplo, matrix_t& A, idx_t kd)
{

    using T = tlapack::type_t<matrix_t>;

    tlapack::Create<matrix_t> new_matrix;

    idx_t numcols = ((kd + 1) * ncols(A) - 1) / kd;
    std::vector<T> CD_;
    auto CD = new_matrix(CD_, kd, numcols);
    printMatrix(CD);


    return CD;
}
*/

//------------------------------------------------------------------------------
    template <typename T, typename uplo_t>
void run(uplo_t uplo, size_t m, size_t n, size_t kd, size_t nb)
{
    using real_t = tlapack::real_type<T>;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = std::pair<idx_t, idx_t>;

    tlapack::Create<matrix_t> new_matrix;

    MatrixMarket mm;

    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);

    std::vector<T> A_orig_;
    auto A_orig = new_matrix(A_orig_, m, n);

    std::vector<T> AB_;
    auto AB = new_matrix(AB_, kd + 1, n);

    std::vector<T> AB_orig_;
    auto AB_orig = new_matrix(AB_orig_, kd + 1, n);

    std::vector<T> CD_;
    idx_t numcols = ((kd + 1) * ncols(A) - 1) / kd;
    auto CD = new_matrix(CD_, kd, numcols);

    std::vector<T> CD_orig_;
    auto CD_orig = new_matrix(CD_orig_, kd, numcols);

    mm.random(A);
    mm.random(AB);
    mm.random(CD);

    //for (idx_t i = 0; i < kd + 1; i++) {
        //for (idx_t j = 0; j < n; j++) {
            //AB(i, j) = static_cast<real_t>(0xCAFEBABE);
        //}
    //}
    //for (idx_t i = 0; i < kd; i++) {
        //for (idx_t j = 0; j < numcols; j++) {
            //CD(i, j) = static_cast<real_t>(0xCAFEBABE);
        //}
    //}

    // make the diagonal real and dominant
    for (idx_t j = 0; j < n; ++j)
        A(j, j) = real_t(n) + real(A(j, j)); 

    //for (idx_t j = 0; j < n; ++j)
    //{
        //for (idx_t i = 0; i < n; ++i)
        //{
            //if (i == j)
            //{
                //// make real and diagonally dominant
                //A(i, j) = static_cast<T>(n) + static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
            //}
            //else
            //{
                //if constexpr (tlapack::is_complex<T>)
                //{
                    //A(i, j) = T(static_cast<real_t>(rand()) / static_cast<real_t>(RAND_MAX), 
                            //static_cast<real_t>(rand()) / static_cast<real_t>(RAND_MAX));
                    //A(j, i) = conj(A(i, j));
                //}
                //else
                //{
                    //A(i, j) = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
                //}
            //}
        //}
    //}

    if (uplo == tlapack::Uplo::Upper) {
        //for (idx_t j = 0; j < n; j++)
            //for (idx_t i = j + 1; i < n; i++)
                //A(i, j) = static_cast<real_t>(0);

        //for (idx_t j = kd + 1; j < n; j++)
            //for (idx_t i = 0; i < j - kd; i++)
                //A(i, j) = static_cast<real_t>(0);

        for (idx_t j = 0; j < n; j++)
        {
            idx_t istart = (j > kd) ? j - kd : 0;
            for (idx_t i = istart; i < j + 1; i++)
                AB(i + kd - j, j) = A(i, j);
        }

        // create the upper blocked compressed martix
        for (idx_t k = 0, jj = 0; k < n; k += kd, jj++)
        {
            idx_t kend = min(k + kd, n);
            for (idx_t i = k; i < kend; i++)
            {
                idx_t jend = min(i + kd + 1, n);
                for (idx_t j = i; j < jend; j++)
                {
                    CD(i % kd, j + jj) = A(i, j);
                }
            }
        }
    }
    else {
        //for (idx_t j = 0; j < n; j++)
            //for (idx_t i = 0; i < j; i++)
                //A(i, j) = static_cast<real_t>(0);

        //for (idx_t j = kd + 1; j < n; j++)
            //for (idx_t i = 0; i < j - kd; i++)
                //A(j, i) = static_cast<real_t>(0);

        for (idx_t j = 0; j < n; j++)
        {
            idx_t iend = min(n, j + kd + 1);
            for (idx_t i = j; i < iend; i++)
                AB(i - j, j) = A(i, j);
        }

        // create the lower blocked compressed martix
        for (idx_t k = 0, jj = 0; k < n; k += kd, ++jj)
        {
            idx_t jend = min(k + kd, n);
            for (idx_t j = k, r = 0; j < jend; j++, r++)
            {
                // diagonal blocks
                idx_t iend = jend;
                for (idx_t i = j; i < iend; i++)
                {
                    CD(i % kd, j + jj) = A(i, j);
                }

                // off diagonal blocks
                iend = min(k + kd + r + 1, n);
                for (idx_t i = k + kd; i < iend; i++)
                {
                    CD(i % kd, j + i / kd) = A(i, j);
                }
            }
        }
    }

    // Create the AB with LDA minus 1 matrix
    auto ABm1 = slice_ABm1(uplo, AB, range(0, kd), range(0, m));

    printMatrix(A);
    printMatrix(AB);
    printMatrix(CD);

    // Save a copy of the original matrix
    lacpy(Uplo::General, A, A_orig);
    lacpy(Uplo::General, AB, AB_orig);
    lacpy(Uplo::General, CD, CD_orig);

    // Check each type of data storage format

    BlockedBandedCholeskyOpts opts;
    opts.nb = nb;

    // Compute norm of A
    real_t normAorig = lanhe(tlapack::Norm::Fro, uplo, A);

    // Full access
    pbtrf(uplo, A, n, opts);

    // Compressed Data Blocks
    pbtrf(uplo, CD, n, opts);

    // LAPACK AB format which is the same as ABm1
    pbtrf(uplo, ABm1, n, opts);


    // HENC:  Need to check that we did not touch any parts that we should not have

    std::vector<T> CD_full_;
    auto CD_full = new_matrix(CD_full_, m, n);

    std::vector<T> ABm1_full_;
    auto ABm1_full = new_matrix(ABm1_full_, m, n);

    for (idx_t i = 0; i < m; ++i) {
        for (idx_t j = 0; j < n; ++j) {
            CD_full(i, j) = static_cast<real_t>(0);
            ABm1_full(i, j) = static_cast<real_t>(0);
        }
    }

    // check
    if (uplo == Uplo::Upper) {
        for (idx_t k = 0, jj = 0; k < n; k += kd, jj++)
        {
            idx_t kend = min(k + kd, n);
            for (idx_t i = k; i < kend; i++)
            {
                idx_t jend = min(i + kd + 1, n);
                for (idx_t j = i; j < jend; j++)
                {
                    CD_full(i, j) = CD(i % kd, j + jj);
                    ABm1_full(i, j) = ABm1(i % kd, j + jj);
                }
            }
        }

        mult_uhu(CD_full);
        mult_uhu(ABm1_full);
        mult_uhu(A);
    }
    else {
        for (idx_t k = 0, jj = 0; k < n; k += kd, ++jj)
        {
            idx_t jend = min(k + kd, n);
            for (idx_t j = k, r = 0; j < jend; j++, r++)
            {
                // diagonal blocks
                idx_t iend = jend;
                for (idx_t i = j; i < iend; i++)
                {
                    CD_full(i, j) = CD(i % kd, j + jj);
                    ABm1_full(i, j) = ABm1(i % kd, j + jj);
                }

                // off diagonal blocks
                iend = min(k + kd + r + 1, n);
                for (idx_t i = k + kd; i < iend; i++)
                {
                    CD_full(i, j) = CD(i % kd, j + i / kd);
                    ABm1_full(i, j) = ABm1(i % kd, j + i / kd);
                }
            }
        }

        mult_llh(CD_full);
        mult_llh(ABm1_full);
        mult_llh(A);
    }

    for (idx_t j = 0; j < n; j++) {
        for (idx_t i = 0; i < n; i++) {
            A(i, j) -= A_orig(i, j);
            CD_full(i, j) -= A_orig(i, j);
            ABm1_full(i, j) -= A_orig(i, j);
        }
    }

    real_t normA = lange(Norm::Fro, A);
    real_t normCD = lange(Norm::Fro, CD_full);
    real_t normABm1 = lange(Norm::Fro, ABm1_full);
    //real_t normA = lanhe(Norm::Fro, uplo, A);
    //real_t normCD = lanhe(Norm::Fro, uplo, CD_full);
    //real_t normABm1 = lanhe(Norm::Fro, uplo, ABm1_full);

    std::cout << std::endl;

    if (uplo == tlapack::Uplo::Upper) {
        std::cout << "||C^H * C - A||_F / || A ||_F = " << normA/normAorig << std::endl;
        std::cout << "||CD^H * CD - A||_F / || A ||_F = " << normCD/normAorig << std::endl;
        std::cout << "||AB^H * AB - A||_F / || A ||_F = " << normABm1/normAorig << std::endl;
    }
    else
    {
        std::cout << "||C * C^H - A||_F / || A ||_F = " << normA/normAorig << std::endl;
        std::cout << "||CD * CD^H - A||_F / || A ||_F = " << normCD/normAorig << std::endl;
        std::cout << "||AB * AB^H - A||_F / || A ||_F = " << normABm1/normAorig << std::endl;
    }

    std::cout << std::endl;
}


//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    using std::size_t;

    using idx_t = size_t;

    //idx_t m = 251;
    //idx_t n = m;
    //idx_t kd = 53;
    //idx_t nb = 32;
    //tlapack::Uplo uplo;

    idx_t m = 13;
    idx_t n = m;
    idx_t kd = 7;
    idx_t nb = 3;
    tlapack::Uplo uplo;


    uplo = tlapack::Uplo::Upper;
    printf("---------Upper---------\n");
    printf("run< float  >( %d, %d )", static_cast<int>(m), static_cast<int>(n));
    std::cout << std::endl;
    run<float>(uplo, m, n, kd, nb);
    printf("-----------------------\n");

    printf("run< double >( %d, %d )\n", static_cast<int>(m), static_cast<int>(n));
    run<double>(uplo, m, n, kd, nb);
    printf("-----------------------\n");

    printf("run< long double >( %d, %d )\n", static_cast<int>(m),
           static_cast<int>(n));
    run<long double>(uplo, m, n, kd, nb);
    printf("-----------------------\n");

    printf("run< complex<float> >( %d, %d )\n", static_cast<int>(m),
           static_cast<int>(n));
    run<std::complex<float>>(uplo, m, n, kd, nb);
    printf("-----------------------\n");

    printf("run< complex<double> >( %d, %d )\n", static_cast<int>(m),
           static_cast<int>(n));
    run<std::complex<double>>(uplo, m, n, kd, nb);
    printf("-----------------------\n");

    uplo = tlapack::Uplo::Lower;
    printf("---------Lower---------\n");
    printf("run< float  >( %d, %d )", static_cast<int>(m), static_cast<int>(n));
    std::cout << std::endl;
    run<float>(uplo, m, n, kd, nb);
    printf("-----------------------\n");

    printf("run< double >( %d, %d )\n", static_cast<int>(m), static_cast<int>(n));
    run<double>(uplo, m, n, kd, nb);
    printf("-----------------------\n");

    printf("run< long double >( %d, %d )\n", static_cast<int>(m),
           static_cast<int>(n));
    run<long double>(uplo, m, n, kd, nb);
    printf("-----------------------\n");

    printf("run< complex<float> >( %d, %d )\n", static_cast<int>(m),
           static_cast<int>(n));
    run<std::complex<float>>(uplo, m, n, kd, nb);
    printf("-----------------------\n");

    printf("run< complex<double> >( %d, %d )\n", static_cast<int>(m),
           static_cast<int>(n));
    run<std::complex<double>>(uplo, m, n, kd, nb);
    printf("-----------------------\n");

    return 0;
}

