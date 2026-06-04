#ifndef TLAPACK_GEQRT3
#define TLAPACK_GEQRT3

namespace tlapack {

template <TLAPACK_STOREV storage_t, TLAPACK_VECTOR vector_t>
inline void larfg_flops(storage_t storeMode,
                        type_t<vector_t>& alpha,
                        vector_t& x,
                        type_t<vector_t>& tau,
                        unsigned int& flops)
{
    // data traits
    using T = type_t<vector_t>;
    using idx_t = size_type<vector_t>;

    // using
    using real_t = real_type<T>;

    // constants
    const real_t one(1);
    const real_t zero(0);
    const real_t safemin = safe_min<real_t>() / uroundoff<real_t>();
    const real_t rsafemin = one / safemin;

    // check arguments
    tlapack_check(storeMode == StoreV::Columnwise ||
                  storeMode == StoreV::Rowwise);

    tau = zero;
    real_t xnorm = nrm2(x);

    if (xnorm > zero || (imag(alpha) != zero)) {
        // First estimate of beta
        real_t temp = (is_real<T>) ? lapy2(real(alpha), xnorm)
                                   : lapy3(real(alpha), imag(alpha), xnorm);
        real_t beta = (real(alpha) < zero) ? temp : -temp;

        // Scale if needed
        idx_t knt = 0;
        if (abs(beta) < safemin) {
            while ((abs(beta) < safemin) && (knt < 20)) {
                knt++;
                scal(rsafemin, x);
                beta *= rsafemin;
                alpha *= rsafemin;
            }
            xnorm = nrm2(x);
            temp = (is_real<T>) ? lapy2(real(alpha), xnorm)
                                : lapy3(real(alpha), imag(alpha), xnorm);
            beta = (real(alpha) < zero) ? temp : -temp;
        }

        // compute tau and y
        tau = (beta - alpha) / beta;
        rscl(alpha - beta, x);
        if (storeMode == StoreV::Rowwise) tau = conj(tau);

        // Scale if needed
        for (idx_t j = 0; j < knt; ++j)
            beta *= safemin;

        // Store beta in alpha
        alpha = beta;
    }
    flops += 3 * (size(x) + idx_t(1));
}

template <TLAPACK_DIRECTION direction_t,
          TLAPACK_STOREV storage_t,
          TLAPACK_VECTOR vector_t,
          enable_if_t<std::is_convertible_v<direction_t, Direction>, int> = 0>
inline void larfg_flops(direction_t direction,
                        storage_t storeMode,
                        vector_t& v,
                        type_t<vector_t>& tau,
                        unsigned int& flops)
{
    using idx_t = size_type<vector_t>;
    using range = pair<idx_t, idx_t>;

    // check arguments
    tlapack_check_false(direction != Direction::Backward &&
                        direction != Direction::Forward);

    const idx_t alpha_idx = (direction == Direction::Forward) ? 0 : size(v) - 1;

    auto x =
        slice(v, (direction == Direction::Forward) ? range(1, size(v))
                                                   : range(0, size(v) - 1));
    type_t<vector_t> alpha = v[alpha_idx];
    larfg_flops(storeMode, alpha, x, tau, flops);
    v[alpha_idx] = alpha;
}

template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixB_t,
          TLAPACK_SCALAR alpha_t,
          class T = type_t<matrixB_t>,
          disable_if_allow_optblas_t<pair<matrixA_t, T>,
                                     pair<matrixB_t, T>,
                                     pair<alpha_t, T> > = 0>
inline void trmm_flops(Side side,
                       Uplo uplo,
                       Op trans,
                       Diag diag,
                       const alpha_t& alpha,
                       const matrixA_t& A,
                       matrixB_t& B,
                       unsigned int& flops)
{
    // data traits
    using TA = type_t<matrixA_t>;
    using TB = type_t<matrixB_t>;
    using idx_t = size_type<matrixA_t>;

    // constants
    const idx_t m = nrows(B);
    const idx_t n = ncols(B);

    // check arguments
    tlapack_check_false(side != Side::Left && side != Side::Right);
    tlapack_check_false(uplo != Uplo::Lower && uplo != Uplo::Upper);
    tlapack_check_false(trans != Op::NoTrans && trans != Op::Trans &&
                        trans != Op::ConjTrans);
    tlapack_check_false(diag != Diag::NonUnit && diag != Diag::Unit);
    tlapack_check_false(nrows(A) != ncols(A));
    tlapack_check_false(nrows(A) != ((side == Side::Left) ? m : n));

    if (side == Side::Left) {
        if (trans == Op::NoTrans) {
            using scalar_t = scalar_type<alpha_t, TB>;
            if (uplo == Uplo::Upper) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t k = 0; k < m; ++k) {
                        const scalar_t alphaBkj = alpha * B(k, j);
                        for (idx_t i = 0; i < k; ++i)
                            B(i, j) += A(i, k) * alphaBkj;
                        B(k, j) = (diag == Diag::NonUnit) ? A(k, k) * alphaBkj
                                                          : alphaBkj;
                    }
                }
            }
            else {  // uplo == Uplo::Lower
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t k = m - 1; k != idx_t(-1); --k) {
                        const scalar_t alphaBkj = alpha * B(k, j);
                        B(k, j) = (diag == Diag::NonUnit) ? A(k, k) * alphaBkj
                                                          : alphaBkj;
                        for (idx_t i = k + 1; i < m; ++i)
                            B(i, j) += A(i, k) * alphaBkj;
                    }
                }
            }
        }
        else if (trans == Op::Trans) {
            using scalar_t = scalar_type<TA, TB>;
            if (uplo == Uplo::Upper) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = m - 1; i != idx_t(-1); --i) {
                        scalar_t sum = (diag == Diag::NonUnit)
                                           ? A(i, i) * B(i, j)
                                           : B(i, j);
                        for (idx_t k = 0; k < i; ++k)
                            sum += A(k, i) * B(k, j);
                        B(i, j) = alpha * sum;
                    }
                }
            }
            else {  // uplo == Uplo::Lower
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = 0; i < m; ++i) {
                        scalar_t sum = (diag == Diag::NonUnit)
                                           ? A(i, i) * B(i, j)
                                           : B(i, j);
                        for (idx_t k = i + 1; k < m; ++k)
                            sum += A(k, i) * B(k, j);
                        B(i, j) = alpha * sum;
                    }
                }
            }
        }
        else {  // trans == Op::ConjTrans
            using scalar_t = scalar_type<TA, TB>;
            if (uplo == Uplo::Upper) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = m - 1; i != idx_t(-1); --i) {
                        scalar_t sum = (diag == Diag::NonUnit)
                                           ? conj(A(i, i)) * B(i, j)
                                           : B(i, j);
                        for (idx_t k = 0; k < i; ++k)
                            sum += conj(A(k, i)) * B(k, j);
                        B(i, j) = alpha * sum;
                    }
                }
            }
            else {  // uplo == Uplo::Lower
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = 0; i < m; ++i) {
                        scalar_t sum = (diag == Diag::NonUnit)
                                           ? conj(A(i, i)) * B(i, j)
                                           : B(i, j);
                        for (idx_t k = i + 1; k < m; ++k)
                            sum += conj(A(k, i)) * B(k, j);
                        B(i, j) = alpha * sum;
                    }
                }
            }
        }
    }
    else {  // side == Side::Right
        using scalar_t = scalar_type<alpha_t, TA>;
        if (trans == Op::NoTrans) {
            if (uplo == Uplo::Upper) {
                for (idx_t j = n - 1; j != idx_t(-1); --j) {
                    {
                        const scalar_t alphaAjj =
                            (diag == Diag::NonUnit) ? alpha * A(j, j) : alpha;
                        for (idx_t i = 0; i < m; ++i)
                            B(i, j) *= alphaAjj;
                    }
                    for (idx_t k = 0; k < j; ++k) {
                        const scalar_t alphaAkj = alpha * A(k, j);
                        for (idx_t i = 0; i < m; ++i)
                            B(i, j) += B(i, k) * alphaAkj;
                    }
                }
            }
            else {  // uplo == Uplo::Lower
                for (idx_t j = 0; j < n; ++j) {
                    {
                        const scalar_t alphaAjj =
                            (diag == Diag::NonUnit) ? alpha * A(j, j) : alpha;
                        for (idx_t i = 0; i < m; ++i)
                            B(i, j) *= alphaAjj;
                    }
                    for (idx_t k = j + 1; k < n; ++k) {
                        const scalar_t alphaAkj = alpha * A(k, j);
                        for (idx_t i = 0; i < m; ++i)
                            B(i, j) += B(i, k) * alphaAkj;
                    }
                }
            }
        }
        else if (trans == Op::Trans) {
            if (uplo == Uplo::Upper) {
                for (idx_t k = 0; k < n; ++k) {
                    for (idx_t j = 0; j < k; ++j) {
                        const scalar_t alphaAjk = alpha * A(j, k);
                        for (idx_t i = 0; i < m; ++i)
                            B(i, j) += B(i, k) * alphaAjk;
                    }
                    {
                        const scalar_t alphaAkk =
                            (diag == Diag::NonUnit) ? alpha * A(k, k) : alpha;
                        for (idx_t i = 0; i < m; ++i)
                            B(i, k) *= alphaAkk;
                    }
                }
            }
            else {  // uplo == Uplo::Lower
                for (idx_t k = n - 1; k != idx_t(-1); --k) {
                    for (idx_t j = k + 1; j < n; ++j) {
                        const scalar_t alphaAjk = alpha * A(j, k);
                        for (idx_t i = 0; i < m; ++i)
                            B(i, j) += B(i, k) * alphaAjk;
                    }
                    {
                        const scalar_t alphaAkk =
                            (diag == Diag::NonUnit) ? alpha * A(k, k) : alpha;
                        for (idx_t i = 0; i < m; ++i)
                            B(i, k) *= alphaAkk;
                    }
                }
            }
        }
        else {  // trans == Op::ConjTrans
            if (uplo == Uplo::Upper) {
                for (idx_t k = 0; k < n; ++k) {
                    for (idx_t j = 0; j < k; ++j) {
                        const scalar_t alphaAjk = alpha * conj(A(j, k));
                        for (idx_t i = 0; i < m; ++i)
                            B(i, j) += B(i, k) * alphaAjk;
                    }
                    {
                        const scalar_t alphaAkk = (diag == Diag::NonUnit)
                                                      ? alpha * conj(A(k, k))
                                                      : alpha;
                        for (idx_t i = 0; i < m; ++i)
                            B(i, k) *= alphaAkk;
                    }
                }
            }
            else {  // uplo == Uplo::Lower
                for (idx_t k = n - 1; k != idx_t(-1); --k) {
                    for (idx_t j = k + 1; j < n; ++j) {
                        const scalar_t alphaAjk = alpha * conj(A(j, k));
                        for (idx_t i = 0; i < m; ++i)
                            B(i, j) += B(i, k) * alphaAjk;
                    }
                    {
                        const scalar_t alphaAkk = (diag == Diag::NonUnit)
                                                      ? alpha * conj(A(k, k))
                                                      : alpha;
                        for (idx_t i = 0; i < m; ++i)
                            B(i, k) *= alphaAkk;
                    }
                }
            }
        }
    }

    if (side == Side::Left)
        flops += 1.0 * nrows(A) * ncols(A) * ncols(B);
    else
        flops += 1.0 * nrows(B) * ncols(B) * nrows(A);
}

template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixB_t,
          TLAPACK_MATRIX matrixC_t,
          TLAPACK_SCALAR alpha_t,
          TLAPACK_SCALAR beta_t,
          class T = type_t<matrixC_t>,
          disable_if_allow_optblas_t<pair<matrixA_t, T>,
                                     pair<matrixB_t, T>,
                                     pair<matrixC_t, T>,
                                     pair<alpha_t, T>,
                                     pair<beta_t, T> > = 0>
inline void gemm_flops(Op transA,
                       Op transB,
                       const alpha_t& alpha,
                       const matrixA_t& A,
                       const matrixB_t& B,
                       const beta_t& beta,
                       matrixC_t& C,
                       unsigned int& flops)
{
    // data traits
    using TA = type_t<matrixA_t>;
    using TB = type_t<matrixB_t>;
    using idx_t = size_type<matrixA_t>;

    // constants
    const idx_t m = (transA == Op::NoTrans) ? nrows(A) : ncols(A);
    const idx_t n = (transB == Op::NoTrans) ? ncols(B) : nrows(B);
    const idx_t k = (transA == Op::NoTrans) ? ncols(A) : nrows(A);

    // check arguments
    tlapack_check_false(transA != Op::NoTrans && transA != Op::Trans &&
                        transA != Op::ConjTrans);
    tlapack_check_false(transB != Op::NoTrans && transB != Op::Trans &&
                        transB != Op::ConjTrans);
    tlapack_check_false((idx_t)nrows(C) != m);
    tlapack_check_false((idx_t)ncols(C) != n);
    tlapack_check_false(
        (idx_t)((transB == Op::NoTrans) ? nrows(B) : ncols(B)) != k);

    if (transA == Op::NoTrans) {
        using scalar_t = scalar_type<alpha_t, TB>;

        if (transB == Op::NoTrans) {
            for (idx_t j = 0; j < n; ++j) {
                for (idx_t i = 0; i < m; ++i)
                    C(i, j) *= beta;
                for (idx_t l = 0; l < k; ++l) {
                    const scalar_t alphaTimesblj = alpha * B(l, j);
                    for (idx_t i = 0; i < m; ++i)
                        C(i, j) += A(i, l) * alphaTimesblj;
                }
            }
        }
        else if (transB == Op::Trans) {
            for (idx_t j = 0; j < n; ++j) {
                for (idx_t i = 0; i < m; ++i)
                    C(i, j) *= beta;
                for (idx_t l = 0; l < k; ++l) {
                    const scalar_t alphaTimesbjl = alpha * B(j, l);
                    for (idx_t i = 0; i < m; ++i)
                        C(i, j) += A(i, l) * alphaTimesbjl;
                }
            }
        }
        else {  // transB == Op::ConjTrans
            for (idx_t j = 0; j < n; ++j) {
                for (idx_t i = 0; i < m; ++i)
                    C(i, j) *= beta;
                for (idx_t l = 0; l < k; ++l) {
                    const scalar_t alphaTimesbjl = alpha * conj(B(j, l));
                    for (idx_t i = 0; i < m; ++i)
                        C(i, j) += A(i, l) * alphaTimesbjl;
                }
            }
        }
    }
    else if (transA == Op::Trans) {
        using scalar_t = scalar_type<TA, TB>;

        if (transB == Op::NoTrans) {
            for (idx_t j = 0; j < n; ++j) {
                for (idx_t i = 0; i < m; ++i) {
                    scalar_t sum(0);
                    for (idx_t l = 0; l < k; ++l)
                        sum += A(l, i) * B(l, j);
                    C(i, j) = alpha * sum + beta * C(i, j);
                }
            }
        }
        else if (transB == Op::Trans) {
            for (idx_t j = 0; j < n; ++j) {
                for (idx_t i = 0; i < m; ++i) {
                    scalar_t sum(0);
                    for (idx_t l = 0; l < k; ++l)
                        sum += A(l, i) * B(j, l);
                    C(i, j) = alpha * sum + beta * C(i, j);
                }
            }
        }
        else {  // transB == Op::ConjTrans
            for (idx_t j = 0; j < n; ++j) {
                for (idx_t i = 0; i < m; ++i) {
                    scalar_t sum(0);
                    for (idx_t l = 0; l < k; ++l)
                        sum += A(l, i) * conj(B(j, l));
                    C(i, j) = alpha * sum + beta * C(i, j);
                }
            }
        }
    }
    else {  // transA == Op::ConjTrans

        using scalar_t = scalar_type<TA, TB>;

        if (transB == Op::NoTrans) {
            for (idx_t j = 0; j < n; ++j) {
                for (idx_t i = 0; i < m; ++i) {
                    scalar_t sum(0);
                    for (idx_t l = 0; l < k; ++l)
                        sum += conj(A(l, i)) * B(l, j);
                    C(i, j) = alpha * sum + beta * C(i, j);
                }
            }
        }
        else if (transB == Op::Trans) {
            for (idx_t j = 0; j < n; ++j) {
                for (idx_t i = 0; i < m; ++i) {
                    scalar_t sum(0);
                    for (idx_t l = 0; l < k; ++l)
                        sum += conj(A(l, i)) * B(j, l);
                    C(i, j) = alpha * sum + beta * C(i, j);
                }
            }
        }
        else {  // transB == Op::ConjTrans
            for (idx_t j = 0; j < n; ++j) {
                for (idx_t i = 0; i < m; ++i) {
                    scalar_t sum(0);
                    for (idx_t l = 0; l < k; ++l)
                        sum += A(l, i) * B(j, l);  // little improvement here
                    C(i, j) = alpha * conj(sum) + beta * C(i, j);
                }
            }
        }
    }
    flops += 2 * m * n * k;
}

template <class T, TLAPACK_MATRIX matrix_a, TLAPACK_MATRIX matrix_h>

void geqrt3_flops(matrix_a& A, matrix_h& Tmatrix, unsigned int& flopsQR)
{
    using std::size_t;
    using matrix_t = LegacyMatrix<T>;
    using idx_t = size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    auto info = 0;
    if (m < n) {
        std::cout << "Error: m < n" << std::endl;
        info = -1;
    }

    if (info != 0) {
        return;
    }

    if (n == 1) {
        // Turn the single column into a vector
        auto a_vector = col(A, 0);

        // Populate matrix T with an elementary reflector
        larfg_flops(Direction::Forward, StoreV::Columnwise, a_vector,
                    Tmatrix(0, 0), flopsQR);
    }
    else {
        // Define slice sizes
        auto n1 = n / 2;
        auto n2 = n - n1;
        auto m1 = n1;
        auto m2 = n2 + n1;
        auto m3 = m;

        // slices
        auto A1 = slice(A, range(0, m), range(0, n1));
        auto A11 = slice(A, range(0, m1), range(0, n1));
        auto A12 = slice(A, range(0, m1), range(n1, n));
        auto A21 = slice(A, range(m1, m2), range(0, n1));
        auto A22 = slice(A, range(m1, m2), range(n1, n));
        auto A22_32 = slice(A, range(m1, m3), range(n1, n));
        auto A31 = slice(A, range(m2, m3), range(0, n1));
        auto A32 = slice(A, range(m2, m3), range(n1, n));
        auto T11 = slice(Tmatrix, range(0, n1), range(0, n1));
        auto T12 = slice(Tmatrix, range(0, n1), range(n1, n));
        auto T22 = slice(Tmatrix, range(n1, n), range(n1, n));

        // Cut down to one leading column
        geqrt3_flops<T>(A1, T11, flopsQR);

        // step 2: Copy A12 into T12
        // no additional flops, just copy
        lacpy(Uplo::General, A12, T12);

        // step 3: A11T * T12 = T12

        trmm_flops(Side::Left, Uplo::Lower, Op::ConjTrans, Diag::Unit,
                   static_cast<T>(1.0), A11, T12, flopsQR);

        // step 4: T12 + (A21T * A22) = T12

        gemm_flops(Op::ConjTrans, Op::NoTrans, static_cast<T>(1.0), A21, A22,
                   static_cast<T>(1.0), T12, flopsQR);
        // T12 + (A31T * A32) = T12

        gemm_flops(Op::ConjTrans, Op::NoTrans, static_cast<T>(1.0), A31, A32,
                   static_cast<T>(1.0), T12, flopsQR);

        // step 5: T11T * T12 = T12
        trmm_flops(Side::Left, Uplo::Upper, Op::ConjTrans, Diag::NonUnit,
                   static_cast<T>(1.0), T11, T12, flopsQR);

        // step 6: A22 - (A21 * T12) = A22
        gemm_flops(Op::NoTrans, Op::NoTrans, static_cast<T>(-1.0), A21, T12,
                   static_cast<T>(1.0), A22, flopsQR);

        // A32 - (A31 * T12) = A32
        gemm_flops(Op::NoTrans, Op::NoTrans, static_cast<T>(-1.0), A31, T12,
                   static_cast<T>(1.0), A32, flopsQR);

        // step 7: A11 * T12 = T12
        trmm_flops(Side::Left, Uplo::Lower, Op::NoTrans, Diag::Unit,
                   static_cast<T>(1.0), A11, T12, flopsQR);

        // step 8: A12 - T12 = A12
        for (idx_t j = 0; j < n2; ++j) {
            for (idx_t i = 0; i < m1; ++i) {
                A12(i, j) -= T12(i, j);
            }
        }
        flopsQR += 1.0 * m1 * n2;

        geqrt3_flops<T>(A22_32, T22, flopsQR);

        // step 10:
        for (idx_t j = 0; j < n2; ++j) {
            for (idx_t i = 0; i < m1; ++i) {
                if constexpr (is_complex<T>)
                    T12(i, j) = std::conj(A21(j, i));
                else
                    T12(i, j) = A21(j, i);
            }
        }

        // step 11:
        trmm_flops(Side::Right, Uplo::Lower, Op ::NoTrans, Diag::Unit,
                   static_cast<T>(1.0), A22, T12, flopsQR);
        // step 12:
        gemm_flops(Op::ConjTrans, Op::NoTrans, static_cast<T>(1.0), A31, A32,
                   static_cast<T>(1.0), T12, flopsQR);
        // step 13:
        trmm_flops(Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
                   static_cast<T>(-1.0), T11, T12, flopsQR);

        // step 14:
        trmm_flops(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
                   static_cast<T>(1.0), T22, T12, flopsQR);
    }
}
}  // namespace tlapack

#endif  // TLAPACK_GEQR2_HH