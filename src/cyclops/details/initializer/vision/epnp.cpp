#include "cyclops/details/initializer/vision/epnp.hpp"
#include "cyclops/details/utils/vision.hpp"

#include "cyclops/details/logging.hpp"

#include <spdlog/spdlog.h>
#include <range/v3/all.hpp>

#include <vector>

/*
 * The code here is non-opencv modified version of:
 * https://github.com/cvlab-epfl/EPnP/tree/5abc3cfa76e8e92e5a8f4be0370bbe7da246065e
 *
 * Copyright (c) 2009, V. Lepetit, EPFL
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are
 * those of the authors and should not be interpreted as representing official
 * policies, either expressed or implied, of the FreeBSD Project.
 */
namespace cyclops::initializer {
  namespace views = ranges::views;

  using std::map;
  using std::vector;

  using Eigen::Matrix3d;
  using Eigen::Matrix4d;
  using Eigen::Vector2d;
  using Eigen::Vector3d;
  using Eigen::Vector4d;

  using Vector5d = Eigen::Matrix<double, 5, 1>;
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  using Vector10d = Eigen::Matrix<double, 10, 1>;
  using Vector12d = Eigen::Matrix<double, 12, 1>;

  using Matrix3Xd = Eigen::Matrix<double, 3, Eigen::Dynamic>;
  using Matrix4Xd = Eigen::Matrix<double, 4, Eigen::Dynamic>;
  using MatrixX12d = Eigen::Matrix<double, Eigen::Dynamic, 12>;
  using Matrix12d = Eigen::Matrix<double, 12, 12>;
  using Matrix3_4d = Eigen::Matrix<double, 3, 4>;
  using Matrix6_3d = Eigen::Matrix<double, 6, 3>;
  using Matrix6_4d = Eigen::Matrix<double, 6, 4>;
  using Matrix6_5d = Eigen::Matrix<double, 6, 5>;
  using Matrix6_10d = Eigen::Matrix<double, 6, 10>;

  static auto const zero = Vector3d(0., 0., 0.);

  static Vector12d selectControlPoints(
    map<LandmarkID, PnpImagePoint> const& image_points) {
    auto landmark_positions = image_points | views::values |
      views::transform([](auto const& info) { return info.position; }) |
      ranges::to<vector<Vector3d>>;
    auto n = image_points.size();
    auto center = (ranges::accumulate(landmark_positions, zero) / n).eval();

    Matrix3Xd X(3, n);
    for (auto const& [i, f] : views::enumerate(landmark_positions))
      X.col(i) = f - center;

    Matrix3d X_XT = X * X.transpose();
    Eigen::JacobiSVD<Matrix3d> svd(X_XT, Eigen::ComputeFullU);
    auto const& s = svd.singularValues();
    auto const& U = svd.matrixU();

    Vector12d result;
    result.head<3>() = center;
    for (size_t i = 0; i < 3; i++) {
      auto s_i = std::sqrt(std::max(s(i), 0.) / n);
      result.segment<3>(3 * i + 3) = center + s_i * U.col(i);
    }
    return result;
  }

  static Vector3d seg3(Vector12d const& x_w, int i) {
    return x_w.segment<3>(3 * i);
  }

  static Matrix4Xd evaluateAlpha(
    Vector12d const& x_w, map<LandmarkID, PnpImagePoint> const& image_points) {
    auto n = image_points.size();

    Matrix3d CC;
    // clang-format off
    CC <<
      seg3(x_w, 1) - seg3(x_w, 0),
      seg3(x_w, 2) - seg3(x_w, 0),
      seg3(x_w, 3) - seg3(x_w, 0)
    ;
    // clang-format on
    Matrix3d const CC_inv = CC.inverse();

    Matrix4Xd result(4, n);
    for (auto const& [i, info] :
         views::enumerate(image_points | views::values)) {
      auto const& p_i = info.position;
      auto _ = Vector3d(CC_inv * (p_i - seg3(x_w, 0)));
      auto a1 = _.x();
      auto a2 = _.y();
      auto a3 = _.z();
      auto a0 = 1 - a1 - a2 - a3;

      result.col(i) << a0, a1, a2, a3;
    }
    return result;
  }

  static MatrixX12d evaluateM(
    Matrix4Xd const& alpha,
    map<LandmarkID, PnpImagePoint> const& image_points) {
    auto n = image_points.size();

    MatrixX12d M(2 * n, 12);
    for (auto const& [i, info] :
         views::enumerate(image_points | views::values)) {
      auto const& u_i = info.observation;
      auto a0 = alpha.col(i).x();
      auto a1 = alpha.col(i).y();
      auto a2 = alpha.col(i).z();
      auto a3 = alpha.col(i).w();

      // clang-format off
      M.row(2 * i) <<
        +a0, +0., -a0 * u_i.x(),
        +a1, +0., -a1 * u_i.x(),
        +a2, +0., -a2 * u_i.x(),
        +a3, +0., -a3 * u_i.x()
      ;
      M.row(2 * i + 1) <<
        +0., +a0, -a0 * u_i.y(),
        +0., +a1, -a1 * u_i.y(),
        +0., +a2, -a2 * u_i.y(),
        +0., +a3, -a3 * u_i.y()
      ;
      // clang-format on
    }
    return M;
  }

  static Matrix6_10d evaluateL6x10(Matrix12d const& U) {
    Matrix6_10d L;

    Vector12d v[4] = {U.col(11), U.col(10), U.col(9), U.col(8)};
    Vector3d dv[4][6];

    for (int i = 0; i < 4; i++) {
      auto _4c2 =
        views::cartesian_product(views::ints(0, 4), views::ints(0, 4)) |
        views::filter([](auto _) {
          auto const& [a, b] = _;
          return a < b;
        }) |
        ranges::to<vector>;
      for (auto [j, comb] : views::enumerate(_4c2)) {
        auto [a, b] = comb;
        dv[i][j] = v[i].segment<3>(3 * a) - v[i].segment<3>(3 * b);
      }
    }
    for (int i = 0; i < 6; i++) {
      L(i, 0) = 1. * dv[0][i].dot(dv[0][i]);
      L(i, 1) = 2. * dv[0][i].dot(dv[1][i]);
      L(i, 2) = 1. * dv[1][i].dot(dv[1][i]);
      L(i, 3) = 2. * dv[0][i].dot(dv[2][i]);
      L(i, 4) = 2. * dv[1][i].dot(dv[2][i]);
      L(i, 5) = 1. * dv[2][i].dot(dv[2][i]);
      L(i, 6) = 2. * dv[0][i].dot(dv[3][i]);
      L(i, 7) = 2. * dv[1][i].dot(dv[3][i]);
      L(i, 8) = 2. * dv[2][i].dot(dv[3][i]);
      L(i, 9) = 1. * dv[3][i].dot(dv[3][i]);
    }
    return L;
  }

  static double dist2(Vector3d const& v1, Vector3d const& v2) {
    Vector3d dv = v1 - v2;
    return dv.dot(dv);
  }

  static Vector6d evaluateRho(Vector12d const& x_w) {
    Vector6d rho;
    // clang-format off
    rho <<
      dist2(seg3(x_w, 0), seg3(x_w, 1)),
      dist2(seg3(x_w, 0), seg3(x_w, 2)),
      dist2(seg3(x_w, 0), seg3(x_w, 3)),
      dist2(seg3(x_w, 1), seg3(x_w, 2)),
      dist2(seg3(x_w, 1), seg3(x_w, 3)),
      dist2(seg3(x_w, 2), seg3(x_w, 3))
    ;
    // clang-format on
    return rho;
  }

  static Vector4d findBetasApprox1(Matrix6_10d const& L, Vector6d const& rho) {
    Matrix6_4d L_6x4;
    L_6x4 << L.col(0), L.col(1), L.col(3), L.col(6);
    Vector4d b4 =
      L_6x4.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(rho);

    double sign = b4.x() < 0 ? -1. : +1.;

    Vector4d beta;
    beta.x() = sqrt(std::max(sign * b4.x(), 0.));
    beta.tail<3>() = sign * b4.tail<3>() / beta.x();
    return beta;
  }

  static Vector4d findBetasApprox2(Matrix6_10d const& L, Vector6d const& rho) {
    Matrix6_3d L_6x3 = L.leftCols<3>();
    Vector3d b3 =
      L_6x3.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(rho);

    Vector4d beta;

    double sign = b3.x() < 0 ? -1. : +1.;
    beta.x() = std::sqrt(std::max(sign * b3.x(), 0.));
    beta.y() = std::sqrt(std::max(sign * b3.z(), 0.));
    if (b3.y() < 0)
      beta.x() = -beta.x();
    beta.z() = 0.;
    beta.w() = 0.;
    return beta;
  }

  static Vector4d findBetasApprox3(Matrix6_10d const& L, Vector6d const& rho) {
    Matrix6_5d L_6x5 = L.leftCols<5>();
    Vector5d b5 =
      L_6x5.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(rho);

    Vector4d beta;

    double sign = b5.x() < 0 ? -1. : +1.;
    beta.x() = std::sqrt(std::max(sign * b5.x(), 0.));
    beta.y() = std::sqrt(std::max(sign * b5.z(), 0.));
    if (b5.y() < 0)
      beta.x() = -beta.x();
    beta.z() = b5.w() / beta.x();
    beta.w() = 0.;
    return beta;
  }

  static Vector4d findBetasApprox(
    int N, Matrix6_10d const& L, Vector6d const& rho) {
    switch (N) {
    case 0:
      return findBetasApprox1(L, rho);
    case 1:
      return findBetasApprox2(L, rho);
    case 2:
      return findBetasApprox3(L, rho);
    default:
      throw "unallowed case N";
    }
  }

  struct GaussNewtonLinearization {
    Matrix6_4d A;
    Vector6d b;
  };

  static GaussNewtonLinearization epnpLinearizeGaussNewton(
    Matrix6_10d const& L_6x10, Vector6d const& rho, Vector4d const& beta) {
    GaussNewtonLinearization result;
    for (int i = 0; i < 6; i++) {
      Matrix4d F;
      // clang-format off
      F <<
#define rowL (L_6x10.row(i))
        2. * rowL(0), 1. * rowL(1), 1. * rowL(3), 1. * rowL(6),
        1. * rowL(1), 2. * rowL(2), 1. * rowL(4), 1. * rowL(7),
        1. * rowL(3), 1. * rowL(4), 2. * rowL(5), 1. * rowL(8),
        1. * rowL(6), 1. * rowL(7), 1. * rowL(8), 2. * rowL(9)
#undef rowL
      ;
      // clang-format on
      result.A.row(i) = F * beta;
    }
    Vector10d b10;

    // clang-format off
    b10 <<
      beta[0] * beta[0],
      beta[0] * beta[1],
      beta[1] * beta[1],
      beta[0] * beta[2],
      beta[1] * beta[2],
      beta[2] * beta[2],
      beta[0] * beta[3],
      beta[1] * beta[3],
      beta[2] * beta[3],
      beta[3] * beta[3]
    ;
    // clang-format on
    result.b = rho - L_6x10 * b10;
    return result;
  }

  static void epnpGaussNewton(
    Vector4d& beta, Matrix6_10d const& L_6x10, Vector6d const& rho,
    size_t iterations = 5) {
    for (int _ = 0; _ < iterations; _++) {
      auto [A, b] = epnpLinearizeGaussNewton(L_6x10, rho, beta);
      beta += A.fullPivHouseholderQr().solve(b);
    }
  }

  static Matrix3_4d evaluateCcs(Matrix12d const& U, Vector4d const& beta) {
    Matrix3_4d result = Matrix3_4d::Zero();
    for (size_t i = 0; i < 4; i++) {
      for (size_t j = 0; j < 4; j++)
        result.col(j) += beta[i] * U.col(11 - i).segment<3>(3 * j);
    }
    return result;
  }

  static vector<Vector3d> evaluatePcs(
    Matrix4Xd const& alpha, Matrix3_4d const& ccs, int n) {
    vector<Vector3d> result;
    result.reserve(n);
    for (int i = 0; i < n; i++)
      result.emplace_back(ccs * alpha.col(i));
    return result;
  }

  static void resolveSign(vector<Vector3d>& pcs, Matrix3_4d& ccs) {
    if (pcs.front().z() < 0.0) {
      ccs = -ccs;
      for (auto& pc_i : pcs)
        pc_i = -pc_i;
    }
  }

  static double evaluateReprojectionError(
    Matrix3d const& R, Vector3d const& t,
    map<LandmarkID, PnpImagePoint> const& image_points) {
    auto result = 0.;
    for (auto const& info : image_points | views::values) {
      auto const& p_w = info.position;
      Vector3d X = R * p_w + t;
      Vector2d u_e = X.head<2>() / X.z();
      Vector2d const& u = info.observation;
      result += std::pow((u - u_e).norm(), 2);
    }
    return std::sqrt(std::max(result / image_points.size(), 0.));
  }

  struct PnpSolution {
    double error;
    RotationPositionPair camera_pose;
  };

  static PnpSolution pnpSolveCameraPose(
    Matrix3_4d const& ccs, vector<Vector3d> const& pcs,
    map<LandmarkID, PnpImagePoint> const& image_points) {
    int n = image_points.size();

    auto landmark_positions = image_points | views::values |
      views::transform([](auto const& info) { return info.position; }) |
      views::cache1;
    auto pc0 = (ranges::accumulate(pcs, zero) / n).eval();
    auto pw0 = (ranges::accumulate(landmark_positions, zero) / n).eval();

    Matrix3d A_BT = Matrix3d::Zero();
    for (auto const& [pc, pw] : views::zip(pcs, landmark_positions))
      A_BT += (pc - pc0) * (pw - pw0).transpose();

    Eigen::JacobiSVD<Matrix3d> svd(
      A_BT, Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto const& U = svd.matrixU();
    auto const& V = svd.matrixV();

    Matrix3d R = U * V.transpose();
    if (R.determinant() < 0)
      R.row(2) = -R.row(2);
    Vector3d t = pc0 - R * pw0;

    auto error = evaluateReprojectionError(R, t, image_points);
    return PnpSolution {
      .error = error,
      .camera_pose = {R.transpose(), -R.transpose() * t},
    };
  }

  static PnpSolution pnpSolveCameraPose(
    Matrix4Xd const& alpha, Matrix12d const& U, Vector4d const& beta,
    map<LandmarkID, PnpImagePoint> const& image_points) {
    int n = image_points.size();
    auto ccs = evaluateCcs(U, beta);
    auto pcs = evaluatePcs(alpha, ccs, n);
    resolveSign(pcs, ccs);
    return pnpSolveCameraPose(ccs, pcs, image_points);
  }

  static vector<PnpSolution> makePnpSolutionCandidates(
    map<LandmarkID, PnpImagePoint> const& image_points,
    int gauss_newton_max_iterations) {
    auto x_w = selectControlPoints(image_points);
    auto alpha = evaluateAlpha(x_w, image_points);
    auto M = evaluateM(alpha, image_points);
    Matrix12d MT_M = M.transpose() * M;

    Eigen::JacobiSVD<Matrix12d> svd(MT_M, Eigen::ComputeFullU);
    auto const& U = svd.matrixU();

    auto L = evaluateL6x10(U);
    auto rho = evaluateRho(x_w);

    // clang-format off
    return views::ints(0, 3)
      | views::transform([&](auto N) {
        auto beta = findBetasApprox(N, L, rho);
        epnpGaussNewton(beta, L, rho, gauss_newton_max_iterations);
        return pnpSolveCameraPose(alpha, U, beta, image_points);
      })
      | ranges::to<vector<PnpSolution>>;
    // clang-format on
  }

  std::optional<RotationPositionPair> solvePnpCameraPose(
    std::map<LandmarkID, PnpImagePoint> const& image_point_set,
    int iterations) {
    if (iterations < 0) {
      __logger__->critical(
        "Invalid EPnP Gauss-Newton max iteration config: {} < 0", iterations);
      return std::nullopt;
    }
    if (image_point_set.size() < 5) {
      __logger__->info(
        "Degenerate pnp: points = {} < 5", image_point_set.size());
      return std::nullopt;
    }

    auto candidates = makePnpSolutionCandidates(image_point_set, iterations);
    if (candidates.empty())
      return std::nullopt;

    auto max = ranges::max_element(
      candidates,
      [](auto const& r1, auto const& r2) { return r1.error > r2.error; });
    return max->camera_pose;
  }
}  // namespace cyclops::initializer
