#include "cyclops/details/initializer/vision_imu/translation_hessian.hpp"
#include "cyclops/details/initializer/vision_imu/translation_analysis.hpp"

namespace cyclops::initializer {
  using Eigen::Matrix3d;
  using Eigen::MatrixXd;
  using Eigen::Vector3d;
  using Eigen::VectorXd;

  using Matrix32d = Eigen::Matrix<double, 3, 2>;

  static std::tuple<Vector3d, int> farthestElementVector(Vector3d const& a) {
    auto max = std::numeric_limits<double>::lowest();
    auto max_i = 0;

    for (int i = 0; i < 3; i++) {
      if (std::abs(a(i)) > max) {
        max = std::abs(a(i));
        max_i = i;
      }
    }

    Vector3d result = Vector3d::Zero();
    result(max_i) = a(max_i) > 0 ? -1 : +1;
    return std::make_tuple(result, max_i);
  }

  static Matrix32d findGravityTangent(Vector3d const& g) {
    // Method in https://math.stackexchange.com/questions/1909536#1909570.
    auto a = g.normalized().eval();
    auto [e, k] = farthestElementVector(a);

    Vector3d v = a - e;
    Vector3d u = v.normalized();
    Matrix3d R = Matrix3d::Identity() - 2 * u * u.transpose();

    Matrix32d T_g;
    int i = 0;
    for (auto j = 0; j < k; j++) {
      T_g.col(i) = R.col(j);
      i++;
    }
    for (auto j = k + 1; j < 3; j++) {
      T_g.col(i) = R.col(j);
      i++;
    }
    return T_g;
  }

  MatrixXd evaluateImuMatchHessian(
    ImuMatchAnalysis const& analysis, double s, VectorXd const& x_I,
    VectorXd const& x_V) {
    auto const& [_1, _2, _3, A_I, B_I, A_V, alpha, beta] = analysis;

    auto n_I = A_I.rows();
    auto m_I = x_I.size();

    auto T_g = findGravityTangent(x_I.head(3));
    auto A_g = A_I.leftCols(3).eval();
    auto A_q = A_I.rightCols(m_I - 3).eval();

    MatrixXd A_I_bar(n_I, m_I - 1);
    A_I_bar << A_g * T_g, A_q;

    auto n_V = A_V.rows();
    auto m_V = x_V.size();

    MatrixXd J(n_I + n_V, m_I + m_V);
    J.block(0, 0, n_I, 1) = (B_I * x_V + beta) * s;
    J.block(0, 1, n_I, m_I - 1) = A_I_bar;
    J.block(0, m_I, n_I, m_V) = B_I * s;
    J.block(n_I, 0, n_V, m_I).setZero();
    J.block(n_I, m_I, n_V, m_V) = A_V;

    return J.transpose() * J;
  }
}  // namespace cyclops::initializer
