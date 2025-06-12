#include "cyclops_tests/random.hpp"

namespace cyclops {
  using Eigen::Matrix2d;
  using Eigen::Matrix3d;
  using Eigen::Quaterniond;
  using Eigen::Vector2d;
  using Eigen::Vector3d;

  template <int dim>
  using Vector = Eigen::Matrix<double, dim, 1>;

  using estimation::LandmarkParameterBlock;
  using estimation::MotionFrameParameterBlock;

  template <int dim, typename concentration_t>
  static Vector<dim> perturbateByConcentration(
    Vector<dim> const& x, concentration_t const& s, std::mt19937& rgen) {
    std::normal_distribution<> random(0);
    Vector<dim> y;
    for (int i = 0; i < dim; i++)
      y[i] = random(rgen);
    return x + s * y;
  }

  static Quaterniond rotateQuaternion(
    Quaterniond const& q, Vector3d const& theta) {
    return q * Eigen::AngleAxisd(theta.norm(), theta.normalized());
  }

  template <typename concentration_t>
  static Quaterniond perturbateByConcentration(
    Quaterniond const& q, concentration_t const& s, std::mt19937& rgen) {
    std::normal_distribution<> random(0);
    return rotateQuaternion(
      q, s * Vector3d(random(rgen), random(rgen), random(rgen)));
  }

  double perturbate(double const& x, double const& s, std::mt19937& rgen) {
    return x + s * std::normal_distribution<>(0)(rgen);
  }

  Vector2d perturbate(Vector2d const& x, double const& S, std::mt19937& rgen) {
    return perturbateByConcentration(x, S, rgen);
  }

  Vector2d perturbate(
    Vector2d const& x, Matrix2d const& S, std::mt19937& rgen) {
    return perturbateByConcentration(x, S, rgen);
  }

  Vector3d perturbate(Vector3d const& x, double const& S, std::mt19937& rgen) {
    return perturbateByConcentration(x, S, rgen);
  }

  Vector3d perturbate(
    Vector3d const& x, Matrix3d const& S, std::mt19937& rgen) {
    return perturbateByConcentration(x, S, rgen);
  }

  Quaterniond perturbate(
    Quaterniond const& q, double const& S, std::mt19937& rgen) {
    return perturbateByConcentration(q, S, rgen);
  }

  Quaterniond perturbate(
    Quaterniond const& q, Matrix3d const& S, std::mt19937& rgen) {
    return perturbateByConcentration(q, S, rgen);
  }

  ImuMotionState perturbate(
    ImuMotionState const& x, double S, std::mt19937& rgen) {
    return ImuMotionState {
      .orientation = perturbate(x.orientation, S, rgen),
      .position = perturbate(x.position, S, rgen),
      .velocity = perturbate(x.velocity, S, rgen),
    };
  }

  MotionFrameParameterBlock makePerturbatedFrameState(
    ImuMotionState const& state, double perturbation, std::mt19937& rgen) {
    MotionFrameParameterBlock x;
    Eigen::Map<Quaterniond>(x.data()) =
      perturbate(state.orientation, perturbation, rgen);
    Eigen::Map<Vector3d>(x.data() + 4) =
      perturbate(state.position, perturbation, rgen);
    Eigen::Map<Vector3d>(x.data() + 7) =
      perturbate(state.velocity, perturbation, rgen);
    Eigen::Map<Vector3d>(x.data() + 10) =
      perturbate(Vector3d::Zero().eval(), perturbation, rgen);
    Eigen::Map<Vector3d>(x.data() + 13) =
      perturbate(Vector3d::Zero().eval(), perturbation, rgen);
    return x;
  }

  LandmarkParameterBlock makePerturbatedLandmarkState(
    Vector3d const& landmark, double perturbation, std::mt19937& rgen) {
    LandmarkParameterBlock x;
    Eigen::Map<Vector3d>(x.data()) = perturbate(landmark, perturbation, rgen);
    return x;
  }
}  // namespace cyclops
