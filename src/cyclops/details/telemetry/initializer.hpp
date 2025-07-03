#pragma once

#include "cyclops/details/type.hpp"

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

namespace cyclops::telemetry {
  class InitializerTelemetry {
  public:
    virtual ~InitializerTelemetry() = default;
    virtual void reset();

    /* Vision initialization telemetry methods */

    struct ImageObservabilityStatistics {
      int common_features;
      double motion_parallax;
    };
    struct ImageObservabilityPretest {
      std::map<FrameID, ImageObservabilityStatistics> frames;
      std::set<FrameID> connected_frames;
    };
    virtual void onImageObservabilityPretest(
      ImageObservabilityPretest const& test);

    enum VisionBootstrapFailureReason {
      NOT_ENOUGH_CONNECTED_IMAGE_FRAMES,
      NOT_ENOUGH_MOTION_PARALLAX,
      BEST_TWO_VIEW_SELECTION_FAILED,
      TWO_VIEW_GEOMETRY_FAILED,
      MULTI_VIEW_GEOMETRY_FAILED,
      BUNDLE_ADJUSTMENT_FAILED,
    };
    struct VisionBootstrapFailure {
      std::set<FrameID> frames;
      VisionBootstrapFailureReason reason;
    };
    virtual void onVisionFailure(VisionBootstrapFailure const& failure);

    struct BestTwoViewSelection {
      std::set<FrameID> frames;

      FrameID frame_id_1;
      FrameID frame_id_2;
    };
    virtual void onBestTwoViewSelection(BestTwoViewSelection const& selection);

    struct TwoViewGeometry {
      bool acceptable;
      bool rotation_prior_test_passed;
      bool triangulation_test_passed;

      double rotation_prior_p_value;
      int triangulation_success_count;

      SE3Transform motion;
    };
    struct TwoViewMotionHypothesis {
      std::set<FrameID> frames;

      FrameID frame_id_1;
      FrameID frame_id_2;

      std::vector<TwoViewGeometry> candidates;
    };
    virtual void onTwoViewMotionHypothesis(
      TwoViewMotionHypothesis const& hypothesis);

    enum TwoViewGeometryModel { EPIPOLAR, HOMOGRAPHY, BOTH };

    struct TwoViewSolverSuccess {
      std::set<FrameID> frames;

      TwoViewGeometryModel initial_selected_model;
      TwoViewGeometryModel final_selected_model;

      int landmarks_count;
      double homography_expected_inliers;
      double epipolar_expected_inliers;

      std::vector<TwoViewGeometry> candidates;
    };
    virtual void onTwoViewSolverSuccess(TwoViewSolverSuccess const& success);

    struct BundleAdjustmentSolution {
      std::map<FrameID, SE3Transform> camera_motions;
      LandmarkPositions landmarks;
    };
    virtual void onBundleAdjustmentSuccess(
      BundleAdjustmentSolution const& solution);

    struct BundleAdjustmentSanity {
      bool acceptable;

      double inlier_ratio;
      double final_cost_significant_probability;
    };
    struct BundleAdjustmentCandidatesSanity {
      std::set<FrameID> frames;

      std::vector<BundleAdjustmentSanity> candidates_sanity;
    };
    virtual void onBundleAdjustmentSanity(
      BundleAdjustmentCandidatesSanity const& sanity);

    /* IMU initialization telemetry methods */

    struct ImuMatchAttempt {
      int degrees_of_freedom;
      std::set<FrameID> frames;

      std::vector<std::tuple<double, double>> landscape;
      std::vector<std::tuple<double, double>> minima;
    };
    virtual void onImuMatchAttempt(ImuMatchAttempt const& argument);

    struct ImuMatchSolutionPoint {
      double scale;
      double cost;

      Eigen::Vector3d gravity;
      Eigen::Vector3d acc_bias;
      Eigen::Vector3d gyr_bias;
      std::map<FrameID, Eigen::Quaterniond> imu_orientations;
      std::map<FrameID, Eigen::Vector3d> imu_body_velocities;
      std::map<FrameID, Eigen::Vector3d> sfm_positions;
    };

    struct ImuMatchUncertainty {
      double final_cost_significant_probability;
      double scale_log_deviation;
      double gravity_max_deviation;
      double bias_max_deviation;
      double body_velocity_max_deviation;
      double scale_symmetric_translation_error_max_deviation;
    };

    struct ImuMatchAmbiguity {
      std::vector<ImuMatchSolutionPoint> solutions;
      std::vector<ImuMatchUncertainty> uncertainties;
    };
    virtual void onImuMatchAmbiguity(ImuMatchAmbiguity const& argument);

    enum ImuMatchCandidateRejectReason {
      UNCERTAINTY_EVALUATION_FAILED,
      COST_PROBABILITY_INSIGNIFICANT,
      UNDERINFORMATIVE_PARAMETER,
      SCALE_LESS_THAN_ZERO,
    };

    struct ImuMatchReject {
      ImuMatchCandidateRejectReason reason;
      ImuMatchSolutionPoint solution;
      std::optional<ImuMatchUncertainty> uncertainty;
    };
    virtual void onImuMatchReject(ImuMatchReject const& argument);
    virtual void onImuMatchCandidateReject(ImuMatchReject const& argument);

    struct ImuMatchAccept {
      ImuMatchSolutionPoint solution;
      ImuMatchUncertainty uncertainty;
    };
    virtual void onImuMatchAccept(ImuMatchAccept const& argument);

    struct VisionSolutionCandidateDigest {
      bool acceptable;
      std::set<FrameID> keyframes;
    };

    struct ImuSolutionCandidateDigest {
      int vision_solution_index;
      bool acceptable;

      double scale;
      std::set<FrameID> keyframes;
    };

    struct OnFailure {
      std::vector<VisionSolutionCandidateDigest> vision_solutions;
      std::vector<ImuSolutionCandidateDigest> imu_solutions;
    };
    virtual void onFailure(OnFailure const& argument);

    struct OnSuccess {
      int vision_solution_index;

      FrameID initial_motion_frame_id;
      Timestamp initial_motion_frame_timestamp;
      std::map<FrameID, SE3Transform> sfm_camera_pose;

      double cost;
      double scale;
      Eigen::Vector3d gravity;
      std::map<FrameID, ImuMotionState> motions;
    };
    virtual void onSuccess(OnSuccess const& argument);

    static std::unique_ptr<InitializerTelemetry> CreateDefault();
  };
}  // namespace cyclops::telemetry
