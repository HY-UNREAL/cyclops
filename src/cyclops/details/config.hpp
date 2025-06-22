#pragma once

#include "cyclops/details/type.hpp"
#include <memory>

namespace cyclops::config::measurement {
  struct KeyframeDetectionThreshold {
    int min_novel_landmarks;
    double min_average_parallax;
  };

  struct KeyframeWindowConfig {
    int optimization_phase_max_keyframes;
    int initialization_phase_max_keyframes;
  };

  struct ImageUpdateThrottlingConfig {
    double update_rate_target;
    double update_rate_smoothing_window_size;
  };
}  // namespace cyclops::config::measurement

namespace cyclops::config::initializer::vision {
  struct TwoViewGeometryModelSelectionConfig {
    int ransac_batch_size;
    double homography_selection_score_threshold;
  };

  struct TwoViewTriangulationSuccessThreshold {
    double min_p_value;
    double max_normalized_deviation;
  };

  struct TwoViewMotionHypothesisTestThreshold {
    int min_triangulation_success;
    double min_imu_rotation_consistency_p_value;
  };

  struct TwoViewConfig {
    TwoViewGeometryModelSelectionConfig model_selection;
    TwoViewTriangulationSuccessThreshold triangulation_acceptance;
    TwoViewMotionHypothesisTestThreshold motion_hypothesis;
  };

  struct MultiViewConfig {
    int bundle_adjustment_max_iterations;
    double bundle_adjustment_max_solver_time;
    double scale_gauge_soft_constraint_deviation;
  };

  struct SolutionAcceptanceThreshold {
    double min_significant_probability;
    double min_inlier_ratio;

    double gyro_motion_min_p_value;
    double gyro_bias_min_p_value;
  };
}  // namespace cyclops::config::initializer::vision

namespace cyclops::config::initializer::imu {
  struct ScaleSamplingConfig {
    double sampling_domain_lowerbound;
    double sampling_domain_upperbound;
    int samples_count;

    double min_evaluation_success_rate;
  };

  struct SolutionRefinementConfig {
    int max_iteration;
    double stepsize_tolerance;
    double gradient_tolerance;

    double duplicate_tolerance;
  };

  struct SolutionCandidateThreshold {
    double cost_significance;
  };

  struct SolutionAcceptanceThreshold {
    double max_scale_log_deviation;
    double max_normalized_gravity_deviation;
    double max_normalized_velocity_deviation;
    double max_sfm_perturbation;

    double translation_match_min_p_value;
  };
}  // namespace cyclops::config::initializer::imu

namespace cyclops::config::initializer {
  struct ObservabilityPretestThreshold {
    int min_landmark_overlap;
    int min_keyframes;

    double min_average_parallax;
  };

  struct VisionSolverConfig {
    double gyro_bias_prior_stddev;

    double feature_point_isotropic_noise;
    double bundle_adjustment_robust_kernel_radius;

    vision::TwoViewConfig two_view;
    vision::MultiViewConfig multiview;
    vision::SolutionAcceptanceThreshold acceptance_test;

    static VisionSolverConfig CreateDefault();
  };

  struct ImuSolverConfig {
    bool imu_only;

    imu::ScaleSamplingConfig sampling;
    imu::SolutionRefinementConfig refinement;
    imu::SolutionCandidateThreshold candidate_test;
    imu::SolutionAcceptanceThreshold acceptance_test;

    static ImuSolverConfig CreateDefault();
  };

  struct InitializationConfig {
    ObservabilityPretestThreshold observability_pretest;

    VisionSolverConfig vision;
    ImuSolverConfig imu;

    static InitializationConfig CreateDefault();
  };
}  // namespace cyclops::config::initializer

namespace cyclops::config::estimation {
  struct OptimizerConfig {
    int max_num_iterations;
    double max_solver_time_in_seconds;
  };

  struct LandmarkAcceptanceThreshold {
    double inlier_min_information_index;
    double inlier_min_depth;
    double inlier_mahalanobis_error;
    double mapping_acceptance_min_eigenvalue;
  };

  struct FaultDetectionThreshold {
    double min_landmark_accept_rate;
    double min_final_cost_p_value;

    int max_landmark_update_failures;
    int max_final_cost_sanity_failures;
  };

  struct EstimatorConfig {
    OptimizerConfig optimizer;
    LandmarkAcceptanceThreshold landmark_acceptance;
    FaultDetectionThreshold fault_detection;

    static EstimatorConfig CreateDefault();
  };
}  // namespace cyclops::config::estimation

namespace cyclops {
  struct SensorStatistics {
    double acc_white_noise;
    double gyr_white_noise;
    double acc_random_walk;
    double gyr_random_walk;
    double acc_bias_prior_stddev;
    double gyr_bias_prior_stddev;
  };

  struct SensorExtrinsics {
    double imu_camera_time_delay;
    SE3Transform imu_camera_transform;
  };

  struct CyclopsConfig {
    double gravity_norm;

    SensorStatistics noise;
    SensorExtrinsics extrinsics;

    config::measurement::KeyframeDetectionThreshold keyframe_detection;
    config::measurement::KeyframeWindowConfig keyframe_window;
    config::measurement::ImageUpdateThrottlingConfig update_throttling;

    config::initializer::InitializationConfig initialization;
    config::estimation::EstimatorConfig estimation;

    static std::unique_ptr<CyclopsConfig> CreateDefault(
      SensorStatistics const& sensor_noise,
      SensorExtrinsics const& sensor_extrinsics);
  };
}  // namespace cyclops
