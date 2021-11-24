#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // State dimension
  n_x_ = 5;
  
  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.45;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // time when the state is true, in us
  time_us_ = 0.0;

  // Weights of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_.fill(0.5 / (lambda_ + n_aug_));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_ , 2 * n_aug_ + 1);

  // Radar Noise init
  radarNoise_Q_ = MatrixXd(3, 3);
  radarNoise_Q_ << std_radr_ * std_radr_, 0, 0,
                  0, std_radphi_ * std_radphi_,0,
                  0, 0, std_radrd_ * std_radrd_;
  
  // LiDAR Noise init
  lidarNoise_Q_ = MatrixXd(2, 2);
  lidarNoise_Q_ << std_laspx_ * std_laspx_ , 0,
                    0, std_laspy_ * std_laspy_;
  
  // Noise matrix
  Q_ = MatrixXd(2, 2);
  Q_ << std_a_ * std_a_ , 0,
        0, std_a_ * std_a_;
  // Radra NIS 
  radarNIS_ = 0.0;

  // LiDAR NIS
  lidarNIS_ = 0.0;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if(! is_initialized_){
    if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
      double rho = meas_package.raw_measurements_[0]; // distance
      double phi = meas_package.raw_measurements_[1]; // angle
      double rho_dot = meas_package.raw_measurements_[2]; // velocity

      double x = rho * cos(phi);
      double y = rho * sin(phi);
      double vx = rho_dot * cos(phi);
      double vy = rho_dot * sin(phi);
      double v = sqrt(pow(vx, 2) + pow(vy, 2));

      x_ << x, y, 0, 0, 0;
    }else{ // Sensor is LiDAR
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    }

    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;
    return;
  }

  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  Prediction(delta_t);

  if(use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR){
    UpdateRadar(meas_package);
  }

  if(use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER){
    UpdateLidar(meas_package);
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  VectorXd x_aug = VectorXd(n_aug_);
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;

  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(2, 2) = Q_;

  // Square root matrix using Cholesky Decomposition
  MatrixXd L = P_aug.llt().matrixL();

  Xsig_aug.col(0) = x_aug;
  for(int i = 0; i < n_aug_ ; ++i){
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }

  for(int i = 0 ; i < 2 * n_aug_ + 1; ++i){
    double px = Xsig_aug(0, i);
    double py = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nuA = Xsig_aug(5, i);
    double nuYawd = Xsig_aug(6, i);

    double px_p, py_p, v_p, yaw_p, yawd_p;

    // avoid division by zero
    if(fabs(yawd) > 0.001)
    {
      px_p = px + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
      py_p = py + v / yawd * (-cos(yaw + yawd*  delta_t) + cos(yaw));
    } 
    else
    {
      px_p = px + v * delta_t * cos(yaw);
      py_p = py + v * delta_t * sin(yaw);
    }
    v_p = v;
    yaw_p = yaw + yawd * delta_t;
    yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5 * nuA * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nuA * delta_t * delta_t * sin(yaw);
    v_p = v_p + nuA * delta_t;
    yaw_p = yaw_p + 0.5 * nuYawd * delta_t * delta_t;
    yawd_p = yawd_p + nuYawd * delta_t;

    // predicted sigma points
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }

  x_.fill(0.0);

  for(int i = 0; i < 2 * n_aug_ + 1; i++)
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  
  P_.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {

    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    x_diff(3) = atan2(sin(x_diff(3)), cos(x_diff(3)));
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  VectorXd z = meas_package.raw_measurements_;

  uint n_z = 2;

  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_+ 1);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    Zsig(0, i) = Xsig_pred_(0, i);
    Zsig(1, i) = Xsig_pred_(1, i);
  }

  VectorXd z_pred= VectorXd(n_z);
  z_pred.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  S = S + lidarNoise_Q_;

  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    VectorXd z_diff = Zsig.col(i) - z_pred;
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  MatrixXd K = Tc * S.inverse();

  VectorXd z_diff = z - z_pred;
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
  
  lidarNIS_ = z_diff.transpose() * S.inverse() * z_diff;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  VectorXd z = meas_package.raw_measurements_;

  int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    double p_x  = Xsig_pred_(0, i);
    double p_y  = Xsig_pred_(1, i);
    double v    = Xsig_pred_(2, i);
    double yaw  = Xsig_pred_(3, i);
    double yawd = Xsig_pred_(4, i);

    double vx = cos(yaw)*v;
    double vy = sin(yaw)*v;

    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);                      // r
    Zsig(1, i) = atan2(p_y, p_x);                              // phi
    Zsig(2, i) = (p_x * vx + p_y * vy)/(sqrt(p_x * p_x + p_y * p_y));  // r_dot
  }

  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    while(z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while(z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  S = S + radarNoise_Q_;

  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for(int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    while(z_diff(1) > M_PI) z_diff(1) -= 2. * M_PI;
    while(z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    while(x_diff(3) > M_PI) x_diff(3) -= 2. * M_PI;
    while(x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  MatrixXd K = Tc * S.inverse();

  VectorXd z_diff = z - z_pred;

  while(z_diff(1) > M_PI)  z_diff(1) -= 2. * M_PI;
  while(z_diff(1) < -M_PI) z_diff(1) += 2. * M_PI;

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  radarNIS_ = z_diff.transpose() * S.inverse() * z_diff;
}