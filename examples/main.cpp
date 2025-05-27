#include <iostream>
#include <cmath>
#include <sqpnl.h>
#include <opencv2/core.hpp>
#include <vector>
#include <chrono>
#include <assert.h>
#include <random>

//
// Generate noisy PnP data
void GenerateSyntheticLines(                           //
    int n,                                             //
    cv::Matx<double, 3, 3> &R,                         //
    cv::Vec<double, 3> &t,                             //
    std::vector<sqpnl::Line> &lines,                   //
    std::vector<sqpnl::Projection> &projections,       //
    std::vector<sqpnl::Projection> &noisy_projections, //
    const double &std_pixel_noise = 0.0,               //
    const double &radius = 1.5)
{
  assert(n > 2);

  cv::Matx<double, 3, 3> K = {               //
                              1400, 0, 1000, //
                              0, 1400, 900,  //
                              0, 0, 1};

  const double std_noise = std_pixel_noise / 1400;
  const double depth = 2.5 * radius; // depth of the barycenter of the points

  const cv::Point3_<double> C(radius / 4, radius / 4, depth);

  // Generate a rotation matrix near the origin
  cv::Vec<double, 3> psi; // = mvnrnd([0; 0; 0], 0.001 * eye(3))';

  static std::random_device r;
  static std::default_random_engine generator(r());
  double sigma_psi = 0.1;
  std::normal_distribution<double> psi_noise(0.0, sigma_psi);
  psi[0] = psi_noise(generator);
  psi[1] = psi_noise(generator);
  psi[2] = psi_noise(generator);

  double sq_norm_psi = psi[0] * psi[0] + psi[1] * psi[1] + psi[2] * psi[2];
  double inv_w = 1.0 / (1 + sq_norm_psi);
  double s = (1 - sq_norm_psi) * inv_w,
         v1 = 2 * psi[0] * inv_w,
         v2 = 2 * psi[1] * inv_w,
         v3 = 2 * psi[2] * inv_w;
  R(0, 0) = s * s + v1 * v1 - v2 * v2 - v3 * v3;
  R(0, 1) = 2 * (v1 * v2 - s * v3);
  R(0, 2) = 2 * (v1 * v3 + s * v2);
  R(1, 0) = 2 * (v1 * v2 + s * v3);
  R(1, 1) = s * s - v1 * v1 + v2 * v2 - v3 * v3;
  R(1, 2) = 2 * (v2 * v3 - s * v1);
  R(2, 0) = 2 * (v1 * v3 - s * v2);
  R(2, 1) = 2 * (v2 * v3 + s * v1);
  R(2, 2) = s * s - v1 * v1 - v2 * v2 + v3 * v3;

  // Generate a translation that's about 1/25 of the depth
  std::normal_distribution<double> camera_position(0.0, depth / 25);
  cv::Vec<double, 3> pos(camera_position(generator), camera_position(generator), camera_position(generator));

  lines.clear();
  projections.clear();
  noisy_projections.clear();

  std::normal_distribution<double> projection_noise(0.0, std_noise);

  while (static_cast<int>(lines.size()) < n)
  {
    std::vector<cv::Vec3d> sampled_line_pts;
    std::vector<cv::Vec2d> sampled_projection_pts;
    std::vector<cv::Vec2d> sampled_noisy_projection_pts;
    bool good_sample = true;
    for (size_t i = 0; i < 2; i++)
    {
      std::normal_distribution<double> point_X(C.x, radius);
      std::normal_distribution<double> point_Y(C.y, radius);
      std::normal_distribution<double> point_Z(C.z, radius);

      cv::Vec<double, 3> Mw(point_X(generator), point_Y(generator), point_Z(generator));
      cv::Vec<double, 3> Mc = R * (Mw - pos);
      if (Mc[2] < 0)
      {
        good_sample = false;
        break;
      }
      cv::Vec<double, 2> proj(Mc[0] / Mc[2], Mc[1] / Mc[2]);
      // Add noise to projection
      cv::Vec<double, 2> noisy_proj = proj;
      noisy_proj[0] += projection_noise(generator);
      noisy_proj[1] += projection_noise(generator);
      sampled_noisy_projection_pts.push_back(noisy_proj);
      sampled_projection_pts.push_back(proj);
      sampled_line_pts.push_back(Mw);
    }
    if (good_sample)
    {
      lines.emplace_back(sampled_line_pts[0], sampled_line_pts[1]);
      projections.emplace_back(sampled_projection_pts[0], sampled_projection_pts[1]);
      noisy_projections.emplace_back(sampled_noisy_projection_pts[0], sampled_noisy_projection_pts[1]);
    }
    else
    {
      std::cerr << "Bad line sample (has negative depth)! Skipping...\n";
    }
  }

  t = -R * pos;
}

// compute the translational and angular error between the pose R,t contained in an SQPnP solution and the (true) pose Rg,tg
static void poseError(const sqp_engine::SQPSolution &solution, const cv::Matx<double, 3, 3> &Rg, const cv::Vec<double, 3> &tg, double &terr, double &aerr)
{
  // translational error
  double a = tg(0) - solution.t(0);
  double b = tg(1) - solution.t(1);
  double c = tg(2) - solution.t(2);
  terr = sqrt(a * a + b * b + c * c);

  /* angular error, defined as the amount of rotation about a unit vector that transfers Rg to R.
   * The (residual) angle is computed with the inverse Rodrigues rotation formula
   */

  // compute trc as the trace of Rg'*R
  a = Rg(0, 0) * solution.r_hat[0] + Rg(1, 0) * solution.r_hat[3] + Rg(2, 0) * solution.r_hat[6];
  b = Rg(0, 1) * solution.r_hat[1] + Rg(1, 1) * solution.r_hat[4] + Rg(2, 1) * solution.r_hat[7];
  c = Rg(0, 2) * solution.r_hat[2] + Rg(1, 2) * solution.r_hat[5] + Rg(2, 2) * solution.r_hat[8];
  const double trc = a + b + c;
  a = 0.5 * (trc - 1.0);
  aerr = acos(std::min(std::max(-1.0, a), 1.0)); // clamp to [-1, 1]
}

int main()
{
  int N = 105;
  int n = 10;
  double std_pixels = sqrt(3);

  std::vector<std::vector<sqpnl::Line>> vlines;
  std::vector<std::vector<sqpnl::Projection>> vprojections;
  std::vector<std::vector<sqpnl::Projection>> vnoisy_projections;

  std::vector<cv::Matx<double, 3, 3>> vRt;
  std::vector<cv::Vec<double, 3>> vtt;

  for (int i = 0; i < N; i++)
  {
    cv::Matx<double, 3, 3> Rt;
    cv::Vec<double, 3> tt;
    std::vector<sqpnl::Line> lines;
    std::vector<sqpnl::Projection> projections;
    std::vector<sqpnl::Projection> noisy_projections;

    GenerateSyntheticLines(n, Rt, tt, lines, projections, noisy_projections, std_pixels);

    vlines.push_back(lines);
    vprojections.push_back(projections);
    vnoisy_projections.push_back(noisy_projections);

    vRt.push_back(Rt);
    vtt.push_back(tt);
  }

  auto start = std::chrono::steady_clock::now();

  sqp_engine::SolverParameters params;
  params.enable_cheirality_check = true;
  params.omega_nullspace_method = sqp_engine::OmegaNullspaceMethod::RRQR;
  std::vector<double> weights(n, 1.0);
  std::vector<Eigen::Vector<double, 3>> cheirality_points;
  double max_sq_error = 0, max_sq_proj_error = 0;
  std::vector<sqp_engine::SQPSolution> solutions;
  for (int i = 0; i < N; i++)
  {
    // example passing weights and parameters to the solver
    sqpnl::PnLSolver solver(vlines[i], vnoisy_projections[i], cheirality_points, weights, params);
    if (solver.IsValid())
    {
      solver.Solve();

      if (solver.SolutionPtr(0))
      {
        max_sq_error = solver.SolutionPtr(0)->sq_error;
        max_sq_proj_error = solver.AverageSquaredProjectionErrors()[0];
        solutions.push_back(*solver.SolutionPtr(0));
      }
    }
  }

  auto finish = std::chrono::steady_clock::now();

  for (int i = 0; i < N; i++)
  {
    double terr, aerr;

    std::cout << i << "-th Solution : " << solutions[i];
    std::cout << i << "-th GT R : " << vRt[i] << std::endl;
    std::cout << i << "-th GT t : " << vtt[i] << "\n";

    poseError(solutions[i], vRt[i], vtt[i], terr, aerr);
    std::cout << i << " translational error : " << terr << "  angular error : " << aerr * 180.0 / 3.14159 << " degrees.\n\n";
  }

  auto diff = finish - start;
  std::cout << " Average execution time : " << std::chrono::duration_cast<std::chrono::microseconds>(diff).count() / N << std::endl;
  std::cout << " Maximum squared error : " << max_sq_error << std::endl;
  std::cout << " Maximum average squared projection error : " << max_sq_proj_error << std::endl;

  return 1;
}
