//
// sqpnp.h
//
// Implementation of the SQPnL algorithm:
//
//    "Fast and Consistently Accurate Perspective-n-Line Pose Estimation"
//     Paper: https://www.researchgate.net/publication/386377725_Fast_and_Consistently_Accurate_Perspective-n-Line_Pose_Estimation
//     Supplementary: https://www.researchgate.net/publication/391902738_sqpnl_supplementarypdf
//
//
// George Terzakis (terzakig-at-hotmail-dot-com), May 2025
// Optimizations by Manolis Lourakis, February 2022, February 2024
//

#ifndef SQPnL_H__
#define SQPnL_H__

#include "types.h"
#include <vector>
#include <assert.h>
#include <iostream>

namespace sqpnl
{

  class PnLSolver
  {

  public:
    static const double SQRT3;

    bool IsValid() const { return flag_valid_; }
    const Eigen::Matrix<double, 9, 9> &Omega() const { return Omega_; }
    const Eigen::Matrix<double, 9, 9> &EigenVectors() const { return U_; }
    const Eigen::Matrix<double, 9, 1> &EigenValues() const { return s_; }
    int NullSpaceDimension() const { return num_null_vectors_; }
    int NumberOfSolutions() const { return num_solutions_; }
    const sqp_engine::SQPSolution *SolutionPtr(int index) const
    {
      return index < 0 || index >= num_solutions_ ? nullptr : &solutions_[index];
    }

    //
    // Return average reprojection errors
    inline std::vector<double> AverageSquaredProjectionErrors() const
    {
      std::vector<double> avg_errors;
      avg_errors.reserve(num_solutions_);
      for (int i = 0; i < num_solutions_; i++)
      {
        avg_errors.emplace_back(AverageSquaredProjectionError(i));
      }
      return avg_errors;
    }
    const std::vector<double> &Weights() const { return weights_; }

    //! Constructor #1: Initialize solver from pairs of points
    template <typename Point3D, typename Point2D, typename Pw = double>
    inline PnLSolver(                                       //
        const std::vector<Point3D> &points1,                //
        const std::vector<Point3D> &points2,                //
        const std::vector<Point2D> &projections1,           //
        const std::vector<Point2D> &projections2,           //
        const std::vector<Pw> &weights = std::vector<Pw>(), //
        const sqp_engine::SolverParameters &parameters = sqp_engine::SolverParameters()) : parameters_(parameters)
    {
      const size_t n = points1.size();

      if (n != points2.size() || n != projections1.size() || n != projections2.size() || n < 3)
      {
        flag_valid_ = false;
        return;
      }

      if (!weights.empty())
      {
        if (n != weights.size())
        {
          flag_valid_ = false;
          return;
        }
        weights_ = weights;
      }
      else
      {
        weights_.resize(n, 1.0);
      }

      flag_valid_ = true;
      lines_.reserve(n);
      projections_.reserve(n);
      num_null_vectors_ = -1; // set to -1 in case we never make it to the decomposition of Omega
      Omega_ = Eigen::Matrix<double, 9, 9>::Zero();

      // Sum of weights
      double sum_w = 0.0;

      Omega_ = Eigen::Matrix<double, 9, 9>::Zero();
      Eigen::Matrix<double, 3, 3> sum_BBt = Eigen::Matrix<double, 3, 3>::Zero();
      Eigen::Matrix<double, 3, 9> sum_BBtM = Eigen::Matrix<double, 3, 9>::Zero();
      Eigen::Matrix<double, 3, 9> sum_M = Eigen::Matrix<double, 3, 9>::Zero();

      cheir_points_mean_ = Eigen::Vector<double, 3>::Zero();
      cheir_points_.resize(projections_.size());

      // Go through the lines and projections now...
      for (size_t i = 0; i < n; i++)
      {
        const double w = weights_[i];
        lines_.emplace_back(points1[i], points2[i]);
        projections_.emplace_back(projections1[i], projections2[i]);

        cheir_points_.emplace_back(0.5 * (points1[i] + points2[i]));

        if (abs(w) < 1e-6)
        {
          continue;
        }

        // Populate Omega, Sum(B*B'), Sum(B*B'*M), sum_w, cheir_points_mean
        AccumulateDataMatrices(w, lines_[i], projections_[i], cheir_points_[i], sum_w, cheir_points_mean_, sum_BBt, sum_BBtM, sum_M);
      }

      FinalizeDataMatrices(sum_w, sum_BBt, sum_BBtM, sum_M);
    }

    //! Constructor (initializes Omega, P and U, s, i.e. the decomposition of Omega)
    template <typename Pp = double, typename Pw = double>
    inline PnLSolver(                                                                                //
        const std::vector<Line> &lines,                                                              //
        const std::vector<Projection> &projections,                                                  //
        const std::vector<Eigen::Vector<Pp, 3>> &cheir_points = std::vector<Eigen::Vector<Pp, 3>>(), //
        const std::vector<Pw> &weights = std::vector<Pw>(),                                          //
        const sqp_engine::SolverParameters &parameters = sqp_engine::SolverParameters()) : parameters_(parameters)
    {
      const size_t n = lines.size();

      if (n != projections.size() || n < 3)
      {
        flag_valid_ = false;
        return;
      }

      if (!weights.empty())
      {
        if (n != weights.size())
        {
          flag_valid_ = false;
          return;
        }
        weights_ = weights;
      }
      else
      {
        weights_.resize(n, 1.0);
      }

      flag_valid_ = true;
      lines_.reserve(n);
      projections_.reserve(n);
      num_null_vectors_ = -1; // set to -1 in case we never make it to the decomposition of Omega
      Omega_ = Eigen::Matrix<double, 9, 9>::Zero();

      // Sum of weights
      double sum_w = 0.0;

      Omega_ = Eigen::Matrix<double, 9, 9>::Zero();
      Eigen::Matrix<double, 3, 3> sum_BBt = Eigen::Matrix<double, 3, 3>::Zero();
      Eigen::Matrix<double, 3, 9> sum_BBtM = Eigen::Matrix<double, 3, 9>::Zero();
      Eigen::Matrix<double, 3, 9> sum_M = Eigen::Matrix<double, 3, 9>::Zero();

      cheir_points_mean_ = Eigen::Vector<double, 3>::Zero();
      cheir_points_.resize(projections_.size());

      // Go through the lines and projections now...
      for (size_t i = 0; i < n; i++)
      {
        const double w = weights_[i];
        lines_.emplace_back(lines[i]);
        projections_.emplace_back(projections[i]);

        if (cheir_points.empty())
        {
          cheir_points_.emplace_back(Line::PointInFrontOfCamera(&lines_[i]));
        }
        else
        {
          cheir_points_.emplace_back(cheir_points_[i][0], cheir_points_[i][1], cheir_points_[i][2]);
        }

        if (abs(w) < 1e-6)
        {
          continue;
        }

        // Populate Omega, Sum(B*B'), Sum(B*B'*M), sum_w, cheir_points_mean
        AccumulateDataMatrices(w, lines_[i], projections_[i], cheir_points_[i], sum_w, cheir_points_mean_, sum_BBt, sum_BBtM, sum_M);
      }

      FinalizeDataMatrices(sum_w, sum_BBt, sum_BBtM, sum_M);
    }

    //! Solve the PnL
    bool Solve();

  private:
    std::vector<Projection> projections_;
    std::vector<Line> lines_;
    //! The cheirality points. Either computed from the lines or provided as a constructor argument.
    std::vector<Eigen::Vector<double, 3>> cheir_points_;
    //! The average of the points on the lines that
    Eigen::Vector<double, 3> cheir_points_mean_;
    std::vector<double> weights_;
    sqp_engine::SolverParameters parameters_;

    Eigen::Matrix<double, 9, 9> Omega_;

    Eigen::Matrix<double, 9, 1> s_;
    Eigen::Matrix<double, 9, 9> U_;
    Eigen::Matrix<double, 3, 9> P_;

    int num_null_vectors_;

    bool flag_valid_;

    sqp_engine::SQPSolution solutions_[18];
    int num_solutions_;

    //! Nearest rotation matrix function. By default, the FOAM method
    std::function<void(const Eigen::Matrix<double, 9, 1> &, Eigen::Matrix<double, 9, 1> &)> NearestRotationMatrix;

    //! Populate data matrices Omega and P in a single iteration
    void AccumulateDataMatrices(                     //
        const double &w,                             //
        const Line &line,                            //
        const Projection &projection,                //
        const Eigen::Vector<double, 3> &cheir_point, //
        double &sum_w,                               //
        Eigen::Vector<double, 3> &cheir_points_mean, //
        Eigen::Matrix<double, 3, 3> &sum_BBt,        //
        Eigen::Matrix<double, 3, 9> &sum_BBtM,       //
        Eigen::Matrix<double, 3, 9> &sum_M);

    //! Finalize the computations for Omega and P (used to recover translation from rotation)
    void FinalizeDataMatrices(                 //
        const double &sum_w,                   //
        Eigen::Matrix<double, 3, 3> &sum_BBt,  //
        Eigen::Matrix<double, 3, 9> &sum_BBtM, //
        const Eigen::Matrix<double, 3, 9> &sum_M);

    //! Handle a newly found solution and populate the list of solutions
    void HandleSolution(sqp_engine::SQPSolution &solution, double &min_sq_error);

    //! Average squared projection error of a given solution as squared distances of the projections of P_hat from the 2D line
    inline double AverageSquaredProjectionError(const int index) const
    {
      double avg = 0.0;
      const auto &r = solutions_[index].r_hat;
      const auto &t = solutions_[index].t;

      for (size_t i = 0; i < lines_.size(); i++)
      {
        const auto &M = lines_[i].P_hat;
        const double Xc = r[0] * M[0] + r[1] * M[1] + r[2] * M[2] + t[0],
                     Yc = r[3] * M[0] + r[4] * M[1] + r[5] * M[2] + t[1],
                     inv_Zc = 1.0 / (r[6] * M[0] + r[7] * M[1] + r[8] * M[2] + t[2]);

        const double dist = abs(projections_[i].n[0] * Xc * inv_Zc + projections_[i].n[1] * Yc * inv_Zc + projections_[i].c);
        avg += dist * dist;
      }

      return avg / lines_.size();
    }

    //
    // Test cheirality on the mean point for a given solution
    inline bool TestPositiveDepth(const sqp_engine::SQPSolution &solution)
    {
      const auto &r = solution.r_hat;
      const auto &t = solution.t;
      const auto &M = cheir_points_mean_;
      return (r[6] * M[0] + r[7] * M[1] + r[8] * M[2] + t[2] > 0);
    }

    //
    // Test cheirality on the majority of points for a given solution
    inline bool TestPositiveMajorityDepths(const sqp_engine::SQPSolution &solution)
    {
      const auto &r = solution.r_hat;
      const auto &t = solution.t;
      int npos = 0, nneg = 0;

      for (size_t i = 0; i < lines_.size(); i++)
      {
        if (abs(weights_[i]) < 1e-6)
        {
          continue;
        }
        const auto &M = cheir_points_[i];
        (r[6] * M[0] + r[7] * M[1] + r[8] * M[2] + t[2] > 0) ? ++npos : ++nneg;
      }

      return npos >= nneg;
    }
  }; // class PnLSolver

}

#endif
