//
// sqpnl.cpp
//
// Implementation of SQPnL as described in the paper:
//
//    "Fast and Consistently Accurate Perspective-n-Line Pose Estimation"
//     Paper: https://www.researchgate.net/publication/386377725_Fast_and_Consistently_Accurate_Perspective-n-Line_Pose_Estimation
//     Supplementary: https://www.researchgate.net/publication/391902738_sqpnl_supplementarypdf
//
//
// George Terzakis (terzakig-at-hotmail-dot-com), May 2025
// Optimizations by Manolis Lourakis, February 2022, February 2024
//

#include "sqpnl.h"

namespace sqpnl
{

  const double PnLSolver::SQRT3 = std::sqrt(3.0);

  void PnLSolver::AccumulateDataMatrices(          //
      const double &w,                             //
      const Line &line,                            //
      const Projection &projection,                //
      const Eigen::Vector<double, 3> &cheir_point, //
      double &sum_w,                               //
      Eigen::Vector<double, 3> &cheir_points_mean, //
      Eigen::Matrix<double, 3, 3> &sum_BBt,        //
      Eigen::Matrix<double, 3, 9> &sum_BBtM,       //
      Eigen::Matrix<double, 3, 9> &sum_M)
  {
    sum_w += w;

    cheir_points_mean_ += w * cheir_point; // @TODO: Should we weight the average for cheirality purposes?

    const double squ1 = line.u[0] * line.u[0], //
        u1u2 = line.u[0] * line.u[1],          //
        u1u3 = line.u[0] * line.u[2],          //
        squ2 = line.u[1] * line.u[1],          //
        u2u3 = line.u[1] * line.u[2],          //
        squ3 = line.u[2] * line.u[2];

    const Eigen::Vector<double, 3> v = (projection.u.cross(projection.P_hat)).normalized();
    const double sqv1 = v[0] * v[0], //
        v1v2 = v[0] * v[1],          //
        v1v3 = v[0] * v[2],          //
        sqv2 = v[1] * v[1],          //
        v2v3 = v[1] * v[2],          //
        sqv3 = v[2] * v[2];

    // Accumulate upper triangle of Omega
    //
    // Row #1
    //
    // [  u1^2*v1^2,  u1^2*v1*v2,  u1^2*v1*v3,  u1*u2*v1^2, u1*u2*v1*v2, u1*u2*v1*v3,  u1*u3*v1^2, u1*u3*v1*v2, u1*u3*v1*v3]
    Omega_(0, 0) += w * squ1 * sqv1;
    Omega_(0, 1) += w * squ1 * v1v2;
    Omega_(0, 2) += w * squ1 * v1v3;
    Omega_(0, 3) += w * u1u2 * sqv1;
    Omega_(0, 4) += w * u1u2 * v1v2;
    Omega_(0, 5) += w * u1u2 * v1v3;
    Omega_(0, 6) += w * u1u3 * sqv1;
    Omega_(0, 7) += w * u1u3 * v1v2;
    Omega_(0, 8) += w * u1u3 * v1v3;
    //
    // Row #2
    //
    // ... u1^2*v2^2,  u1^2*v2*v3, u1*u2*v1*v2,  u1*u2*v2^2, u1*u2*v2*v3, u1*u3*v1*v2,  u1*u3*v2^2, u1*u3*v2*v3]
    Omega_(1, 1) += w * squ1 * sqv2;
    Omega_(1, 2) += w * squ1 * v2v3;
    Omega_(1, 3) += w * u1u2 * v1v2;
    Omega_(1, 4) += w * u1u2 * sqv2;
    Omega_(1, 5) += w * u1u2 * v2v3;
    Omega_(1, 6) += w * u1u3 * v1v2;
    Omega_(1, 7) += w * u1u3 * sqv2;
    Omega_(1, 8) += w * u1u3 * v2v3;
    //
    // Row #3
    //
    // ... u1^2*v3^2, u1*u2*v1*v3, u1*u2*v2*v3,  u1*u2*v3^2, u1*u3*v1*v3, u1*u3*v2*v3,  u1*u3*v3^2]
    Omega_(2, 2) += w * squ1 * sqv3;
    Omega_(2, 3) += w * u1u2 * v1v3;
    Omega_(2, 4) += w * u1u2 * v2v3;
    Omega_(2, 5) += w * u1u2 * sqv3;
    Omega_(2, 6) += w * u1u3 * v1v3;
    Omega_(2, 7) += w * u1u3 * v2v3;
    Omega_(2, 8) += w * u1u3 * sqv3;
    //
    // Row #4
    //
    // ... u2^2*v1^2,  u2^2*v1*v2,  u2^2*v1*v3,  u2*u3*v1^2, u2*u3*v1*v2, u2*u3*v1*v3]
    Omega_(3, 3) += w * squ2 * sqv1;
    Omega_(3, 4) += w * squ2 * v1v2;
    Omega_(3, 5) += w * squ2 * v1v3;
    Omega_(3, 6) += w * u2u3 * sqv1;
    Omega_(3, 7) += w * u2u3 * v1v2;
    Omega_(3, 8) += w * u2u3 * v1v3;
    //
    // Row #5
    //
    // ... u2^2*v2^2,  u2^2*v2*v3, u2*u3*v1*v2,  u2*u3*v2^2, u2*u3*v2*v3]
    Omega_(4, 4) += w * squ2 * sqv2;
    Omega_(4, 5) += w * squ2 * v2v3;
    Omega_(4, 6) += w * u2u3 * v1v2;
    Omega_(4, 7) += w * u2u3 * sqv2;
    Omega_(4, 8) += w * u2u3 * v2v3;
    //
    // Row #6
    //
    // ... u2^2*v3^2, u2*u3*v1*v3, u2*u3*v2*v3,  u2*u3*v3^2]
    Omega_(5, 5) += w * squ2 * sqv3;
    Omega_(5, 6) += w * u2u3 * v1v3;
    Omega_(5, 7) += w * u2u3 * v2v3;
    Omega_(5, 8) += w * u2u3 * sqv3;
    //
    // Row #7
    //
    // ... u3^2*v1^2,  u3^2*v1*v2,  u3^2*v1*v3]
    Omega_(6, 6) += w * squ3 * sqv1;
    Omega_(6, 7) += w * squ3 * v1v2;
    Omega_(6, 8) += w * squ3 * v1v3;
    //
    // Row #8
    //
    // u3^2*v2^2,  u3^2*v2*v3]
    Omega_(7, 7) += w * squ3 * sqv2;
    Omega_(7, 8) += w * squ3 * v2v3;
    //
    // Row #9
    //
    // ... u3^2*v3^2]
    Omega_(8, 8) += w * squ3 * sqv3;

    // Construct the orthonormal basis matrix Bi
    //
    // NOTE: P_hat and u are already orthogonal. Just need to normalize P_hat.
    Eigen::Matrix<double, 3, 2> Bi;
    const double inv_normP_hat = 1.0 / projection.P_hat.norm();
    Bi(0, 0) = projection.P_hat[0] * inv_normP_hat;
    Bi(1, 0) = projection.P_hat[1] * inv_normP_hat;
    Bi(2, 0) = projection.P_hat[2] * inv_normP_hat;

    Bi(0, 1) = projection.u[0];
    Bi(1, 1) = projection.u[1];
    Bi(2, 1) = projection.u[2];

    Eigen::Matrix<double, 3, 3> BBt;
    BBt(0, 0) = Bi(0, 0) * Bi(0, 0) + Bi(0, 1) * Bi(1, 0);
    BBt(0, 1) = Bi(0, 0) * Bi(1, 0) + Bi(0, 1) * Bi(1, 1);
    BBt(0, 2) = Bi(0, 0) * Bi(2, 0) + Bi(0, 1) * Bi(2, 1);

    BBt(1, 1) = Bi(1, 0) * Bi(1, 0) + Bi(1, 1) * Bi(1, 1);
    BBt(1, 2) = Bi(1, 0) * Bi(2, 0) + Bi(1, 1) * Bi(2, 1);

    BBt(2, 2) = Bi(2, 0) * Bi(2, 0) + Bi(2, 1) * Bi(2, 1);

    sum_BBt(0, 0) += w * BBt(0, 0);
    sum_BBt(0, 1) += w * BBt(0, 1);
    sum_BBt(0, 2) += w * BBt(0, 2);

    sum_BBt(1, 1) += w * BBt(1, 1);
    sum_BBt(1, 2) += w * BBt(1, 2);

    sum_BBt(2, 2) += w * BBt(2, 2);

    // Sum(wi*Bi*Bi'*Mi) where the matrix Mi is such that. Mi*r = R'*Pi (Pi is the line's Pi_hat)
    sum_BBtM(0, 0) += w * line.P_hat[0] * BBt(0, 0);
    sum_BBtM(0, 1) += w * line.P_hat[0] * BBt(0, 1);
    sum_BBtM(0, 2) += w * line.P_hat[0] * BBt(0, 2);
    sum_BBtM(0, 3) += w * line.P_hat[1] * BBt(0, 0);
    sum_BBtM(0, 4) += w * line.P_hat[1] * BBt(0, 1);
    sum_BBtM(0, 5) += w * line.P_hat[1] * BBt(0, 2);
    sum_BBtM(0, 6) += w * line.P_hat[2] * BBt(0, 0);
    sum_BBtM(0, 7) += w * line.P_hat[2] * BBt(0, 1);
    sum_BBtM(0, 8) += w * line.P_hat[2] * BBt(0, 2);

    sum_BBtM(1, 1) += w * line.P_hat[0] * BBt(1, 1);
    sum_BBtM(1, 2) += w * line.P_hat[0] * BBt(1, 2);
    sum_BBtM(1, 4) += w * line.P_hat[1] * BBt(1, 1);
    sum_BBtM(1, 5) += w * line.P_hat[1] * BBt(1, 2);
    sum_BBtM(1, 7) += w * line.P_hat[2] * BBt(1, 1);
    sum_BBtM(1, 8) += w * line.P_hat[2] * BBt(1, 2);

    sum_BBtM(2, 2) += w * line.P_hat[0] * BBt(2, 2);
    sum_BBtM(2, 5) += w * line.P_hat[1] * BBt(2, 2);
    sum_BBtM(2, 8) += w * line.P_hat[2] * BBt(2, 2);

    // Sum(wi*Mi)
    sum_M(0, 0) += w * line.P_hat[0];
    sum_M(0, 3) += w * line.P_hat[1];
    sum_M(0, 6) += w * line.P_hat[2];
    sum_M(1, 1) += w * line.P_hat[0];
    sum_M(1, 4) += w * line.P_hat[1];
    sum_M(1, 7) += w * line.P_hat[2];
    sum_M(2, 2) += w * line.P_hat[0];
    sum_M(2, 5) += w * line.P_hat[1];
    sum_M(2, 8) += w * line.P_hat[2];
  }

  void PnLSolver::FinalizeDataMatrices(      //
      const double &sum_w,                   //
      Eigen::Matrix<double, 3, 3> &sum_BBt,  //
      Eigen::Matrix<double, 3, 9> &sum_BBtM, //
      const Eigen::Matrix<double, 3, 9> &sum_M)
  {
    // 3D point weighted mean of cheirality points (for quick cheirality checks)
    cheir_points_mean_ *= 1.0 / sum_w;

    // Filling lower triangles in sum_BBt and sum_BBtM
    for (size_t r = 1; r < 3; r++)
    {
      for (size_t c = 0; c < r; c++)
      {
        sum_BBt(r, c) = sum_BBt(c, r);

        sum_BBtM(r, c) = sum_BBtM(c, r);
        sum_BBtM(r, c + 3) = sum_BBtM(c, r + 3);
        sum_BBtM(r, c + 6) = sum_BBtM(c, r + 6);
      }
    }
    // Q = sum(wi)*eye(3) - Sum(Bi*Bi')
    Eigen::Matrix<double, 3, 3> Q = sum_w * Eigen::Matrix<double, 3, 3>::Identity() - sum_BBt;
    Eigen::Matrix<double, 3, 3> Qinv;
    sqp_engine::InvertSymmetric3x3(Q, Qinv);

    // Compute P = -inv( Sum(wi)*eye(3)-Sum(wi*Bi*Bi') ) * ( Sum(wi*Mi) - Sum(wi*Bi*Bi'*Mi) ) = -Qinv * sum_BBtM
    P_ = -Qinv * (sum_M - sum_BBtM);

    //  Fill lower triangle of Omega
    for (size_t r = 1; r < 9; r++)
    {
      for (size_t c = 0; c < r; c++)
      {
        Omega_(r, c) = Omega_(c, r);
      }
    }

    // Finally, decompose Omega with the chosen method
    if (parameters_.omega_nullspace_method == sqp_engine::OmegaNullspaceMethod::RRQR)
    {
      // Rank revealing QR nullspace computation with full pivoting.
      // This is slightly less accurate compared to SVD but x2 faster
      Eigen::FullPivHouseholderQR<Eigen::Matrix<double, 9, 9>> rrqr(Omega_);
      U_ = rrqr.matrixQ();

      Eigen::Matrix<double, 9, 9> R = rrqr.matrixQR().template triangularView<Eigen::Upper>();
      s_ = R.diagonal().array().abs();
    }
    else if (parameters_.omega_nullspace_method == sqp_engine::OmegaNullspaceMethod::CPRRQR)
    {
      // Rank revealing QR nullspace computation with column pivoting.
      // This is potentially less accurate compared to RRQR but faster
      Eigen::ColPivHouseholderQR<Eigen::Matrix<double, 9, 9>> cprrqr(Omega_);
      U_ = cprrqr.householderQ();

      Eigen::Matrix<double, 9, 9> R = cprrqr.matrixR().template triangularView<Eigen::Upper>();
      s_ = R.diagonal().array().abs();
    }
    else // if ( parameters_.omega_nullspace_method == sqp_engine::OmegaNullspaceMethod::SVD )
    {
      // SVD-based nullspace computation. This is the most accurate but slowest option
      Eigen::JacobiSVD<Eigen::Matrix<double, 9, 9>> svd(Omega_, Eigen::ComputeFullU);
      U_ = svd.matrixU();
      s_ = svd.singularValues();
    }

    // Find dimension of null space; the check guards against overly large rank_tolerance
    while (7 - num_null_vectors_ >= 0 && s_[7 - num_null_vectors_] < parameters_.rank_tolerance)
    {
      num_null_vectors_++;
    }
    // Dimension of null space of Omega must be <= 6
    if (++num_null_vectors_ > 6)
    {
      flag_valid_ = false;
    }

    // Assign nearest rotation method
    NearestRotationMatrix = parameters_.sqp_config.NearestRotationMatrix;
  }

  void PnLSolver::HandleSolution(sqp_engine::SQPSolution &solution, double &min_sq_error)
  {
    if (!parameters_.enable_cheirality_check ||                              //
        TestPositiveDepth(solution) || TestPositiveMajorityDepths(solution)) // check the majority if the check with a single points fails
    {

      solution.sq_error = (Omega_ * solution.r_hat).dot(solution.r_hat);
      if (fabs(min_sq_error - solution.sq_error) > parameters_.equal_squared_errors_diff)
      {
        if (min_sq_error > solution.sq_error)
        {
          min_sq_error = solution.sq_error;
          solutions_[0] = solution;
          num_solutions_ = 1;
        }
      }
      else // look for a solution that's almost equal to this
      {
        bool found = false;
        for (int i = 0; i < num_solutions_; i++)
        {
          if ((solutions_[i].r_hat - solution.r_hat).squaredNorm() < parameters_.equal_vectors_squared_diff)
          {
            if (solutions_[i].sq_error > solution.sq_error)
            {
              solutions_[i] = solution;
            }
            found = true;
            break;
          }
        }
        if (!found)
        {
          solutions_[num_solutions_++] = solution;
        }
        if (min_sq_error > solution.sq_error)
        {
          min_sq_error = solution.sq_error;
        }
      }
    }
  }

  bool PnLSolver::Solve()
  {
    if (!flag_valid_)
    {
      return false;
    }

    double min_sq_error = std::numeric_limits<double>::max();
    int num_eigen_points = num_null_vectors_ > 0 ? num_null_vectors_ : 1;
    // clear solutions
    num_solutions_ = 0;

    for (int i = 9 - num_eigen_points; i < 9; i++)
    {
      // NOTE: No need to scale by sqrt(3) here, but better be there for other computations (i.e., orthogonality test)
      const Eigen::Matrix<double, 9, 1> e = SQRT3 * Eigen::Map<Eigen::Matrix<double, 9, 1>>(U_.block<9, 1>(0, i).data());
      double orthogonality_sq_error = sqp_engine::OrthogonalityError(e);
      // Find nearest rotation vector
      sqp_engine::SQPSolution solution[2];

      // Avoid SQP if e is orthogonal
      if (orthogonality_sq_error < parameters_.orthogonality_squared_error_threshold)
      {
        solution[0].r_hat = sqp_engine::Determinant9x1(e) * e;
        solution[0].t = P_ * solution[0].r_hat;
        solution[0].num_iterations = 0;

        HandleSolution(solution[0], min_sq_error);
      }
      else
      {
        NearestRotationMatrix(e, solution[0].r);
        solution[0] = sqp_engine::RunSQP(Omega_, solution[0].r, parameters_.sqp_config);
        solution[0].t = P_ * solution[0].r_hat;
        HandleSolution(solution[0], min_sq_error);

        NearestRotationMatrix(-e, solution[1].r);
        solution[1] = sqp_engine::RunSQP(Omega_, solution[1].r, parameters_.sqp_config);
        solution[1].t = P_ * solution[1].r_hat;
        HandleSolution(solution[1], min_sq_error);
      }
    }

    int index, c = 1;
    while ((index = 9 - num_eigen_points - c) > 0 && min_sq_error > 3 * s_[index])
    {
      const Eigen::Matrix<double, 9, 1> e = Eigen::Map<Eigen::Matrix<double, 9, 1>>(U_.block<9, 1>(0, index).data());
      sqp_engine::SQPSolution solution[2];

      NearestRotationMatrix(e, solution[0].r);
      solution[0] = sqp_engine::RunSQP(Omega_, solution[0].r, parameters_.sqp_config);
      solution[0].t = P_ * solution[0].r_hat;
      HandleSolution(solution[0], min_sq_error);

      NearestRotationMatrix(-e, solution[1].r);
      solution[1] = sqp_engine::RunSQP(Omega_, solution[1].r, parameters_.sqp_config);
      solution[1].t = P_ * solution[1].r_hat;
      HandleSolution(solution[1], min_sq_error);

      c++;
    }

    // Transpose rotations in solutions
    for (size_t i = 0; i < num_solutions_; i++)
    {
      solutions_[i].r = sqp_engine::TransposeRotationVector(solutions_[i].r);
      solutions_[i].r_hat = sqp_engine::TransposeRotationVector(solutions_[i].r_hat);
    }

    return true;
  }

} // namespace sqpnl
