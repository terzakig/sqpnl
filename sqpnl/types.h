//
// Types.h
//
// Implementation of SQPnL as described in the paper:
//
//    "Fast and Consistently Accurate Perspective-n-Line Pose Estimation"
//     Paper: https://www.researchgate.net/publication/386377725_Fast_and_Consistently_Accurate_Perspective-n-Line_Pose_Estimation
//     Supplementary: https://www.researchgate.net/publication/391902738_sqpnl_supplementarypdf
//
// George Terzakis, May, 2025
//

#ifndef _SQPNLTYPES__H_
#define _SQPNLTYPES__H_

#ifdef HAVE_OPENCV

#include <opencv2/core.hpp>

#endif

#include <iostream>
#include <Eigen/Dense>

#include "sqp_engine/sqp_engine.h"

namespace Eigen
{
  template <typename P, int n>
  using Vector = Matrix<P, n, 1>;
}

namespace sqpnl
{

  //! Line representation
  struct Line
  {
    //! Nearest point to the origin
    Eigen::Matrix<double, 3, 1> P_hat;
    //! Line direction
    Eigen::Matrix<double, 3, 1> u;

    //! Empty line
    inline Line() : P_hat(Eigen::Vector<double, 3>::Zero()), u(Eigen::Vector<double, 3>::Zero())
    {
    }

    //! Construct from two points
    inline Line(const Eigen::Vector<double, 3> &_P1, const Eigen::Vector<double, 3> &_P2)
    {
      u = (_P2 - _P1).normalized();
      P_hat = _P1 - _P1.dot(u) * u;
    }

    //! Build from arbitrary point and direction ('factory' method)
    inline static Line FromPointAndDirection(const Eigen::Vector<double, 3> &_P, const Eigen::Vector<double, 3> &_u)
    {
      Line line;

      line.u = _u.normalized();
      line.P_hat = _P - _P.dot(line.u) * line.u;

      return line;
    }

#ifdef HAVE_OPENCV
    //! Construct from two points stored in OpenCV types
    inline Line(const cv::Vec<double, 3> &_P1, const cv::Vec<double, 3> &_P2)
    {
      const Eigen::Vector<double, 3> P2 = Eigen::Vector<double, 3>(_P2[0], _P2[1], _P2[2]);
      const Eigen::Vector<double, 3> P1 = Eigen::Vector<double, 3>(_P1[0], _P1[1], _P1[2]);

      u = (P2 - P1).normalized();
      P_hat = P1 - P1.dot(u) * u;
    }

    //! Build from arbitrary point and direction stored in OpenCV types
    inline static Line FromPointAndDirection(const cv::Vec<double, 3> &_P, const cv::Vec<double, 3> &_u)
    {
      Line line;
      const Eigen::Vector<double, 3> P = Eigen::Vector<double, 3>(_P[0], _P[1], _P[2]);

      line.u = Eigen::Vector<double, 3>(_u[0], _u[1], _u[2]).normalized();
      line.P_hat = P - P.dot(line.u) * line.u;

      return line;
    }

#endif

    //! Find the 3D point on the line that projects to the nearest point to the center of the image (such that it lies in front)
    inline static Eigen::Vector<double, 3> PointInFrontOfCamera(const Line *line)
    {
      const double X = line->P_hat[0];
      const double Y = line->P_hat[1];
      const double Z = line->P_hat[2];
      const double u1 = line->u[0];
      const double u2 = line->u[1];
      const double u3 = line->u[2];

      //@NOTE: This should not be zero under normal circumstances ( Z > 1 and u1 and u2 cannot be zero at the same time).
      double lambda0 = (u3 * X * X - Z * u1 * X + u3 * Y * Y - Z * u2 * Y) / (Z * u1 * u1 - X * u3 * u1 + Z * u2 * u2 - Y * u3 * u2);

      // If depth turns out negative, then the oposite direction will bring the 3D point in front of the camera
      if (Z + lambda0 * u3 < 0)
      {
        lambda0 = -lambda0;
      }
      return line->P_hat + lambda0 * line->u;
    }
  };

  //! Projection of a line on the Z=1 Euclidean plane
  struct Projection : public Line
  {
    //! Hesse constant
    double c;
    //! Hesse normal
    Eigen::Matrix<double, 2, 1> n;

    //! Default constructor
    inline Projection() : Line(), c(0.), n(Eigen::Vector<double, 2>::Zero())
    {
    }

    //! Construct from points on the Euclidean proj. plane at Z=1
    inline Projection(const Eigen::Vector<double, 2> &_p1, const Eigen::Vector<double, 2> &_p2) :                                                  //
                                                                                                  Line(                                            //
                                                                                                      Eigen::Vector<double, 3>(_p1[0], _p1[1], 1), //
                                                                                                      Eigen::Vector<double, 3>(_p2[0], _p2[1], 1))
    {
      n[0] = -u[1];
      n[1] = u[0];
      n.normalize();
      c = (abs(n[0]) > abs(n[1])) ? -P_hat[0] / n[0] : -P_hat[1] / n[1];
    }

    //! Construct a projection from a line
    inline Projection(const Line &line)
    {
      const double D = line.P_hat[2] * line.P_hat[2] + line.u[2] * line.u[2];
      if (D < 1e-10)
      {
        P_hat = line.P_hat;
        P_hat[2] = 1;
        u = line.u;
      }
      else
      {
        const double iD = 1.0 / D;
        const double isqrtD = sqrt(iD);
        P_hat = iD * (P_hat[2] * P_hat + u[2] * u);
        u = isqrtD * (-u[2] * P_hat + P_hat[2] * u);
      }
      n[0] = -u[1];
      n[1] = u[0];
      n.normalize();
      c = (abs(n[0]) > abs(n[1])) ? -P_hat[0] / n[0] : -P_hat[1] / n[1];
    }

    //! Construct projection from Hesse coordinates (line equation as constant and 2D normal vector)
    inline Projection(const double &_c, const Eigen::Vector<double, 2> &_n) : c(_c), n(_n)
    {
      P_hat[0] = -c * n[0];
      P_hat[1] = -c * n[1];
      P_hat[2] = 1;
      u[0] = -n[1];
      u[1] = n[0];
      u[2] = 0;
    }

    //! Build projection from 2D point and direction vectors ('factory' method)
    inline static Projection FromPointAndDirection(const Eigen::Vector<double, 2>& m, const Eigen::Vector<double, 2>& d)
    {
      Eigen::Vector<double, 2> n(-d[1], d[0]);
      n.normalize();
      double c = -n.dot(m);

      return Projection(c, n);
    }

#ifdef HAVE_OPENCV
    //! Construct from points on the Euclidean proj. plane at Z=1 with OpenCV types
    inline Projection(const cv::Vec<double, 2> &_p1, const cv::Vec<double, 2> &_p2) :                                                  //
                                                                                      Line(                                            //
                                                                                          Eigen::Vector<double, 3>(_p1[0], _p1[1], 1), //
                                                                                          Eigen::Vector<double, 3>(_p2[0], _p2[1], 1))
    {
      n[0] = -u[1];
      n[1] = u[0];
      n.normalize();
      c = (abs(n[0]) > abs(n[1])) ? -P_hat[0] / n[0] : -P_hat[1] / n[1];
    }

    //! Construct projection from Hesse coordinates (line equation as constant and 2D normal vector) in OpenCV vectors
    inline Projection(const double &_c, const cv::Vec<double, 2> &_n) : c(_c), n(Eigen::Vector<double, 2>(_n[0], _n[1]))
    {
      P_hat[0] = -c * n[0];
      P_hat[1] = -c * n[1];
      P_hat[2] = 1;
      u[0] = -n[1];
      u[1] = n[0];
      u[2] = 0;
    }

    //! Build projection from 2D point and direction in OpenCV vectors
    inline static Projection FromPointAndDirection(const cv::Vec<double, 2>& m, const cv::Vec<double, 2>& d)
    {
      Eigen::Vector<double, 2> n(-d[1], d[0]);
      n.normalize();
      double c = -n.dot(Eigen::Vector<double, 2>(m[0], m[1]));

      return Projection(c, n);
    }
#endif
  };

}

#endif
