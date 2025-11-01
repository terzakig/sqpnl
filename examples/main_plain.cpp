//
// main.cpp
//
// Manolis Lourakis (lourakis **at** ics forth gr), September 2020 (updated with line data, May 2025)
//
// Example demo program for SQPnL with data point-pairs loaded from plain 2D arrays
//

#include <vector>
#include <iostream>
#include <chrono>

#include <types.h>
#include <sqpnl.h>

int main()
{
  const int num_points = 8; // points (lines = 8/2 = 4)

  /*
     R = [ -0.580579, -0.558763, 0.592209;
     -0.777322, 0.596819, -0.198944;
     -0.242279, -0.57584, -0.780839 ]

     t = [ -0.0820076; 0.0651061; 6.22306 ]
  */

  double pts3[num_points][3] = {
      {-0.429857595273321, -0.441798127281825, 0.714342354521372},
      {-2.1568268264648, 0.113521604867983, -0.148634122716948},
      {0.694636908485644, -0.737067927134015, -1.38877746946909},
      {-1.07051455287146, -1.2122304801284, -0.841002964233812},
      {0.509844073252947, -1.07097319594739, 0.675410167109412},
      {0.40951585099, 2.2300713816052, 0.365229861025625},
      {2.04320214188098, 1.11847674401846, 0.623432173763436},
      {1.04320214188098, -1.11847674401846, 0.433432173763436}};

  double pts2[num_points][2] = {
#if 0
      // no noise
      {0.139024436737141, -0.00108631784422283},
      {0.149897105048989, 0.270584578309815},
      {-0.118448642309468, -0.0844116551810971},
      {0.0917181969674735, 0.0435196877212059},
      {0.100243308685939, -0.178506520365217},
      {-0.296312157121094, 0.220675975198136},
      {-0.331509880499455, -0.213091587841007},
      {0.030908054191591, -0.238937196785852}
#else
      // noisy
      {0.138854772853285, -0.00157437083896972},
      {0.149353089173631, 0.269826809256435},
      {-0.118391028248405, -0.0842834292914752},
      {0.0937833539430025, 0.0473371294380393},
      {0.101410594775151, -0.179030803711188},
      {-0.294749181228375, 0.221134043355639},
      {-0.334084299358372, -0.21071853326318},
      {0.031064918027501, -0.238965960143722}
#endif
  };

  std::vector<sqpnl::Line> lines;
  std::vector<sqpnl::Projection> line_projections;

  const int nlines = num_points / 2;
  lines.reserve(nlines);
  line_projections.reserve(nlines);

  for (int i = 0; i < nlines; i++)
  {
    const Eigen::Vector3d M1(pts3[2 * i][0], pts3[2 * i][1], pts3[2 * i][2]);
    const Eigen::Vector3d M2(pts3[2 * i + 1][0], pts3[2 * i + 1][1], pts3[2 * i + 1][2]);

    const Eigen::Vector2d p1(pts2[2 * i][0], pts2[2 * i][1]);
    const Eigen::Vector2d p2(pts2[2 * i + 1][0], pts2[2 * i + 1][1]);

    lines.emplace_back(M1, M2);
    line_projections.emplace_back(p1, p2);

    /*
    // alternative using points and directions:
    lines.emplace_back(sqpnl::Line::FromPointAndDirection(M1, M2-M1));
    line_projections.emplace_back(sqpnl::Projection::FromPointAndDirection(p1, p2-p1));
    */
  }

  auto start = std::chrono::high_resolution_clock::now();

  // demonstration of passing parameters to the solver
  sqp_engine::EngineParameters engine_params;
  engine_params.omega_nullspace_method = sqp_engine::OmegaNullspaceMethod::RRQR;
  sqpnl::Parameters sqpnl_params;
  sqpnl_params.translation_method = sqpnl::TranslationMethod::MIRZAEI;
  // equal weights for all points
  sqpnl::PnLSolver solver(                      //
      lines,                                    //
      line_projections,                         //
      std::vector<Eigen::Vector3d>(),  //
      std::vector<double>(nlines, 1.0), //
      engine_params,
      sqpnl_params);

  auto stop = std::chrono::high_resolution_clock::now();

  if (solver.IsValid())
  {
    solver.Solve();
    stop = std::chrono::high_resolution_clock::now();
    std::cout << "SQPnL found " << solver.NumberOfSolutions() << " solution(s)" << std::endl;
    for (int i = 0; i < solver.NumberOfSolutions(); i++)
    {
      std::cout << "\nSolution " << i << ":\n";
      std::cout << *solver.SolutionPtr(i) << std::endl;
      std::cout << " Average squared projection error : " << solver.AverageSquaredProjectionErrors()[i] << std::endl;
    }
  }
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Time taken by SQPnL: " << duration.count() << " microseconds" << std::endl
            << std::endl;

  return 0;
}
