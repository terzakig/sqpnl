# SQPnL 
C++ Implementation of the SQPnL algorithm.

The algorithm is the generic Perspective-n-Line (PnL) solver described in the paper ["Fast and Consistently Accurate Perspective-n-Line Pose Estimation"](https://www.researchgate.net/publication/383692992_Fast_and_Consistently_Accurate_Perspective-n-Line_Pose_Estimation). Supplementary material [here](https://www.researchgate.net/publication/383693236_Supplementary_material_for_the_SQPnL_paper).

## Required libraries
SQPnL requires the [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) library to build. Besides [rank revealing](https://nhigham.com/2021/05/19/what-is-a-rank-revealing-factorization/) QR and optionally SVD, the use of Eigen is confined to matrix addition, transposition and multiplication.
Choosing Eigen was motivated by its increasing popularity and lightweight character. There are also two examples of using the solver in this repository.
The first example requires OpenCV while the second uses standard arrays only. Build will proceed with either one of 1) or 2), depending on whether OpenCV is found or not.

## Build
-----
The repository uses a shared lbrary (with [sqpnp](https://github.com/terzakig/sqpnp)) called **SQPEngine** which is included as a submodule. In the sqpnl root, initialize the submodules recursively,

``git submodule update --init --recursive``

Create a ``build`` directory in the root of the cloned repository and run ``cmake``:

``mkdir build``

``cd build``

``cmake ..``

or, for a *release* build,

``cmake .. -DCMAKE_BUILD_TYPE=Release``

The latter will allow for more accurate timing of average execution time. Finally build everything:

``make``

To run the PnL example(s), once in the ``build`` directory,

``./examples/sqpnl_example``

## Non-default parameters
See ``struct SolverParameters`` in ``SQPEngine/sqp_engin/sqp_engine.h`` which contains SQPnL's parameters that can be specified by the caller.
For instance, to use SVD instead of the default RRQR for the nullspace basis of Omega, the following fragment can be used:
```c++
  // call solver with user-specified parameters (and equal weights for all lines). Note that lines and projections can be defined by pairs of points (points1-points2 and projections1-projections2)
  //  or with vectors of sqpnl::Line and sqpnl::Projection objects. 
  sqp_engine::SolverParameters params;
  params.omega_nullspace_method = sqp_engine::OmegaNullspaceMethod::SVD;
  sqp_engine::PnLSolver solver(points1, points2, projections1, projections2, std::vector<Eigen::Vector3d>(), std::vector<double>(n, 1.0), params);
  // sqp_engine::PnLSolver solver(lines, projections, std::vector<Eigen::Vector3d>(), std::vector<double>(n, 1.0), params);
```
Similarly, to use SVD in place of [FOAM](https://www.researchgate.net/publication/316445722_An_efficient_solution_to_absolute_orientation) for the nearest rotation matrix computations, use
```c++
params.nearest_rotation_method = sqp_engine::NearestRotationMethod::SVD;
```

## Cite as
If you use this code in your published work, please cite the following paper:<br><br>
<pre>
  @inproceedings{terzakis2025fast,
  title={Fast and Consistently Accurate Perspective-n-Line Pose Estimation},
  author={Terzakis, George and Lourakis, Manolis},
  booktitle={International Conference on Pattern Recognition},
  pages={97--112},
  year={2025},
  organization={Springer}
}
</pre>

