#include <igl/opengl/glfw/Viewer.h>
#include <igl/triangulated_grid.h>
#include <Spectra/SymGEigsShiftSolver.h>
#include <igl/vector_area_matrix.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/repdiag.h>
#include <igl/sort.h>
#include "LUSymShiftInvert.h"
#include <igl/matlab_format.h>


int main(int argc, char *argv[])
{
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  igl::read_triangle_mesh(argc == 1 ? "../beetle.off" : argv[1], V, F);

  // Assemble the area matrix (note that A is #Vx2 by #Vx2)
  Eigen::SparseMatrix<double> A;
  igl::vector_area_matrix(F,A);

  // Assemble the cotan laplacian matrix
  Eigen::SparseMatrix<double> L;
  igl::cotmatrix(V,F,L);

  Eigen::SparseMatrix<double> L_flat;
  igl::repdiag(L,2,L_flat);

  // Minimize the LSCM energy
  Eigen::SparseMatrix<double> Q = -L_flat - 2.*A;

  Eigen::SparseMatrix<double> M;
  igl::massmatrix(V,F,igl::MASSMATRIX_TYPE_DEFAULT,M);
  Eigen::SparseMatrix<double> M2;
  igl::repdiag(M,2,M2);


  Eigen::MatrixXd U;
  Eigen::MatrixXd S;
  {
    const Eigen::SparseMatrix<double> & A = Q;
    const Eigen::SparseMatrix<double> & B = M2;
    Eigen::SparseLU<Eigen::SparseMatrix<double>>* lu_solver = new  Eigen::SparseLU<Eigen::SparseMatrix<double>>();

    MatProd Bop(B);
    LUSymShiftInvert op(A, B, lu_solver);
    int r = 5;
    Spectra::SymGEigsShiftSolver<LUSymShiftInvert, MatProd, Spectra::GEigsMode::ShiftInvert> geigs(op, Bop, r, r*2.0, 0);

    
    // Adapted from Otman Benchekroun's example:
    op.set_mat(A, B);
    Bop.set_mat(B);
    op.set_custom_shift(0);

    geigs.init();
    // int nconv = geigs_umf.compute(SortRule::LargestMagn);
    std::cout << "Computing eigenvalues/eigenvectors using shift invert mode..." << std::endl;
    int nconv = geigs.compute(Spectra::SortRule::LargestMagn);
    //B_spectra.resize(V.rows() * 3, 1);
    //B_spectra.setZero();
    if (geigs.info() == Spectra::CompInfo::Successful)
    {
      U = geigs.eigenvectors();
      S = geigs.eigenvalues();
      Eigen::MatrixXd S_mat = Eigen::MatrixXd(S.cwiseAbs());
      //sort these according to S.abs()
      Eigen::MatrixXi I;
      Eigen::MatrixXd S_sorted;
      igl::sort(S_mat, 1, true, S_sorted, I);
      Eigen::MatrixXd U_sorted = Eigen::MatrixXd::Zero(U.rows(), U.cols());
      for (int i = 0; i < S.rows(); i++)
      {
          U_sorted.col(i) = U.col(I(i));
      }
      S = S_sorted;
      U = U_sorted;
    }else
    {
      printf("Mode computation failed!");
    }
  }

  Eigen::MatrixXd UV(V.rows(),2);
  UV<< U.col(2).head(V.rows()),U.col(2).tail(V.rows());

  // Plot the mesh
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(UV, F);
  viewer.data().set_face_based(true);
  viewer.launch();
}
