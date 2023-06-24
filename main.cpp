#include <igl/opengl/glfw/Viewer.h>
#include <igl/triangulated_grid.h>
#include <Spectra/SymGEigsShiftSolver.h>
#include <igl/massmatrix.h>
#include <igl/sort.h>
#include <igl/matlab_format.h>
#include <igl/sort.h>
#include <igl/slice.h>

#include <igl/eigs.h>
#include <cassert>
namespace igl
{
namespace spectra
{
  template <
    typename EigsScalar,
    typename DerivedU,
    typename DerivedS,
    typename Solver = Eigen::SparseLU<Eigen::SparseMatrix<EigsScalar>> >
  IGL_INLINE bool eigs(
    const Eigen::SparseMatrix<EigsScalar> & A,
    const Eigen::SparseMatrix<EigsScalar> & B,
    const size_t k,
    const EigsType type,
    Eigen::PlainObjectBase<DerivedU> & U,
    Eigen::PlainObjectBase<DerivedS> & S)
  {
    assert(type == igl::EIGS_TYPE_SM && "Only SM supported");

    class SparseMatProd
    {
      public:
        using Scalar = EigsScalar;
        const Eigen::SparseMatrix<Scalar> & m_B;
        SparseMatProd(const Eigen::SparseMatrix<Scalar> & B) : m_B(B) {}
        int rows() const { return m_B.rows(); }
        int cols() const { return m_B.cols(); }
        void perform_op(const Scalar *x_in, Scalar *y_out) const
        {
          typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXS;
          Eigen::Map<const VectorXS> x(x_in, m_B.cols());
          Eigen::Map<      VectorXS> y(y_out, m_B.rows());
          y = m_B * x;
        }
    };

    // Solver must expose .compute(A) and .solve(x)
    class ShiftInvert
    {
    public:
      using Scalar = EigsScalar;
    private:
      const Eigen::SparseMatrix<Scalar> & m_A;
      const Eigen::SparseMatrix<Scalar> & m_B;
      Scalar m_sigma;
      Solver m_solver;
    public:
      bool m_solver_is_successfully_factorized;
      ShiftInvert(
          const Eigen::SparseMatrix<Scalar>& A, 
          const Eigen::SparseMatrix<Scalar>& B, 
          const Scalar sigma):
          m_A(A), m_B(B)
      {
        assert(m_A.rows() == m_A.cols() && "A must be square");
        assert(m_B.rows() == m_B.cols() && "B must be square");
        assert(m_A.rows() == m_B.cols() && "A and B must have the same size");
        set_shift(sigma, true);
      }
      void set_shift(const Scalar & sigma, const bool force = false)
      {
        if(sigma == m_sigma && !force)
        {
          return;
        }
        m_sigma = sigma;
        const Eigen::SparseMatrix<Scalar> C = m_A + m_sigma * m_B;
        m_solver.compute(C);
        m_solver_is_successfully_factorized = (m_solver.info() == Eigen::Success);
      }
      int rows() const { return m_A.rows(); }
      int cols() const { return m_A.cols(); }
      void perform_op(const Scalar* x_in,Scalar* y_out) const
      {
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXS;
        Eigen::Map<const VectorXS>x(x_in, m_A.cols());
        Eigen::Map<VectorXS>y(y_out, m_A.rows());
        y = m_solver.solve(x);
      }
    };

    const EigsScalar sigma = 0;
    SparseMatProd Bop(B);
    ShiftInvert op(A, B, sigma);
    if(!op.m_solver_is_successfully_factorized)
    {
      return false;
    }
    Spectra::SymGEigsShiftSolver<ShiftInvert, SparseMatProd, Spectra::GEigsMode::ShiftInvert> geigs(op, Bop, k, 2*k, sigma);

    geigs.init();
    int nconv = geigs.compute(Spectra::SortRule::LargestMagn);
    if (geigs.info() != Spectra::CompInfo::Successful)
    {
      return false;
    }
    U = geigs.eigenvectors().template cast<typename DerivedU::Scalar>();
    S = geigs.eigenvalues().template cast<typename DerivedS::Scalar>();
    Eigen::MatrixXd S_mat = Eigen::MatrixXd(S.cwiseAbs());
    //sort these according to S.abs()
    Eigen::VectorXi I;
    igl::sort( Eigen::VectorXd(S.cwiseAbs()), 1, true, S, I);
    igl::slice(Eigen::MatrixXd(U),I,2,U);
    return true;
  }
}
}

#include <igl/vector_area_matrix.h>
#include <igl/cotmatrix.h>
#include <igl/repdiag.h>
namespace igl
{
template <typename DerivedV, typename DerivedF, typename QScalar>
void lscm_hessian(
  const Eigen::MatrixBase<DerivedV> & V,
  const Eigen::MatrixBase<DerivedF> & F,
  Eigen::SparseMatrix<QScalar> & Q)
{
  // Assemble the area matrix (note that A is #Vx2 by #Vx2)
  Eigen::SparseMatrix<QScalar> A;
  igl::vector_area_matrix(F,A);
  // Assemble the cotan laplacian matrix
  Eigen::SparseMatrix<QScalar> L;
  igl::cotmatrix(V,F,L);
  Eigen::SparseMatrix<QScalar> L_flat;
  igl::repdiag(L,2,L_flat);
  Q = -L_flat - 2.*A;
}
}

int main(int argc, char *argv[])
{
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  igl::read_triangle_mesh(argc == 1 ? "../beetle.off" : argv[1], V, F);

  Eigen::SparseMatrix<double> Q;
  igl::lscm_hessian(V,F,Q);
  Eigen::SparseMatrix<double> M;
  igl::massmatrix(V,F,igl::MASSMATRIX_TYPE_DEFAULT,M);
  Eigen::SparseMatrix<double> M2;
  igl::repdiag(M,2,M2);

  Eigen::MatrixXd U;
  Eigen::VectorXd S;
  igl::spectra::eigs(Q,M2,3,igl::EIGS_TYPE_SM,U,S);

  Eigen::MatrixXd UV(V.rows(),2);
  UV<< U.col(2).head(V.rows()),U.col(2).tail(V.rows());

  // Plot the mesh
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(UV, F);
  viewer.data().set_face_based(true);
  viewer.launch();
}
