#pragma once
// Adapted from Otman Benchekroun's adaptation of:
// Copyright (C) 2020-2021 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.


#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <stdexcept>
#include <type_traits>  // std::conditional, std::is_same

#include "Spectra/Util/CompInfo.h"
using namespace Eigen;



class MatProd
{
public:
    using Scalar = double;
private:

    // unnecessary copy
    SparseMatrix<double> m_matB;
    const int m_n;

public:
    ///
    /// Constructor to create the matrix operation object.
    ///
    MatProd(const SparseMatrix<double>& B) :
        m_matB(B), m_n(B.rows())
    {
        if (m_n != B.rows())
            throw std::invalid_argument("LUSymShiftInvert: A and B must be square matrices of the same size");
    }
    int rows() const { return m_n; }
    int cols() const { return m_n; }
    void set_mat(const Eigen::SparseMatrix<double>& B)
    {
        m_matB = B;
    }
    void perform_op(const double* x_in, double* y_out) const
    {
        Eigen::Map<const Eigen::VectorXd>x(x_in, m_n);
        Eigen::Map<Eigen::VectorXd> y(y_out, m_n);
        y = m_matB*(x);
    }
};

class LUSymShiftInvert
{
public: 
    using Scalar = double;
private:
    // unnecessary copy
    SparseMatrix<double> m_matA;
    // unnecessary copy
    SparseMatrix<double> m_matB;
    // mildly unnecessary copy
    Eigen::SparseMatrix<double> C;
    
    const int m_n;
    Eigen::SparseLU<Eigen::SparseMatrix<double>>* m_solver;

public:
    ///
    /// Constructor to create the matrix operation object.
    ///
    LUSymShiftInvert(const SparseMatrix<double>& A, const SparseMatrix<double>& B, Eigen::SparseLU<Eigen::SparseMatrix<double>>* solver) :
        m_matA(A), m_matB(B), m_n(A.rows()), m_solver(solver)
    {
        if (m_n != A.cols() || m_n != B.rows() || m_n != B.cols())
            throw std::invalid_argument("LLUSymShiftInvert: A and B must be square matrices of the same size");
    }
    int rows() const { return m_n; }
    int cols() const { return m_n; }
    void set_mat(const Eigen::SparseMatrix<double>& A, const Eigen::SparseMatrix<double> & B)
    {
        m_matA = A;
        m_matB = B;
    }
    void set_solver(Eigen::SparseLU<Eigen::SparseMatrix<double>>* solver)
    {
        m_solver = solver;
    }

    void set_custom_shift(const double& sigma)
    {
        C = m_matA + sigma * m_matB;
        m_solver->compute(C);
        const bool success = (m_solver->info() == Eigen::Success);
        if (!success)
            throw std::invalid_argument("LUSymShiftInvert: factorization failed with the given shift");
    }

    void set_shift(const double& sigma)
    {
      //  C = m_matA + sigma * m_matB;
      //  m_solver->compute(C);
   //     const bool success = (m_solver->info() == Eigen::Success);
     //   if (!success)
      //      throw std::invalid_argument("UMFPACKSymShiftInvert: factorization failed with the given shift");
    }
    void perform_op(const double* x_in,double* y_out) const
    {
        Eigen::Map<const Eigen::VectorXd>x(x_in, m_n);
        Eigen::Map<Eigen::VectorXd>y(y_out, m_n);
        y = m_solver->solve(x);
    }
};

