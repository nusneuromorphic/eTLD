#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>

class Matrix
{
    public:
        Matrix(int row, int col);
        // this class need a copy constructor and a copy assignment operator.
        ~Matrix();
        void printMat() const;

    public:
        int _row;
        int _col;
        double **_matrix;
};
#endif
