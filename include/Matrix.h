#ifndef MATRIX_H
#define MATRIX_H

#include<iostream>
using namespace std;

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

Matrix::Matrix(int row = 5, int col = (5)) :_row(row), _col(col)
{
	_matrix = new double*[_row];
	for (int i = 0; i<_row; i++)
	{
		_matrix[i] = new double[_col];
		for (int j = 0; j<_col; j++)
		{
			_matrix[i][j] = 0;
		}
	}
}

Matrix::~Matrix()
{
	for (int i = 0; i<_row; i++)
	{
		delete[] _matrix[i];
		_matrix[i] = NULL;
		//cout << "~Matrix(): "<< i << endl;
	}
	delete[] _matrix;
	_matrix = NULL;
	//cout << "~Matrix(): final" << endl;
}

void Matrix::printMat() const
{
	for (int i = 0; i<_row; i++)
	{
		for (int j = 0; j<_col; j++)
		{
			cout << _matrix[i][j] << "\t";
		}
		cout << endl;
	}
}

#endif 