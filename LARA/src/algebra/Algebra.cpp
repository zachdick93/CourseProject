#include "../../include/algebra/Algebra.hpp"
#include <iostream>
#include <iomanip>

using std::cout;
using std::endl;
/*
	This code is a quick filler to replace a linear algebra library used in the original project. Sources of this code below.
		Source of the determinant function logic: https://www.tutorialspoint.com/cplusplus-program-to-compute-determinant-of-a-matrix
		Source of the logic for the other functions: https://www.geeksforgeeks.org/adjoint-inverse-matrix/
*/


namespace algebra {
double Algebra::determinant(const vector<vector<double>> &matrix, size_t n) {
    int det = 0;
    vector<vector<double>> submatrix(n-1);
    if (n == 2)
        return ((matrix[0][0] * matrix[1][1]) - (matrix[1][0] * matrix[0][1]));
    else {
        for (int x = 0; x < n; x++) {
             int subi = 0;
            for (int i = 1; i < n; i++) {
                int subj = 0;
                submatrix[subi].resize(n-1);
                for (int j = 0; j < n; j++) {
                    if (j == x)
                        continue;
                    submatrix[subi][subj] = matrix[i][j];
                    subj++;
                }
                subi++;
            }
            det = det + (pow(-1, x) * matrix[0][x] * determinant( submatrix, n - 1 ));
        }
   }
   return det;
}

vector<vector<double>> Algebra::getCofactor(const vector<vector<double>> &matrix, size_t p, size_t q, size_t n) 
{ 
	size_t i = 0, j = 0; 
    vector<vector<double>> temp = matrix;

	// Looping for each element of the matrix 
	for (size_t row = 0; row < n; row++) 
	{ 
		for (size_t col = 0; col < n; col++) 
		{ 
			// Copying into temporary matrix only those element 
			// which are not in given row and column 
			if (row != p && col != q) 
			{ 
				temp[i][j++] = matrix[row][col]; 

				// Row is filled, so increase row index and 
				// reset col index 
				if (j == n - 1) 
				{ 
					j = 0; 
					i++; 
				} 
			} 
		} 
	}
    return temp; 
}

// Function to get adjoint
vector<vector<double>> Algebra::adjoint(const vector<vector<double>> &matrix, size_t n) 
{
    vector<vector<double>> temp = matrix;
    vector<vector<double>> adj = matrix;
	if (n == 1) 
	{ 
		adj[0][0] = 1; 
		return adj; 
	} 

	// temp is used to store cofactors of A[][] 
	int sign = 1; 

	for (size_t i=0; i<n; i++) 
	{ 
		for (size_t j=0; j<n; j++) 
		{ 
			// Get cofactor of A[i][j] 
			temp = getCofactor(matrix, i, j, n); 

			// sign of adj[j][i] positive if sum of row 
			// and column indexes is even. 
			sign = ((i+j)%2==0)? 1: -1; 

			// Interchanging rows and columns to get the 
			// transpose of the cofactor matrix 
			adj[j][i] = (sign)*(determinant(temp, n-1)); 
		} 
	}
    return adj;
} 

// Function to calculate and store inverse. Empty container returned if singular
vector<vector<double>> Algebra::inverse(const vector<vector<double>> &matrix, size_t n) 
{ 
    vector<vector<double>> inverse = matrix;
	// Find determinant of A[][] 
	int det = determinant(matrix, n); 

	if (det == 0) 
	{ 
		cout << "Singular matrix, can't find its inverse"; 
		return vector<vector<double>>(); 
	} 

	// Find adjoint 
	vector<vector<double>> adj = adjoint(matrix, n); 

	// Find Inverse using formula "inverse(A) = adj(A)/det(A)" 
	for (size_t i=0; i<n; i++) 
		for (size_t j=0; j<n; j++) 
			inverse[i][j] = adj[i][j] / det; 

	return inverse; 
} 

}




