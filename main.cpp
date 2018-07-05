//
// -----------------------------------------------
// Provides Extreme Learning Machine algorithm.
// -----------------------------------------------
//
// Refer to:
// G.-B. Huang et al., “Extreme Learning Machine for Regression and Multiclass Classification,” IEEE TSMC, vol. 42, no. 2, pp. 513-529, 2012.
//
// Vladislavs D.
//


/////////////////////////////////////
//                                 //
// modified by LI LIANG at 2018/7  //
//                                 //
/////////////////////////////////////

#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>

#define  NN 581012
#define LL 100
#define CC 0.1

#include "Eigen/Core"
#include "Eigen/Cholesky"

using namespace std;
using namespace Eigen;

int compare(const void *a, const void *b);

//template <typename Derived>
MatrixXd buildTargetMatrix(double *Y, int nLabels);

// entry function to train the ELM model
// INPUT: X, Y, nhn, C
// OUTPUT: inW, bias, outW
template<typename Derived>
int elmTrain(double *X, int dims, int nsmp,
             double *Y,
             const int nhn, const double C,
             MatrixBase<Derived> &inW, MatrixBase<Derived> &bias, MatrixBase<Derived> &outW) {

// map the samples into the matrix object
    MatrixXd mX = Map<MatrixXd>(X, dims, nsmp);


// build target matrix
    MatrixXd mTargets = buildTargetMatrix(Y, nsmp);

// generate random input weight matrix - inW
    inW = MatrixXd::Random(nhn, dims);

// generate random bias vectors
    bias = MatrixXd::Random(nhn, 1);
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
// compute the pre-H matrix
    MatrixXd preH = inW * mX + bias.replicate(1, nsmp);

// compute hidden neuron output
    MatrixXd H = (1 + (-preH.array()).exp()).cwiseInverse();

// build matrices to solve Ax = b
    MatrixXd A = (MatrixXd::Identity(nhn, nhn)).array() * (1 / C) + (H * H.transpose()).array();
    MatrixXd b = H * mTargets.transpose();

// solve the output weights as a solution to a system of linear equations
    outW = A.llt().solve(b);
	end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Training time: " << elapsed_seconds.count() << "s\n";
    return 0;

}

// entry function to predict class labels using the trained ELM model on test data
// INPUT : X, inW, bias, outW
// OUTPUT : scores
template<typename Derived>
int elmPredict(double *X, int dims, int nsmp,
               MatrixBase<Derived> &mScores,
               MatrixBase<Derived> &inW, MatrixBase<Derived> &bias, MatrixBase<Derived> &outW) {

    // map the sample into the Eigen's matrix object
    MatrixXd mX = Map<MatrixXd>(X, dims, nsmp);
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
    // build the pre-H matrix
    MatrixXd preH = inW * mX + bias.replicate(1, nsmp);

    // apply the activation function
    MatrixXd H = (1 + (-preH.array()).exp()).cwiseInverse();

    // compute output scores
    mScores = (H.transpose() * outW).transpose();
	end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
	std::cout << "Predict time: " << elapsed_seconds.count() << "s\n";
    return 0;
}


// --------------------------
// Helper functions
// --------------------------

// compares two integer values
//int compare( const void* a, const void *b ) {
//	return ( *(int *) a - *(int *) b );
//}

int compare(const void *a, const void *b) {
    const double *da = (const double *) a;
    const double *db = (const double *) b;
    return (*da > *db) - (*da < *db);
}

// builds 1-of-K target matrix from labels array
//template <typename Derived>
MatrixXd buildTargetMatrix(double *Y, int nLabels) {

    // make a temporary copy of the labels array
    double *tmpY = new double[nLabels];
    double *Y_unique = new double[nLabels];
    for (int i = 0; i < nLabels; i++) {
        tmpY[i] = Y[i];
    }
    // sort the array of labels
    qsort(tmpY, nLabels, sizeof(double), compare);

    // count unique labels
    int nunique = 0;
    Y_unique[0] = tmpY[0];
    for (int i = 0; i < nLabels - 1; i++) {
        if (tmpY[i] != tmpY[i + 1]) {
            nunique++;
            Y_unique[nunique] = tmpY[i + 1];
        }
    }
    nunique++;

    delete[] tmpY;

    MatrixXd targets(nunique, nLabels);
    targets.fill(0);


    // fill in the ones
    for (int i = 0; i < nLabels; i++) {
        int index = 0;
        while (index++ < nunique) {
            if (Y[i] == Y_unique[index]) {
                targets(index, i) = 1;
                break;
            }
        }
    }
    delete[] Y_unique;
    // normalize the targets matrix values (-1/1)
    targets *= 2;
    targets.array() -= 1;
    return targets;

}

int main() {	
    cout<<Eigen::nbThreads()<<"\n";
    double *x = (double *) malloc(31374648 * sizeof(double));
    double *y = (double *) malloc(581012 * sizeof(double));

    std::ifstream fin("covtype.data");
    std::string line;

    long long int row = 0;
    int column;
    int data[55];
    while (getline(fin, line)) {
        char cstr[100000];
        strcpy(cstr, line.c_str());
        char *p = strtok(cstr, ",");
        column = 0;
        while (p) {
            data[column] = atoi(p);

            p = strtok(NULL, ",");
            if (column < 54)
                x[row * 54 + column] = data[column];
            else
                y[row] = data[column];
            ++column;
        }
        ++row;
    }

    MatrixXd inW;    // input weight
    MatrixXd bias;   // b
    MatrixXd outW;   // output weight
    MatrixXd mScore;  //predict result
    
    elmTrain(x, 54, NN, y, LL, CC, inW, bias, outW);
    elmPredict(x, 54, NN, mScore, inW, bias, outW);
    return 0;
}
