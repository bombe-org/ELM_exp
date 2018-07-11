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

#define  N_COUNT 581012
#define IN_COUNT 54
#define L_COUNT 1000
#define C_COUNT 0.1

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

	std::chrono::time_point<std::chrono::system_clock> start, end;
// map the samples into the matrix object
    MatrixXd mX = Map<MatrixXd>(X, dims, nsmp);


// build target matrix
    MatrixXd mTargets = buildTargetMatrix(Y, nsmp);

// generate random input weight matrix - inW
	start = std::chrono::system_clock::now();
    inW = MatrixXd::Random(nhn, dims);

	end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "random w time: " << elapsed_seconds.count() << "s\n";
// generate random bias vectors
	start = std::chrono::system_clock::now();
    bias = MatrixXd::Random(nhn, 1);

	end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    std::cout << "random b time: " << elapsed_seconds.count() << "s\n";
// compute the pre-H matrix
start = std::chrono::system_clock::now();
    MatrixXd preH = inW * mX + bias.replicate(1, nsmp);
end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    std::cout << "computing H  time: " << elapsed_seconds.count() << "s\n";
// compute hidden neuron output
start = std::chrono::system_clock::now();
    MatrixXd H = (1 + (-preH.array()).exp()).cwiseInverse();
end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    std::cout << "computing g(H)  time: " << elapsed_seconds.count() << "s\n";

// build matrices to solve Ax = b
start = std::chrono::system_clock::now();
    MatrixXd A = (MatrixXd::Identity(nhn, nhn)).array() * (1 / C) + (H * H.transpose()).array();
end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    std::cout << "computing A  time: " << elapsed_seconds.count() << "s\n";
start = std::chrono::system_clock::now();
    MatrixXd b = H * mTargets.transpose();
end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    std::cout << "computing b  time: " << elapsed_seconds.count() << "s\n";

// solve the output weights as a solution to a system of linear equations
start = std::chrono::system_clock::now();
    outW = A.llt().solve(b);
end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    std::cout << "solving time: " << elapsed_seconds.count() << "s\n";
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

int main(int argc,const char* argv[]) {	
Eigen::setNbThreads(atoi(argv[1])); 
cout<<Eigen::nbThreads()<<"\n";
    double *x = (double *) malloc(IN_COUNT * N_COUNT * sizeof(double));
    double *y = (double *) malloc(N_COUNT * sizeof(double));

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
            if (column < IN_COUNT)
                x[row * IN_COUNT + column] = data[column];
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
    
    elmTrain(x, IN_COUNT, N_COUNT, y, L_COUNT, C_COUNT, inW, bias, outW);
    elmPredict(x, IN_COUNT, N_COUNT, mScore, inW, bias, outW);
    return 0;
}
