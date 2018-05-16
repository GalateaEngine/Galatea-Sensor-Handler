#include "stdafx.h"

struct hPoint
{
	double x, y, z, w;
};

struct rotationVals
{
	double cosx, sinx, cosy, siny, cosz, sinz;
};

void BuildLinearSystem(Mat &H, Mat &b, Mat x, Mat jacobian, Mat error, Mat information)
{
	//calculate b
	b = error.t() * information * jacobian;

	//Calc H
	H = jacobian.t() * information * jacobian;
}

//simple cholesky decomposition
cv::Mat MatrixSqrt(cv::Mat a)
{
	int n = a.cols;

	cv::Mat ret(n, n, CV_64FC1);
	for (int r = 0; r < n; r++)
	{
		for (int c = 0; c <= r; c++)
		{
			if (c == r)
			{
				double sum = 0;
				for (int j = 0; j < c; j++)
				{
					sum += ret.at<double>(c, j) * ret.at<double>(c, j);
				}
				ret.at<double>(c, c) = sqrt(a.at<double>(c, c) - sum);
			}
			else
			{
				double sum = 0;
				for (int j = 0; j < c; j++)
					sum += ret.at<double>(r, j) * ret.at<double>(c, j);
				ret.at<double>(r, c) = (1.0 / ret.at<double>(c, c)) * (a.at<double>(r, c) - sum);
			}
		}
	}
	return ret;
}

//code roughly from my math lib
void InvertLowerTriangluar(Mat &l)
{
	int i, j, k;
	double sum;
	int n = l.rows;

	//Invert the diagonal elements of the lower triangular matrix L.

	for (k = 0; k < n; k++) {
		 l.at<double>(k,k) = 1.0 / l.at<double>(k, k);
	}

	//Invert the remaining lower triangular matrix L row by row.

	for (i = 1; i < n; i++) 
	{
		for (j = 0; j < i; j++) 
		{
			sum = 0.0;
			for (k = j; k < i; k++)
				sum += l.at<double>(i, k) * l.at<double>(k, j);
			l.at<double>(i, j) = -l.at<double>(i, i) * sum;
		}
	}
}

//solve for delta x in (H * dX = -b)
Mat SolveSparseMatrix(Mat H, Mat b)
{
	Mat l = MatrixSqrt(H);
	InvertLowerTriangluar(l);
	return -b * (l.t() * l);
}

Mat CalcErrorVec(Mat expectedPos, Mat landmarks)
{
	int objects = expectedPos.cols;
	Mat errorVec(1, objects, CV_64FC1);
	Mat splitError = expectedPos - landmarks;
	//calc difference
	for (int i = 0; i < objects; i++)
	{
		errorVec.at<double>(0, i) = splitError.at<double>(0, i);
		for (int j = 1; j < 6; j++)
		{
			errorVec.at<double>(j, i) += splitError.at<double>(j, i);
		}
	}
	return errorVec;
}

double CalcErrorVal(Mat errorVec, Mat informationMatrix)
{
	return Mat(errorVec.t() * informationMatrix * errorVec).at<double>(0,0);
}

rotationVals ComputeRobotRotationValues(Mat robotPose)
{
	rotationVals vals;
	vals.cosx = cos(robotPose.at<double>(3, 0));
	vals.cosy = cos(robotPose.at<double>(4, 0));
	vals.cosz = cos(robotPose.at<double>(5, 0));

	vals.sinx = sin(robotPose.at<double>(3, 0));
	vals.siny = sin(robotPose.at<double>(4, 0));
	vals.sinz = sin(robotPose.at<double>(5, 0));
}

Mat ComputeJacobian(rotationVals vals)
{
	Mat jacobian(3, 3, CV_64FC1);

	//col 1
	jacobian.at<double>(0, 0) = vals.siny * vals.sinz;
	jacobian.at<double>(1, 0) = -vals.siny * vals.cosz;
	jacobian.at<double>(2, 0) = vals.cosy;

	//col 2
	jacobian.at<double>(0, 1) = vals.cosx * vals.cosy * vals.sinz + vals.sinx * vals.cosz;
	jacobian.at<double>(1, 1) = -vals.cosx * vals.cosy * vals.cosz + vals.sinx * vals.sinz;
	jacobian.at<double>(2, 1) = -vals.cosx * vals.siny;

	//col 3
	jacobian.at<double>(0, 2) = -vals.sinx * vals.cosy * vals.sinz + vals.cosx * vals.cosz;
	jacobian.at<double>(1, 2) = vals.sinx * vals.cosy * vals.cosz + vals.cosx * vals.sinz;
	jacobian.at<double>(2, 2) = vals.sinx * vals.siny;


	return jacobian;
}

double lambdaInit = 1;
Mat optimizeError(Mat expectedPos, Mat landmarks, Mat informationMatrix)
{
	bool converged = false;
	double lambda = lambdaInit;
	Mat ident = Mat::eye(expectedPos.rows, expectedPos.cols, expectedPos.type);
	double minError = std::numeric_limits<double>::infinity();
	int errorCount = 0;
	while (!converged)
	{
		Mat H, b;
		rotationVals vals = ComputeRobotRotationValues(expectedPos);
		Mat jacobian = ComputeJacobian(vals);
		Mat errorVec = CalcErrorVec(expectedPos, landmarks);
		BuildLinearSystem(H, b, expectedPos, jacobian, errorVec, informationMatrix);
		double error = CalcErrorVal(errorVec, informationMatrix);
		Mat xOld = expectedPos;
		Mat deltaX = SolveSparseMatrix(H + (lambda * ident), b);
		expectedPos += deltaX;
		if (error < CalcErrorVal(expectedPos, informationMatrix))
		{
			expectedPos = xOld;
			lambda *= 2;
		}
		else
		{
			lambda /= 2;
		}
		if (minError > error) 
		{
			minError = error;
			errorCount = 0;
		}
		else errorCount++;
		if (errorCount > 10) converged = true;
	}
	return expectedPos;
}

//The main function for LS Graph SLAM. Takes input in the form of camera frames, and returns a matrix with the approximate position of the camera. 
Ptr<SURF> detector;
Mat LS_Graph_SLAM(Mat cameraFrame)
{
	detector = SURF::create();

	std::vector<KeyPoint> keypoints_1;

	detector->detect(cameraFrame, keypoints_1);

	//get aproximate movement from last frame till now

	//construct expected pos mat

	//construct information matrix

	//convert keypoints into landmarks

	//optimize error

	//return aprox position 

}

//Sets up matrices and other th ings
void Initialize_LS_Graph_SLAM()
{

}