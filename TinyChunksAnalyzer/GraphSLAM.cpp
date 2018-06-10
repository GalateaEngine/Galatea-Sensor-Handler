#include "stdafx.h"

class KeyFrame
{
public:
	Mat inverseDepthD;
	Mat scaledImageI;//mean 1
	Mat inverseDepthVarianceV;
	Mat cameraTransformationAndScaleS; //taken and scaled with repsect to the world frame W aka the first frame. This is an element of Sim(3)
};

Mat makeHomo(Mat input);
Mat piInv(Mat input, double invDepth);
Mat pi(Mat input);

double findY(Mat cameraParams, Mat cameraPose, Point pixelU, Mat depthMat, KeyFrame keyframe, Mat image, double rmean)
{
	double residualSum = 0.0;

	//calculate initial parameters
	double depthAtU = depthMat.at<double>(pixelU);
	double keyDepthAtU = keyframe.inverseDepthD.at<double>(pixelU);
	Mat p = cameraPose * keyframe.cameraTransformationAndScaleS.t() * piInv(makeHomo(pixelU.operator cv::Vec<int, 2>), depthAtU);
	double r = calcPhotometricResidual(cameraParams, pixelU, p, cameraPose, keyframe, image, rmean);
	//calculate variance of pixel u
	Mat p2 = cameraPose * keyframe.cameraTransformationAndScaleS.t() * piInv(makeHomo(pixelU.operator cv::Vec<int, 2>), keyDepthAtU);
	double r2 = calcPhotometricResidual(cameraParams, pixelU, p2, cameraPose, keyframe, image, rmean);
	double photoDeriv = (r - r2) / (depthAtU - keyDepthAtU);
	double pixelVar = pixelIntensityNoise + (photoDeriv * photoDeriv) * keyframe.inverseDepthVarianceV.at<double>(pixelU);

	//divide terms
	residualSum += HuberNorm(r / pixelVar, 1);
	return residualSum;
}

//turns the given point into a homogeneouse point
Mat makeHomo(Mat x)
{
	int size = x.rows;
	Mat xbar(size + 1, 1, CV_64FC1);
	int i = 0;
	for (; i < size; i++)
	{
		xbar.at<double>(i, 0) = x.at<double>(i, 0);
	}
	xbar.at<double>(i + 1, 0) = 1.0;
	return xbar;
}

Mat deHomo(Mat xbar)
{
	int size = xbar.rows;
	int cols = xbar.cols;
	Mat x(size - 1, cols, CV_64FC1);
	for (int j = 0; j < cols; j++)
	{
		int i = 0;
		double scale = xbar.at<double>(size - 1, j);
		for (; i < size - 1; i++)
		{
			x.at<double>(i, j) = xbar.at<double>(i, j) / scale;
		}
	}
}

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
		l.at<double>(k, k) = 1.0 / l.at<double>(k, k);
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

double CalcErrorVal(Mat errorVec, Mat informationMatrix)
{
	return Mat(errorVec.t() * informationMatrix * errorVec).at<double>(0, 0);
}

double alpha = 1e-6;
double derivative(Mat cameraParams, Mat cameraPose, Point pixelU, Mat depthMat, KeyFrame keyframe, Mat image, double rmean, int bIndexX, int bIndexY) //b is our guessed position, x a given pixel
{
	Mat bCopy = cameraPose.clone();
	bCopy.at<double>(bIndexX, bIndexY) += alpha;
	double y1 = findY(cameraParams, bCopy, pixelU, depthMat, keyframe, image, rmean);
	bCopy = cameraPose.clone();
	bCopy.at<double>(bIndexX, bIndexY) -= alpha;
	double y2 = findY(cameraParams, bCopy, pixelU, depthMat, keyframe, image, rmean);
	return (y1 - y2) / (2 * alpha);
}

Mat ComputeJacobian(Mat cameraParams, Mat depthMat, KeyFrame keyframe, Mat image, double rmean, Mat b, int numRes)//b is our guessed position, x is our pixel info, y is residual
{
	Mat jc(numRes, 8, CV_32FC1); //8 is fro the number of numbers in a sim(3) var

	for (int i = 0; i < b.rows; i++) 
	{
		for (int j = 0; j < b.cols; j++)
		{
			//run through all of Sim(3) variable
			for (int k = 0; k < 4; k++) 
			{
				for (int l = 0; l < 2; l++)
				{
					jc.at<double>((i * b.rows) + j, k) = derivative(cameraParams, b, Point(i, j), depthMat, keyframe, image, rmean, k, l);
				}
			}
		}
	}
	return jc;
}

double lambdaInit = 1;
//Mat optimizeError(Mat expectedPos, Mat landmarks, Mat informationMatrix)
//{
//	bool converged = false;
//	double lambda = lambdaInit;
//	Mat ident = Mat::eye(expectedPos.rows, expectedPos.cols, expectedPos.type);
//	double minError = std::numeric_limits<double>::infinity();
//	int errorCount = 0;
//	while (!converged)
//	{
//		Mat H, b;
//		Mat jacobian = ComputeJacobian();
//		Mat errorVec = CalcErrorVec(expectedPos, landmarks);
//		BuildLinearSystem(H, b, expectedPos, jacobian, errorVec, informationMatrix);
//		double error = CalcErrorVal(errorVec, informationMatrix);
//		Mat xOld = expectedPos;
//		Mat deltaX = SolveSparseMatrix(H + (lambda * ident), b);
//		expectedPos += deltaX;
//		if (error < CalcErrorVal(expectedPos, informationMatrix))
//		{
//			expectedPos = xOld;
//			lambda *= 2;
//		}
//		else
//		{
//			lambda /= 2;
//		}
//	}
//	return expectedPos;
//}



class PoseGraph
{
	std::list<KeyFrame> V;
	std::list<Mat> E; //sim(3) constraints between keyframes
};

//predicts new position based on dampened approximate velocity
void forwardPredictPosition(Mat &lastPos, Mat &Velocity)
{
	float dt = 0.5;
	float a = 0.85, b = 0.005;

	float xk, vk, rk;
	float xm;

	for (int i = 0; i < 6; i++)
	{
		xm = lastPos.at<double>(0, i);
		vk = Velocity.at<double>(0, i);
		xk = xm + (vk * dt);

		rk = xm - xk;

		xk += a * rk;
		vk += (b * rk) / dt;

		lastPos.at<double>(0, i) = xk;
		Velocity.at<double>(0, i) = vk;
	}
}

class Node
{
public:
	std::vector<Node> subNodes;
	Point3f location;
};


//projects 3d point input into 2d space
Mat pi(Mat input)
{
	Mat output(2, 1, CV_64FC1);
	//Not sure we need the dbz protection, test this later, perhaps we can pixle offset higher up the chain
	output.at<double>(0, 0) = input.at<double>(0, 0) / (input.at<double>(2, 0) + 0.00001);
	output.at<double>(1, 0) = input.at<double>(1, 0) / (input.at<double>(2, 0) + 0.00001);
	return output;
}

Mat piInv(Mat input, double invDepth)
{
	Mat output(3, 1, CV_64FC1);
	//Not sure we need the dbz protection, test this later, perhaps we can pixle offset higher up the chain
	output.at<double>(0, 0) = input.at<double>(0, 0) / invDepth;
	output.at<double>(1, 0) = input.at<double>(1, 0) / invDepth;
	output.at<double>(2, 0) = input.at<double>(2, 0) / invDepth;
	return output;
}

//puts the projected point from pi into camera space
Mat projectWorldPointToCameraPointU(Mat cameraParamsK, Mat cameraPoseT, Mat wPointP)
{
	Mat pBar = makeHomo(wPointP);
	//3x3 * 4x4 * 3x1 ???????? How can you dehomogenize an SE3 element?
	//DUH 3x3 * dehomo(4x4 * 4x1) = 3x3 * 3x1 = 3x1
	Mat notationalClarity = deHomo(cameraPoseT * pBar);
	return pi(cameraParamsK * notationalClarity);
}


double HuberNorm(double x, double epsilon)
{
	double huberMin = HuberNorm(x, 2.0);
	if (huberMin <= epsilon) return huberMin * huberMin;
	return HuberNorm(x, 1) - (epsilon / 2.0);
}

Mat CalcErrorVec(Mat cameraParams, KeyFrame kf, Mat image, Mat depthMap, Mat cameraPose)
{
	int objects = image.rows * image.cols;
	Mat errorVec(1, objects, CV_64FC1);
	//calc difference
	for (int x = 0; x < image.rows; x++)
	{
		for (int y = 0; y < image.cols; y++)
		{
			Point pixelU = Point(x, y);
			Mat worldPoint = cameraPose * kf.cameraTransformationAndScaleS.t() * piInv(makeHomo(pixelU.operator cv::Vec<int, 2>), depthMap.at<double>(pixelU));
			Point keyframePoint = Point(projectWorldPointToCameraPointU(cameraParams, kf.cameraTransformationAndScaleS, worldPoint).operator cv::Vec<int, 2>);
			errorVec.at<double>(0, (x*image.rows) + y) = depthMap.at<double>(pixelU) - kf.inverseDepthD.at<double>(keyframePoint);
		}
	}
	return errorVec;
}

//pixel U is in fact an index
double calcPhotometricResidual(Mat cameraParams, Point pixelU, Mat worldPointP, Mat cameraPose, KeyFrame keyframe, Mat imageT, double globalResidue)
{
	double r;//single pixel
	Point projectedpoint = projectWorldPointToCameraPointU(cameraParams, keyframe.cameraTransformationAndScaleS , worldPointP).operator cv::Vec<double, 2>;
	r = keyframe.scaledImageI.at<uchar>(pixelU) - imageT.at<uchar>(projectedpoint) - globalResidue; //hot DAMN
	return r;
}

double pixelIntensityNoise = 1.0;
Mat CalcGNPosOptimization(Mat image, Mat depthMat, KeyFrame keyframe)
{
	//u: pixel index
	//T: position of camera
	//p: projected 3d point of pixle u
	//sigma: variance of pixle u
	//r: photometric residual

	//set initial camera pose
	Mat cameraPose = keyframe.cameraTransformationAndScaleS;

	//set the median photometric residual
	uchar rmean;

	//run gauss-newton optomization
	double residualSum = 0.0;
	double oldResidual = 1.0;
	while (fabs(residualSum - oldResidual) > 0)//while we have not converged
	{
		oldResidual = residualSum;
		
		//calculate all residualsand the sum
		Mat residuals;

		//update pose estimate
		/*Mat H, b;
		Mat jacobian = ComputeJacobian(depthMat, keyframe, image, rmean, cameraPose, image.cols * image.rows);
		Mat errorVec = CalcErrorVec(expectedPos, landmarks);
		BuildLinearSystem(H, b, expectedPos, jacobian, errorVec, keyframe.inverseDepthVarianceV);
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
		}*/

	}

}

Mat lastPos;
Mat velocity;
PoseGraph keyframes;
KeyFrame lastKey;
//The main function for LS Graph SLAM. Takes input in the form of camera frames, and returns a matrix with the approximate position of the camera. 
//Also builds a map behind the scenes for which the point cloud can be accessed by the helper functions
//enhanced implementation of https://groups.csail.mit.edu/rrg/papers/greene_icra16.pdf
//K: is a 3x3 real mat with the camera parameters
//pi: perspective projection function
Mat LS_Graph_SLAM(Mat cameraFrame)
{


	//lastPos = CalcGNPosOptimization(cameraFrame);

	//construct expected pos mat based on cameras predicted position (load in the points that SHOULD be in view)
	//store all keyframe points into temp array
	//match to existing dots based on movement prediction


	//construct information matrix

	//convert keypoints into landmarks

	//optimize error

	//store knew last pos

	//recalculate aproximate velocity

	//return aprox position 

}

//Sets up matrices and other things
void Initialize_LS_Graph_SLAM(Mat cameraFrame)
{
	//initialize lastpost
	//initialize velocity
	//initialize posegraph
}