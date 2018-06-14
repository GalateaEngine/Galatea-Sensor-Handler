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

//for holding processed pixels that appear in both the keyframe and the current frame
class pPixel
{
public:
	Point imagePixel;
	Point keyframePixel;
	Mat worldPoint;
	double depth;
	double residualSum;
	double keyframeIntensity;
	double imageIntensity;
	//16 is for the number of numbers in a sim(3) var
	double derivatives[16];
};

double findY(Point pixelU, Point projectedPoint, KeyFrame keyframe, Mat image, double rmean)
{
	//calc photometric residue
	double r = calcPhotometricResidual(pixelU, projectedPoint, keyframe, image, rmean);

	//Im not exactly sure if this is right? It's asking the derivate of a constant with respect to a constant. 
	//The literal meaning of derviate would lead me to believe it's the calculated photometric residual, but why be so unclear?
	//(See eq. 6 in the paper)
	double photoDeriv = r;
	double pixelVar = pixelIntensityNoise + (photoDeriv * photoDeriv) * keyframe.inverseDepthVarianceV.at<double>(pixelU);

	return HuberNorm(r / pixelVar, 1);
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
	//solve for cp such that cp = cpo + dif
	//dif = 

	//calculate b
	//1xi * ixi * ix16 =  
	b = error.t() * information * jacobian;

	//Calc H
	//16xi * ixi * ix16 = 16x16
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
	//16xi * ixi
	return -b * (l.t() * l);
}

double CalcErrorVal(std::vector<pPixel> residuals)
{
	double error = 0;
	for (int i = 0; i < residuals.size; i++)
	{
		error += residuals[i].residualSum * residuals[i].residualSum;
	}
	return error;
}

double alpha = 1e-6;
double derivative(Mat cameraPose, Point pixelU, Point projectedPoint, KeyFrame keyframe, Mat image, double rmean, int bIndexX, int bIndexY) //b is our guessed position, x a given pixel
{
	Mat bCopy = cameraPose.clone();
	bCopy.at<double>(bIndexX, bIndexY) += alpha;
	double y1 = findY(pixelU, projectedPoint, keyframe, image, rmean);
	bCopy = cameraPose.clone();
	bCopy.at<double>(bIndexX, bIndexY) -= alpha;
	double y2 = findY(pixelU, projectedPoint, keyframe, image, rmean);
	return (y1 - y2) / (2 * alpha);
}

std::vector<pPixel> ComputeJacobian(Mat cameraParams, Mat cameraPose, KeyFrame keyframe, Mat image, double rmean, int numRes)//b is our guessed position, x is our pixel info, y is residual
{
	std::vector<pPixel> jacobianResults;

	//for our image
	for (int x = 0; x < image.cols; x++)
	{
		for (int y = 0; y < image.rows; y++)
		{
			//get pixel location with respect to our new frame
			Point pixelU = Point(x, y);
			double keyDepthAtU = keyframe.inverseDepthD.at<double>(pixelU);
			//project into world space
			Mat p = keyframe.cameraTransformationAndScaleS.t() * piInv(makeHomo(pixelU.operator cv::Vec<int, 2>), keyDepthAtU);
			//project into new image
			Point projectedPoint = projectWorldPointToCameraPointU(cameraParams, cameraPose, p).operator cv::Vec<double, 2>;
			//do a bounds check, continue if we are out of range
			if (projectedPoint.x < 0 || projectedPoint.y < 0 || projectedPoint.x > image.rows || projectedPoint.y > image.cols) continue;
			//set inital pixel
			pPixel npixel;
			npixel.keyframePixel = pixelU;
			npixel.imagePixel = projectedPoint;
			npixel.worldPoint = p;
			npixel.depth = keyframe.inverseDepthD.at<double>(pixelU);;
			npixel.keyframeIntensity = keyframe.scaledImageI.at<double>(pixelU);
			jacobianResults.push_back(pPixel());
			//for the sim(3) vars
			for (int i = 0; i < cameraPose.rows; i++)
			{
				for (int j = 0; j < cameraPose.cols; j++)
				{
					//compute this section of the jacobian and store for jacobian compilation
					//jc.at<double>((x * image.rows) + y, (i * b.rows) + j) = derivative(cameraParams, b, pixelU, projectedPoint, keyframe, image, rmean, i, j);
					jacobianResults.back().derivatives[(i * cameraPose.cols) + j] = derivative(cameraPose, pixelU, projectedPoint, keyframe, image, rmean, i, j);
				}
			}
		}
	}
	return jacobianResults;
}

std::vector<pPixel> ComputeResiduals(Mat cameraParams, Mat cameraPose, KeyFrame keyframe, Mat image, double rmean)//b is our guessed position, x is our pixel info, y is residual
{
	std::vector<pPixel> results;

	//for our image
	for (int x = 0; x < image.cols; x++)
	{
		for (int y = 0; y < image.rows; y++)
		{
			//get pixel location with respect to our new frame
			Point pixelU = Point(x, y);
			double keyDepthAtU = keyframe.inverseDepthD.at<double>(pixelU);
			//project into world space
			Mat p = keyframe.cameraTransformationAndScaleS.t() * piInv(makeHomo(pixelU.operator cv::Vec<int, 2>), keyDepthAtU);
			//project into new image
			Point projectedPoint = projectWorldPointToCameraPointU(cameraParams, cameraPose, p).operator cv::Vec<double, 2>;
			//do a bounds check, continue if we are out of range
			if (projectedPoint.x < 0 || projectedPoint.y < 0 || projectedPoint.x > image.rows || projectedPoint.y > image.cols) continue;
			//set inital pixel
			pPixel npixel;
			npixel.keyframePixel = pixelU;
			npixel.imagePixel = projectedPoint;
			npixel.worldPoint = p;
			npixel.depth = keyframe.inverseDepthD.at<double>(pixelU);;
			npixel.keyframeIntensity = keyframe.scaledImageI.at<double>(pixelU);
			npixel.residualSum = findY(pixelU, projectedPoint, keyframe, image, rmean);
			results.push_back(pPixel());
		}
	}
	return results;
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

Mat CalcErrorVec(std::vector<pPixel> pixels)
{
	int objects = pixels.size;
	Mat errorVec(1, objects, CV_64FC1);
	//calc difference
	for (int x = 0; x < objects; x++)
	{
		errorVec.at<double>(0, x) = pixels[x].imageIntensity - pixels[x].keyframeIntensity;
	}
	return errorVec;
}

//pixel U is in fact an index
double calcPhotometricResidual(Point pixelU, Point projectedPoint, KeyFrame keyframe, Mat imageT, double globalResidue)
{
	double r;//single pixel
	r = keyframe.scaledImageI.at<uchar>(pixelU) - imageT.at<uchar>(projectedPoint) - globalResidue;
	return r;
}

void ComputeMedianResidualAndCorrectedPhotometricResiduals(Mat cameraPose, Mat image, KeyFrame kf, std::vector<pPixel> & results, double & median)
{
	// max heap to store the higher half elements 
	std::priority_queue<double> max_heap_left;

	// min heap to store the lower half elements
	std::priority_queue<double, std::vector<double>, std::greater<double>> min_heap_right;
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			//calc residual
			//get pixel location with respect to our new frame
			Point pixelU = Point(i, j);
			double keyDepthAtU = kf.inverseDepthD.at<double>(pixelU);
			//project into world space
			Mat p = kf.cameraTransformationAndScaleS.t() * piInv(makeHomo(pixelU.operator cv::Vec<int, 2>), keyDepthAtU);
			//project into new image
			Point projectedPoint = projectWorldPointToCameraPointU(cameraParams, cameraPose, p).operator cv::Vec<double, 2>;
			//do a bounds check, continue if we are out of range
			if (projectedPoint.x < 0 || projectedPoint.y < 0 || projectedPoint.x > image.rows || projectedPoint.y > image.cols) continue;
			//set inital pixel
			pPixel npixel;
			npixel.keyframePixel = pixelU;
			npixel.imagePixel = projectedPoint;
			npixel.worldPoint = p;
			npixel.depth = kf.inverseDepthD.at<double>(pixelU);;
			npixel.keyframeIntensity = kf.scaledImageI.at<double>(pixelU);
			npixel.residualSum = kf.scaledImageI.at<double>(pixelU) - image.at<double>(projectedPoint);
			results.push_back(pPixel());
			double x = npixel.residualSum;
			// case1(left side heap has more elements)
			if (max_heap_left.size() > min_heap_right.size())
			{
				if (x < median)
				{
					min_heap_right.push(max_heap_left.top());
					max_heap_left.pop();
					max_heap_left.push(x);
				}
				else
					min_heap_right.push(x);

				median = ((double)max_heap_left.top()
					+ (double)min_heap_right.top()) / 2.0;
			}
			else if (max_heap_left.size() == min_heap_right.size())
			{
				if (x < median)
				{
					max_heap_left.push(x);
					median = (double)max_heap_left.top();
				}
				else
				{
					min_heap_right.push(x);
					median = (double)min_heap_right.top();
				}
			}
			else
			{
				if (x > median)
				{
					max_heap_left.push(min_heap_right.top());
					min_heap_right.pop();
					min_heap_right.push(x);
				}
				else
					max_heap_left.push(x);

				median = ((double)max_heap_left.top()
					+ (double)min_heap_right.top()) / 2.0;
			}
		}
	}
}


//computes the update
Mat TransformJacobian(Mat jacobian, Mat residuals)
{
	Mat JT = jacobian.t(); // JT
	Mat JTJ = JT * jacobian; // JT * J
	Mat l = MatrixSqrt(JTJ);
	InvertLowerTriangluar(l);
	Mat JTJi = l.t() * l; // (JT * J)^-1
	Mat JTJiJT = JTJi * JT; // (JT * J)^-1 * JT
	return JTJiJT * residuals; // (JT * J)^-1 * JT * r
}

double pixelIntensityNoise = 1.0;
Mat cameraParams;
Mat CalcGNPosOptimization(Mat image, KeyFrame keyframe)
{
	//set initial camera pose
	Mat cameraPose = keyframe.cameraTransformationAndScaleS;

	//run gauss-newton optimization
	double residualSum = 0.0;
	double oldResidual = 1.0;
	double lambda = 1.0;
	while (fabs(residualSum - oldResidual) > 0)//while we have not converged
	{
		oldResidual = residualSum;

		//calculate all residuals and the sum
		std::vector<pPixel> residuals;
		double rmean = 0;
		ComputeMedianResidualAndCorrectedPhotometricResiduals(cameraPose, image, keyframe, residuals, rmean);
		//calculate error with current residuals
		//Mat errorVec = CalcErrorVec(residuals);
		double error = CalcErrorVal(residuals);

		//update pose estimate
		std::vector<pPixel> jacobianRes = ComputeJacobian(cameraParams, cameraPose, keyframe, image, rmean, image.cols * image.rows);

		//place jacobians and residuals into matrices
		Mat jacobianMat(jacobianRes.size, 16, CV_64FC1);
		Mat residualsMat(jacobianRes.size, 1, CV_64FC1);
		for (int i = 0; i < jacobianRes.size; i++)
		{
			residualsMat.at<double>(i, 0) = residuals[i].residualSum;
			for (int j = 0; j < 16; j++)
			{
				residualsMat.at<double>(i, j) = residuals[i].derivatives[j];
			}
		}
		//calculate deltax from derivatives
		Mat deltaX = TransformJacobian(jacobianMat, residualsMat);
		//SolveSparseMatrix(H + lambda, b);


		//store position
		Mat camOld = cameraPose;
		//increment camera pose
		//for the sim(3) vars
		for (int i = 0; i < cameraPose.rows; i++)
		{
			for (int j = 0; j < cameraPose.cols; j++)
			{
				cameraPose.at<double>(i,j) += deltaX.at<double>((i * cameraPose.cols) + j) * lambda;
			}
		}
		//compute new residuals
		std::vector<pPixel> nresiduals = ComputeResiduals(cameraParams, cameraPose, keyframe, image, rmean);
		if (error < CalcErrorVal(nresiduals))
		{
			cameraPose = camOld;
			lambda *= 2;
		}
		else
		{
			lambda /= 2;
		}

	}
	return cameraPose;
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


	Mat position = CalcGNPosOptimization(cameraFrame, lastKey);

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