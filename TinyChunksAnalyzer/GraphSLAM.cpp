#pragma once
#include "stdafx.h"

//turns the given point into a homogeneouse point
cv::Mat GraphSLAMer::makeHomo(cv::Mat x)
{
	int size = x.rows;
	cv::Mat xbar(size + 1, 1, CV_64FC1);
	int i = 0;
	if (x.type() == 4)
	{
		for (; i < size; i++)
		{
			xbar.at<double>(i, 0) = x.at<int>(i, 0);
		}
	}
	else
	{
		for (; i < size; i++)
		{
			xbar.at<double>(i, 0) = x.at<double>(i, 0);
		}
	}
	xbar.at<double>(i, 0) = 1.0;
	return xbar;
}

cv::Mat GraphSLAMer::deHomo(cv::Mat xbar)
{
	int size = xbar.rows;
	int cols = xbar.cols;
	cv::Mat x(size - 1, cols, CV_64FC1);
	for (int j = 0; j < cols; j++)
	{
		int i = 0;
		double scale = xbar.at<double>(size - 1, j);
		for (; i < size - 1; i++)
		{
			x.at<double>(i, j) = xbar.at<double>(i, j) / scale;
		}
	}
	return x;
}

void printMat(Mat &a)
{
	std::cout << a;
	std::cout << std::endl;
}

void printMat(const Mat &a)
{
	std::cout << a;
	std::cout << std::endl;
}

//simple cholesky decomposition
cv::Mat GraphSLAMer::MatrixSqrt(cv::Mat a)
{
	int n = a.cols;
	//printMat(a);
	cv::Mat ret = cv::Mat::eye(n, n, CV_64FC1);
	for (int r = 0; r < n; r++)
	{
		for (int c = 0; c <= r; c++)
		{
			//printMat(ret);
			if (c == r)
			{
				double sum = 0;
				for (int j = 0; j < c; j++)
				{
					double tval = ret.at<double>(c, j);
					sum += tval * tval;
				}
				double tval = a.at<double>(c, c) - sum;
				ret.at<double>(c, c) = sqrt(tval);
			}
			else
			{
				double sum = 0;
				for (int j = 0; j < c; j++)
				{
					double v1 = ret.at<double>(r, j);
					double v2 = ret.at<double>(c, j);
					sum += v1 * v2;
				}
				double v1 = (1.0 / ret.at<double>(c, c));
				double v2 = a.at<double>(r, c);
				ret.at<double>(r, c) = v1 * (v2 - sum);
			}
		}
	}
	return ret;
}

//code roughly from "my math lib"
void GraphSLAMer::InvertLowerTriangluar(cv::Mat &l)
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
cv::Mat GraphSLAMer::SolveSparseMatrix(cv::Mat H, cv::Mat b)
{
	cv::Mat l = MatrixSqrt(H);
	InvertLowerTriangluar(l);
	//16xi * ixi
	return -b * (l.t() * l);
}

double GraphSLAMer::CalcErrorVal(cv::Mat residuals)
{
	double sum = 0.0;
	for (int i = 0; i < residuals.cols; i++)
	{
		sum += residuals.at<double>(0, i);
	}
	return sum;
}


cv::Mat GraphSLAMer::derivative(SIM3 & cameraPose, KeyFrame & keyframe, cv::Mat & image, double rmean, int bIndexX) //b is our guessed position, x a given pixel
{
	Mat y1(keyframe.quadTreeNodeCount, 1, CV_64FC1);
	Mat y2(keyframe.quadTreeNodeCount, 1, CV_64FC1);
	double ty1, ty2;
	int tries = 0;
	Mat nCamPose;
	Mat nParams = cameraPose.getParameters();
	double * paraIndex = &nParams.ptr<double>(0)[bIndexX];
	paraIndex[0] += alpha[bIndexX];
	cameraPose.setParameters(nParams);
	nCamPose = cameraPose.getlieMatrix();
	y1 = ComputeResiduals(nCamPose, keyframe, image, true, rmean);
	ty1 = ComputeResidualError(nCamPose, keyframe, image, true, rmean);
	paraIndex[0] -= (2 * alpha[bIndexX]);
	cameraPose.setParameters(nParams);
	nCamPose = cameraPose.getlieMatrix();
	y2 = ComputeResiduals(nCamPose, keyframe, image, true, rmean);
	ty2 = ComputeResidualError(nCamPose, keyframe, image, true, rmean);
	paraIndex[0] += alpha[bIndexX];
	cameraPose.setParameters(nParams);
	return (y1 - y2) / (alpha[bIndexX]);
}

cv::Mat computeHessian()
{
	return Mat();
}


cv::Mat GraphSLAMer::ComputeJacobian(SIM3 & cameraPose, KeyFrame keyframe, cv::Mat & image, double rmean, int numRes)//b is our guessed position, x is our pixel info, y is residual
{
	cv::Mat result(keyframe.quadTreeNodeCount, 0, CV_64FC1);
	bool badJacob = false;
	//for the sim(3) vars

	for (int i = 0; i < 6; i++)
	{
		//compute this section of the jacobian and store for jacobian compilation
		cv::Mat tmat = derivative(cameraPose, keyframe, image, rmean, i);
		if (cv::countNonZero(tmat) < 1)
		{
			alpha[i] *= 2;
		}
		result.push_back(tmat);
	}

	return result;
}


cv::Mat GraphSLAMer::ComputeResiduals(Mat & cameraPose, KeyFrame keyframe, cv::Mat & image, bool dcheck, double rmean)//b is our guessed position, x is our pixel info, y is residual
{
	cv::Mat keyTrans;
	keyTrans = keyframe.cameraTransformationAndScaleS.getlieMatrix();

	Mat residuals(1, keyframe.quadTreeNodeCount, CV_64FC1);
	double * resRowIndex = residuals.ptr<double>(0);

	double * cPosePoint0 = cameraPose.ptr<double>(0);
	double * cPosePoint1 = cameraPose.ptr<double>(1);
	double * cPosePoint2 = cameraPose.ptr<double>(2);
	double cameraScale = cameraPose.at<double>(3, 3);

	double * kfPosePoint0 = keyTrans.ptr<double>(0);
	double * kfPosePoint1 = keyTrans.ptr<double>(1);
	double * kfPosePoint2 = keyTrans.ptr<double>(2);
	double kfScale = keyTrans.at<double>(3, 3);

	Mat * processedImage = &keyframe.scaledImageI;


	//for our image
	for (int x = 0; x < keyframe.quadTreeNodeCount; x++)
	{
		QuadTreeNode * leaf = &keyframe.quadTreeLeaves[x];
		//get pixel location with respect to our new frame
		double keyDepthAtU = 1.0 / leaf->meanDepth;
		if (keyDepthAtU == 0 || isinf(keyDepthAtU))
		{
			resRowIndex[x] = 0;
			continue;
		}

		uchar *kfImage = (*processedImage).ptr<uchar>(leaf->position.x);
		//project into world space
		cv::Point * pref = &(leaf->position);
		double xv, yv, zv;
		int xIndex, yIndex;
		projectCameraPointToWorldPointP(kfPosePoint0, kfPosePoint1, kfPosePoint2, kfScale, pref->x, pref->y, keyDepthAtU, xv, yv, zv);
		projectWorldPointToCameraPointU(cPosePoint0, cPosePoint1, cPosePoint2, cameraScale, xv, yv, zv, xIndex, yIndex);
		//do a bounds check, continue if we are out of range
		if (xIndex < 0 || yIndex < 0 || xIndex >= image.rows || yIndex >= image.cols)
		{
			resRowIndex[x] = 0;
			continue;
		}

		//calc photometric residue
		double r = (kfImage[leaf->position.y] - image.ptr<uchar>(xIndex)[yIndex]) / 255.0;
		resRowIndex[x] = HuberNorm(leaf->weightValue * (r * r) * (1.0 / leaf->depthDeviation), 3); //I guess 3 is just a maagggiiicc number (2 sources doing visual odometry have used it so who am I to question it?)
	}
	return residuals;
}


double GraphSLAMer::ComputeResidualError(Mat & cameraPose, KeyFrame keyframe, cv::Mat & image, bool dcheck, double rmean)//b is our guessed position, x is our pixel info, y is residual
{
	cv::Mat keyTrans;
	keyTrans = keyframe.cameraTransformationAndScaleS.getlieMatrix();

	double residual = 0;
	int residualCount = 0;

	double * cPosePoint0 = cameraPose.ptr<double>(0);
	double * cPosePoint1 = cameraPose.ptr<double>(1);
	double * cPosePoint2 = cameraPose.ptr<double>(2);
	double cameraScale = cameraPose.at<double>(3, 3);

	double * kfPosePoint0 = keyTrans.ptr<double>(0);
	double * kfPosePoint1 = keyTrans.ptr<double>(1);
	double * kfPosePoint2 = keyTrans.ptr<double>(2);
	double kfScale = keyTrans.at<double>(3, 3);

	Mat * processedImage = &keyframe.scaledImageI;


	//for our image
	for (int x = 0; x < keyframe.quadTreeNodeCount; x++)
	{
		QuadTreeNode * leaf = &keyframe.quadTreeLeaves[x];
		//get pixel location with respect to our new frame
		double keyDepthAtU = 1.0 / leaf->meanDepth;
		if (keyDepthAtU == 0 || isinf(keyDepthAtU)) continue;

		uchar *kfImage = (*processedImage).ptr<uchar>(leaf->position.x);
		//project into world space
		cv::Point * pref = &(leaf->position);
		double xv, yv, zv;
		int xIndex, yIndex;
		projectCameraPointToWorldPointP(kfPosePoint0, kfPosePoint1, kfPosePoint2, kfScale, pref->x, pref->y, keyDepthAtU, xv, yv, zv);
		projectWorldPointToCameraPointU(cPosePoint0, cPosePoint1, cPosePoint2, cameraScale, xv, yv, zv, xIndex, yIndex);
		//do a bounds check, continue if we are out of range
		if (xIndex < 0 || yIndex < 0 || xIndex >= image.rows || yIndex >= image.cols) continue;

		//calc photometric residue
		double r = (kfImage[leaf->position.y] - image.ptr<uchar>(xIndex)[yIndex]) / 255.0;
		residual += r * r;
		residualCount++;
	}
	if (residualCount <= 10) return 1e+300;
	return residual;
}

//predicts new position based on dampened approximate velocity
void GraphSLAMer::forwardPredictPosition(cv::Mat &lastPos, cv::Mat &Velocity)
{
	double dt = 0.5;
	double a = 0.85, b = 0.005;

	double xk, vk, rk;
	double xm;

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



//puts the projected point from pi into camera space
void GraphSLAMer::projectWorldPointToCameraPointU(double * poseRow1, double * poseRow2, double * poseRow3, double poseScale, double inX, double inY, double inZ, int & returnX, int & returnY)
{
	//apply reverse rotation
	inX = (inX * poseRow1[0] + inX * poseRow1[1] + inX * poseRow1[2]) * poseScale;
	inY = (inY * poseRow2[0] + inY * poseRow2[1] + inY * poseRow2[2]) * poseScale;
	inZ = (inZ * poseRow3[0] + inZ * poseRow3[1] + inZ * poseRow3[2]) * poseScale;


	//apply reverse translation
	inX -= poseRow1[3] * poseScale;
	inY -= poseRow2[3] * poseScale;
	inZ -= poseRow3[3] * poseScale;

	//3dx * fx_d / depth + cx_d = x
	returnX = inX * fx_d / inZ + cx_d;
	returnY = inY * fy_d / inZ + cy_d;
}

//puts the projected point from pi into camera space
void GraphSLAMer::projectCameraPointToWorldPointP(double * poseRow1, double * poseRow2, double * poseRow3, double poseScale, int inX, int inY, double depth, double & returnX, double & returnY, double & returnZ)
{
	returnX = (inX - cx_d) * depth / fx_d;
	returnY = (inY - cy_d) * depth / fy_d;
	returnZ = depth;

	//apply rotation
	returnX = (returnX * poseRow1[0] + returnX * poseRow2[0] + returnX * poseRow3[0]) * poseScale;
	returnY = (returnY * poseRow1[1] + returnY * poseRow2[1] + returnY * poseRow3[1]) * poseScale;
	returnZ = (returnZ * poseRow1[2] + returnZ * poseRow2[2] + returnZ * poseRow3[2]) * poseScale;

	//apply translation
	returnX += poseRow1[3] * poseScale;
	returnY += poseRow2[3] * poseScale;
	returnZ += poseRow3[3] * poseScale;
}


double GraphSLAMer::HuberNorm(double x, double epsilon)
{
	if (abs(x) <= epsilon) return (x * x) / (2 * epsilon);
	return abs(x) - (epsilon / 2.0);
}


cv::Mat GraphSLAMer::applyVarianceWeights(cv::Mat & jacobianTranspose, KeyFrame kf)
{
	int residualSize = jacobianTranspose.cols;
	int rowSize = jacobianTranspose.rows;

	//compute the constant from our gausian to probability conversion
	const double prefix = 2 / (sqrt(3.14159265));
	const double lnConst = log(10);

	cv::Mat results(rowSize, residualSize, CV_64FC1);
	for (int r = 0; r < rowSize; r++)
	{
		double * rowPtr = jacobianTranspose.ptr<double>(r);
		double * resPtr = results.ptr<double>(r);
		for (int c = 0; c < residualSize; c++)
		{
			if (kf.quadTreeLeaves[c].needsWeightUpdate)
			{
				//this long and confusing calculation is a result of caluclating the derivative of a log probability
				//I worked it out on paper, it SHOULD work. This is calculating (d log P(ri))/(d ri) * 1 / ri which gives us a weight that can pull a 
				//residual between 0 and its original value depending on how certain we are its a match
				//the weired number and complextiy comes from subbing the simple (r-mean)/std into the error function and a log and then taking the derivative
				double stdSquared = kf.quadTreeLeaves[c].depthDeviation * kf.quadTreeLeaves[c].depthDeviation;
				double changedResidual = -resPtr[c] * 0.707106781186547524401;//M_SQRT1_2
				double mean = kf.quadTreeLeaves[c].meanDepth;
				double exponent = (changedResidual * changedResidual - 4 * changedResidual * mean - mean * mean) / stdSquared;
				//awweee ya I just remembered chain rule
				double g = prefix * exp(-exponent);
				double gprime = (4 * changedResidual * mean * g) / stdSquared;
				double fprime = gprime / (g * lnConst);
				kf.quadTreeLeaves[c].weightValue = (fprime / resPtr[c]);
				kf.quadTreeLeaves[c].needsWeightUpdate = false;
			}

			resPtr[c] = resPtr[c] * kf.quadTreeLeaves[c].weightValue;
			if (isinf(resPtr[c]) || isnan(resPtr[c]))
			{
				resPtr[c] = resPtr[c];
			}
		}
	}
	return results;
}

//computes the update
cv::Mat GraphSLAMer::TransformJacobian(cv::Mat & jacobian, KeyFrame kf, cv::Mat residuals)
{
	//double max = 0, min = 0;
	//cv::minMaxIdx(jacobian, &min, &max);
	//jacobian = (jacobian - min) / max;
	//findDependantVals(jacobian);
	//printMat(JT);
	cv::Mat ojtj = jacobian * jacobian.t();
	cv::Mat JTJ = applyVarianceWeights(jacobian, kf) *  jacobian.t(); // because our javobian is already the wrong way lol
	//printMat(JTJ);
	//invert using decomposition then lower matrix inverse
	//cv::Mat l = MatrixSqrt(JTJ);
	//std::cout << l << std::endl;
	//InvertLowerTriangluar(l);

	//we use the transpose because the hessian of of a vector jacobian.t * a vector jacobian = H.t
	cv::Mat JTJi = -JTJ.inv();//l*l.t(); // (JT * J)^-1

	std::cout << cv::determinant(JTJ) << std::endl;
	//printMat(JTJi);
	kf.posCovariance = JTJi;
	cv::Mat JTJiJT = JTJi * jacobian; // (JT * J)^-1 * JT
	return applyVarianceWeights(JTJiJT, kf) * residuals.t(); // (JT * J)^-1 * JT * r
}


GraphSLAMer::SIM3 GraphSLAMer::CalcGNPosOptimization(cv::Mat & image, KeyFrame keyframe)
{
	//set initial camera pose
	SIM3 cameraPose(keyframe.cameraTransformationAndScaleS.getRotationMat(), keyframe.cameraTransformationAndScaleS.getTranslation(), keyframe.cameraTransformationAndScaleS.getParameters());

	//run gauss-newton optimization
	Mat camPose = cameraPose.getlieMatrix();
	Mat oldCamParas;
	double rmean = 0;
	//ComputeMedianResidualAndCorrectedPhotometricResiduals(cameraPose, image, keyframe, rmean);
	double residualSum = 1;//ComputeResidualError(camPose, keyframe, image, false, rmean);
	//if (residualSum < 0.010) return keyframe.cameraTransformationAndScaleS;
	double oldResidual = 0;
	double lambda = 1;
	while (fabs(oldResidual - residualSum) / oldResidual > 0.01)//while we have not converged
	{
		oldResidual = residualSum;

		camPose = cameraPose.getlieMatrix();
		//calculate all residuals and the sum

		//calculate error with current residuals
		//cv::Mat errorVec = CalcErrorVec(residuals);
		cv::Mat residuals = ComputeResiduals(camPose, keyframe, image, false, rmean);
		double error = CalcErrorVal(residuals);
		residualSum = error;

		if (error > 1.0e+200)
		{
			std::cout << "hwat";
		}

		//update pose estimate
		cv::Mat jacobianRes = ComputeJacobian(cameraPose, keyframe, image, rmean, image.cols * image.rows);

		//calculate deltax from derivatives
		cv::Mat deltaX = TransformJacobian(jacobianRes, keyframe, residuals).t();
		//SolveSparsecv::Matrix(H + lambda, b);
		double lnerror = 0;

		cameraPose = SIM3(deltaX) * cameraPose;

		/*cv::Mat deltaX = (rmean - error) / jacobianMat;
		SIM3 oldCam(cameraPose.getRotationMat(), cameraPose.getTranslation(), cameraPose.getParameters());
		for (int i = 5; i >= 0; i--)
		{
			Mat selectedvalue = cv::Mat::zeros(1, 6, CV_64FC1);
			int fcount = 0;
			lambda = 0.01;
			selectedvalue.at<double>(0,i) = deltaX.at<double>(0, i);
			while (fcount < 10)
			{
				cameraPose.addParameters(selectedvalue * lambda);
				camPose = cameraPose.getlieMatrix();

				double nerror = ComputeResidualError(camPose, keyframe, image, false, rmean);

				fcount++;

				lnerror = nerror;

				if (error <= nerror)
				{
					cameraPose = oldCam;
					lambda /= 2;
				}
				else
				{
					error = nerror;
					oldCam = cameraPose;
				}
			}
		}*/
	}
	return cameraPose;
}

double getPercent(double x1, double x2)
{
	if (x1 > x2)
	{
		return (x1 - x2) / x1;
	}
	else
	{
		return (x2 - x1) / x2;
	}
}


void GraphSLAMer::ConstructQuadtreeForKeyframe(KeyFrame &kf)
{
	double thresholdSquared = 0.01;// 0.01;//10% post square
	cv::Mat image = kf.scaledImageI;
	int numPixles = image.rows * image.cols;
	int treeSize = numPixles;
	int tpixels = numPixles;
	for (int i = 0; i < quadTreeDepth; i++)
	{
		tpixels /= 4;
		treeSize += tpixels;
	}
	int index = treeSize - numPixles;
	std::vector<QuadTreeNode> nodes(treeSize);

	//first set up our image pyramid
	int prows = image.rows;
	int pcols = image.cols;
	cv::Mat lastImage = image;

	//push back original image
	cv::Mat pimage(prows, pcols, CV_8UC1);
	for (int x = 0; x < prows; x++)
	{
		for (int y = 0; y < pcols; y++)
		{
			pimage.at<uchar>(x, y) = lastImage.at<uchar>(x, y);
		}
	}
	lastImage = pimage.clone();
	kf.pyramid.push_back(pimage.clone());

	//create power pyramid
	for (int i = 0; i < quadTreeDepth; i++)
	{
		cv::Mat pimage(prows / 2, pcols / 2, CV_8UC1);
		for (int x = 0; x < prows; x += 2)
		{
			for (int y = 0; y < pcols; y += 2)
			{
				double avg = 0;
				for (int j = 0; j < 2; j++)
				{
					for (int k = 0; k < 2; k++)
					{
						avg += lastImage.at<uchar>(x + j, y + k);
					}
				}
				pimage.at<uchar>(x / 2, y / 2) = avg / 4.0;
			}
		}
		lastImage = pimage.clone();
		double val = lastImage.at<uchar>(0, 0);
		double val2 = pimage.at<uchar>(0, 0);
		kf.pyramid.push_back(pimage.clone());
		prows /= 2;
		pcols /= 2;
	}

	//place image into quadtree
	for (int x = 0; x < image.rows; x += 2)
	{
		for (int y = 0; y < image.cols; y += 2)
		{
			for (int i = 0; i < 2; i++)
			{
				for (int j = 0; j < 2; j++)
				{
					QuadTreeNode leaf;
					leaf.numChildren = 0;
					leaf.strikes = 0;
					leaf.avgIntensity = image.at<uchar>(x + i, y + j);

					//calculate and store gradient
					double x1, x2, y1, y2;
					//bounds check gradiant, use current value if tripped
					if (x + i - 1 > 0) x1 = image.at<uchar>(x + i - 1, y + j);
					else x1 = leaf.avgIntensity;

					if (x + i + 1 < image.rows) x2 = image.at<uchar>(x + i + 1, y + j);
					else x2 = leaf.avgIntensity;

					if (y + j - 1 > 0) y1 = image.at<uchar>(x + i, y + j - 1);
					else y1 = leaf.avgIntensity;

					if (y + j + 1 < image.cols) y2 = image.at<uchar>(x + i, y + j + 1);
					else y2 = leaf.avgIntensity;

					leaf.xGradient = x1 - x2;
					leaf.yGradient = y1 - y2;

					leaf.layer = 0;
					leaf.position = Point(x + i, y + j);
					leaf.width = 1;
					leaf.length = 1;

					double savg = (x1 + x2 + y1 + y2) / 4;

					double percent = getPercent(savg, leaf.avgIntensity);
					percent *= percent;
					if (percent > thresholdSquared)
					{
						leaf.fLeaf = true;
					}
					else leaf.fLeaf = false;
					leaf.valid = true;

					nodes[index] = leaf;
					index++;
				}
			}
		}
	}

	//construct higher levels of the quad tree
	int groupSizeX = image.rows;
	int groupSizeY = image.cols;
	index = treeSize - numPixles;
	std::vector<QuadTreeNode> finalNodes;
	for (int l = 0; l < quadTreeDepth; l++)
	{
		groupSizeX /= 2;
		groupSizeY /= 2;
		int indexOff = 0;
		int curGroupSize = groupSizeX * groupSizeY;
		for (int x = 0; x < groupSizeX; x += 2)
		{
			for (int y = 0; y < groupSizeY; y += 2)
			{
				for (int i = 0; i < 2; i++)
				{
					for (int j = 0; j < 2; j++)
					{
						QuadTreeNode branch;
						branch.numChildren = 4;
						double avgIntensity = 0.0;
						double xGradient = 0;
						double yGradient = 0;
						branch.fLeaf = false;
						bool fleafSkip = false; //check if we can skip rollup 

						//index is the start of the lower layer
						//xyOffset is the offset from our x and y values, every X must skip double group size of y, and y is also doubled
						//this is to account fort he lower level being twice the size of the current
						//the i and j offsets are to vary our insertion order, they ca be directly applied to x and y
						int xyOffset = ((x + i) * 4 * groupSizeY) + 4 * (y + j);

						//set children from the lower group
						for (int k = 0; k < 4; k++)
						{

							//our sub offset k must read the first 4 values in order, the branches are ordered in the proper for mat on insertion
							branch.children[k] = &nodes[index + xyOffset + k];
							branch.children[k]->parent = &branch;
							avgIntensity += branch.children[k]->avgIntensity;
							xGradient += branch.children[k]->xGradient;
							yGradient += branch.children[k]->yGradient;
							//if any of the children are final leaves, we cannot roll up any further
							if (branch.children[k]->fLeaf)
							{
								fleafSkip = true;
							}

						}
						branch.avgIntensity = avgIntensity / 4.0;
						branch.layer = l + 1;
						branch.strikes = 0;
						//this is an approximation and WRONG, in fact we should be using max found instead of avg
						//but this might be close enough, we can check later
						branch.xGradient = xGradient / 4;
						branch.yGradient = yGradient / 4;

						if (fleafSkip)
						{
							//since we are skipping, add all non-fleaf children to the final vector and set this node's fleaf value to true
							for (int k = 0; k < 4; k++)
							{
								if (branch.children[k]->fLeaf && branch.children[k]->valid)
								{
									branch.children[k];
									finalNodes.push_back(*branch.children[k]);
								}
								else if (!branch.children[k]->fLeaf)
								{
									branch.children[k]->fLeaf = true;
									branch.children[k]->valid = true;
									finalNodes.push_back(*branch.children[k]);
								}
							}
							branch.fLeaf = true;
							branch.valid = false;
						}
						else
						{
							//set branch position and structure info
							branch.position = branch.children[0]->position;
							branch.length = branch.children[0]->length * 2;
							branch.width = branch.children[0]->width * 2;

							//do the threshold check for trim otherwise
							//check if we need to trim branch
							bool trim = true;
							for (int i = 0; i < 4; i++)
							{
								double percent = getPercent(branch.children[i]->avgIntensity, branch.avgIntensity);
								percent *= percent;
								if (percent > thresholdSquared)
								{
									trim = false;
									break;
								}
							}

							//trim if nesseccary
							if (trim)
							{
								branch.numChildren = 0;
							}
							else
							{
								//check if final leaf, and add to final vector if true
								//a branch is a final leaf if it is the first branch from the end not to be trimmed
								//wait, to get here we must have not encountered any fleaf children (see above) so we don;t need a check lol
								branch.fLeaf = true;
								branch.valid = true;
							}

						}

						//store branch in proper group pattern (00,01,10,11)
						//2 in a row, skip y size, do 2
						nodes[(index - curGroupSize) + (x * groupSizeY) + (2 * y) + (i * 2) + j] = branch;
						indexOff++;
					}
				}

			}
		}
		index -= curGroupSize;
	}

	//default case for when we are looking at basicly a solid colour (or testing why things are breaking ;) )
	//changing this in order to  search the final top level and add any nodes that are not yet fleafed
	//if (finalNodes.size() == 0)
	{
		int l = quadTreeDepth - 1;
		for (int x = 0; x < groupSizeX * groupSizeY; x++)
		{
			if (!nodes[x].fLeaf)
			{
				nodes[x].valid = true;
				nodes[x].fLeaf = true;
				finalNodes.push_back(nodes[x]);
			}
		}
	}

	//now the finalNodes vector contains all our leaves that we are using in the keyframe
	kf.quadTreeNodeCount = finalNodes.size();
	kf.quadTreeLeaves = new QuadTreeNode[kf.quadTreeNodeCount];
	std::copy(finalNodes.begin(), finalNodes.end(), kf.quadTreeLeaves);
}

double maxDepth = 0;;
//calculates the depths by comparing the image, after plcement into a power of 2 pyramid, against the keyframe quadtree leaves
void GraphSLAMer::computeDepthsFromStereoPair(KeyFrame & kf, cv::Mat & image, cv::Mat & cameraParams, SIM3 cameraPos, bool initialize)
{
	int prows = image.rows;
	int pcols = image.cols;
	std::vector<cv::Mat> pyramid;
	cv::Mat lastImage = image;

	//push back original image
	cv::Mat pimage(prows, pcols, CV_8UC1);
	for (int x = 0; x < prows; x++)
	{
		for (int y = 0; y < pcols; y++)
		{
			pimage.at<uchar>(x, y) = lastImage.at<uchar>(x, y);
		}
	}
	lastImage = pimage.clone();
	pyramid.push_back(pimage.clone());

	//create power pyramid
	for (int i = 0; i < quadTreeDepth; i++)
	{
		cv::Mat pimage(prows / 2, pcols / 2, CV_8UC1);
		for (int x = 0; x < prows; x += 2)
		{
			for (int y = 0; y < pcols; y += 2)
			{
				double avg = 0;
				for (int j = 0; j < 2; j++)
				{
					for (int k = 0; k < 2; k++)
					{
						avg += lastImage.at<uchar>(x + j, y + k);
					}
				}
				pimage.at<uchar>(x / 2, y / 2) = avg / 4.0;
			}
		}
		lastImage = pimage.clone();
		double val = lastImage.at<uchar>(0, 0);
		double val2 = pimage.at<uchar>(0, 0);
		pyramid.push_back(pimage.clone());
		prows /= 2;
		pcols /= 2;
	}

	//first generate the fundamental matrix
	//get offset from keyframe to image
	cv::Mat newCamMat = cameraPos.getlieMatrix();
	cv::Mat translation = kf.cameraTransformationAndScaleS.getTranslation() - cameraPos.getTranslation();
	cv::Mat rotation = kf.cameraTransformationAndScaleS.getRotationMat() - cameraPos.getRotationMat();
	double* transPtr = translation.ptr<double>(0);


	double sum = 0;
	double direction = (transPtr[0] > 0) - (transPtr[0] < 0);

	if (!initialize)
	{
		sum = transPtr[0] * transPtr[0];
		sum += transPtr[1] * transPtr[1];
		sum += transPtr[2] * transPtr[2];
	}
	else sum = 0.001; //for initialization purposes

	double baseline = sqrt(sum);

	//see baseline disparity depth calculation
	//we may need to recitfy our image sections, but the paper says the difference is small enough not to matter
	double focalXTimesBase = fx_d * baseline;


	//extract s = promote translate from vector to mat in cross multiply format
	cv::Mat S = cv::Mat::zeros(3, 3, CV_64FC1);

	if (!initialize)
	{
		S.at<double>(0, 0) = 0;
		S.at<double>(1, 0) = transPtr[2];
		S.at<double>(2, 0) = -transPtr[1];

		S.at<double>(0, 1) = -transPtr[2];
		S.at<double>(1, 1) = 0;
		S.at<double>(2, 1) = transPtr[0];

		S.at<double>(0, 2) = transPtr[1];
		S.at<double>(1, 2) = -transPtr[0];
		S.at<double>(2, 2) = 0;
	}
	else
	{
		//S = cv::Mat::zeros(3, 3, CV_64FC1);
		S.at<double>(2, 1) = 0.01; //we can say it moved 1 x unit
		S.at<double>(1, 2) = -0.01; //we can say it moved 1 x unit
		//S.at<double>(0, 2) = 1; //we can say it moved 1 y unit
		//S.at<double>(2, 0) = -1; //we can say it moved 1 y unit
	}



	//no initial rotation
	cv::Mat R;

	if (initialize)
	{
		R = cv::Mat::eye(3, 3, CV_64FC1);
	}
	else
	{
		R = cv::Mat(3, 3, CV_64FC1);
		for (int x = 0; x < 3; x++)
			for (int y = 0; y < 3; y++)
				R.at<double>(x, y) = rotation.at<double>(x, y);
	}

	//calculate Mi inverse
	//3x3 * 3x4 = 3x4
	//cv::Mat imageParamsTimesPose = cameraParams * cameraPos.getExtrinsicMatrix();
	//cv::Mat imageParamsTimesPoseInv = imageParamsTimesPose.inv();

	//Construct F = Mk^(-T)EMi^(-1), E = RS, M = intrinsic camera params
	//3x3 * (3x3 * 3x3) * 3x3
	cv::Mat E = (S * R);
	//cv::Mat test = cameraParams * cameraParamsInv;
	cv::Mat F = cameraParams.t() * E * cameraParams;

	double totalInvDepth = 0.0;

	//for each fleaf in the keyframe we search for a match
	for (int i = 0; i < kf.quadTreeNodeCount; i++)
	{

		//extract the leaf
		QuadTreeNode leaf = kf.quadTreeLeaves[i];

		double pixelMod = imageScale * pow(2, leaf.layer);


		//skip if the node is invalid
		if (!leaf.valid) continue;

		//store the value we will be comparing against
		double kValue = leaf.avgIntensity;

		//extract the image pyramid layer we will be comparing against
		cv::Mat pimage = pyramid[leaf.layer];

		//get the run of pixels we need for our leaf
		double kfPixels[5];
		int kfRightShift = 5 - leaf.position.y / leaf.length;
		int kfLeftShift = pimage.cols - (leaf.position.y / leaf.length + 5);
		int shift = 0.0;
		if (kfLeftShift < 0) shift = kfLeftShift;
		else if (kfRightShift > 0) shift = kfRightShift;
		for (int p = 0; p < 5; p++)
		{
			kfPixels[p] = kf.pyramid[leaf.layer].ptr<uchar>(leaf.position.x / leaf.length)[leaf.position.y / leaf.length + p + shift];
		}


		//calculate line equation
		cv::Mat position(1, 3, CV_64FC1);
		position.at<double>(0, 0) = (leaf.position.y * imageScale);// +(320 / imageScale);
		position.at<double>(0, 1) = (leaf.position.x * imageScale);// +(240 / imageScale);
		position.at<double>(0, 2) = 1;
		cv::Mat lineParams = position * F;

		//store constant values for epipolar line
		double a = lineParams.at<double>(0, 0);
		double b = lineParams.at<double>(0, 1);
		double c = lineParams.at<double>(0, 2);

		//values for storing our max and pixel position
		int bestPosX;
		int bestPosY;
		//this stores our minimum variances for threshold checking
		double bestPhotoValue = 0;
		double bestPhotoError = 0;
		double bestGeoError = 0;
		//this stores the selcted best-fit pixel's error
		double selectGeoError = 0;
		double selectPhotoError = 0;
		double minSSD = std::numeric_limits<double>::infinity();

		//to store locations of points along the epipolar line
		int goodCount = 0;
		double * eLineVals = new double[pimage.cols];
		int * eLineLocsX = new int[pimage.cols];
		int * eLineLocsY = new int[pimage.cols];
		double * geoError = new double[pimage.cols];
		double * photoError = new double[pimage.cols];
		//for the entire epipolar line try and find our best match
		//changing this to search the proper region
		/*for (int x = 0; x < pimage.cols; x++)
		{
			int y = (((-x * pixelMod * a) - c) / b) / pixelMod;*/

			//set up line information to limit epipolar search to small baseline and also to run EXTREMELY fast
		double lineConst = -c / b;
		double yIncrement = -a / b;
		double infDepthPoint = (lineConst + (leaf.position.x * pixelMod) * yIncrement) / pixelMod;
		int minOffset = 0;
		int maxOffset = pimage.cols;
		int y = infDepthPoint;
		int x = leaf.position.x;
		double yOffset = lineConst;
		if (leaf.meanDepth != 0)
		{
			if (leaf.aprox3dPosition.cols == 0)
			{
				cv::Mat Pos3D(1, 3, CV_64FC1);
				cv::Mat tPose = kf.cameraTransformationAndScaleS.getlieMatrix();
				projectCameraPointToWorldPointP(tPose.ptr<double>(0), tPose.ptr<double>(1), tPose.ptr<double>(2), 1, leaf.position.x, leaf.position.y, leaf.meanDepth, Pos3D.ptr<double>(0)[0], Pos3D.ptr<double>(0)[1], Pos3D.ptr<double>(0)[2]);
				leaf.aprox3dPosition = Pos3D;
			}
			projectWorldPointToCameraPointU(newCamMat.ptr<double>(0), newCamMat.ptr<double>(1), newCamMat.ptr<double>(2), 1, leaf.aprox3dPosition.at<double>(0, 0), leaf.aprox3dPosition.at<double>(0, 1), leaf.aprox3dPosition.at<double>(0, 2), x, y);
			int halfSearchRange = 2 * leaf.depthDeviation * fx_d;
			minOffset = x - halfSearchRange;
			maxOffset = x + halfSearchRange;

			//save us some multiplications if we can
			yOffset = infDepthPoint - halfSearchRange * yIncrement;
		}

		//get normalized line vector for error calculation
		double vecSum = a + b;
		double normA = -a / vecSum;
		double normB = b / vecSum;

		//term for carrying forward the last image intensity so we can compute the epipolar gradiant
		double lastIntensity = pimage.ptr<double>(infDepthPoint)[minOffset];

		//set y increment to move based on the relative position of our images
		yIncrement *= direction / pixelMod;

		for (int stepX = minOffset; stepX < maxOffset; stepX++)
		{
			yOffset += yIncrement;
			y = yOffset;

			/*Matx31d np(x, fullY, 1);

			Mat hz = position.t() * F * Mat(np);

			if (hz.at<double>(0) > 0.0001)
			{
				hz = hz;
			}*/
			//YES THANK YOU HZ


			if (y < 1 || y >= pimage.rows - 1 || stepX < 1 || stepX >= pimage.cols)
			{
				break;
			}

			//calculate variance to see if even at our best position it's worth updating

			//calc geometric error
			double xGrad = (pimage.ptr<double>(y + 1)[stepX] - pimage.ptr<double>(y - 1)[stepX]) / 2;
			double yGrad = (pimage.ptr<double>(y)[stepX + 1] - pimage.ptr<double>(y)[stepX - 1]) / 2;
			geoError[goodCount] = kf.posVariance / pow((xGrad * normA + yGrad * normB), 2.0);

			//calc photo. disperity error
			double nextIntensity = pimage.ptr<uchar>(y + yIncrement)[stepX + 1];
			double epipolarGrad = (nextIntensity - lastIntensity) / 2;
			lastIntensity = pimage.ptr<double>(stepX)[y];
			photoError[goodCount] = (2 * leaf.intensityVariance) / epipolarGrad;


			//grab and store point on row
			eLineVals[goodCount] = lastIntensity;
			eLineLocsX[goodCount] = y;
			eLineLocsY[goodCount] = stepX;
			goodCount++;
		}

		int numValues = goodCount - 5;
		for (int x = 0; x < numValues; x++)
		{
			double curSSD = 0;
			double minGeoErr = 1e+50;
			double minPhotoError = 1e+50;
			for (int y = 0; y < 5; y++)
			{
				double diff = eLineVals[x + y] - kfPixels[y];
				curSSD += sqrt(diff * diff);
				if (geoError[x + y] < minGeoErr)
				{
					minGeoErr = geoError[x + y];
				}
				if (photoError[x + y] < minPhotoError)
				{
					minPhotoError = photoError[x + y];
				}
			}

			//update our min if we need too
			if (curSSD < minSSD)
			{
				bestPosX = eLineLocsX[x] - shift;
				bestPosY = eLineLocsY[x];
				bestGeoError = minGeoErr;
				bestPhotoValue = eLineVals[3 - shift] - kfPixels[3 - shift];
				bestPhotoError = minPhotoError;
				selectGeoError = geoError[x];
				selectPhotoError = photoError[x];
				minSSD = curSSD;
			}
		}

		if (isinf(minSSD))
		{
			minSSD = minSSD;
		}

		delete[] eLineVals;
		delete[] eLineLocsX;
		delete[] eLineLocsY;

		//kf.quadTreeLeaves[i].depth = minSSD;
		//continue;

		//calc pixel to inverse depth conversion variance
		//the exact text is "the length of the searched inverse depth interval" over "the length of the searched epipolar line segment"
		//the bottom is the pixel disparity in terms of 3d space
		//and the top portion is the z distance (depth of course) minus the 3d coordiante best pixel location
		//this might seem inverted but it's because we are calculating the inverse depth, remember
		//also, this isn't mentioned in either paper by Engel et al., but we can compare the magnitude of the x and y change of the slope of the epipolar line
		//to determine which dimensions to calculate disparity over, thus giving more opportunity for parallax to affect depth
		//...why not true triangular disparity? We'll work this over in our head. Maybe it's just overkill, or unnecessarily adds more error from the dimension with less parallax?
		//It's worth a try, but I don't see aannnyyyyone else doing it
		double alpha;
		double depth;
		if (a * a > b * b)
		{
			double worldPosY = ((bestPosX * pixelMod) - cy_d) / fy_d;
			double disparity = fy_d * (leaf.position.x * imageScale - bestPosX * pixelMod);
			//we multiply z by our pos y because the closer we are to the edge of the camera the further along the curve of the lens we are, 
			//which means the more the z interval affects our pixel until we are parallel right at the edge and it's 100% included
			double baseline = (transPtr[2] * worldPosY) - transPtr[1];
			depth = disparity / baseline; //1.0 / (focalXTimesBase / disparity);
			//since our increments are already in pixel space, we will do the whole calculation there
			//square the baseline to give us magnitude
			//this is the same trick as above to get the amount the components are affecting the search interval
			double yRotated = fy_d * (rotation.ptr<double>(1)[0] * leaf.position.x + rotation.ptr<double>(1)[1] * leaf.position.y + rotation.ptr<double>(1)[2]);
			double zRotated = fy_d * (rotation.ptr<double>(2)[0] * leaf.position.x + rotation.ptr<double>(2)[1] * leaf.position.y + rotation.ptr<double>(2)[2]);
			alpha = b * (yRotated * transPtr[2] - zRotated * transPtr[1]);

		}
		else
		{
			double worldPosX = ((bestPosY * pixelMod) - cx_d) / fx_d;
			double disparity = fx_d * (leaf.position.y * imageScale - bestPosY * pixelMod);
			double baseline = (transPtr[2] * worldPosX) - transPtr[1];
			depth = disparity / baseline;
			double xRotated = fx_d * (rotation.ptr<double>(0)[0] * leaf.position.x + rotation.ptr<double>(0)[1] * leaf.position.y + rotation.ptr<double>(0)[2]);
			double zRotated = fx_d * (rotation.ptr<double>(2)[0] * leaf.position.x + rotation.ptr<double>(2)[1] * leaf.position.y + rotation.ptr<double>(2)[2]);
			alpha = -a * (xRotated * transPtr[2] - zRotated * transPtr[0]);
		}

		double obsVar = alpha * alpha * (selectGeoError + selectPhotoError);

		//we now have our best ssd value and the most likley location
		//thus we can kalman update our depth map and variances,
		//or if the ssd value is too large put a strike against the current leaf
		//Finally, if a leaf has too many strikes we rule it invalid
		if (obsVar <= alpha * alpha * (bestGeoError + bestPhotoError)) //arbitrary threshold, currently
		{

			//calculate pixel diff
			//this is the vector created from the camera centers towards the pixels selected, subtracted from one another
			double xDiff = abs(leaf.position.y * imageScale - bestPosY * pixelMod);


			//see baseline disparity depth calculation
			//we may need to recitfy our image sections, but the paper says the difference is small enough not to matter
			double depth = 1 / ((focalXTimesBase) / xDiff);

			totalInvDepth += depth;

			if (depth > maxDepth) maxDepth = depth;

			if (depth == 0 || isnan(depth))
			{
				depth = depth;
			}

			kf.quadTreeLeaves[i].needsProjecting = true;
			kf.quadTreeLeaves[i].needsWeightUpdate = true;

			//if depth is uninitialized just set it
			if (kf.quadTreeLeaves[i].meanDepth == 0)
			{
					kf.quadTreeLeaves[i].depthDeviation = 0.9;//we can attenuate or strengthen sensor dependancy later
					kf.quadTreeLeaves[i].meanDepth = depth;
					kf.quadTreeLeaves[i].intensityVariance = 0.9;
					kf.quadTreeLeaves[i].intensityMean = bestPhotoValue;

			}
			else
			{
				//>hurf durf use a kalman filter for a single value
				//no
				//update depth based on variance, and update variance too
				//I'm an idtiot how dare I diss the great and mighty kalman filter 
				//(especially considering it's basicly the best and most strightfoward way of combining gaussian distributions like this I really am an idiot)
				if (!isnan(depth))
				{
					kf.quadTreeLeaves[i].updateCount++;

					//update depth info
					double curMean = kf.quadTreeLeaves[i].meanDepth;
					double curError = kf.quadTreeLeaves[i].depthDeviation;
					double newMean = (curError * depth + obsVar * curMean) / (obsVar + curError);
					double newVariance = (obsVar * curError) / (obsVar + curError);
					kf.quadTreeLeaves[i].depthDeviation = newVariance;
					if (kf.quadTreeLeaves[i].depthDeviation == 0 || isnan(kf.quadTreeLeaves[i].depthDeviation))
					{
						kf.quadTreeLeaves[i].depthDeviation = kf.quadTreeLeaves[i].depthDeviation;
					}
					kf.quadTreeLeaves[i].meanDepth = newMean;

					//update photometric stuff
					kf.quadTreeLeaves[i].intensityMean = (kf.quadTreeLeaves[i].intensityVariance * bestPhotoValue + bestPhotoError * kf.quadTreeLeaves[i].intensityMean) / (bestPhotoError + kf.quadTreeLeaves[i].intensityVariance);
					kf.quadTreeLeaves[i].intensityVariance = (bestPhotoError * kf.quadTreeLeaves[i].intensityVariance) / (bestPhotoError + kf.quadTreeLeaves[i].intensityVariance);
				}
			}
		}
		else
		{
			//mark this node with a strike
			kf.quadTreeLeaves[i].strikes++;
			if (kf.quadTreeLeaves[i].strikes > 10)//another arbitrary threshold...
			{
				kf.quadTreeLeaves[i].valid = false;
			}
		}
	}

	//normalize depths
	for (int i = 0; i < kf.quadTreeNodeCount; i++)
	{
		//kf.quadTreeLeaves[i].depth /= 
	}
}

void GraphSLAMer::projectDepthNodesToDepthMap(KeyFrame & kf)
{
	std::list<int> invalidChunks;
	std::list<int> validChunks;
	std::list<int> retryList;
	for (int i = 0; i < kf.quadTreeNodeCount; i++)
	{
		//paper uses interpolation for assignment, lets try skipping it for now, and we can use our own fast poly algo later
		QuadTreeNode * qtn = &kf.quadTreeLeaves[i];
		int x = qtn->position.x;
		int y = qtn->position.y;
		int xSize = qtn->position.x + qtn->length;
		int ySize = qtn->position.y + qtn->width;

		//only set if valid
		if (qtn->valid)
		{
			if (qtn->needsProjecting)
			{
				for (; x < xSize; x++)
				{
					for (y = qtn->position.y; y < ySize; y++)
					{
						kf.inverseDepthD.at<double>(x, y) = qtn->meanDepth;
					}
				}
				validChunks.push_back(i);
			}
		}
		else
		{
			invalidChunks.push_back(i);
		}
	}

	//keeping trying to interpolate nodes until all have a value for everything
	int lastSize = 0;
	while (invalidChunks.size() != lastSize)
	{
		lastSize = invalidChunks.size();
		auto it = std::begin(invalidChunks);
		while (it != std::end(invalidChunks))
		{
			QuadTreeNode * qtn = &kf.quadTreeLeaves[*it];

			//find surrounding nodes
			double avgDepth = 0.0;
			int invalidCount = 0;
			int invalidLimit = qtn->length * 2;
			int numValues = qtn->length * 4;

			//west side
			int tlength = qtn->position.y + qtn->length;
			cv::Point2f pt(qtn->position.x, qtn->position.y);
			if (pt.x == 0)
			{
				pt.x = 1;
				pt.y = tlength;
				numValues -= qtn->length;
			}
			else
			{
				pt.x -= 1;

				for (; pt.y < tlength; pt.y++)
				{
					double value = kf.inverseDepthD.at<double>(pt.x, pt.y);
					if (value == 0)
					{
						invalidCount++;
						numValues--;
						continue;
					}
					avgDepth += value;
				}
				pt.x++;
			}


			//south side
			tlength = qtn->position.x + qtn->length;
			if (pt.y == kf.inverseDepthD.cols)
			{
				pt.x = tlength;
				pt.y = kf.inverseDepthD.rows - 1;
				numValues -= qtn->length;
			}
			else
			{
				for (; pt.x < tlength; pt.x++)
				{
					double value = kf.inverseDepthD.at<double>(pt.x, pt.y);
					if (value == 0)
					{
						invalidCount++;
						numValues--;
						continue;
					}
					avgDepth += value;
				}

				if (invalidCount > invalidLimit) { it++;  continue; };
				pt.y--;
			}


			//east side
			tlength = qtn->position.y;
			if (pt.x == kf.inverseDepthD.rows)
			{
				pt.y = tlength;
				pt.x = qtn->position.x;
				numValues -= qtn->length;
			}
			else
			{
				for (; pt.y > tlength; pt.y--)
				{
					double value = kf.inverseDepthD.at<double>(pt.x, pt.y);
					if (value == 0)
					{
						invalidCount++;
						numValues--;
						continue;
					}
					avgDepth += value;
				}

				if (invalidCount > invalidLimit) { it++;  continue; }
				pt.x--;
			}

			//north side
			if (pt.y != 0)
			{
				tlength = qtn->position.x;
				for (; pt.x > tlength; pt.x--)
				{
					double value = kf.inverseDepthD.at<double>(pt.x, pt.y);
					if (value == 0)
					{
						invalidCount++;
						numValues--;
						continue;
					}
					avgDepth += value;
				}

				if (invalidCount > invalidLimit) { it++;  continue; };
			}
			else
			{
				numValues -= qtn->length;
			}


			//set to interpolated value(gradiant?)
			qtn->depth = avgDepth / numValues;

			//remove from invalid list and add to retry list and valid list
			qtn->valid = true;
			qtn->strikes = 0;
			retryList.push_back(*it);
			it = invalidChunks.erase(it);
		}
	}

	//cycle through the retry list and project all the depths back onto the depthmap 
	for (int i : retryList)
	{
		QuadTreeNode * qtn = &kf.quadTreeLeaves[i];
		int x = qtn->position.x;
		int y = qtn->position.y;
		int xSize = qtn->position.x + qtn->length;
		int ySize = qtn->position.y + qtn->width;
		//set up covariance matrix for this node
		for (int j = 0; j < kf.quadTreeNodeCount; j++)
		{
			QuadTreeNode * subNode = &kf.quadTreeLeaves[j];
		}
		for (; x < xSize; x++)
		{
			for (; y < ySize; y++)
			{
				kf.inverseDepthD.at<double>(x, y) = 1 / qtn->meanDepth;
			}
		}
	}


}

//call this after quad tree creation to map the old depths to the new map as best as possible
void GraphSLAMer::transplantDepthsToNewKeyFrame(KeyFrame & newKF, KeyFrame & oldKF)
{
	cv::Mat nKeyTrans = newKF.cameraTransformationAndScaleS.getlieMatrix();
	cv::Mat oKeyTrans = oldKF.cameraTransformationAndScaleS.getlieMatrix();

	double * nPosePoint0 = nKeyTrans.ptr<double>(0);
	double * nPosePoint1 = nKeyTrans.ptr<double>(1);
	double * nPosePoint2 = nKeyTrans.ptr<double>(2);

	double * oPosePoint0 = oKeyTrans.ptr<double>(0);
	double * oPosePoint1 = oKeyTrans.ptr<double>(1);
	double * oPosePoint2 = oKeyTrans.ptr<double>(2);

	//for storing position info
	double x3, y3, z3 = 0;
	int x2, y2 = 0;

	double zMove = oldKF.cameraTransformationAndScaleS.getTranslation().at<double>(0, 2) - newKF.cameraTransformationAndScaleS.getTranslation().at<double>(0, 2);

	//accumulate a sum of the final depths so we can normalize the mean to 1 and set the scale of our keyframe
	double sum = 0;

	for (int i = 0; i < oldKF.quadTreeNodeCount; i++)
	{

		QuadTreeNode * qtn = &oldKF.quadTreeLeaves[i];
		int x = qtn->position.x;
		int y = qtn->position.y;
		int xSize = qtn->position.x + qtn->length;
		int ySize = qtn->position.y + qtn->width;
		double depth = qtn->depth;

		//might be worth setting up an epipolar line and incrementing over it, but for now this
		if (qtn->valid)
		{
			for (; x < xSize; x++)
			{
				for (y = qtn->position.y; y < ySize; y++)
				{
					projectCameraPointToWorldPointP(oPosePoint0, oPosePoint1, oPosePoint2, 1, x, y, depth, x3, y3, z3);
					projectWorldPointToCameraPointU(nPosePoint0, nPosePoint1, nPosePoint2, 1, x3, y3, z3, x2, y2);

					//bounds check
					if (x2 < 0 || y2 < 0 || x2 >= newKF.inverseDepthD.rows || y2 >= newKF.inverseDepthD.cols) continue;

					//approximatley correct(?), Maybe we should grab the angular components of x and y as well
					double newDepth = depth - zMove;
					double newVariance = pow((newDepth/depth),4) * qtn->depthDeviation + qtn->intensityVariance;

					//if we have no current entry just assign
					if (newKF.inverseDepthD.at<double>(x2, y2) == 0)
					{
						newKF.inverseDepthD.at<double>(x2, y2) = newDepth;
						newKF.depthVarianceV.at<double>(x2, y2) = newVariance;
					}
					else if (newKF.inverseDepthD.at<double>(x2, y2) != newDepth) //if an entry is already here
					{
						double curDepth = newKF.inverseDepthD.at<double>(x2, y2);
						double curVar = newKF.depthVarianceV.at<double>(x2, y2);
						if (abs(depth - newKF.inverseDepthD.at<double>(x2, y2)) < 2 * newKF.depthVarianceV.at<double>(x2, y2))
						{
							//if the depth is within 2 standard deviations(95% of guassian range) then we assume it's the same point with some noise applied, treat it like a new observation
							newKF.inverseDepthD.at<double>(x2, y2) = (curVar * newDepth + newVariance * curDepth) / (newVariance + curVar);
							newKF.depthVarianceV.at<double>(x2, y2) = (newVariance * curVar) / (newVariance + curVar);
						}
						else
						{
							//otherwise we take the nearest point to the camera, assuming occlusion
							if (newDepth < curDepth)
							{
								newKF.inverseDepthD.at<double>(x2, y2) = newDepth;
								newKF.depthVarianceV.at<double>(x2, y2) = newVariance;
							}
						}
					}
					else if (newKF.depthVarianceV.at<double>(x2, y2) > newVariance) //snag the new variance if we can from an equally far but more well known point(this probably won't happen really but it't worth a test I think?)
					{
						newKF.depthVarianceV.at<double>(x2, y2) = newVariance;
					}
					
				}
			}
		}
	}


	//iterate over each leaf in the new kf and get its depth from the depth map
	for (int i = 0; i < newKF.quadTreeNodeCount; i++)
	{
		//for saving us from looping over the entire vector sets each time we want to check a depth
		double lastDepth = 0;
		double lastIndex = -1;

		//for storing storing sub-region changes in depth and variance
		std::vector<double> depths;
		std::vector<double> variances;
		std::vector<double> count;

		QuadTreeNode * qtn = &oldKF.quadTreeLeaves[i];
		int x = qtn->position.x;
		int y = qtn->position.y;
		int xSize = qtn->position.x + qtn->length;
		int ySize = qtn->position.y + qtn->width;
		double depth = qtn->depth;

		//might be worth setting up an epipolar line and incrementing over it, but for now this
		if (qtn->valid)
		{
			for (; x < xSize; x++)
			{
				for (y = qtn->position.y; y < ySize; y++)
				{
					
					if (newKF.inverseDepthD.at<double>(x, y) != lastDepth)
					{
						lastDepth = newKF.inverseDepthD.at<double>(x, y);
						int index = 0;
						for (; index < depths.size(); index++)
						{
							//if we find an existing match 
							if (depths[index] == depth)
							{
								count[index]++;
								lastIndex = index;
								break;
							}
						}

						if (index == depths.size())//we had no match
						{
							depths.push_back(depth);
							variances.push_back(newKF.depthVarianceV.at<double>(x, y));
							count.push_back(1);
						}
					}
					else
					{
						count[lastIndex]++;
					}
				}
			}

			//using the gathered sub-pixel data we need to get a single value for the node
			//the largest case where we will have subpixels is when there's a loss of data from the camera moving to far away to resolve details
			//in which case our values SHOULD be close together since its the same surface(probably) and we can fuse using the above kalman method
			//if there exists one outside the 95% interval and it is in the minority it's probably an outlier and we can ignore it
			//first grab our highest occuring index
			int highCount = 0;
			int highIndex = 0;
			for (int j = 0; j < count.size(); j++)
			{
				if (count[j] > highCount)
				{
					highCount = count[j];
					highIndex = j;
				}
			}

			double resDepth = depths[highIndex];
			double resVar = variances[highIndex];

			for (int j = 0; j < depths.size(); j++)
			{
				if (j == highIndex) continue;

				double newDepth = depths[j];
				double newVar = variances[j];
				if (abs(resDepth - newDepth) < 2 * resVar)
				{
					//if the depth is within 2 standard deviations(95% of guassian range) then we assume it's the same point with some noise applied, treat it like a new observation
					resDepth = (resVar * newDepth + newVar * resDepth) / (newVar + resVar);
					resVar = (newVar * resVar) / (newVar + resVar);
				}
			}

			//last but not least actually assign the depths
			qtn->depth = resDepth;
			qtn->depthDeviation = resVar;
			sum += resDepth * (qtn->width * qtn->length);
		}
	}

	//normailze depth mean to 1 and set the scale of our key frame
	double scale = sum / (newKF.scaledImageI.rows * newKF.scaledImageI.cols);
	//iterate over each leaf in the new kf and get its depth from the depth map
	for (int i = 0; i < newKF.quadTreeNodeCount; i++)
	{

		QuadTreeNode * qtn = &oldKF.quadTreeLeaves[i];
		qtn->meanDepth /= scale;
		qtn->depthDeviation /= scale;
	}
	newKF.cameraTransformationAndScaleS.setScale(scale);
}

//The main function for LS Graph SLAM. Takes input in the form of camera frames, and returns a matrix with the approximate position of the camera. 
//Also builds a map behind the scenes for which the point cloud can be accessed by the helper functions
//enhanced implementation of https://groups.csail.mit.edu/rrg/papers/greene_icra16.pdf
//actually not to fond of that paper so now it's based on many different generic SLAM and photometric reconstruction algos
//seriously that paper is not clear on a lot of things I had to take 4 university courses
//K: is a 3x3 real mat with the camera parameters
//pi: perspective projection function
double increment = 0;
GraphSLAMer::SIM3 GraphSLAMer::LS_Graph_SLAM(cv::Mat cameraFrame)
{
	bool makeNewKeyframe;

	Mat magFrame;
	cvtColor(cameraFrame, magFrame, CV_BGR2GRAY);


	//find most likley position of camera based on last keyframe
	SIM3 position = CalcGNPosOptimization(magFrame, lastKey);

	if (!(cv::countNonZero(position.getParameters() != lastKey.cameraTransformationAndScaleS.getParameters()) == 0))
	{
		//construct depth quadtrees based on the stereo pairs
		computeDepthsFromStereoPair(lastKey, cameraFrame, cameraParams, position);

		//convert the nodes into a pixel map
		projectDepthNodesToDepthMap(lastKey);
	}

	//run makenewkeyframe check against image quality
	//makeNewKeyframe = !(cv::countNonZero(position.getParameters() != lastKey.cameraTransformationAndScaleS.getParameters()) == 0);
	makeNewKeyframe = false;
	if (makeNewKeyframe)
	{

		KeyFrame newKey;

		newKey.cameraTransformationAndScaleS = position;

		//add image te new  keyframe
		newKey.scaledImageI = cameraFrame;


		//set variance to one
		newKey.inverseDepthD = cv::Mat::ones(cameraFrame.rows, cameraFrame.cols, CV_64FC1);

		//computes the power tree for the image, allowing for fast analysis 
		ConstructQuadtreeForKeyframe(newKey);

		//construct depth quadtrees based on the stereo pairs
		computeDepthsFromStereoPair(newKey, lastKey.scaledImageI, cameraParams, lastKey.cameraTransformationAndScaleS);

		newKey.depthVarianceV = cv::Mat::zeros(cameraFrame.rows, cameraFrame.cols, CV_64FC1);

		//convert the nodes into a pixel map AND create information amtrix
		projectDepthNodesToDepthMap(newKey);

		//loop closure check (TO DO)

		keyframes.V.push_back(newKey);
		keyframes.E.push_back(position.getlieMatrix());

		//add new keyframe and constraints to list

		lastKey = newKey;
	}

	std::cout << "Current estimated camera position:" << std::endl << position.getParameters() << std::endl << std::endl;

	return position;
}

//Sets up matrices and other things
void GraphSLAMer::Initialize_LS_Graph_SLAM(cv::Mat cameraFrame, cv::Mat cameraFrame2)
{
	srand(1111);

	//initialize our pixel scale relative to the incoming frame
	imageScale = (2 * cameraParams.at<double>(0, 2)) / cameraFrame.cols;

	//initialize lastKey
	KeyFrame newKey;

	//set invDepth
	newKey.inverseDepthD = cv::Mat::ones(cameraFrame.rows, cameraFrame.cols, CV_64FC1);

	//set position to 0,0,0

	//add image te new  keyframe

	cvtColor(cameraFrame, newKey.scaledImageI, CV_BGR2GRAY);
	newKey.origImage = cameraFrame;

	//computes the power tree for the image, allowing for fast analysis 
	ConstructQuadtreeForKeyframe(newKey);


	SIM3 position;
	computeDepthsFromStereoPair(newKey, cameraFrame2, cameraParams, position, true);


	projectDepthNodesToDepthMap(newKey);

	//initialize posegraph
	keyframes = PoseGraph();

	lastKey = newKey;

	keyframes.E.push_back(position.getlieMatrix());
	keyframes.V.push_back(lastKey);

	std::ofstream myfile;
	myfile.open("./depths.txt");
	//cycle through the depth maps, converting the depths into points using the camera position
	for (int kfi = 0; kfi < keyframes.V.size(); kfi++)
	{

		cv::Mat depths = 1.0 / keyframes.V[kfi].inverseDepthD;
		for (int px = 0; px < depths.rows; px++)
		{
			for (int py = 0; py < depths.cols; py++)
			{
				double depth = depths.at<double>(px, py);
				myfile << depth << ", ";
			}
			myfile << std::endl;
		}
	}
	myfile.close();

	//initlalize alpha values
	alpha[0] = 0.00001 / 3.141592;
	alpha[1] = 0.00001 / 3.141592;
	alpha[2] = 0.00001 / 3.141592;
	alpha[3] = 0.00001;
	alpha[4] = 0.00001;
	alpha[5] = 0.00001;
	alpha[6] = 0.00001;
}

int cloudOffset = 0;

//passes over keyframes and constraints and returns a list of points
void GraphSLAMer::get3dPointsAndColours(std::vector<cv::Point3d> & pcloud_est, std::vector<cv::Vec3b> & colours)
{

	pcloud_est.clear();
	colours.clear();
	std::ofstream myfile;
	//myfile.open("./depths.txt");
	//cycle through the depth maps, converting the depths into points using the camera position
	for (int kfi = 0; kfi < keyframes.V.size(); kfi++)
	{

		cv::Mat depths = 1.0 / keyframes.V[kfi].inverseDepthD;
		for (int px = 0; px < depths.rows; px++)
		{
			for (int py = 0; py < depths.cols; py++)
			{
				double depth = depths.at<double>(px, py);
				//myfile << depth << ", ";
				if (depth == 1)
				{
					continue;
				}
				Point3d tpoint(py, px, depth * 10); //= projectCameraPointToWorldPointP(cameraParams, keyframes.E[kfi], Point(px, py), depth);
				pcloud_est.push_back(tpoint);
				Vec3b cpoint = (keyframes.V[kfi].origImage.at<Vec3b>(px, py));
				colours.push_back(cpoint);
			}
			//myfile << std::endl;
		}
	}
	//myfile.close();

}

void GraphSLAMer::get3dColours(std::vector<cv::Vec3b> & pcloud_est)
{

	pcloud_est.clear();

	//cycle through the depth maps, converting the depths into points using the camera position
	for (int kfi = 0; kfi < keyframes.V.size(); kfi++)
	{
		cv::Mat depths = 1.0 / keyframes.V[kfi].inverseDepthD;
		for (int px = 0; px < depths.rows; px++)
		{
			for (int py = 0; py < depths.cols; py++)
			{
				Vec3b tpoint = (keyframes.V[kfi].scaledImageI.at<Vec3d>(px, py) * 255);
				pcloud_est.push_back(tpoint);
			}
		}
	}

}