#pragma once
#include "stdafx.h"


	double GraphSLAMer::findY(cv::Point2f pixelU, cv::Point2f projectedPoint, GraphSLAMer::KeyFrame keyframe, cv::Mat image, double rmean)
	{
		double pixelIntensityNoise = 1.0;

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
	cv::Mat GraphSLAMer::makeHomo(cv::Mat x)
	{
		int size = x.rows;
		cv::Mat xbar(size + 1, 1, CV_64FC1);
		int i = 0;
		for (; i < size; i++)
		{
			xbar.at<double>(i, 0) = x.at<double>(i, 0);
		}
		xbar.at<double>(i + 1, 0) = 1.0;
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

	//simple cholesky decomposition
	cv::Mat GraphSLAMer::MatrixSqrt(cv::Mat a)
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

	double GraphSLAMer::CalcErrorVal(std::vector<pPixel> residuals)
	{
		double error = 0;
		for (int i = 0; i < residuals.size(); i++)
		{
			error += residuals[i].residualSum * residuals[i].residualSum;
		}
		return error;
	}

	
	double GraphSLAMer::derivative(cv::Mat cameraPose, cv::Point2f pixelU, cv::Point2f projectedPoint, KeyFrame keyframe, cv::Mat image, double rmean, int bIndexX, int bIndexY) //b is our guessed position, x a given pixel
	{
		double alpha = 1e-6;
		cv::Mat bCopy = cameraPose.clone();
		bCopy.at<double>(bIndexX, bIndexY) += alpha;
		double y1 = findY(pixelU, projectedPoint, keyframe, image, rmean);
		bCopy = cameraPose.clone();
		bCopy.at<double>(bIndexX, bIndexY) -= alpha;
		double y2 = findY(pixelU, projectedPoint, keyframe, image, rmean);
		return (y1 - y2) / (2 * alpha);
	}

	std::vector<GraphSLAMer::pPixel> GraphSLAMer::ComputeJacobian(cv::Mat cameraParams, SE3 cameraPose, KeyFrame keyframe, cv::Mat image, double rmean, int numRes)//b is our guessed position, x is our pixel info, y is residual
	{
		std::vector<pPixel> jacobianResults;

		//for our image
		for (int x = 0; x < image.cols; x++)
		{
			for (int y = 0; y < image.rows; y++)
			{
				//get pixel location with respect to our new frame
				cv::Point2f pixelU = cv::Point(x, y);
				double keyDepthAtU = keyframe.inverseDepthD.at<double>(pixelU);
				//project into world space
				cv::Mat p = keyframe.cameraTransformationAndScaleS.getExtrinsicMatrix().t() * piInv(makeHomo(cv::Mat(pixelU)), keyDepthAtU);
				//project into new image
				cv::Point2f projectedPoint = cv::Point(projectWorldPointToCameraPointU(cameraParams, cameraPose, p));
				//do a bounds check, skip if we are out of range
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
				for (int i = 0; i < cameraPose.getlieMatrix().rows; i++)
				{
					for (int j = 0; j < cameraPose.getlieMatrix().cols; j++)
					{
						//compute this section of the jacobian and store for jacobian compilation
						//jc.at<double>((x * image.rows) + y, (i * b.rows) + j) = derivative(cameraParams, b, pixelU, projectedPoint, keyframe, image, rmean, i, j);
						jacobianResults.back().derivatives[(i * cameraPose.getlieMatrix().cols) + j] = derivative(cameraPose.getlieMatrix(), pixelU, projectedPoint, keyframe, image, rmean, i, j);
					}
				}
			}
		}
		return jacobianResults;
	}

	std::vector<GraphSLAMer::pPixel> GraphSLAMer::ComputeResiduals(cv::Mat cameraParams, SE3 cameraPose, KeyFrame keyframe, cv::Mat image, double rmean)//b is our guessed position, x is our pixel info, y is residual
	{
		std::vector<pPixel> results;

		//for our image
		for (int x = 0; x < image.cols; x++)
		{
			for (int y = 0; y < image.rows; y++)
			{
				//get pixel location with respect to our new frame
				cv::Point2f pixelU = cv::Point(x, y);
				double keyDepthAtU = keyframe.inverseDepthD.at<double>(pixelU);
				//project into world space
				cv::Mat p = keyframe.cameraTransformationAndScaleS.getExtrinsicMatrix().t() * piInv(makeHomo(cv::Mat(pixelU)), keyDepthAtU);
				//project into new image
				cv::Point2f projectedPoint = cv::Point(projectWorldPointToCameraPointU(cameraParams, cameraPose, p));
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

	


	//projects 3d point input into 2d space
	cv::Mat GraphSLAMer::pi(cv::Mat input)
	{
		cv::Mat output(2, 1, CV_64FC1);
		//Not sure we need the dbz protection, test this later, perhaps we can pixle offset higher up the chain
		output.at<double>(0, 0) = input.at<double>(0, 0) / (input.at<double>(2, 0) + 0.00001);
		output.at<double>(1, 0) = input.at<double>(1, 0) / (input.at<double>(2, 0) + 0.00001);
		return output;
	}

	cv::Mat GraphSLAMer::piInv(cv::Mat input, double invDepth)
	{
		cv::Mat output(3, 1, CV_64FC1);
		//Not sure we need the dbz protection, test this later, perhaps we can pixle offset higher up the chain
		output.at<double>(0, 0) = input.at<double>(0, 0) / invDepth;
		output.at<double>(1, 0) = input.at<double>(1, 0) / invDepth;
		output.at<double>(2, 0) = input.at<double>(2, 0) / invDepth;
		return output;
	}

	//puts the projected point from pi into camera space
	cv::Mat GraphSLAMer::projectWorldPointToCameraPointU(cv::Mat cameraParamsK, SE3 cameraPoseT, cv::Mat wPointP)
	{
		cv::Mat pBar = makeHomo(wPointP);
		//3x3 * 4x4 * 3x1 ???????? How can you dehomogenize an SE3 element?
		//DUH 3x3 * dehomo(4x4 * 4x1) = 3x3 * 3x1 = 3x1
		cv::Mat notationalClarity = deHomo(cameraPoseT.getlieMatrix() * pBar);
		return pi(cameraParamsK * notationalClarity);
	}


	double GraphSLAMer::HuberNorm(double x, double epsilon)
	{
		if (abs(x) <= epsilon) return (x * x) / (2 * epsilon);
		return abs(x) - (epsilon / 2.0);
	}

	cv::Mat GraphSLAMer::CalcErrorVec(std::vector<pPixel> pixels)
	{
		int objects = pixels.size();
		cv::Mat errorVec(1, objects, CV_64FC1);
		//calc difference
		for (int x = 0; x < objects; x++)
		{
			errorVec.at<double>(0, x) = pixels[x].imageIntensity - pixels[x].keyframeIntensity;
		}
		return errorVec;
	}

	//pixel U is in fact an index
	double GraphSLAMer::calcPhotometricResidual(cv::Point2f pixelU, cv::Point2f projectedPoint, KeyFrame keyframe, cv::Mat imageT, double globalResidue)
	{
		double r;//single pixel
		r = keyframe.scaledImageI.at<uchar>(pixelU) - imageT.at<uchar>(projectedPoint) - globalResidue;
		return r;
	}

	void GraphSLAMer::ComputeMedianResidualAndCorrectedPhotometricResiduals(cv::Mat cameraParams, SE3 cameraPose, cv::Mat image, KeyFrame kf, std::vector<pPixel> & results, double & median)
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
				cv::Point pixelU = cv::Point(i, j);
				double keyDepthAtU = kf.inverseDepthD.at<double>(pixelU);
				//project into world space
				cv::Mat p = kf.cameraTransformationAndScaleS.getExtrinsicMatrix().t() * piInv(makeHomo(cv::Mat(pixelU)), keyDepthAtU);
				//project into new image
				cv::Point2f projectedPoint = cv::Point(projectWorldPointToCameraPointU(cameraParams, cameraPose, p));
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
	cv::Mat GraphSLAMer::TransformJacobian(cv::Mat jacobian, cv::Mat residuals)
	{
		cv::Mat JT = jacobian.t(); // JT
		cv::Mat JTJ = JT * jacobian; // JT * J
		cv::Mat l = MatrixSqrt(JTJ);
		InvertLowerTriangluar(l);
		cv::Mat JTJi = l.t() * l; // (JT * J)^-1
		cv::Mat JTJiJT = JTJi * JT; // (JT * J)^-1 * JT
		return JTJiJT * residuals; // (JT * J)^-1 * JT * r
	}


	GraphSLAMer::SE3 GraphSLAMer::CalcGNPosOptimization(cv::Mat image, KeyFrame keyframe)
	{
		//set initial camera pose
		SE3 cameraPose = keyframe.cameraTransformationAndScaleS;

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
			ComputeMedianResidualAndCorrectedPhotometricResiduals(cameraParams, cameraPose, image, keyframe, residuals, rmean);
			//calculate error with current residuals
			//cv::Mat errorVec = CalcErrorVec(residuals);
			double error = CalcErrorVal(residuals);

			//update pose estimate
			std::vector<pPixel> jacobianRes = ComputeJacobian(cameraParams, cameraPose, keyframe, image, rmean, image.cols * image.rows);

			//place jacobians and residuals into cv::Matrices
			cv::Mat jacobianMat(jacobianRes.size(), 16, CV_64FC1);
			cv::Mat residualsMat(jacobianRes.size(), 1, CV_64FC1);
			for (int i = 0; i < jacobianRes.size(); i++)
			{
				residualsMat.at<double>(i, 0) = residuals[i].residualSum;
				for (int j = 0; j < 16; j++)
				{
					residualsMat.at<double>(i, j) = residuals[i].derivatives[j];
				}
			}
			//calculate deltax from derivatives
			cv::Mat deltaX = TransformJacobian(jacobianMat, residualsMat);
			//SolveSparsecv::Matrix(H + lambda, b);


			//store position
			SE3 camOld = cameraPose;
			cv::Mat deltaMat(4, 4, CV_64FC1);
			//increment camera pose
			//for the sim(3) vars
			for (int i = 0; i < 4; i++)
			{
				for (int j = 0; j < 4; j++)
				{
					deltaMat.at<double>(i, j) += deltaX.at<double>((i * 4) + j) * lambda;
				}
			}
			cameraPose.addLie(deltaMat);
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

	

	
	void GraphSLAMer::ComputeQuadtreeForKeyframe(KeyFrame &kf)
	{
		double thresholdSquared = 0.01;//10% post square
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
						leaf.avgIntensity = cv::sum(image.at<Vec3b>(x + i, y + j))[0];

						//calculate and store gradient
						double x1, x2, y1, y2;
						//bounds check gradiant, use current value if tripped
						if (x + i - 1 > 0) x1 = cv::sum(image.at<Vec3b>(x + i - 1, y + j))[0];
						else x1 = leaf.avgIntensity;

						if (x + i + 1 < image.rows) x2 = cv::sum(image.at<Vec3b>(x + i + 1, y + j))[0];
						else x2 = leaf.avgIntensity;

						if (y + j - 1 > 0) y1 = cv::sum(image.at<Vec3b>(x + i, y + j - 1))[0];
						else y1 = leaf.avgIntensity;

						if (y + j + 1 < image.cols) y2 = cv::sum(image.at<Vec3b>(x + i, y + j + 1))[0];
						else y2 = leaf.avgIntensity;

						leaf.xGradient = x1 - x2;
						leaf.yGradient = y1 - y2;

						leaf.layer = 0;
						leaf.position = Point(x + i, y + j);
						leaf.width = 1;
						leaf.length = 1;
						leaf.fLeaf = false;
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
		for (int l = 0; l < quadTreeDepth - 1; l++)
		{
			groupSizeX /= 2;
			groupSizeY /= 2;
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
							bool fleafSkip = false; //check if we can skip rollup for 
							//set children from the lower group
							for (int k = 0; k < 4; k++)
							{
								//index is the start of the lower layer
								//xyOffset is the offset from our x and y values, every X must skip double group size of y, and y is also doubled
								//this is to account fort he lower level being twice the size of the current
								//the i and j offsets are to vary our insertion order, they ca be directly applied to x and y
								int xyOffset = 2 * ((x + i) * groupSizeY + (y + j));
								//our sub offset k must read the first 4 values in order, the branches are ordered in the proper forcv::Mat on insertion
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
							branch.avgIntensity /= 4;
							branch.layer = l + 1;
							branch.strikes = 0;
							//this is an approximation and WRONG, in fact we should be using max found instead of avg
							//but this might be close enough, we can check later
							branch.xGradient /= 4;
							branch.yGradient /= 4;

							if (fleafSkip)
							{
								//since we are skipping, add all non-fleaf children to the final vector and set this node's fleaf value to true
								for (int k = 0; k < 4; k++)
								{
									if (!branch.children[k]->fLeaf)
									{
										branch.children[k]->fLeaf = true;
										branch.children[k]->valid = true;
										finalNodes.push_back(*branch.children[k]);
									}
								}
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
									double percent = branch.avgIntensity / branch.children[i]->avgIntensity;
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
									finalNodes.push_back(branch);
								}

							}

							//store branch in proper group pattern (11,12,21,22)
							//2 in a row, skip y size, do 2
							nodes[(index - curGroupSize) + ((x + i) * groupSizeY) + (y + j)] = branch;
						}
					}

				}
			}
			index -= curGroupSize;
		}

		//now the finalNodes vector contains all our leaves that we are using in the keyframe
		kf.quadTreeLeaves = finalNodes;

	}

	//calculates the depths by comparing the image, after plcement into a power of 2 pyramid, against the keyframe quadtree leaves
	void GraphSLAMer::computeDepthsFromStereoPair(KeyFrame kf, cv::Mat image, cv::Mat cameraParams, SE3 cameraPos)
	{
		int prows = image.rows;
		int pcols = image.cols;
		std::vector<cv::Mat> pyramid;
		cv::Mat lastImage = image;

		//create power pyramid
		for (int i = 0; i < quadTreeDepth; i++)
		{
			cv::Mat pimage(prows / 2, pcols / 2, CV_64FC1);
			for (int x = 0; x < prows; x += 2)
			{
				for (int y = 0; y < pcols; y += 2)
				{
					double avg = 0;
					for (int j = 0; j < 2; j++)
					{
						for (int k = 0; k < 2; k++)
						{
							avg += lastImage.at<double>(x + j, k + k);
						}
					}
					pimage.at<double>(x, y) += avg / 4.0;
				}
			}
			lastImage = pimage;
			pyramid.push_back(pimage);
		}

		//first generate the fundamental matrix
		//get offset from keyframe to image
		cv::Mat transform = kf.cameraTransformationAndScaleS.getlieMatrix() - cameraPos.getlieMatrix();

		//extract s = promote translate from vector to mat in cross multiply format
		cv::Mat S(3, 3, CV_64FC1);

		S.at<double>(0, 0) = 0;
		S.at<double>(1, 0) = transform.at<double>(2, 3);
		S.at<double>(2, 0) = -transform.at<double>(1, 3);

		S.at<double>(0, 1) = -transform.at<double>(2, 3);
		S.at<double>(1, 1) = 0;
		S.at<double>(2, 1) = transform.at<double>(0, 3);

		S.at<double>(0, 2) = transform.at<double>(1, 3);
		S.at<double>(1, 2) = -transform.at<double>(0, 3);
		S.at<double>(2, 2) = 0;

		//extract R
		cv::Mat R(3, 3, CV_64FC1);
		for (int x = 0; x < 3; x++)
			for (int y = 0; y < 3; y++)
				R.at<double>(x, y) = transform.at<double>(x, y);

		//calculate Mi inverse
		//3x3 * 3x3 = 3x3
		cv::Mat imageParamsTimesPose = cameraParams * cameraPos.getExtrinsicMatrix();
		cv::Mat imageParamsTimesPoseInv = imageParamsTimesPose.inv();

		//Construct F = Mk^(-T)EMi^(-1), E = RS
		//3x3 * (3x3 * 3x3) * 3x3
		cv::Mat F = kf.paramsTimesPoseInv.t() * (R * S) * cameraParamsInv;

		//for each fleaf in the keyframe we search for a match
		for (int i = 0; i < kf.quadTreeLeaves.size(); i++)
		{

			//extract the leaf
			QuadTreeNode leaf = kf.quadTreeLeaves[i];

			//skip if the node is invalid
			if (leaf.valid) continue;

			//store the value we will be comparing against
			double kValue = leaf.avgIntensity;

			//extract the image pyramid layer we will be comparing against
			cv::Mat pimage = pyramid[leaf.layer];

			//find epipolar x coordinate so we know where to start
			//3x3 * 3x1
			cv::Mat e = deHomo(imageParamsTimesPose * kf.cameraTransformationAndScaleS.getTranslation().t());
			int x = floor(e.at<double>(0, 0));

			//calculate line equation
			cv::Mat position(1, 3, CV_64FC1);
			position.at<double>(0, 0) = leaf.position.x;
			position.at<double>(0, 1) = leaf.position.y;
			position.at<double>(0, 2) = 1;
			cv::Mat lineParams = F * position;

			//store contant values for epipolar line
			double xC = lineParams.at<double>(0, 0);
			double yC = lineParams.at<double>(0, 1);
			double C = lineParams.at<double>(0, 2);

			//values for storing our max and pixel position
			cv::Point2f bestPos;
			double minSSD = std::numeric_limits<double>::infinity();

			//5 sample window and value for updating ssd
			std::queue<double> window;
			double curSSD = 0;

			//for the entire epipolar line try and find our best match
			for (; x < pimage.rows; x++)
			{
				//calculate y
				int y = round(((-x * xC) - C) / yC);
				if (y < 0 || y > pimage.cols)
				{
					//aside from MAYBE odd rounding errors this should NEVER happen if we are calculating the epipolar line right
					throw;
				}

				//calc ssd
				double iValue = pimage.at<double>(x, y);
				double diff = iValue - kValue;
				double ssd = diff * diff;

				//make sure we start with the window initially full
				while (window.size() < 5)
				{
					window.push(diff);
				}

				//store the ssd
				window.push(ssd);
				//add the new value to the counter
				curSSD += ssd;
				//remove the old ssd value
				curSSD -= window.front();

				//update our min if we need too
				if (curSSD < minSSD)
				{
					bestPos = cv::Point(x, y);
					minSSD = curSSD;
				}
			}

			//we now have our best ssd value and the most likley location
			//thus we can kalman update our depth map and variances,
			//or if the ssd value is too large put a strike against the current leaf
			//Finally, if a leaf has too many strikes we rule it invalid
			if (minSSD < 50) //arbitrary threshold, currently
			{
				//triangulate depth
				//calc baseline = dist between cameras
				cv::Mat baseOffset = cameraPos.getTranslation() - kf.cameraTransformationAndScaleS.getTranslation();
				double sum = 0;
				for (int l = 0; l < 3; l++)
				{
					sum += baseOffset.at<double>(0, l) * baseOffset.at<double>(0, l);
				}
				double baseline = sqrt(sum);

				//calculate pixel diff
				//this is the vector created from the camera centers towards the pixels selected, subtracted from one another
				double pDiff = bestPos.x - leaf.position.x;

				double focalLength = cameraParams.at<double>(0, 0);

				//see baseline disparity depth calculation
				//we may need to recitfy our image sections, but the paper says the difference is small enough not to matter
				double depth = (baseline * focalLength) / pDiff;

				//if depth is uninitialized just set it
				if (kf.quadTreeLeaves[i].depth == 0 && kf.quadTreeLeaves[i].depthVariance == 0)
				{
					kf.quadTreeLeaves[i].depth = depth;
					kf.quadTreeLeaves[i].depthVariance = 0.5;//we can attenuate or strengthen sensor dependancy later
				}
				else
				{
					//>hurf durf use a kalman filter for a single value
					//no
					//update depth based on variance, and update variance too
					double error = kf.quadTreeLeaves[i].depth - depth;
					//square the error
					error *= error;
					//update based on current variance
					double updateMod = (1 / kf.quadTreeLeaves[i].depthVariance);
					double keepMod = 1.0 - updateMod;
					//update depth
					kf.quadTreeLeaves[i].depth = (kf.quadTreeLeaves[i].depth * keepMod) + (depth * updateMod);
					//update variance
					kf.quadTreeLeaves[i].depthVariance = (kf.quadTreeLeaves[i].depthVariance * keepMod) + (error * updateMod);
					//may need to change this to a more advanced filter later, but we'll try this for now
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
	}

	void GraphSLAMer::projectDepthNodesToDepthMap(KeyFrame kf)
	{
		std::list<QuadTreeNode*> invalidChunks;
		std::list<QuadTreeNode*> validChunks;
		std::list<QuadTreeNode*> retryList;
		for (int i = 0; i < kf.quadTreeLeaves.size(); i++)
		{
			//paper uses interpolation for assignment, lets try skipping it for now, and we can use our own fast poly algo later
			QuadTreeNode qtn = kf.quadTreeLeaves[i];
			int x = qtn.position.x;
			int y = qtn.position.y;
			int xSize = qtn.position.x + qtn.length;
			int ySize = qtn.position.y + qtn.width;
			//only set if valid
			if (qtn.valid)
			{
				for (; x < xSize; x++)
				{
					for (; y < ySize; y++)
					{
						kf.inverseDepthD.at<double>(x, y) = 1 / qtn.depth;
						kf.inverseDepthVarianceV.at<double>(x, y) = qtn.depthVariance;
					}
				}
				validChunks.push_back(&qtn);
			}
			else
			{
				invalidChunks.push_back(&qtn);
			}
		}

		//keeping trying to interpolate nodes until all have a value for everything
		while (invalidChunks.size() > 0)
		{
			auto it = std::begin(invalidChunks);
			while (it != std::end(invalidChunks))
			{
				QuadTreeNode *qtn = *it;

				//find surrounding nodes
				double avgDepth = 0.0;
				int invalidCount = 0;
				int invalidLimit = qtn->length * 2;
				int numValues = qtn->length * 4;

				//west side
				int tlength = qtn->position.y + qtn->length;
				cv::Point2f pt = qtn->position;
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
						double value = kf.inverseDepthD.at<double>(pt);
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
				if (pt.y + qtn->length == kf.inverseDepthD.rows)
				{
					pt.x = tlength;
					pt.y = kf.inverseDepthD.rows - 1;
					numValues -= qtn->length;
				}
				else
				{
					for (; pt.x < tlength; pt.x++)
					{
						double value = kf.inverseDepthD.at<double>(pt);
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
				if (pt.x + qtn->length == kf.inverseDepthD.rows)
				{
					pt.y = tlength;
					pt.x = qtn->position.x;
					numValues -= qtn->length;
				}
				else
				{
					for (; pt.y > tlength; pt.y--)
					{
						double value = kf.inverseDepthD.at<double>(pt);
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
						double value = kf.inverseDepthD.at<double>(pt);
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
				retryList.push_back(qtn);
				it = invalidChunks.erase(it);
			}
		}

		//cycle through the retry list and project all the depths back onto the depthmap 
		for (QuadTreeNode* const& qtn : retryList)
		{
			int x = qtn->position.x;
			int y = qtn->position.y;
			int xSize = qtn->position.x + qtn->length;
			int ySize = qtn->position.y + qtn->width;
			for (; x < xSize; x++)
			{
				for (; y < ySize; y++)
				{
					kf.inverseDepthD.at<double>(x, y) = 1 / qtn->depth;
					kf.inverseDepthVarianceV.at<double>(x, y) = qtn->depthVariance;
				}
			}
		}

	}

	
	//The main function for LS Graph SLAM. Takes input in the form of camera frames, and returns a matrix with the approximate position of the camera. 
	//Also builds a map behind the scenes for which the point cloud can be accessed by the helper functions
	//enhanced implementation of https://groups.csail.mit.edu/rrg/papers/greene_icra16.pdf
	//K: is a 3x3 real mat with the camera parameters
	//pi: perspective projection function
	GraphSLAMer::SE3 GraphSLAMer::LS_Graph_SLAM(cv::Mat cameraFrame)
	{
		bool makeNewKeyframe;

		//find most likley position of camera based on last keyframe
		SE3 position = CalcGNPosOptimization(cameraFrame, lastKey);

		//construct depth quadtrees based on the stereo pairs
		computeDepthsFromStereoPair(lastKey, cameraFrame, cameraParams, position);

		//convert the nodes into a pixel map
		projectDepthNodesToDepthMap(lastKey);

		//run makenewkeyframe check against image quality
		makeNewKeyframe = false;
		if (makeNewKeyframe)
		{
			KeyFrame newKey;

			//add image te new  keyframe
			newKey.scaledImageI = cameraFrame;

			//computes the power tree for the image, allowing for fast analysis 
			ComputeQuadtreeForKeyframe(newKey);

			//generate constraints between old keyframe and new one(Aka stores the cameras approximate new position)
			cv::Mat constraints = position.getlieMatrix();

			//loop closure check (TO DO)

			//add new keyframe and constraints to list
			keyframes.E.push_back(constraints);
			keyframes.V.push_back(newKey);
			lastKey = newKey;
		}

		return position;
	}

	//Sets up matrices and other things
	void GraphSLAMer::Initialize_LS_Graph_SLAM(cv::Mat cameraFrame)
	{
		//initialize lastKey
		KeyFrame newKey;

		//add image te new  keyframe
		newKey.scaledImageI = cameraFrame;

		//computes the power tree for the image, allowing for fast analysis 
		ComputeQuadtreeForKeyframe(newKey);

		//generate constraints between old keyframe and new one(Aka stores the cameras approximate new position)
		SE3 position;
		cv::Mat constraints = position.getlieMatrix();

		//add new keyframe and constraints to list
		keyframes.E.push_back(constraints);
		keyframes.V.push_back(newKey);
		lastKey = newKey;

		//initialize velocity

		//initialize posegraph
		keyframes = PoseGraph();

		//initialize camera params
		//done at the top, hardcoded
	}

	//passes over keyframes and constraints and returns a list of points
	cv::Mat GraphSLAMer::get3dPoints()
	{
		std::vector<cv::Vec3f> pcloud_est;
		//extract some important values from our camera matrix
		float cx_d = cameraParams.at<float>(0, 2);
		float cy_d = cameraParams.at<float>(1, 2);
		float fx_d = cameraParams.at<float>(0, 0);
		float fy_d = cameraParams.at<float>(1, 1);
		
		//cycle through the depth maps, converting the depths into points using the camera position
		for (int kfi = 0; kfi < keyframes.V.size(); kfi++)
		{
			cv::Mat depths = 1.0/keyframes.V[kfi].inverseDepthD;
			for (int px = 0; px < depths.rows; px++)
			{
				for (int py = 0; py < depths.cols; py++)
				{
					float x = (px - cx_d) * depths.at<float>(px, py) / fx_d;//(x_d - cx_d) * depth(x_d, y_d) / fx_d;
					float y = (py - cy_d) * depths.at<float>(px, py) / fy_d;
					float z = depths.at<float>(px, py);
					pcloud_est.push_back(cv::Vec3f(x,y,z));
				}
			}
		}

		return cv::Mat(pcloud_est);
	}