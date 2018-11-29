#pragma once
#include "stdafx.h"

class GraphSLAMer
{
public:
	//These are my calculated intrinsic camera parameters (Webcam: Logitech C920)
	//to properly use this program you'll need to find the intrinsic camera parameters for your own webcam/camera, assuming it's a different model
	//google search "opencv camera calibration" for an easy guide
	//(Todo: add autocv::Matic camera calibration)
	cv::Mat cameraParams = (cv::Mat_<double>(3, 3) << 5.7481157594243552e+02, 0.0, 320.0, 0.0, 5.7481157594243552e+02, 240.0, 0.0, 0.0, 1.0);


	cv::Mat lastPos;
	cv::Mat velocity;
	int quadTreeDepth = 4;



	cv::Mat cameraParamsInv = cameraParams.inv();

	//look up "lie groups" for more information
	//contains both the translation and rotation for a given object in 3d space
	//methods can export the lie matrix, its individual components, or it's applied 3x3 extrinsic matrix

	class SE3
	{
	private:
		cv::Mat rotation;
		cv::Mat translation;
		cv::Mat lieMat;
	public:

		SE3()
		{
			//create proper size matrices at origin
			translation = cv::Mat::zeros(1, 3, CV_64FC1);
			rotation = cv::Mat::eye(3, 3, CV_64FC1);

			lieMat = cv::Mat::zeros(4, 4, CV_64FC1);
			//set rotation
			for (int x = 0; x < 3; x++)
				for (int y = 0; y < 3; y++)
					lieMat.at<double>(x, y) = rotation.at<double>(x, y);
			//set translation
			for (int i = 0; i < 3; i++)
				lieMat.at<double>(3, i) = translation.at<double>(0, i);

			lieMat.at<double>(3, 3) = 1;
		}

		SE3(cv::Mat _rotation, cv::Mat _translation)
		{
			rotation = _rotation;
			translation = _translation;

			lieMat = cv::Mat::zeros(4, 4, CV_64FC1);
			//set rotation
			for (int x = 0; x < 3; x++)
				for (int y = 0; y < 3; y++)
					lieMat.at<double>(x, y) = rotation.at<double>(x, y);
			//set translation
			for (int i = 0; i < 3; i++)
				lieMat.at<double>(3, i) = translation.at<double>(0, i);

			lieMat.at<double>(3, 3) = 1;
		}

		void addRotation(cv::Mat rotationAdd)
		{
			for (int x = 0; x < 3; x++)
				for (int y = 0; y < 3; y++)
				{
					lieMat.at<double>(x, y) += rotationAdd.at<double>(x, y);
					rotation.at<double>(x, y) += rotationAdd.at<double>(x, y);
				}
		}

		void addTranslation(cv::Mat translationAdd)
		{
			for (int i = 0; i < 3; i++)
			{
				lieMat.at<double>(3, i) = translationAdd.at<double>(0, i);
			}
		}

		void addLie(cv::Mat lieAdd)
		{
			for (int x = 0; x < 3; x++)
				for (int y = 0; y < 3; y++)
				{
					lieMat.at<double>(x, y) += lieAdd.at<double>(x, y);
					rotation.at<double>(x, y) += lieAdd.at<double>(x, y);
				}
		}

		cv::Mat getRotation()
		{
			return rotation;
		}

		cv::Mat getTranslation()
		{
			return translation;
		}

		//constructs and returns the lie matrix
		cv::Mat getlieMatrix()
		{
			//std::cout << lieMat;
			return lieMat;
		}

		//calculatres the 3x4 extrinsic matrix
		cv::Mat getExtrinsicMatrix()
		{
			cv::Mat eMat = cv::Mat::zeros(3, 4, CV_64FC1);
			
			for (int x = 0; x < 3; x++)
				for (int y = 0; y < 4; y++)
				{
					eMat.at<double>(x, y) = lieMat.at<double>(x, y);
				}

			return eMat;
		}
	};

	class QuadTreeNode
	{
	public:
		double avgIntensity;
		double xGradient;
		double yGradient;
		double depth;
		double depthVariance;
		int numChildren;
		cv::Point2f position;
		int length;
		int width;
		bool fLeaf;
		bool valid;
		int strikes;
		int layer;
		QuadTreeNode *parent;
		QuadTreeNode *children[4];
	};

	class KeyFrame
	{
	public:
		cv::Mat inverseDepthD;
		cv::Mat scaledImageI;//mean 1
		cv::Mat inverseDepthVarianceV;
		SE3 cameraTransformationAndScaleS; //taken and scaled with repsect to the world frame W aka the first frame. This is an element of Sim(3)
		std::vector<QuadTreeNode> quadTreeLeaves; //contains the significant quad tree nodes
		cv::Mat paramsTimesPose;
		cv::Mat paramsTimesPoseInv;
	};

	//for holding processed pixels that appear in both the keyframe and the current frame
	class pPixel
	{
	public:
		pPixel()
		{
			for (int i = 0; i < 16; i++)
			{
				derivatives[i] = 0.0;
			}
		}
		cv::Point2f imagePixel;
		cv::Point2f keyframePixel;
		cv::Point3d worldPoint;
		double depth;
		double residualSum;
		double keyframeIntensity;
		double imageIntensity;
		//16 is for the number of numbers in a sim(3) var
		double derivatives[16];
	};

	class PoseGraph
	{
	public:
		std::vector<KeyFrame> V;
		std::vector<cv::Mat> E; //sim(3) constraints between keyframes
	};

	class Node
	{
	public:
		std::vector<Node> subNodes;
		cv::Point3f location;
	};


	PoseGraph keyframes;
	KeyFrame lastKey;

	double findY(double kfPixelSum, double kfPixelVariance, cv::Point2i projectedPoint, cv::Mat &image, double rmean);

	//turns the given point into a homogeneouse point
	cv::Mat makeHomo(cv::Mat x);

	cv::Mat deHomo(cv::Mat xbar);

	//simple cholesky decomposition
	cv::Mat MatrixSqrt(cv::Mat a);

	//code roughly from "my math lib"
	void InvertLowerTriangluar(cv::Mat &l);

	//solve for delta x in (H * dX = -b)
	cv::Mat SolveSparseMatrix(cv::Mat H, cv::Mat b);

	double CalcErrorVal(std::vector<pPixel> residuals);


	double derivative(cv::Mat & cameraPose, cv::Point3d worldPointP, double pixelSum, double kfPixelVariance, cv::Mat & image, double rmean, int bIndexX, int bIndexY);

	std::vector<pPixel> ComputeJacobian(cv::Mat & cameraParams, SE3 cameraPose, KeyFrame keyframe, cv::Mat & image, double rmean, int numRes);

	std::vector<pPixel> ComputeResiduals(cv::Mat & cameraParams, SE3 cameraPose, KeyFrame keyframe, cv::Mat & image, double rmean);


	//predicts new position based on dampened approximate velocity
	void forwardPredictPosition(cv::Mat &lastPos, cv::Mat &Velocity);


	//turns the world point into a pixel value
	cv::Point projectWorldPointToCameraPointU(cv::Mat & cameraParamsInv, cv::Mat & cameraPoseT, cv::Point3d wPointP);

	//turns a pixel value into a world point
	cv::Point3d projectCameraPointToWorldPointP(cv::Mat & cameraParamsK, cv::Mat & cameraPoseT, cv::Point cPointU, double depth);


	double HuberNorm(double x, double epsilon);

	cv::Mat CalcErrorVec(std::vector<pPixel> pixels);

	//pixel U is in fact an index
	double calcPhotometricResidual(double kfPixelSum, cv::Point2i projectedPoint, cv::Mat & imageT, double globalResidue);

	void ComputeMedianResidualAndCorrectedPhotometricResiduals(cv::Mat & cameraParams, SE3 cameraPose, cv::Mat & image, KeyFrame kf, std::vector<pPixel> & results, double & median);


	//computes the update
	cv::Mat TransformJacobian(cv::Mat & jacobian, cv::Mat & residuals);

	SE3 CalcGNPosOptimization(cv::Mat & image, KeyFrame keyframe);

	void ComputeQuadtreeForKeyframe(KeyFrame &kf);

	//calculates the depths by comparing the image, after plcement into a power of 2 pyramid, against the keyframe quadtree leaves
	void computeDepthsFromStereoPair(KeyFrame kf, cv::Mat & image, cv::Mat & cameraParams, SE3 cameraPos);

	void projectDepthNodesToDepthMap(KeyFrame kf);


	//The main function for LS Graph SLAM. Takes input in the form of camera frames, and returns a matrix with the approximate position of the camera. 
	//Also builds a map behind the scenes for which the point cloud can be accessed by the helper functions
	//enhanced implementation of https://groups.csail.mit.edu/rrg/papers/greene_icra16.pdf
	//K: is a 3x3 real mat with the camera parameters
	//pi: perspective projection function
	SE3 LS_Graph_SLAM(cv::Mat cameraFrame);

	//Sets up matrices and other things
	void Initialize_LS_Graph_SLAM(cv::Mat cameraFrame);

	//passes over keyframes and constraints and returns a list of points
	std::vector<cv::Point3d> get3dPoints();
};