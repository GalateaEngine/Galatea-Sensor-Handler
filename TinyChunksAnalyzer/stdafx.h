// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once
#include "targetver.h"
#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\viz.hpp>
#include <opencv2\features2d.hpp>
#include <map>
#include <limits>
#include <list>
#include <iomanip>
#include <thread>
#include <chrono>
#include <cmath>
#include <math.h>
#include "GraphSLAMer.h"

namespace std {
	inline bool operator<(const cv::Point2f a, const cv::Point2f b)
	{
		if (a.x != b.x) return a.x < b.x;
		else return a.y < b.y;
	}

	inline bool operator==(const cv::Point2f a, const cv::Point2f b)
	{
		return (a.x == b.x) && (a.y == b.y);
	}
}

using namespace cv;

// TODO: reference additional headers your program requires here
