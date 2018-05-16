// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once
#include "targetver.h"

#include <stdio.h>
#include <tchar.h>
#include <opencv2\opencv.hpp>
#include <opencv2\xfeatures2d.hpp>
#include <map>
#include <limits>

namespace std {
	inline bool operator<(const cv::Point a, const cv::Point b)
	{
		if (a.x != b.x) return a.x < b.x;
		else return a.y < b.y;
	}

	inline bool operator==(const cv::Point a, const cv::Point b)
	{
		return (a.x == b.x) && (a.y == b.y);
	}
}

using namespace cv;
using namespace cv::xfeatures2d;
// TODO: reference additional headers your program requires here
