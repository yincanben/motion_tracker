/*************************************************************************
	> File Name: tracker.h
	> Author: yincanben
	> Mail: yincanben@163.com
	> Created Time: 2015年01月07日 星期三 15时58分42秒
 ************************************************************************/

#ifndef _TRACKER_H
#define _TRACKER_H
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <vector>
class Tracker{
    public:
        Tracker();
        std::vector<cv::DMatch> match_and_filter(const cv::Mat& descriptors);
        void new_image(cv::Mat& gray_img);
    private:
        cv::Ptr<cv::FeatureDetector> featureDetector_;
        cv::Ptr<cv::DescriptorExtractor> descriptorExtractor_;
        cv::Ptr<cv::DescriptorMatcher> descriptorMatcher_;
        std::vector<cv::KeyPoint> last_keypoints_;
};

#endif
