/*************************************************************************
	> File Name: tracker.cpp
	> Author: yincanben
	> Mail: yincanben@163.com
	> Created Time: 2015年01月07日 星期三 16时01分03秒
 ************************************************************************/
#include "tracker.h"

Tracker::Tracker(){
    featureDetector_ = cv::FeatureDetector::create( "SURF" );
    descriptorExtractor_ = cv::DescriptorExtractor::create( "SURF" );
    descriptorMatcher_ = cv::DescriptorMatcher::create( "BruteForce");
    if(featureDetector_.empty()|| descriptorExtractor_.empty()|| descriptorMatcher_.empty() )
        std::cerr << "Error creating detector, extractor or matcher.\n";
}


void Tracker::new_image(cv::Mat& gray_img){
    cv::initModule_nonfree();
    std::vector<cv::KeyPoint> keypoints;
    
    featureDetector_->detect(gray_img, keypoints);
    std::cout << "Found " << keypoints.size() << " keypoints\n";
    if(keypoints.size() == 0)
        return;

    //Compute descriptors for keypoints
    
    cv::Mat descriptors(1, keypoints.size(), CV_32FC3);
    descriptorExtractor_->compute(gray_img, keypoints, descriptors);
    cv::drawKeypoints(gray_img, keypoints, gray_img);
    if(!last_keypoints_.empty()){ //First Frame
        std::vector<cv::DMatch> matches = match_and_filter(descriptors);
        std::cout << "Matches:\n" << matches.size() << std::endl;
        if(matches.size() > 0){
            for(int i = 0; i < matches.size(); i++){
                cv::Scalar color(255);
                cv::line( gray_img, last_keypoints_[matches[i].trainIdx].pt, keypoints[matches[i].queryIdx].pt, color,  2 );
            }
        }
    }

    std::vector<cv::Mat> tmp;
    tmp.push_back(descriptors);
    descriptorMatcher_->clear();
    descriptorMatcher_->add(tmp);
    last_keypoints_.swap(keypoints);
    
}


std::vector<cv::DMatch> Tracker::match_and_filter(const cv::Mat& descriptors){
    std::vector<cv::DMatch> result;
    if(last_keypoints_.empty()) { //First frame
        return result;
    }
    //For each keypoint of the new image
    //get two matches in the database
    std::vector<std::vector<cv::DMatch> > pairs_of_matches;
    descriptorMatcher_->knnMatch(descriptors, pairs_of_matches, 2);
    for(unsigned int i=0; i< pairs_of_matches.size(); i++){
        float ratio = pairs_of_matches[i][0].distance /
        pairs_of_matches[i][1].distance;
        if(ratio < 0.5){
            result.push_back(pairs_of_matches[i][0]);
        }
    }
    return result;
}
