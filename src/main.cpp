/*************************************************************************
	> File Name: main.cpp
	> Author: yincanben
	> Mail: yincanben@163.com
	> Created Time: 2015年01月07日 星期三 16时33分00秒
 ************************************************************************/


#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/openni_grabber.h>
#include "tracker.h"
typedef pcl::PointXYZRGBA point_type;
typedef pcl::PointCloud<point_type> cloud_type;


void convert_to_img(const cloud_type& cloud, cv::Mat& gray_img){
    gray_img.create(cloud.height, cloud.width, CV_8UC1);
    for(int row = 0; row < cloud.height;row++){
        for(int col = 0; col < cloud.width;col++){
            const point_type& pt = cloud.at(col, row);
            gray_img.at<unsigned char>(row, col) = 0.3*pt.r + 0.6*pt.g + 0.1*pt.b;
        }//for col
    }//for row
}


void cloud_callback(cloud_type::ConstPtr cloud,Tracker* tracker){
    cv::Mat gray_img; //Create a grayscale image for feature extraction
    convert_to_img(*cloud, gray_img); //Extract 2D information
    tracker->new_image(gray_img);
    cv::imshow("Current View", gray_img);
    //cv::waitKey(1) ;
    if(cv::waitKey(200) > 0)
        exit(0);
}


int main (int argc, const char** argv){
    Tracker tracker;
    boost::function<void (const cloud_type::ConstPtr&)> f = boost::bind(cloud_callback, _1, &tracker);
    pcl::OpenNIGrabber interface;
    interface.registerCallback (f);
    interface.start();
    while(true)
        sleep(1);
    interface.stop();
    return 0;
}
