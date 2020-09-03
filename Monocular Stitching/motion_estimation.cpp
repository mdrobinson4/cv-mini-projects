#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2/sfm/triangulation.hpp>

#include <iostream>
#include <ctype.h>
#include <cstring>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

const float MAX_FEATURES = 500;
const float GOOD_MATCH_PERCENT = 0.15f;

/* combine the inputted images */
Mat getHomography(Mat im1, Mat im2) {
  int minHessian = 400;
  Mat dp1, dp2, H;
  Mat im1Gray, im2Gray;
  vector<KeyPoint> kp1, kp2;
  vector<DMatch> matches, goodMatches;
  vector< vector<DMatch> > knnMatches;
  vector<Point2f> points1, points2;
  // Detect ORB features and compute descriptrs.
  Ptr<SURF> detector = SURF::create( minHessian );
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
  cvtColor(im1, im1Gray, COLOR_BGR2GRAY);
  cvtColor(im2, im2Gray, COLOR_BGR2GRAY);
  // Get descriptors
  detector->detectAndCompute(im1Gray, Mat(), kp1, dp1);
  detector->detectAndCompute(im2Gray, Mat(), kp2, dp2);
  // Match features
  matcher->knnMatch(dp1, dp2, knnMatches, 2);
  // Sort matches by score
  for (size_t i = 0; i < knnMatches.size(); i++) {
    if (knnMatches[i][0].distance < 0.75f * knnMatches[i][1].distance) {
      goodMatches.push_back(knnMatches[i][0]);
    }
  }
  // Extract location of good matches
  for (size_t i = 0; i < goodMatches.size(); i++) {
    points1.push_back(kp1[goodMatches[i].queryIdx].pt);
    points2.push_back(kp2[goodMatches[i].trainIdx].pt);
  }
  // Find homography
  H = findHomography(points1, points2);
  return H;
}

Mat getCornerTransform(Mat im1, Mat im2, Mat H) {
  Mat H2 = Mat::eye(3, 3, CV_64F);
  vector<Point2f> corners(4);
  corners[0] = Point2f(0, 0);
  corners[1] = Point2f(0, im1.rows);
  corners[2] = Point2f(im1.cols, 0);
  corners[3] = Point2f(im1.cols, im1.rows);
  vector<Point2f> cornersTransform(4);
  perspectiveTransform(corners, cornersTransform, H);
  double offsetX = 0.0;
  double offsetY = 0.0;
  //Get max offset outside of the image
  for(size_t i = 0; i < 4; i++) {
    if(cornersTransform[i].x < offsetX) {
      offsetX = cornersTransform[i].x;
    }
    if(cornersTransform[i].y < offsetY) {
      offsetY = cornersTransform[i].y;
    }
  }
  offsetX = -offsetX;
  offsetY = -offsetY;
  H2.at<double>(0,2) = offsetX;
  H2.at<double>(1,2) = offsetY;
  return H2;
}

void estimateMotionFromImages1(const char folder[]) {
  Mat im1, im2, mask, H, H2, position, C;
  Mat T = Mat::eye(3, 3, CV_64F);
  Mat C0 = Mat::eye(3, 3, CV_64F);
  Mat ht = Mat::zeros(1, 4, CV_64F);
  vector<Point2f> points1, points2;

  if (auto dir = opendir(folder)) {
    while (auto f = readdir(dir)) {
      if (!f->d_name || f->d_name[0] == '.')
          continue;
      im2 = imread(string(folder) + f->d_name);
      // keep getting frames until we have three
      if (!im1.empty()) {
        H = getHomography(im1, im2);
        //H2 = getCornerTransform(im1, im2);
        T = T * H;
        C = C0 * T;
        position = Mat::zeros(3, 1, CV_64F);
        position.at<double>(2,0) = 1;
        position = C * position;
        position.at<double>(0,0) = position.at<double>(0,0)/position.at<double>(2,0);
        position.at<double>(1,0) = position.at<double>(1,0)/position.at<double>(2,0);
        position.at<double>(2,0) = position.at<double>(2,0)/position.at<double>(2,0);
        cout << position.at<double>(0,0) << ", " << position.at<double>(1,0) << ";" << endl;
      }
      else {
        C0.at<double>(0,2) = (im2.cols+1)/2;
        C0.at<double>(1,2) = (im2.rows+1)/2;
        cout << C0 << endl << endl;
      }
      im1 = im2.clone();
      im2.release();
    }
  }
  return;
}

int main(int argc, char **argv) {
  string saveLocation, imageFolder;
  imageFolder = "./Orthos/3cm_ortho(1)/";
  estimateMotionFromImages1(imageFolder.c_str());
  return 0;
}
