#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
#define CERES_FOUND 1
#include <opencv2/sfm.hpp>

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

Point2f multiply(Mat H, Point2f point) {
  Point2f point2;
  Mat result;
  Mat point1 = Mat::ones(3, 1, CV_64F);
  point1.at<double>(0, 0) = point.x;
  point1.at<double>(1, 0) = point.y;
  result = H * point1;
  point2.x = result.at<double>(0, 0) / result.at<double>(2, 0);
  point2.y = result.at<double>(1, 0) / result.at<double>(2, 0);
  return point2;
}

/* combine the inputted images */
void matchFeatures(Mat im1, Mat im2, vector<Point2f> &points1, vector<Point2f> &points2) {
  int minHessian = 400;
  Mat dp1, dp2, dp3;
  Mat im1Gray, im2Gray, im3Gray;
  vector<KeyPoint> kp1, kp2, kp3;
  vector<Point2f> points1_a, points2_a, points2_b, points3_a;
  vector<DMatch> goodMatches_12, goodMatches_23;
  vector< vector<DMatch> > knnMatches_12, knnMatches_23;
  // Detect ORB features and compute descriptrs.
  Ptr<SIFT> detector = SIFT::create( minHessian );
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

  cvtColor(im1, im1Gray, COLOR_BGR2GRAY);
  cvtColor(im2, im2Gray, COLOR_BGR2GRAY);

  // Get descriptors
  detector->detectAndCompute(im1Gray, Mat(), kp1, dp1);
  detector->detectAndCompute(im2Gray, Mat(), kp2, dp2);
  // Match features
  matcher->knnMatch(dp1, dp2, knnMatches_12, 2);
  // Sort matches by score 1->2
  for (size_t i = 0; i < knnMatches_12.size(); i++) {
    if (knnMatches_12[i][0].distance < 0.75f * knnMatches_12[i][1].distance) {
      goodMatches_12.push_back(knnMatches_12[i][0]);
    }
  }
  // Extract location of good matches
  for (size_t i = 0; i < goodMatches_12.size(); i++) {
    points1.push_back(kp1[goodMatches_12[i].queryIdx].pt);
    points2.push_back(kp2[goodMatches_12[i].trainIdx].pt);
  }
  return;
}

/*
float getNorm(Mat point) {
  float norm;
  norm = sqrt(pow(point.at<double>(0,0), 2) + pow(point.at<double>(1,0), 2) + pow(point.at<double>(2,0), 2));
  return norm;
}

float getScale(Mat points3d) {
  float denom, scale;
  denom = 0;
  scale = 0;
  for (int i = 0; i < points3d[0].size(); i++) {
    for (int j = 0; j < points3d[0].size(); j++) {
      if (i != j) {
        scale += norm(points3d[0][i] - points3d[0][j]);
        denom += 1;
      }
    }
  }
}
*/

vector<Mat> getSfm(vector<Point2f> points1, vector<Point2f> points2) {
  vector<Mat> points;
  Mat points1Mat = (Mat_<double>(2,1));
  Mat points2Mat = (Mat_<double>(2,1));
  //
  for (int i=0; i < points1.size(); i++) {
      Point2f myPoint1 = points1.at(i);
      Point2f myPoint2 = points2.at(i);
      Mat matPoint1 = (Mat_<double>(2,1) << myPoint1.x, myPoint1.y);
      Mat matPoint2 = (Mat_<double>(2,1) << myPoint2.x, myPoint2.y);
      hconcat(points1Mat, matPoint1, points1Mat);
      hconcat(points2Mat, matPoint2, points2Mat);
  }
  //
  points.push_back(points1Mat);
  points.push_back(points2Mat);
  return points;
}

void getExtrinsics(Mat &extrinsics, Mat cameraMatrix, vector<Point2f> points1, vector<Point2f> points2) {
  Mat E, R, t, mask, projMat;
  // get the essential matrix
  E = findEssentialMat(points1, points2, cameraMatrix, RANSAC, 0.999, 1.0, mask);
  // get the relative rotation and translation
  recoverPose(E, points1, points2, cameraMatrix, R, t, mask);
  // build extrinsic matrix
  hconcat(R, t, extrinsics);
  // add the points to 2d vector
}

void estimateMotionFromImages(const char folder[], string saveLocation) {
  Mat im1, im2, im3, mask;
  float fx, fy, cx, cy;
  vector<Point2f> points1, points2;

  Mat T = Mat::eye(4, 4, CV_64F);
  Mat C = Mat::eye(4, 4, CV_64F);
  Mat h = Mat::zeros(1, 4, CV_64F);
  h.at<double>(0,3) = 1;


  Mat R = Mat::eye(3, 3, CV_64F);
  Mat t = Mat::zeros(3, 1, CV_64F);
  Mat extrinsics_1, extrinsics_2, points3d;

  // intrinsic parameters
  fx = 24;
  fy = 24;
  cx = 2736;
  cy = 1824;

  Mat cameraMatrix = (Mat1d(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

  hconcat(R, t, extrinsics_1);

  if (auto dir = opendir(folder)) {
    while (auto f = readdir(dir)) {
      if (!f->d_name || f->d_name[0] == '.')
          continue; // Skip everything that starts with a dot
      // read in the first image
      im2 = imread(string(folder) + f->d_name);
      // resize frame
      //resize(im3, im3, Size(), 0.5, 0.5);
      // keep getting frames until we have three
      if (!im1.empty()) {
        // get matching feature points between three images
        matchFeatures(im1, im2, points1, points2);
        cout << points1.size() << endl;
        // combine the data
        vector<Mat> points = getSfm(points1, points2);
        Mat Ps, out;
        sfm::reconstruct(points, Ps, points3d, cameraMatrix, true);
        cout << "432" << endl;
        cout << Ps << endl;
        vconcat(Ps, h, out);
        T = T * out;
        C = C * T;
        extrinsics_1 = extrinsics_2.clone();
      }
      im1 = im2.clone();
    }
    closedir(dir);
  }
}

int main(int argc, char **argv) {
  string saveLocation, imageFolder;
  imageFolder = "./Orthos/3cm_ortho/";
  saveLocation = "./results/motionData.txt";
  estimateMotionFromImages(imageFolder.c_str(), saveLocation);
  return 0;
}
