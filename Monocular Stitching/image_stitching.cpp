#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"

#include <iostream>
#include <cstring>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

const float MAX_FEATURES = 500;
const float GOOD_MATCH_PERCENT = 0.15f;

Point2f multiply(Mat H, Point2f point) {
  Point2f point2;
  Mat point1(3, 1, CV_64F), result;
  point1.at<double>(0, 0) = point.x;
  point1.at<double>(1, 0) = point.y;
  point1.at<double>(2, 0) = 1;
  result = H * point1;
  point2.x = result.at<double>(0, 0);
  point2.y = result.at<double>(1, 0);
  return point2;
}

void cropFrame(Mat &src) {
  int largest_area=0;
  int largest_contour_index=0;
  Rect bounding_rect;

  Mat thr(src.rows,src.cols,CV_8UC1);
  Mat dst(src.rows,src.cols,CV_8UC1,Scalar::all(0));
  cvtColor(src,thr,COLOR_BGR2GRAY); //Convert to gray
  threshold(thr, thr, 25, 255,THRESH_BINARY); //Threshold the gray
  vector<vector<cv::Point>> contours; // Vector for storing contour
  vector<Vec4i> hierarchy;
  findContours( thr, contours, hierarchy,RETR_LIST, CHAIN_APPROX_SIMPLE ); // Find the contours in the image
  for(int i = 0; i< contours.size(); i++ ) {
  	double a=contourArea( contours[i],false);  //  Find the area of contour
    if (a > largest_area) {
    	largest_area=a;
    	largest_contour_index=i;                //Store the index of largest contour
    	bounding_rect=boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
    }
  }
  Scalar color( 255,255,255);
  drawContours( dst, contours,largest_contour_index, color, FILLED, 8, hierarchy ); // Draw the largest contour using previously stored index.
  rectangle(src, bounding_rect,  Scalar(0,0,0),1, 8,0);
  src = Mat(src, bounding_rect);
}

void updateCameraCenter(vector<Point2f> &cameraCenters, Mat H2, Mat newFrame) {
  Point2f currentCenter, newFrameCenter;
  newFrameCenter = Point2f(newFrame.cols/2, newFrame.rows/2);
  currentCenter = multiply(H2, newFrameCenter);
  cameraCenters.push_back(currentCenter);
}

Mat stitchFrames(const Mat &img1, const Mat &img2, Mat &mask, const Mat H, Mat H2, double maxX, double maxY) {
  Size size_warp(maxX, maxY);
  Mat panorama(size_warp, CV_8UC3), half;
  double offsetX = H2.at<double>(0,2);
  double offsetY = H2.at<double>(1,2);

  warpPerspective(img2, panorama, H2*H, size_warp);
  //ROI for img1
  Rect img1_rect(offsetX, offsetY, img1.cols, img1.rows);
  //First iteration
  //Copy img1 in the panorama using the ROI
  half = cv::Mat(panorama, img1_rect);
  img1.copyTo(half);
  //Create the new mask matrix for the panorama
  //mask = cv::Mat::ones(img2.size(), CV_8U)*255;
  //warpPerspective(mask, mask, H2*H, size_warp);
  //rectangle(mask, img1_rect, cv::Scalar(255), -1);
  if (mask.empty()) {
    //Copy img1 in the panorama using the ROI
    half = cv::Mat(panorama, img1_rect);
    img1.copyTo(half);
  }
  return panorama;
}

void getCornerTransform(Mat panorama, Mat newFrame, Mat H, Mat &H2, double &maxX, double &maxY) {
  H2 = Mat::eye(3, 3, CV_64F);
  vector<Point2f> corners(4);
  corners[0] = Point2f(0, 0);
  corners[1] = Point2f(0, panorama.rows);
  corners[2] = Point2f(panorama.cols, 0);
  corners[3] = Point2f(panorama.cols, panorama.rows);
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
  //Get max width and height for the new size of the panorama
  maxX = max((double) newFrame.cols+offsetX, (double) max(cornersTransform[2].x, cornersTransform[3].x)+offsetX);
  maxY = max((double) newFrame.rows+offsetY, (double) max(cornersTransform[1].y, cornersTransform[3].y)+offsetY);
}

/* combine the inputted images */
void getHomography(Mat panorama, Mat newFrame, Mat &H, Mat &H2, double &maxX, double &maxY) {
  int minHessian = 400;
  Mat dp1, dp2;
  Mat panoramaGray, newFrameGray;
  vector<KeyPoint> kp1, kp2;
  vector<DMatch> matches, goodMatches;
  vector< vector<DMatch> > knnMatches;
  vector<Point2f> points1, points2;
  // Detect ORB features and compute descriptrs.
  Ptr<SURF> detector = SURF::create( minHessian );
  //Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
  // Descriptor Matcher
  //Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

  cvtColor(panorama, panoramaGray, COLOR_BGR2GRAY);
  cvtColor(newFrame, newFrameGray, COLOR_BGR2GRAY);
  // Get descriptors
  //getDescriptors(panorama, newFrame, dp1, dp2, kp1, kp2);
  detector->detectAndCompute(panoramaGray, Mat(), kp1, dp1);
  detector->detectAndCompute(newFrameGray, Mat(), kp2, dp2);
  // Match features
  matcher->knnMatch(dp1, dp2, knnMatches, 2);
  // Sort matches by score
  //std::sort(matches.begin(), matches.end());
  // Remove not so good matches
  //const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
  //matches.erase(matches.begin() + numGoodMatches, matches.end());
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
  H = findHomography(points1, points2, RANSAC);
  if (H.empty()) {
    return;
  }
  getCornerTransform(panorama, newFrame, H, H2, maxX, maxY);
}

void panoramaFromVideo(string filename, string saveLocation) {
  int frameCount;
  double maxX, maxY;
  Mat H, H2, mask;
  Mat frame, prevFrame, panorama, frameToSave;
  vector<Point2f> cameraCenter;
  VideoCapture cap(filename);

  if (!cap.isOpened()) {
		return;
	}
  // get first two frames
  cap >> panorama;
  cap >> frame;
  // get the center of the image
  cameraCenter.push_back(Point2f(panorama.cols / 2, panorama.rows / 2));
  frameCount = 1;
  // loop over the images
  while (!frame.empty()) {
    if (frameCount % 50 == 0) {
      // Get the transform from frame i to frame i-1
      getHomography(panorama, frame, H, H2, maxX, maxY);
      // Combine frame(i) with frame(0:i-1)
      panorama = stitchFrames(frame, panorama, mask, H, H2, maxX, maxY);
      cropFrame(panorama);
      // Update the center of the camera in the new combined frame
      updateCameraCenter(cameraCenter, H2, frame);
      // Save new frame with camera center added
      frameToSave = panorama.clone();
      circle(frameToSave, cameraCenter.back(), 10, Scalar(0, 0, 255), FILLED);
      imwrite(saveLocation + to_string(frameCount) + ".jpg", frameToSave);
    }
    cap >> frame;
    frameCount += 1;
  }
  return;
}

void panoramaFromImages(char const folder[], string saveLocation) {
  int frameCount, frameTotal;
  double maxX, maxY;
  double rows, cols;
  Mat H, H2, mask;
  Mat currentFrame, panorama, frameToSave;
  string imageName;
  vector<Point2f> cameraCenter;
  Point2f center = Point2f(-1, -1);

  Mat T = Mat::eye(3, 3, CV_64F);
  Mat C = Mat::eye(3, 3, CV_64F);

  // get the center of the image
  frameCount = 0;
  // loop over the images
  if (auto dir = opendir(folder)) {
    while (auto f = readdir(dir)) {
        if (!f->d_name || f->d_name[0] == '.')
            continue; // Skip everything that starts with a dot
        currentFrame = imread(string(folder) + f->d_name);

        // wait till we get second image till we start creating panorama
        if (!panorama.empty()) {
          // Get the transform from frame i to frame i-1
          getHomography(panorama, currentFrame, H, H2, maxX, maxY);
          if (!H.empty()) {
            // Combine frame(i) with frame(0:i-1)
            panorama = stitchFrames(currentFrame, panorama, mask, H, H2, maxX, maxY);
            cropFrame(panorama);
          }
        }
        else {
          panorama = currentFrame;
        }
        frameCount += 1;
    }
    closedir(dir);
  }
  imwrite("panorama.jpg", panorama);
}

int main(int argc, char **argv) {
  int len;
  char location[] = "./map/";
  string saveLocation, command;
    // create the panoramas!!!
    saveLocation = "./panoramas/" + to_string(0) + "/";
    command = "mkdir -p " + saveLocation;
    system(command.c_str());
    panoramaFromImages(location, saveLocation);
  return 0;
}
