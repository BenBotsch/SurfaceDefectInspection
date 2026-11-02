#include "classifier.hpp"

#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace sdi {

cv::Mat preprocess(const cv::Mat& gray) {
  if (gray.empty() || gray.type() != CV_8UC1) {
    throw std::invalid_argument("expected a non-empty 8-bit greyscale image");
  }
  cv::Mat resized;
  cv::resize(gray, resized, cv::Size(kSize, kSize), 0, 0, cv::INTER_LINEAR);
  cv::Mat scaled;
  resized.convertTo(scaled, CV_32F, 1.0 / 255.0);
  scaled = (scaled - kMean) / kStd;
  return cv::dnn::blobFromImage(scaled);
}

Classifier::Classifier(const std::string& model_path) : net_(cv::dnn::readNetFromONNX(model_path)) {
  net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

std::vector<float> Classifier::probabilities(const cv::Mat& gray) {
  net_.setInput(preprocess(gray), "image");
  cv::Mat out = net_.forward("probabilities");
  return std::vector<float>(out.begin<float>(), out.end<float>());
}

}  // namespace sdi
