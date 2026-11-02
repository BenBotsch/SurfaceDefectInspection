// Classify NEU-DET images with the ONNX model exported by `sdi export`, using OpenCV DNN only.
#pragma once

#include <array>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

namespace sdi {

// Same order as sdi.dataset.CLASSES.
inline const std::array<std::string, 6> kClasses = {
    "crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"};

// Same constants as sdi.preprocess.
constexpr int kSize = 224;
constexpr float kMean = 0.449F;
constexpr float kStd = 0.226F;

// Resize an 8-bit greyscale image bilinearly and normalise it into a 1x1xHxW blob.
cv::Mat preprocess(const cv::Mat& gray);

class Classifier {
 public:
  explicit Classifier(const std::string& model_path);

  // Softmax probabilities per class for one 8-bit greyscale image.
  std::vector<float> probabilities(const cv::Mat& gray);

 private:
  cv::dnn::Net net_;
};

}  // namespace sdi
