// sdi-classify MODEL IMAGE...: print the predicted class and its probability per image.
#include <algorithm>
#include <cstdio>
#include <exception>

#include <opencv2/imgcodecs.hpp>

#include "classifier.hpp"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s MODEL.onnx IMAGE...\n", argv[0]);
    return 2;
  }
  try {
    sdi::Classifier classifier(argv[1]);
    for (int i = 2; i < argc; ++i) {
      cv::Mat gray = cv::imread(argv[i], cv::IMREAD_GRAYSCALE);
      if (gray.empty()) {
        std::fprintf(stderr, "cannot read %s\n", argv[i]);
        return 1;
      }
      auto p = classifier.probabilities(gray);
      auto best = std::max_element(p.begin(), p.end()) - p.begin();
      std::printf("%s\t%s\t%.4f\n", argv[i], sdi::kClasses[best].c_str(), p[best]);
    }
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
  return 0;
}
