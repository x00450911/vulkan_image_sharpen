#ifndef MOTION_VECTOR_OPENCV_H
#define MOTION_VECTOR_OPENCV_H

#include <opencv2/opencv.hpp>
#include <opencv2/optflow.hpp>
#include <vector>

/**
 * OpenCV-based motion vector calculation using optical flow
 * Provides more advanced algorithms like Farneback, Lucas-Kanade, etc.
 */
class MotionVectorOpenCV {
public:
    enum class FlowMethod {
        FARNEBACK,      // Dense optical flow (Farneback)
        LUCAS_KANADE,   // Sparse optical flow (Lucas-Kanade)
        DUAL_TVL1,      // Dense optical flow (Dual TV-L1)
        DIS             // Dense inverse search
    };
    
    /**
     * Constructor
     * @param method Optical flow method to use
     */
    explicit MotionVectorOpenCV(FlowMethod method = FlowMethod::FARNEBACK);
    
    /**
     * Calculate dense motion vectors using optical flow
     * @param prev_frame Previous frame (grayscale)
     * @param curr_frame Current frame (grayscale)
     * @return Motion vectors as a 2-channel Mat (dx, dy) for each pixel
     */
    cv::Mat calculateDenseFlow(const cv::Mat& prev_frame, const cv::Mat& curr_frame);
    
    /**
     * Calculate sparse motion vectors at feature points
     * @param prev_frame Previous frame
     * @param curr_frame Current frame
     * @param points Feature points to track
     * @return Motion vectors for each point
     */
    std::vector<cv::Point2f> calculateSparseFlow(
        const cv::Mat& prev_frame,
        const cv::Mat& curr_frame,
        std::vector<cv::Point2f>& points
    );
    
    /**
     * Convert dense flow to block-based motion vectors
     * @param flow Dense optical flow (2-channel Mat)
     * @param block_size Size of blocks
     * @return Block-based motion vectors
     */
    std::vector<cv::Point2f> flowToBlockVectors(const cv::Mat& flow, int block_size = 16);
    
    /**
     * Visualize motion vectors
     * @param frame Frame to draw on
     * @param flow Motion vectors (dense or block-based)
     * @param step Step size for visualization
     * @return Visualization image
     */
    cv::Mat visualizeFlow(const cv::Mat& frame, const cv::Mat& flow, int step = 16);
    
    void setMethod(FlowMethod method) { method_ = method; }
    FlowMethod getMethod() const { return method_; }

private:
    FlowMethod method_;
    cv::Ptr<cv::optflow::DualTVL1OpticalFlow> tvl1_;
    cv::Ptr<cv::optflow::DISOpticalFlow> dis_;
};

#endif // MOTION_VECTOR_OPENCV_H
