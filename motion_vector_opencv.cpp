#include "motion_vector_opencv.h"
#include <opencv2/imgproc.hpp>

MotionVectorOpenCV::MotionVectorOpenCV(FlowMethod method)
    : method_(method) {
    
    // Initialize optical flow algorithms
    tvl1_ = cv::optflow::createOptFlow_DualTVL1();
    dis_ = cv::optflow::createOptFlow_DIS(cv::optflow::DISOpticalFlow::PRESET_MEDIUM);
}

cv::Mat MotionVectorOpenCV::calculateDenseFlow(
    const cv::Mat& prev_frame,
    const cv::Mat& curr_frame) {
    
    cv::Mat flow;
    
    // Ensure frames are grayscale
    cv::Mat prev_gray, curr_gray;
    if (prev_frame.channels() == 3) {
        cv::cvtColor(prev_frame, prev_gray, cv::COLOR_BGR2GRAY);
    } else {
        prev_gray = prev_frame.clone();
    }
    
    if (curr_frame.channels() == 3) {
        cv::cvtColor(curr_frame, curr_gray, cv::COLOR_BGR2GRAY);
    } else {
        curr_gray = curr_frame.clone();
    }
    
    switch (method_) {
        case FlowMethod::FARNEBACK: {
            // Farneback dense optical flow
            cv::calcOpticalFlowFarneback(
                prev_gray, curr_gray, flow,
                0.5,  // pyramid scale
                3,    // levels
                15,   // winsize
                3,    // iterations
                5,    // poly_n
                1.2,  // poly_sigma
                0     // flags
            );
            break;
        }
        
        case FlowMethod::DUAL_TVL1: {
            if (tvl1_) {
                tvl1_->calc(prev_gray, curr_gray, flow);
            }
            break;
        }
        
        case FlowMethod::DIS: {
            if (dis_) {
                dis_->calc(prev_gray, curr_gray, flow);
            }
            break;
        }
        
        case FlowMethod::LUCAS_KANADE:
            // Lucas-Kanade is sparse, handled separately
            flow = cv::Mat::zeros(prev_gray.size(), CV_32FC2);
            break;
    }
    
    return flow;
}

std::vector<cv::Point2f> MotionVectorOpenCV::calculateSparseFlow(
    const cv::Mat& prev_frame,
    const cv::Mat& curr_frame,
    std::vector<cv::Point2f>& points) {
    
    cv::Mat prev_gray, curr_gray;
    if (prev_frame.channels() == 3) {
        cv::cvtColor(prev_frame, prev_gray, cv::COLOR_BGR2GRAY);
    } else {
        prev_gray = prev_frame.clone();
    }
    
    if (curr_frame.channels() == 3) {
        cv::cvtColor(curr_frame, curr_gray, cv::COLOR_BGR2GRAY);
    } else {
        curr_gray = curr_frame.clone();
    }
    
    std::vector<cv::Point2f> next_points;
    std::vector<uchar> status;
    std::vector<float> err;
    
    // Lucas-Kanade optical flow
    cv::calcOpticalFlowPyrLK(
        prev_gray, curr_gray,
        points, next_points,
        status, err,
        cv::Size(21, 21),  // window size
        3,                 // max pyramid level
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
        0,                 // flags
        0.001              // min eigen threshold
    );
    
    // Calculate motion vectors
    std::vector<cv::Point2f> motion_vectors;
    for (size_t i = 0; i < points.size(); ++i) {
        if (status[i]) {
            cv::Point2f mv = next_points[i] - points[i];
            motion_vectors.push_back(mv);
        } else {
            motion_vectors.push_back(cv::Point2f(0, 0));
        }
    }
    
    // Update points for next iteration
    points = next_points;
    
    return motion_vectors;
}

std::vector<cv::Point2f> MotionVectorOpenCV::flowToBlockVectors(
    const cv::Mat& flow,
    int block_size) {
    
    std::vector<cv::Point2f> block_vectors;
    
    int blocks_x = (flow.cols + block_size - 1) / block_size;
    int blocks_y = (flow.rows + block_size - 1) / block_size;
    
    for (int by = 0; by < blocks_y; ++by) {
        for (int bx = 0; bx < blocks_x; ++bx) {
            int x = bx * block_size;
            int y = by * block_size;
            
            // Average motion vector in the block
            cv::Point2f sum(0, 0);
            int count = 0;
            
            for (int py = y; py < std::min(y + block_size, flow.rows); ++py) {
                for (int px = x; px < std::min(x + block_size, flow.cols); ++px) {
                    const cv::Vec2f& f = flow.at<cv::Vec2f>(py, px);
                    sum.x += f[0];
                    sum.y += f[1];
                    count++;
                }
            }
            
            if (count > 0) {
                block_vectors.push_back(cv::Point2f(sum.x / count, sum.y / count));
            } else {
                block_vectors.push_back(cv::Point2f(0, 0));
            }
        }
    }
    
    return block_vectors;
}

cv::Mat MotionVectorOpenCV::visualizeFlow(
    const cv::Mat& frame,
    const cv::Mat& flow,
    int step) {
    
    cv::Mat vis = frame.clone();
    if (vis.channels() == 1) {
        cv::cvtColor(vis, vis, cv::COLOR_GRAY2BGR);
    }
    
    for (int y = step / 2; y < flow.rows; y += step) {
        for (int x = step / 2; x < flow.cols; x += step) {
            const cv::Vec2f& f = flow.at<cv::Vec2f>(y, x);
            
            cv::Point2f pt1(x, y);
            cv::Point2f pt2(x + f[0], y + f[1]);
            
            // Draw arrow
            cv::arrowedLine(vis, pt1, pt2, cv::Scalar(0, 255, 0), 1, 8, 0, 0.3);
        }
    }
    
    return vis;
}
