import numpy as np
import torch
import time
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize VADER sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Matrix Multiplication Implementations

def matrix_multiply_loops(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    if cols_A != rows_B:
        raise ValueError("Matrix dimensions incompatible for multiplication")
    C = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return C

def matrix_multiply_comprehension(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    if cols_A != rows_B:
        raise ValueError("Matrix dimensions incompatible for multiplication")
    return [[sum(A[i][k] * B[k][j] for k in range(cols_A)) for j in range(cols_B)] for i in range(rows_A)]

def matrix_multiply_numpy(A, B):
    return np.dot(np.array(A), np.array(B))

def matrix_multiply_gpu(A, B):
    A_tensor = torch.tensor(A, dtype=torch.float32)
    B_tensor = torch.tensor(B, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A_tensor = A_tensor.to(device)
    B_tensor = B_tensor.to(device)
    result = torch.matmul(A_tensor, B_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return result.cpu().numpy()

# YOLO-like Efficient Model (Simple version for benchmarking)

class EfficientConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(EfficientConvolution, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(x)

class YOLODetectionLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(YOLODetectionLayer, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.img_size = img_size

    def forward(self, x):
        batch_size, grid_size = x.size(0), x.size(2)
        prediction = x.view(batch_size, len(self.anchors), self.num_classes + 5, grid_size, grid_size)
        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()
        prediction[..., 4] = torch.sigmoid(prediction[..., 4])
        prediction[..., 5:] = torch.sigmoid(prediction[..., 5:])
        return prediction

class SimpleYOLO(nn.Module):
    def __init__(self, num_classes):
        super(SimpleYOLO, self).__init__()
        self.backbone = nn.Sequential(
            EfficientConvolution(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            EfficientConvolution(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            EfficientConvolution(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        self.detection = nn.Sequential(
            EfficientConvolution(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            EfficientConvolution(128, 3 * (5 + num_classes), 1, 1, 0)
        )
        self.anchors = torch.tensor([[10., 13.], [16., 30.], [33., 23.]])
        self.detection_layer = YOLODetectionLayer(self.anchors.float(), num_classes=num_classes, img_size=416)

    def forward(self, x):
        features = self.backbone(x)
        detection_output = self.detection(features)
        return self.detection_layer(detection_output)

# Sentiment Analysis Function

def analyze_sentiment(text):
    sentiment = sia.polarity_scores(text)
    return sentiment

# User Input Section

def get_user_matrix_input():
    size = int(input("Enter the size of the square matrices (e.g., 2 for 2x2): "))
    print("Enter values for Matrix A (row-wise):")
    A = [[float(input(f"A[{i}][{j}]: ")) for j in range(size)] for i in range(size)]

    print("Enter values for Matrix B (row-wise):")
    B = [[float(input(f"B[{i}][{j}]: ")) for j in range(size)] for i in range(size)]
    return A, B, size

# Benchmark YOLO

def test_yolo_performance():
    print("\nTesting YOLO model performance:")
    model = SimpleYOLO(num_classes=80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_sizes = [1, 4, 8]
    yolo_times = []
    for batch_size in batch_sizes:
        input_tensor = torch.randn(batch_size, 3, 416, 416).to(device)
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_tensor)
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        inference_time = (time.time() - start) / 10
        print(f"  Batch size {batch_size}: {inference_time:.5f} seconds per inference")
        yolo_times.append((batch_size, inference_time))
    return yolo_times

# Main Execution

if __name__ == "__main__":
    # Ask the user for a sentence to analyze sentiment
    user_input = input("Enter a sentence for sentiment analysis: ")
    sentiment = analyze_sentiment(user_input)
    print(f"Sentiment Analysis: {sentiment}")
    
    A, B, size = get_user_matrix_input()
    methods = []
    timings = []

    if size <= 100:
        start = time.time()
        result_loops = matrix_multiply_loops(A, B)
        loop_time = time.time() - start
        print("\nResult (Loops Method):", result_loops)
        methods.append('GPU')
        timings.append(loop_time)

        start = time.time()
        result_comp = matrix_multiply_comprehension(A, B)
        comp_time = time.time() - start
        print("\nResult (Comprehension Method):", result_comp)
        methods.append('NumPy')
        timings.append(comp_time)

    start = time.time()
    result_numpy = matrix_multiply_numpy(A, B)
    numpy_time = time.time() - start
    print("\nResult (NumPy Method):", result_numpy)
    methods.append('NumPy')
    timings.append(numpy_time)

    try:
        start = time.time()
        result_gpu = matrix_multiply_gpu(A, B)
        gpu_time = time.time() - start
        print("\nResult (GPU Method):", result_gpu)
        methods.append('Comprehensive loops')
        timings.append(gpu_time)
    except Exception as e:
        print("\nGPU multiplication skipped (no CUDA support or error)")

    # Plotting Matrix Benchmark Results
    if methods and timings:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, timings, color='skyblue')
        plt.title('Matrix Multiplication Timing Comparison')
        plt.ylabel('Time (seconds)')
        plt.xlabel('Method')
        plt.grid(True, axis='y')
        for bar, time_taken in zip(bars, timings):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{time_taken:.6f}', va='bottom', ha='center')
        plt.tight_layout()
        plt.show()

    # YOLO Benchmark
    yolo_results = test_yolo_performance()
    if yolo_results:
        batch_sizes, yolo_times = zip(*yolo_results)
        plt.figure(figsize=(10, 6))
        bars = plt.bar([str(bs) for bs in batch_sizes], yolo_times, color='salmon')
        plt.title('YOLO Inference Time per Batch Size')
        plt.ylabel('Time (seconds)')
        plt.xlabel('Batch Size')
        plt.grid(True, axis='y')
        for bar, time_taken in zip(bars, yolo_times):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{time_taken:.5f}', va='bottom', ha='center')
        plt.tight_layout()
        plt.show()
