# Matrix, Sentiment, and YOLO Benchmark Suite

This Python script provides a multi-purpose benchmarking and demonstration suite featuring:

- **Sentiment Analysis** using NLTK's VADER
- **Matrix Multiplication** using various methods (loops, list comprehensions, NumPy, and GPU with PyTorch)
- **YOLO-like Model**: A simple, efficient YOLO-inspired neural network for benchmarking inference speed

## Features

- **Sentiment Analysis**: Enter a sentence and get its sentiment scores (positive, negative, neutral, compound).
- **Matrix Multiplication**: Enter two square matrices and compare the speed of different multiplication methods.
- **YOLO Benchmark**: Run a simple YOLO-like model on random data and measure inference time for different batch sizes.
- **Visualization**: Benchmark results are plotted using Matplotlib.

## Requirements

- Python 3.7+
- NumPy
- PyTorch
- Matplotlib
- NLTK

## Installation

1. **Clone or download this repository** and place the script in your working directory.

2. **Install dependencies:**

   ```bash
   pip install numpy torch matplotlib nltk
   ```

3. **Download NLTK VADER Lexicon**  
   The script will automatically download the VADER lexicon the first time it runs.

## Usage

Run the script:

```bash
python your_script_name.py
```

### What happens:

1. **Sentiment Analysis**:  
   - You will be prompted to enter a sentence.
   - The script will print the sentiment analysis result.

2. **Matrix Multiplication**:  
   - You will be prompted to enter the size of the square matrices (e.g., 2 for 2x2).
   - Enter the values for both matrices A and B, row by row.
   - The script will compute and print the results using different methods, and plot a timing comparison.

3. **YOLO Benchmark**:  
   - The script will run a simple YOLO-like model on random data for batch sizes 1, 4, and 8.
   - Inference times will be printed and plotted.

### Controls

- Close the Matplotlib plot windows to continue or finish the script.

## Notes

- **GPU Support**: If CUDA is available, PyTorch will use the GPU for matrix multiplication and YOLO benchmarking.
- **Error Handling**: If GPU is not available, the script will skip GPU-based multiplication.
- **YOLO Model**: The YOLO-like model is a simplified version for benchmarking only, not for real object detection.

## Example Output

```
Enter a sentence for sentiment analysis: I love this product!
Sentiment Analysis: {'neg': 0.0, 'neu': 0.294, 'pos': 0.706, 'compound': 0.6696}

Enter the size of the square matrices (e.g., 2 for 2x2): 2
Enter values for Matrix A (row-wise):
A[0][0]: 1
A[0][1]: 2
A[1][0]: 3
A[1][1]: 4
Enter values for Matrix B (row-wise):
B[0][0]: 5
B[0][1]: 6
B[1][0]: 7
B[1][1]: 8

Result (Loops Method): [[19.0, 22.0], [43.0, 50.0]]
...
```

## License

This project is for educational and benchmarking purposes.

---

**Enjoy experimenting with sentiment analysis, matrix math, and neural network benchmarking!**
