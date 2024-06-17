# Handwritten-Eq_solver


# Handwritten Equation Solver

This project aims to create a Handwritten Equation Solver using Convolutional Neural Networks (CNNs). The system recognizes and solves simple handwritten equations, consisting of digits (0-9) and operators (+, -, x).

## Project Structure

The project is divided into three main parts:
1. **Data Extraction**: Extracts and preprocesses handwritten images for training.
2. **Model Training**: Trains a CNN to recognize digits and operators from the preprocessed images.
3. **Equation Recognition and Evaluation**: Recognizes and evaluates equations from images using the trained model.

## Getting Started

### Prerequisites

- Python 3.6 or later
- TensorFlow 2.x
- OpenCV
- Pandas
- Matplotlib
- scikit-learn

### Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/Handwritten-Equation-Solver.git
cd Handwritten-Equation-Solver
pip install -r requirements.txt
```

### Directory Structure

```
Handwritten-Equation-Solver/
│
├── data_extr.py              # Data extraction and preprocessing script
├── train.py                  # Model training script
├── CNN_test.py               # Equation recognition and evaluation script
├── extracted_images/         # Directory containing training images
│   ├── 0/                    # Images of digit 0
│   ├── 1/                    # Images of digit 1
│   ├── ...                   # Other digits and operators
│   ├── +/                    # Images of the plus operator
│   ├── -/                    # Images of the minus operator
│   └── times/                # Images of the multiplication operator
├── train_final.csv           # Preprocessed data for training
└── README.md                 # Project README file
```

## Data Extraction

### Running Data Extraction

Extract and preprocess images for training:

```bash
python data_extr.py
```

This script processes images from the `extracted_images` directory and saves the preprocessed data to `train_final.csv`.

## Model Training

### Running Model Training

Train the CNN model:

```bash
python train.py
```

This script reads `train_final.csv`, applies data augmentation, and trains the CNN model. The trained model is saved as `model_final.json` and `model_final.weights.h5`.

## Equation Recognition and Evaluation

### Running the Recognition and Evaluation

Recognize and evaluate equations from images in a specified folder:

```bash
python CNN_test.py
```

This script processes all `.jpg` and `.png` images in the specified folder (`test_images` by default), recognizes the equations, and prints the results.

### Sample Usage

For example, if you have a folder `test_images` with handwritten equations:

```bash
python CNN_test.py
```

### Output

The script will output recognized equations and their evaluated results for each image in the folder.

## Contribution

Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

- Thanks to the developers of TensorFlow, OpenCV, and other libraries used in this project.

---

Replace the placeholder links and usernames with your actual GitHub repository link and username. This README provides a clear overview of the project, instructions for setup and usage, and a directory structure for easy navigation.
