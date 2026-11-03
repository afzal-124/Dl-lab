
Deep Learning Lab Programs
A collection of 8 Deep Learning laboratory experiments implemented in Python, based on the VTU Deep Learning Lab Manual for B.E. CSE (Artificial Intelligence & Machine Learning).
The repository covers word embeddings, neural-network classification, CNNs, autoencoders, text classification, time-series forecasting, transfer learning, and sentiment analysis.

📚 Experiments
No.	Program	Description
01	Word Embeddings using Word2Vec	Trains a Word2Vec model on a small document corpus and displays word vectors and similar words.
02	Deep Neural Network Classification	Builds and trains a feedforward neural network to classify the Iris dataset.
03	CNN Image Classification	Uses a Convolutional Neural Network to classify handwritten digits from the MNIST dataset.
04	Image Compression using Autoencoder	Compresses MNIST images into a smaller representation and reconstructs them using an autoencoder.
05	Text Document Classification	Uses an embedding-based deep learning model to classify IMDB movie reviews.
06	Time-Series Forecasting using LSTM	Trains an LSTM network on generated time-series data and compares predicted values with actual values.
07	Transfer Learning using MobileNetV2	Uses an ImageNet-pretrained MobileNetV2 model for image classification on CIFAR-10.
08	Positive/Negative Review Classification	Converts review text into TF-IDF features and classifies sentiment using Logistic Regression.
🗂️ Repository Structure
Deep-Learning-Lab/
│
├── 01_word_embeddings.py
├── 02_deep_neural_network.py
├── 03_cnn_mnist.py
├── 04_autoencoder_mnist.py
├── 05_text_document_classification.py
├── 06_lstm_time_series_forecasting.py
├── 07_transfer_learning_cifar10.py
├── 08_review_sentiment_classification.py
│
├── requirements.txt
└── README.md
🛠️ Technologies Used
Language: Python
Deep Learning: TensorFlow / Keras
Machine Learning: Scikit-learn
Word Embeddings: Gensim (Word2Vec)
Data Processing: NumPy
Visualization: Matplotlib
⚙️ Setup and Installation
1. Clone the repository
git clone https://github.com/YOUR-USERNAME/Deep-Learning-Lab.git
cd Deep-Learning-Lab
Replace YOUR-USERNAME with your GitHub username.
2. Create a virtual environment (recommended)
macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
Windows
python -m venv .venv
.venv\Scripts\activate
3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
Python compatibility: TensorFlow support depends on the Python version and operating system. Python 3.10–3.12 is a practical starting point; check the TensorFlow installation guide if installation fails on your setup.
▶️ Run the Programs
Run each experiment separately from the repository folder.
For example:

python 01_word_embeddings.py
To run another experiment, replace the filename:
python 02_deep_neural_network.py
python 03_cnn_mnist.py
python 04_autoencoder_mnist.py
python 05_text_document_classification.py
python 06_lstm_time_series_forecasting.py
python 07_transfer_learning_cifar10.py
python 08_review_sentiment_classification.py
Some experiments download datasets or pretrained weights the first time they run. An internet connection is required for those downloads. Training time depends on your computer.
📌 Notes
Run one program at a time.
The MNIST, IMDB, and CIFAR-10 datasets are provided through TensorFlow/Keras dataset utilities.
Program 6 generates synthetic time-series data for forecasting.
Program 7 downloads MobileNetV2 ImageNet weights when they are not already cached.
Program 8 uses a small demonstration review dataset. Its accuracy is for learning purposes and should not be treated as real-world sentiment performance.
Model accuracy and printed output may vary with software versions, hardware, and training settings.
🎯 Learning Outcomes
By completing these experiments, students get practical experience with:
Generating and exploring word embeddings.
Building and evaluating deep neural networks.
Understanding convolution and pooling in CNNs.
Learning image compression and reconstruction with autoencoders.
Applying deep learning to text data.
Forecasting sequential data using LSTM networks.
Reusing pretrained models through transfer learning.
Classifying text reviews by sentiment.
👨‍🎓 Academic Information
Course: Deep Learning
Programme: B.E. – Computer Science and Engineering (AI & ML)
Institution: AMC Engineering College
University: Visvesvaraya Technological University (VTU)
Academic Year: 2025–2026
🤝 Contributions
This repository is intended for academic learning and laboratory practice. You may fork the repository, experiment with the programs, and suggest improvements.
If you find an issue, feel free to open a GitHub issue or submit a pull request.

⭐ If this repository helps with your learning, consider giving it a star!
