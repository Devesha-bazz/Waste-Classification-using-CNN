# ♻️ Waste Management using CNN (Convolutional Neural Network)

This project focuses on classifying waste images into 'Organic' and 'Recyclable' using a Convolutional Neural Network (CNN) built with TensorFlow/Keras. It uses image data from Kaggle with 22,564 labeled images and achieves nearly 90% validation accuracy. The dataset is loaded using OpenCV and preprocessed for training and validation. The CNN consists of multiple convolutional layers, ReLU activation, max pooling, dropout to avoid overfitting, and a sigmoid output for binary classification. Key libraries include pandas, numpy, matplotlib, seaborn, cv2, tqdm, and TensorFlow.

## 🔧 Technologies Used:
- Python
- TensorFlow/Keras
- OpenCV
- pandas, numpy, matplotlib, seaborn
- Google Colab / Jupyter Notebook

## 📁 Dataset:
**Source:** [Waste Classification Data on Kaggle](https://www.kaggle.com/datasets/techsash/waste-classification-data)
- Categories: Organic (O) and Recyclable (R)
**Total Images:** 22,564
  - **Organic (O):** 12,565 images
  - **Recyclable (R):** 9,999 images
- Size: ~450 MB

## ✅ Features:
- CNN for waste image classification
- High validation accuracy (~89.85%)
- Visualizations and prediction function
- Accuracy/Loss plots for evaluation

## 💡 Future Scope:
- Apply transfer learning
- Real-time camera prediction
- Multi-class waste classification

