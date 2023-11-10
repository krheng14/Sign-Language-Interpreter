# SignTalk - A ML-Powered Sign-Language-Interpreter

# 1.0 Introduction

Communication is a fundamental human need. For the deaf and hard of hearing, sign language is a vital tool for daily communication. Yet, the majority of hearing people do not understand sign language, which creates a communication barrier.

# 2.0 How it Works
<p float="left">
  <img src="/Images/MediaPipe_Tracking.png" alt="Using MediaPipe to track feature keypoints" width="400" />
  <img src="/Images/SignTalk.gif" alt="Interpretation `Hello`, `Thank you` and `ILoveYou`" width="400" /> 
</p>

SignTalk is a real-time sign language detection system that bridges this gap. It utilizes advanced machine learning techniques, employing MediaPipe Holistic Keypoints to detect body language and hand signs, which are then processed by an LSTM-based action detection model to interpret the signs accurately.

# 3.0 Features

## 3.1 Real-Time Sign Language Detection

Using a webcam, SignTalk captures live video feed and processes it in real-time. The system identifies and interprets individual signs through the detection of hand and body keypoints, providing instant text translation of the signed content.

## 3.2 LSTM-Powered Action Detection

In SignTalk, our LSTM model is intricately designed to interpret sign language gestures with high accuracy. The model consists of a sequential neural network architecture specifically tuned to process time-series data of body and hand movements.

### Model Architecture
The LSTM model in SignTalk is built using the following layers:
- **Input Layer**: The input to the model is a sequence of keypoints (total 1662 features per frame) representing the position and movement of various body parts. Each input sequence corresponds to a sign language gesture.
- **LSTM Layers**: 
  - The first LSTM layer has 64 nodes and returns sequences, allowing the subsequent LSTM layers to receive sequences of data.
  - The second LSTM layer is composed of 128 nodes, also returning sequences.
  - The third LSTM layer has 64 nodes and does not return sequences, effectively distilling the temporal information into a single context vector.
- **Dense Layers**: 
  - Following the LSTM layers, the model includes a dense layer with 64 nodes.
  - Another dense layer follows with 32 nodes.
  - Both these layers use ReLU (Rectified Linear Unit) as the activation function.
- **Output Layer**: The final layer is a dense layer with a softmax activation function, corresponding to the number of actions (gestures) the model is trained to recognize.

### Training Process
- **Dataset**: For training, we capture 30 sequences for each hand sign, with each sequence comprising 30 frames. This comprehensive dataset ensures that the model learns a wide range of motion dynamics for each gesture.
- **Epochs**: The model is trained for 2000 epochs, allowing it to iteratively learn and improve its accuracy in interpreting sign language gestures.
- **Optimization and Loss Function**: The model uses the Adam optimizer and categorical cross-entropy as the loss function, which are standard choices for multi-class classification tasks.

## 3.3 TensorFlow and Keras Integration

SignTalk is built using TensorFlow and Keras, harnessing their powerful neural network capabilities. TensorFlow provides the computational backbone, enabling efficient processing of our extensive dataset. Keras, with its user-friendly API, simplifies the implementation of our LSTM model, making the development process more intuitive and streamlined.

Through TensorFlow and Keras, the LSTM model is meticulously trained on our dataset. This training involves feeding the sequences of keypoints into the model and optimizing its parameters through backpropagation. The comprehensive training enables the model to recognize and predict a wide range of sign language gestures accurately.

Once trained, the model can make real-time predictions. As the MediaPipe Holistic model processes new video frames, the LSTM model interprets the sequences of keypoints and predicts the corresponding sign language gestures. This integration allows SignTalk to provide real-time, accurate sign language interpretation, facilitating effective communication for the deaf and hard-of-hearing community.

# 4.0 How to Use

## 4.1 Setting up the Environment

This project is designed to be cross-platform, working on most operating systems with minimal setup required.

### <u>4.1.1 Python Environment</u>

We recommend setting up a virtual Python environment to manage dependencies:

```
python3 -m venv signTalk-env
source signTalk-env/bin/activate
```

### <u>4.1.2 Installing Dependencies</u>

Install all the required Python packages with pip:

```
pip install -r requirements.txt
```

## 4.2 Running SignTalk

Once the environment is set up and dependencies are installed, you can run SignTalk using:

```
python SignTalk.py /path/to/your/model.h5
```

The application will start and you will be prompted to allow webcam access for real-time sign language detection.

## 4.3 Interacting with the Application

The user interface is intuitive, with live feedback displayed on the screen. Sign language detected by the webcam will be translated and shown in text form in real-time.

# 5.0 Contributing

We welcome contributions from the community to help improve SignTalk. If you would like to contribute, please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Submit a pull request

# 6.0 Future Enhancements

- [ ] Support for multiple sign languages - Currently, the model can only detect signs like `I love you`, `Hello` and `Thank you`.
- [ ] Improved accuracy with deep learning optimizations - The model is only trained on my facial and body features. To let it be universal for all users, I would need to train with more data and video clips.
- [ ] Mobile app for on-the-go interpretation - For now, it is only executable from desktop terminal.
- [ ] Integration with speech synthesis for two-way communication

Your feedback and contributions are what will make SignTalk a powerful tool for breaking communication barriers. Join us in creating a world where everyone has the power to communicate freely.
