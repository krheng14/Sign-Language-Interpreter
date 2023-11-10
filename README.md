# SignTalk - A ML-Powered Sign-Language-Interpreter

# 1.0 Introduction

Communication is a fundamental human need. For the deaf and hard of hearing, sign language is a vital tool for daily communication. Yet, the majority of hearing people do not understand sign language, which creates a communication barrier.

# 2.0 How it Works

![Alt Text](/images/signTalkDemo.gif)

SignTalk is a real-time sign language detection system that bridges this gap. It utilizes advanced machine learning techniques, employing MediaPipe Holistic Keypoints to detect body language and hand signs, which are then processed by an LSTM-based action detection model to interpret the signs accurately.

# 3.0 Features

## 3.1 Real-Time Sign Language Detection

Using a webcam, SignTalk captures live video feed and processes it in real-time. The system identifies and interprets individual signs through the detection of hand and body keypoints, providing instant text translation of the signed content.

## 3.2 LSTM-Powered Action Detection

Our LSTM model analyzes temporal sequences of movements to understand and predict complex sign language gestures, ensuring accurate translation even for nuanced expressions.

## 3.3 TensorFlow and Keras Integration

SignTalk is built using TensorFlow and Keras, harnessing their powerful neural network capabilities to learn and improve sign language interpretation over time with more data input.

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
python app.py
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

- [ ] Support for multiple sign languages
- [ ] Improved accuracy with deep learning optimizations
- [ ] Mobile app for on-the-go interpretation
- [ ] Integration with speech synthesis for two-way communication

Your feedback and contributions are what will make SignTalk a powerful tool for breaking communication barriers. Join us in creating a world where everyone has the power to communicate freely.
