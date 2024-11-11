# Captcha Recognition using Convolutional and Recurrent Neural Networks in Pytorch

The Project aims to develop a Deep learning model to recognize and decode CAPTCHA images. It uses a Convolutional Neural Network and a Recurrent Neural Network to process and predict characters from captcha images.

## Model Architecture
The Captcha DL model combines CNN layers for image feature extraction and a GRU layer for sequential character prediction, outputting probabilities across character classes.

## Training and Evaluation
A custom engine module provides training and evaluation functions to train the model and compute losses.
During training, the optimizer and learning rate scheduler are used to adjust model weights.
After each epoch, the model’s predictions are decoded into human-readable text using the decode_preds customized function, and the results are evaluated against actual CAPTCHA text to monitor performance.


##Dataset Link
https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip

## Cloning this Repository
git clone https://github.com/shanunrandev123/Captcha-Recognition.git
cd /src





















## Output snippet

![image](https://github.com/user-attachments/assets/64884dee-9adf-4460-bb97-21ef80c31dc7)

