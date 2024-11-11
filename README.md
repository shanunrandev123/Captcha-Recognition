# Captcha Recognition using Convolutional and Recurrent Neural Networks in Pytorch

The Project aims to develop a Deep learning model to recognize and decode CAPTCHA images. It uses a Convolutional Neural Network and a Recurrent Neural Network to process and predict characters from captcha images.

## Model Architecture
The Captcha DL model combines CNN layers for image feature extraction and a GRU layer for sequential character prediction, outputting probabilities across character classes.

## Training and Evaluation
A custom engine module provides training and evaluation functions to train the model and compute losses.
During training, the optimizer and learning rate scheduler are used to adjust model weights.
After each epoch, the model’s predictions are decoded into human-readable text using the decode_preds customized function, and the results are evaluated against actual CAPTCHA text to monitor performance.

## CTC Loss (Connectionist Temporal Classification)

In this CAPTCHA recognition model, CTC Loss is used as the loss function.

CTC is particularly useful for sequence-to-sequence problems where the alignment between input and output sequences is unknown (which is common in tasks like speech recognition and handwritten text recognition). CTC allows the network to output a sequence of probabilities and automatically learns the best alignment between the input (image of CAPTCHA text) and the output (the predicted text).

The model outputs a probability distribution for each time step in the input sequence.
CTC decodes the predicted sequence by considering all possible alignments of the input and output.
The CTC loss function then computes the probability of the correct output sequence given the input sequence and minimizes this loss.
In our case, the output sequence is the CAPTCHA text, and the input sequence is the image of the CAPTCHA.


## Dataset Link
https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip

## Cloning this Repository
git clone https://github.com/shanunrandev123/Captcha-Recognition.git

cd /src





















## Output snippet

![image](https://github.com/user-attachments/assets/64884dee-9adf-4460-bb97-21ef80c31dc7)

