# TransformerGAN

## About

This project aims to foster the usage of transformer architectures as autoencoders for multivariate time series. The goal is to enable generative sampling processes to create artificial, but plausible time series signals. These artificial signals can then for example serve to broaden datasets or to represent local neighborhoods of a certain time series signal. A specific use case lies in the field of explainable machine learning, in which a local neighborhood for a time series input query is sampled to construct a new dataset of similar signals. If some of these signals activate another class in a blackbox decision making system, minimal changes of the original time series that cause another classification result can be revealed (counterfactual explanations). 

Transformer models follow the Encoder-Decoder principle, where the Encoder projects the input time series into a meaningful memory. A subsequent feed-forward-network then serves as a compressed latent space of the memory, which gives more control about the most important, latent features. Using this compressed memory, the Decoder learns to reconstruct the original time series. Thereby, the compressed memory latent space is shaped during training using a GAN-principle: A Generator samples a random point from the compressed latent space and constructs a time series using the Transformer-Decoder. A Discriminator then decides if a given time series is real (from the dataset) or not. This procedure shifts random points from the latent space near to the true distribution of the given dataset, such that plausible interpolations between learned representations can be sampled.

To compare this procedure with other existing strategies, a convolutional autoencoder is implemented based on multiple convolutional layers with different numbers of kernels and kernel sizes [[1]](#1).

<img width="500" alt="epoch29" src="https://user-images.githubusercontent.com/56418155/206883314-1adc2da2-e1d8-4e5d-a53d-a4e9634ca1ed.png">

During the experiments the NATOS dataset is used, which contains body sensor recordings of gestures used as aircraft handling signals [[2]](#2).
http://groups.csail.mit.edu/mug/natops/

## References
<a id="1">[1]</a> 
R. Guidotti et al. (2020). 
Explaining Any Time Series Classifier.
IEEE Second International Conference on Cognitive Machine Intelligence (CogMI)

<a id="2">[2]</a> 
Yale Song, David Demirdjian, and Randall Davis (2011).
Tracking Body and Hands For Gesture Recognition: NATOPS Aircraft Handling Signals Database.
In Proceedings of the 9th IEEE International Conference on Automatic Face and Gesture Recognition.
