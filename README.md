# SiameseNet
Siamese networks for one-shot image recognition.

## Implementation
- trains a Siamese network using Tensorflow on MNIST for image identification/recognition: https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
- network embeds a pair of 28x28 images (a point in 784D) into a lower dimensional space through a series of conv-maxpool blocks with shared weights across the image pair
- compares the two candidates via subtraction or concatenation to see if they represent the same number (same individual)
- uses binary cross-entropy between true and predicted similarities
- adapted from: https://github.com/ywpkwon/siamese_tf_mnist

## Data
- MNIST 28x28 grayscale images
- load 2 random training batches, assign examples from the 1st (2nd) as input tensor of the 1st (2nd) image of the pair
- create GT label by comparing the actual number label (0: dissimilar individuals, 1: same individual) 

## Results
- after 55 epochs of training on a laptop GPU: OA: 99.72, Mean F1-score: 0.992 (on separate test set)

Evolution of OA on validation set:
![](figures/val_OA.png?raw=true "")

Example of results on test set:
True label: predicted similarity (float in [0, 1])
![](figures/sanity_check_step5_plot0_top.png?raw=true "")
![](figures/sanity_check_step5_plot0_bottom.png?raw=true "")


