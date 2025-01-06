# Hackathon 2024 - Submission of Group *Darmstadt Intelligence Technologies*
Team members:
- Philipp Hinz
- Janina Fritzke
## Installation
```bash
pip install -r requirements.txt
pip install git+https://github.com/qubvel/segmentation_models.git
```
## Description
This project provides a robust solution for processing images of metal parts to extract precise 
coordinates (x, y) and orientation (angle) of grippers, which was developed as part of a hackathon 
challenge. The approach combines advanced deep learning for image segmentation, computer vision
techniques for post-processing and optimization techniques to get the solution. 
Here's an overview of the solution:

**1. Image Segmentation**:
We fine-tuned a DeepLabV3-Plus model to generate masks of the input image. 

**2. Post-Processing**:
The model's output was processed using thresholding and erosion to generate binary masks.

**3. Optimization**:
Using simulated annealing, we computed the optimal x, y, and angle.

## How to Run
To execute the solution, use the following command:
```bash
python solution/main.py path/to/input/tasks.csv output/solutions.csv
```

## Pipeline

### Step 1: Fine-Tuning the DeepLabV3-Plus Model

To achieve high-quality segmentation, we fine-tuned the DeepLabV3-Plus model with custom masks:
- Dataset Creation: We hand-painted masks for 42 images, paired each mask with its original 
and inverted image, resulting in 84 image-mask pairs. Then, we augmented the data with 10 
variations per pair.
- Training Setup: The model was fine-tuned on Google Colab using an A100 GPU.

### Step 2: Post-Processing
The model output is not a binary mask, so we applied thresholding and erosion to generate a 
binary mask.

### Step 3: Simulated Annealing
After generating binary masks, simulated annealing was used to optimize the x, y, and angle 
for the task. This technique efficiently explored the solution space, balancing exploration 
and exploitation.

## Decision Process

### Preprocessing Approaches to get binary mask
Several approaches were tested for preprocessing, including:

#### 1. Classical Computer Vision Techniques:
- Applied Gaussian blur, thresholding, and edge detection (Canny).
- These methods struggled with noisy, dusty images and were ultimately discarded.

#### 2. UNet Fine-Tuning (ResNet34):
- Initial experiments with fine-tuning UNet were conducted with a small dataset of 
10 hand-painted masks.
- The results were suboptimal, possibly due to limitations of the pre-trained model.

#### DeepLabV3-Plus Fine-Tuning (Final Approach):
- Provided better results than UNet. The reason might be that the DeepLabV3-Plus model is

#### Segment-Anything fine-tuning
- Tried to run a fine-tuning script on Google Colab, but was facing to many issues, so we could
not run it.
- Downloaded the vit_b (base) and vit_h (huge) pre-trained model to try it without fine-tuning.
The code was running so slowly on the CPU, that it was now option to use it.

### Optimization Approaches for x, y, and Angle

We evaluated multiple strategies to compute x, y, and angle after obtaining binary masks:
#### 1. Brute Force:
- Evaluated all combinations of x, y, and angle. Computationally expensive and discarded.

#### 2. scipy.optimize.minimize:
- Leveraged the minimize function with a custom objective function. While promising, 
preprocessing bottlenecks limited its success.
- The objective function had similar ideas to the simulated annealing approach. It returned
inf if the gripper collided with the mask and the distance to the middle of the image if not.

#### 3. Improved Brute Force:
- Optimized brute force by using convolutions to identify potential positions. Still too slow 
for practical use.
- We tried to find all possible positions of a gripper, where one dot can fit using convolution
and then computed the distance between 2 of the dots in the gripper, created a mask with a circle
in data radius and checked with bitwise and which angles have a possible fitting next gripper dot.
With those angles we tried out the rest of the gripper dots for each returned angle to find the
angles where every dot in the gripper fits. But this was also very slow.

#### Simulated Annealing (Final Approach):
The best idea was simulated annealing. Here neighbor positions based on
an initial position get tried out and are accepted as new current position
if the score of the objective function is lower as the current or with a 
probability (temperature). The temperature gets smaller with every step, so
at the beginning also bad decisions get accepted and the at the end a good 
and more stable solution is preferred. This results in a bigger exploration 
at the beginning of the computation and more exploitation at the end. This solution
was very fast but not perfect and sometimes did not find a good solution.
[More optimization]

## Challenges

### GPU
The computation power for the fine-tuning was a big issue. We dont have a 
Nvidia GPU, so had to rent one on AWS or Google Colab. At first I tried AWS, which
was my first time, so I did not know I had to request additional vCPUs to get an
instance with a GPU. I tried to request one, explaining that I want to fine-tune
a UNet for a hackathon. But I got rejected and therefore was not able to pay for
a GPU. Thats why I used Google Colab. The free trial of 
Google Colab was not enough and the free GPUs on Google Colab are still very
slow compared to the A100. Therefore, I needed to pay for it, because fine-tuning
multiple models and debugging took a lot of compute power.


// optional, e.g., design decisions, challenges you faced, etc. //
