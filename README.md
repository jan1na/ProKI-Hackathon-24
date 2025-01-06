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
The approach is to first pre-process the input image using a fine-tuned
DeepLabV3-Plus model. This gives us a mask that is still not binary, so we 
post-process it using thresholding and erosion. After we have the binary mask
we compute the perfect x, y and angle using simulated annealing.

## How to Run
// How to run your code, e.g., `python solution/main.py path/to/input/tasks.csv output/solutions.csv` //

## Pipeline

### STEP 1: Fine-Tuning of DeepLabV3-Plus model
To fine-tune it to output masks of the image, we hand
painted masks for 42 images. And also used the inverted images for the same mask, 
so we had 2 images for one mask, so we had 84 image mask pairs. Then applied 10 
augmentations per image mask pair to create more training data. Then we run
the training script on Google Colab on an A100.

### STEP 2: Simulated Annealing

## Decision Process
We tried out multiple solutions for preprocessing and finding x, y, angle

### Preprocessing Approaches to get binary mask

#### Classical computer vision techniques
To get rid of dust we used gaussian blur. And then thresholding. Also 
we tried to find connected areas with cv2 to get rid of noise and dust.
After that we used Canny to detect edges and contours. All of that 
was not leading to something, that could work for all those dusty images.
Therefore we discarded this approach.

#### UNet fine-tuning (Resnet34)
The first idea was to finetune a UNet, because they are used for segmentation.
The approach was to create hand painted masks. To try it out we only
painted 10 masks, but the goal was to paint 42, to get every part.
The fine-tuning did not produces usable mask. But the idea was still the
best so maybe the pre-trained model was not the right one. [include images]

#### DeepLabV3-Plus fine-tuning
This was the next model we pretrained. Here the results were not bad, but
the post-processing was the hardest part, to get a good binary out of the
model output. 

#### Segment-Anything fine-tuning
Another idea we tried out.


### Approaches to get x, y and angle after having the binary mask

#### Bruteforce
The first approach was to bruteforce all different combinations of 
x, y and angle. But even after making steps of 10 for x and y and 36 for
the angle it was ways to slow, so we quickly discarded this solution.

#### scipy.optimize.minimize
The second approach was to use the minimize function of scipy with a 
defined objective function. Here we returned inf if the gripper collided
with the mask and if not the distance to the middle of the image. This
approach was not far from our final solution (simulated annealing) but at
that time our biggest problem was the preprocessing of the image to get
a good binary mask.

#### Improved Bruteforce
Before bruteforce we found all possible positions of a gripper, where
one dot can fit using convolution of a mask of one dot. Then we computed
the distance between 2 of the dots in the gripper, created a mask with 
a circle in data radius and checked with bitwise and which angles have 
a possible fitting next gripper dot. With those angles we tried out the
rest of the gripper dots for each returned angle to find the angles 
where every dot in the gripper fits. But this was also very slow.

#### Simulated annealing (FINAL)
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
