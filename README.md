# Data Augmented AI-Generated Image Detector

## Project Description
I, along with Sami Nourji, Sujith Pakala, and Everest Yang developed a novel AI-Generated Image Detector which distinguishes real images from AI-generated ones, addressing challenges posed by the rise of hyperrealistic content
produced by generative AI. Using the CIFAKE dataset, we implement a CNN architecture with Fourier Transform features to evaluate the model's accuracy in identifying synthetic images. Our hypothesis is that incorporating frequency information via Fourier Transforms, in addition to spatial domain information, into a CNN can enhance the detection of AI-generated images by leveraging subtle frequency inconsistencies.
This was disproved by our research, as our best-performing baseline CNN achieved a testing accuracy of 98.58%, while our Fourier-based model reached an accuracy of 98.50%. Our findings highlight that incorporating Fourier features into the detection pipeline provides valuable insights, although the overall accuracy depends mostly on the CNN architecture. This research aims to encourage future research in the growing field concerning digital authenticity.

## Technical Implementation
<img src="https://github.com/user-attachments/assets/e49587d2-8e0d-4fb6-97b2-327b3153d5b4" alt="Image" width="786" height="251"> <br />
Original Architecture - 98.58%<br /><br />

<img src="https://github.com/user-attachments/assets/c061c428-1826-4d1a-9cbf-d89ae8527c73" alt="Image" width="786" height="251"> <br />
Fourier Transform - 82.59%<br /><br />

<img src="https://github.com/user-attachments/assets/55e2629d-f755-49ae-aa78-78a8088d0a65" alt="Image" width="789" height="362"> <br />
Experiment 1 (concatenation) - Actual Fourier: 95.36% | Random Noise: 50.35%<br /><br />

<img src="https://github.com/user-attachments/assets/55144b09-928a-4854-8ba4-2a73a12f7f68" alt="Image" width="784" height="387"> <br />
Experiment 2 (combination) - Actual Fourier: 98.50% | Random Noise: 98.50%
