Neural Architecture Search for Image De-raining
This repository implements a neural architecture search (NAS) approach to develop efficient and effective neural networks for single image de-raining tasks. By leveraging NAS, the model aims to automatically discover optimal architectures that can remove rain streaks from images, enhancing visibility and performance in subsequent computer vision applications.

Repository Structure
block/: Contains modules defining various building blocks used in the network architectures.
genetic_algorithm/: Implements the genetic algorithm employed for the neural architecture search process.
library_imports/: Manages external library imports and dependencies required for the project.
unet_model/: Provides implementations of U-Net based models tailored for image de-raining tasks.
validation/: Includes scripts and tools for validating and evaluating the performance of the trained models.
