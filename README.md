# Dataset Structure

The project relies on the VisDrone dataset, which is meticulously divided into training, validation, and testing subsets to ensure unbiased model evaluation. Inside the main repository, the files are systematically organized into distinct image and label folders, where each image has a corresponding text file containing localized bounding box coordinates. This structured layout allows the YOLOv8 architecture to parse training pairs instantly without any manual path mapping.

# Preprocessing and Augmentation Steps

Before entering the neural network, all drone imagery is dynamically downscaled to a uniform $640 \times 640$ resolution to balance feature detection with computational speed on an integrated GPU. To increase model robustness, on-the-fly data augmentations like horizontal flipping, color channel adjustments, and mosaic combinations are applied. These variations simulate different weather conditions and drone vantage points, teaching the network to generalize well even without millions of physical images.

<img width="1920" height="1920" alt="train_batch0" src="https://github.com/user-attachments/assets/5a320d99-2085-4699-abf9-db2ad7d6b8a0" />
Figure: Automated YOLOv8 Preprocessing Batch. This image demonstrates the on-the-fly spatial resizing to $640 \times 640$ pixels, normalized bounding box boundary checks, and Mosaic Data Augmentation applied to the drone imagery before entering the training layers.

# Challenges Noticed in the Dataset

The primary bottleneck in this dataset is the extreme variance in scale, as objects captured from high-altitude drone perspectives often appear as microscopic clusters of pixels. Furthermore, dense environments create massive object occlusion, where overlapping pedestrians or cars passing underneath trees severely degrade initial confidence scores. These attributes make it difficult for lightweight models to maintain stable tracking without a significant number of training epochs.

# Training Approach

The model training approach utilizes a fine-tuning strategy by leveraging a pre-trained YOLOv8 object detection framework optimized for drone telemetry processing. The training pipeline feeds localized $640 \times 640$ resolution image tensors and matching bounding box annotation coordinates through the neural network layers across 20 supervised learning epochs. During each forward pass, the model calculates detection errors using a composite loss function that combines bounding box regression variance with class categorization accuracy. An AdamW optimizer then executes backpropagation, adjusting the model's internal weights to minimize these errors and stabilize the feature extraction layers. This methodology allows the model to learn the spatial features of tiny, high-altitude targets like pedestrians and cars efficiently, without requiring a prolonged training cycle from scratch.

<img width="935" height="543" alt="image" src="https://github.com/user-attachments/assets/4000e05f-bfdf-4582-9e3e-4d45c8841a62" />

Figure: Model Training approach


<img width="1360" height="765" alt="0000006_01111_d_0000003" src="https://github.com/user-attachments/assets/4eb38a81-02b3-4d4b-91a7-816d05ae4be8" />

Figure: Sample prediction

#  Human & Car Detection with Counting

An interactive execution pipeline that processes the drone imagery sequentially while calculating real-time localization metrics. The pipeline uses our fine-tuned YOLOv8 model weights to perform frame-by-frame inference, dynamically tracking structural count statistics for both pedestrians and vehicles.

* State-Preserving Navigation: Implements an automated local session-tracking mechanism (viewer_progress.txt) that logs your current viewing index, allowing seamless application resumes upon restart.

* Interactive Control Loop: Integrated robust key-listeners ('d' for next, 'a' for previous, 'ESC' to quit) to give developers total manual playback flexibility.

* Dynamic HUD Diagnostics: Draws color-coded bounding boxes directly on targeted entities alongside a real-time statistical dashboard overlay displaying active categorical metrics.

<img width="1349" height="796" alt="image" src="https://github.com/user-attachments/assets/6b0b5992-6049-47cd-8ea4-42cece86dd29" />
Figure: It shows the counting of both car and human

# Object Tracking (ByteTrack) Implementation & Improvements

This phase upgrades our system from a frame-by-frame object detector into a temporal tracking pipeline using the high-performance ByteTrack framework. By changing our detection core from standard .predict() to .track(persist=True), the pipeline bridges the gap between consecutive frames by mapping spatial movements and assigning unique, persistent ID numbers to individual assets.

* Advanced Data Association (ByteTrack Logic): Unlike traditional trackers that discard low-confidence bounding boxes, our implementation processes both high and low scores. This ensures that when a car passes under a tree or a human is briefly hidden in a crowd (occlusion), the tracker maintains their unique identity instead of treating them as brand-new objects.

* Persistent Memory Optimization (persist=True): We integrated memory persistence directly into the YOLOv8 tracking engine. This forces the model to actively carry over its tracked ID registry from the previous image frame into the current processing tensor, creating a cohesive, simulated video stream.

* Creative Sequence Simulation Workaround: Because a continuous drone video file wasn't directly available, the system was creatively engineered to sort, stitch, and sequence the high-altitude test images together into an automated loop. By applying a programmatic $100\text{ ms}$ processing delay (cv2.waitKey(100)), we successfully simulated a live telemetry video feed right on the desktop to demonstrate full tracking viability.

# Results & Discussion

Strengths

The system demonstrates high portability and modularity by utilizing dynamic, relative path routing that allows the codebase to execute on any host computer without manual directory reconfiguration. Integrating YOLOv8's native ByteTrack module eliminates fragile third-party dependencies, guaranteeing stable tracking associations directly out of the box. Furthermore, the creative sequencing workaround successfully simulates a continuous live video stream from independent static image files, proving full end-to-end pipeline viability.

Limitations

Operating the heavy neural network framework entirely on a CPU architecture introduces a significant processing bottleneck, restricting inference speeds below real-time deployment requirements. Due to the abbreviated 10-epoch training constraints, the model struggles with low recall scores, occasionally dropping detections on highly condensed clusters or microscopic background assets. Additionally, because the test dataset consists of discontinuous scenes rather than a real video file, the tracking IDs naturally reset when transitioning between completely different drone environments.

Challenges Faced

The primary technical challenge involved a recurring FileNotFoundError caused by transferring absolute workspace directories between different development environments, which was resolved by implementing Python's automated os.path path-builders. We also encountered system freezes when closing the OpenCV GUI window via the standard window 'X' button, requiring the integration of dynamic window property checks into our key-listener loops to force clean application terminations. Finally, initial object dropouts were successfully mitigated by tuning the inference confidence threshold down to a precise conf=0.15, optimizing the model's sensitivity to smaller targets.

Evaluation Metrices

<img width="523" height="112" alt="image" src="https://github.com/user-attachments/assets/0b603c29-2dc8-447a-8906-9dc54179ec27" />

