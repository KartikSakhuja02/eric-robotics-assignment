# Object Detection and Distance Estimation

**Kartik Sakhuja**

ERIC Robotics technical assignment · 14 October 2026

## What I built

I built a pipeline that detects cones, barriers and stop signs, estimates their depth from the camera, and draws the results on the input image. I fine-tuned a pretrained YOLO11n model and compared its performance on CPU and GPU. I also exported the trained model to OpenVINO and tested INT8 quantization.

I used dataset images for training, evaluation and the demonstration. For nuImages, the camera parameters came from the dataset metadata. For BDD100K, I used explicit camera assumptions and marked those estimates in the output.

The final baseline achieved a test mAP50 of **0.7136** and mAP50–95 of **0.4567**. Including depth estimation and drawing, GPU FP32 ran at **30.92 FPS**. The OpenVINO INT8 model ran at **32.01 FPS** on the CPU. These timings exclude reading images from disk and encoding video.

## Choosing the model and data

I used YOLO11n because the assignment required transfer learning and my laptop had an RTX 2050 with 4 GB of VRAM. Its small model size made it a practical starting point for both training and the CPU comparison. I initialized it from COCO-pretrained weights and adapted the detector to three output classes:

```text
0: cone
1: barrier
2: stop_sign
```

BDD100K required some care. Its detection annotations contain a general `traffic sign` category, which includes signs other than STOP. Using that category directly would have trained the wrong class. I used BDD images for reviewed stop-sign examples and added nuImages for its explicit cone and barrier annotations:

| Target class | Source annotation |
|---|---|
| Cone | nuImages: `movable_object.trafficcone` |
| Barrier | nuImages: `movable_object.barrier` |
| Stop sign | Reviewed BDD traffic-sign boxes, with supplemental stop boxes from nuImages |

This also gave the distance-estimation part access to per-image camera intrinsics and camera-to-vehicle transforms from nuImages.

I ran the project in a Python 3.11.9 virtual environment on Windows, using PyTorch 2.10.0 with CUDA 12.8, Ultralytics 8.4.177 and OpenCV 5.0. The export and quantization stages used OpenVINO 2026.4.1 and NNCF 3.4.0. The dependency versions are recorded in `requirements.txt` and `requirements.lock.txt`.

## Preparing the dataset with limited storage

I had around 26 GB of free disk space when I started preparing the downloaded data. Extracting the complete nuImages samples archive would have used too much of that space, so I selected the frames first and extracted only the required files.

The preparation script reads the metadata tables, identifies CAM_FRONT keyframes containing cones or barriers, and builds a selection list. It uses seed 42 and caps the selection at 20 frames per recording. The selected frames came from 252 training logs and 64 held-out logs.

The final nuImages subset contained 1,000 frames from the official training partition and 300 from its validation partition. The extraction added approximately 1.01 GB of metadata and 220 MB of selected images. The original archives were retained. The extractor checks available space before writing a file and reserves 15 GiB during extraction. That reserve applies to the extraction process; Windows and other applications can still change the available space.

I kept the image preparation stages separate from the final dataset. The scripts attempt to hardlink images between stages and fall back to copying when linking is unavailable. Labels are written separately so that editing one set of annotations does not change another.

The box conversion uses the YOLO detection format:

```text
class_id centre_x centre_y width height
```

Coordinates are normalized by the image dimensions. Boxes are clipped to the image boundary, and invalid boxes are rejected. The conversion also retains each frame's recording ID and camera metadata.

### Stop-sign annotations

I used a model-assisted workflow to find likely stop signs in the BDD candidate pool. A pretrained COCO detector proposed stop boxes, which were matched to the existing traffic-sign annotations. The candidate crops were then visually reviewed. No-entry signs, yield signs, arrows, sign backs and ambiguous crops were excluded. Recognizable red octagonal stop faces, including raised-hand stop signs, were retained.

The accepted BDD boxes kept their source coordinates; the review supplied the stop-sign subclass. An additional audit found 14 stop signs in the selected nuImages frames. Those supplemental boxes came from the proposal model, with visual verification of the class. Their localization therefore has a different level of independence from the original nuImages cone and barrier annotations.

The first stop-sign holdout was too small, so I expanded the BDD validation candidate search before training. The final dataset used 70 unique BDD images with 85 reviewed stop boxes, alongside the 1,300 nuImages frames.

This selection process has a limitation: a teacher model can miss signs before they ever reach review. The stop-sign results therefore describe this reviewed subset, rather than all stop signs in either source dataset.

### Splitting and class imbalance

I split the data by recording group to keep frames from the same recording together. Official nuImages training logs remained in training. The selected official validation logs were assigned to validation or test using a stable hash of the log ID. BDD training examples remained in training, while its validation examples were divided between validation and test by recording ID.

Stop signs were much less common than cones and barriers. I repeated the 28 BDD training images eight times to give that class more exposure during training. This produced 196 additional training entries. The training set therefore has **1,224 entries but 1,028 unique images**. Validation and test contain no repeated copies.

| Split | Image entries | Cone boxes | Barrier boxes | Stop-sign boxes |
|---|---:|---:|---:|---:|
| Training | 1,224 | 4,152 | 5,630 | 301 |
| Validation | 177 | 578 | 892 | 32 |
| Test | 165 | 783 | 795 | 18 |

The 301 training stop boxes include repeated examples; there are 49 distinct stop boxes in the training split. Oversampling changes how often the model sees those examples, but it does not add new scenes.

Before training, I checked image decoding, class IDs, box dimensions, missing labels and group separation. The final inspection reported no structural errors or recording-group overlap between splits. All selected images contain a target object, so this dataset does not establish performance on entirely negative scenes.

## Fine-tuning the detector

I first ran a one-epoch trial to check that the dataset and CUDA training path worked. After that completed, I trained the baseline for 30 epochs.

| Setting | Value |
|---|---|
| Starting weights | COCO-pretrained `yolo11n.pt` |
| Training image size | 640 × 640 |
| Batch size | 4 |
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Final learning-rate fraction | 0.1 |
| Weight decay | 0.0005 |
| Precision | Automatic mixed precision |
| Seed | 42 |
| Data workers | 2 |
| Early-stopping patience | 8 epochs |

The detector was fine-tuned using the framework's detection loss. The DFL projection remained frozen by the framework. Training used mosaic augmentation, which was disabled for the final five epochs. I selected the best checkpoint using validation fitness and kept the test split for final evaluation.

The run completed all 30 epochs in approximately 37 minutes. I saved the best checkpoint as `models/baseline.pt`. The training arguments and epoch results are retained in [training_args.yaml](outputs/metrics/training_args.yaml) and [training_results.csv](outputs/metrics/training_results.csv).

The measured run used two workers. The training script also supports zero workers for lower host-memory use, checkpoint resumption, a time limit, and a free-space stop at an epoch boundary. These controls make another run easier to manage on the same laptop.

## Estimating depth from the camera

I kept the distance calculation separate from the detector. The detector provides the class and bounding box; the geometry module uses that box together with the available camera parameters.

The quantity displayed is **camera-forward depth, Z**, measured along the optical axis. It is different from straight-line range for an object far to the side of the image. I used this definition consistently in the overlays and saved JSON results.

### Cones and barriers with supplied calibration

For a cone or barrier, I use the bottom-centre of the detected box as an approximate ground-contact point. The pixel is undistorted using the supplied camera matrix and distortion coefficients, producing a camera ray:

```text
r = [x_n, y_n, 1]
```

Let `R` rotate from camera coordinates to the vehicle's coordinate frame, and let `t` be the camera translation in that frame. A point on the ray is:

```text
p_vehicle = t + Z R r
```

Assuming a local road plane at vehicle z = 0, the depth follows from the vertical component:

```text
Z = -t_z / (R r)_z
```

Because the third component of the camera ray is 1, the solved parameter is camera-forward depth in metres.

This method depends on the road being approximately flat and the box bottom representing a visible ground contact. A slope, curb, occluded base or loose box can move that contact point and affect the estimate. Rays that do not intersect the ground in front of the camera, clipped contacts and estimates outside the configured range return `N/A`.

### Stop signs and the size-based fallback

For stop signs, I estimate depth from the projected height of the sign face. The same calculation is available for cones and barriers when camera extrinsics are unavailable:

```text
Z ≈ H / |y_n,bottom - y_n,top|
```

With a pinhole camera and negligible distortion, this becomes:

```text
Z ≈ f_y H / h_pixels
```

Here, `H` is the assumed physical height, `f_y` is the vertical focal length in pixels, and `h_pixels` is the box height. For example, a 0.75 m sign face spanning 200 pixels with `f_y = 800` gives an estimated depth of 3 m. This is a calculation example, not a measurement from a dataset image.

The configuration uses these illustrative height priors:

| Class | Assumed height |
|---|---:|
| Cone | 0.50 m |
| Barrier | 0.90 m |
| Stop-sign face | 0.75 m |

These sizes were not measured in the source images. Changing a height prior changes the estimated depth proportionally, and tilted or partly hidden objects introduce further error. Boxes less than 10 pixels tall and vertically clipped objects return `N/A`.

### Images without calibration

BDD camera calibration was unavailable in this workflow. I used a 90° horizontal field of view, square pixels, a centred principal point and zero distortion as the fallback camera model. For image width `W`, the assumed focal length is:

```text
f_x = f_y = W / (2 tan(90° / 2))
```

Those estimates carry `*` in the overlay and `calibration_source: assumed_fov` in the JSON. The `~` before a depth value marks it as approximate. The fallback can also be disabled in `configs/camera.json`, in which case missing calibration produces `N/A`.

I did not calculate distance MAE or RMSE because the selected annotations provide no object-distance ground truth. The geometry tests check known constructed cases, resizing and invalid inputs; they do not establish real-world distance accuracy.

## Evaluating detection accuracy

I evaluated the saved checkpoint on the 165-image test split using batch size 1, 640 × 640 input, NMS IoU 0.7 and a maximum of 50 detections per image. AP evaluation uses a confidence floor of 0.001 to build the precision-recall curve. The demonstration uses a confidence threshold of 0.25.

The baseline test results were:

| Class | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|
| Cone | 0.7550 | 0.5377 | 0.6052 | 0.2909 |
| Barrier | 0.6624 | 0.5182 | 0.5517 | 0.2890 |
| Stop sign | 0.8778 | 1.0000 | 0.9839 | 0.7902 |

Overall mAP50 was **0.7136**, and mAP50–95 was **0.4567**. Precision and recall in this table use the framework's best mean-F1 confidence, so they are not fixed-threshold scores at 0.25.

The class breakdown matters here. Cone and barrier recall was around 0.52–0.54, which leaves substantial room for improvement. Stop-sign performance was higher, but the test set contains only 18 stop boxes and the candidate selection used a pretrained teacher. I would not use that stop-sign score as evidence of broad performance on unseen driving scenes. mAP is a detection metric, not a percentage of images classified correctly.

The full results and curves are stored in [outputs/metrics](outputs/metrics) and [outputs/evaluation](outputs/evaluation).

## Testing the optimization options

I compared PyTorch FP32 on CPU and GPU, PyTorch FP16 on GPU, and OpenVINO FP32 and INT8 on CPU. The OpenVINO exports use a fixed input shape of `1 × 3 × 640 × 640`, with NMS handled by the common prediction pipeline.

For INT8 calibration, I selected 320 unique training images. This included the reviewed BDD stop-sign training images and excluded their repeated copies. Validation and test images were kept out of calibration. I then evaluated each runtime on the same test split.

To measure speed, I preloaded the same 16 test images for every configuration. Each configuration had a separate initialization, 20 warmup iterations, and three repeats of 100 timed iterations. CUDA was synchronized around GPU timings. FPS was calculated from the mean latency across the timed iterations.

I recorded two timings. **Detector FPS** includes preprocessing, the network and NMS. **Annotated FPS** also includes the depth calculation and drawing the boxes and labels. Both exclude disk I/O, image/video decoding, video encoding, model loading and camera capture.

The hardware was an AMD Ryzen 7 7435HS with 16 logical CPUs and an NVIDIA RTX 2050 with 4 GB VRAM. PyTorch and OpenCV used four CPU threads. OpenVINO used LATENCY mode and reported eight inference threads, so the PyTorch/OpenVINO comparison also involves a runtime and thread-setting change.

| Runtime | mAP50 | mAP50–95 | Detector FPS | Annotated FPS | Artifact size |
|---|---:|---:|---:|---:|---:|
| PyTorch CPU FP32 | 0.7135 | 0.4567 | 14.42 | 13.68 | 5.23 MiB |
| PyTorch GPU FP32 | 0.7136 | 0.4567 | 34.84 | 30.92 | 5.23 MiB |
| PyTorch GPU FP16 | 0.7116 | 0.4477 | 32.31 | 28.84 | 5.23 MiB |
| OpenVINO CPU FP32 | 0.7135 | 0.4567 | 26.32 | 24.37 | 10.31 MiB |
| OpenVINO CPU INT8 | 0.7088 | 0.4478 | 35.51 | 32.01 | 3.27 MiB |

The INT8 model was **1.31 times faster** than OpenVINO FP32 for the annotated pipeline. Its mAP50–95 decreased by **0.0089**, or about **0.89 percentage points**, and its artifact size fell by about 68%. That is the comparison I use to describe the quantization benefit, since both models run through OpenVINO.

FP16 did not improve GPU speed in this run. It was slightly slower than FP32 and had a lower test mAP50–95, so I kept GPU FP32 as the demonstration default. These measurements apply to this laptop and workload; I did not benchmark a Jetson or a deployed robot.

The PyTorch rows use the same checkpoint file with different compute settings. Their identical file sizes do not mean that FP32 and FP16 inference use the same precision. OpenVINO artifact sizes include the XML graph, binary weights and metadata.

![Measured speed and accuracy](outputs/benchmark_comparison.png)

## Demonstration and bird's-eye view

The final demonstration uses 16 held-out images selected with a fixed seed, including both dataset sources. The selection was not filtered by prediction correctness. The output contains annotated images, a gallery, a JSON file with boxes and depth estimates, and a 32-second slideshow.

The slideshow contains independent images displayed at 2 FPS. Its playback rate is separate from the measured inference FPS, and it does not demonstrate temporal tracking.

![Detection and depth examples](outputs/demo/gallery.jpg)

I also added a road-plane bird's-eye projection using nuImages calibration. After undistorting the image, I used the camera-to-vehicle transform to construct the ground-plane homography:

```text
H_ground_to_image = K [Rᵀe_x, Rᵀe_y, -Rᵀt]
```

The displayed region covers vehicle-forward X = 2–30 m and lateral Y = −10 to +10 m. Road-plane points follow the transform; elevated objects stretch because their pixels do not lie on the assumed plane. The projection is saved in [outputs/bev/road_bev.jpg](outputs/bev/road_bev.jpg), together with its geometry metadata.

## Code structure and checks

I separated data preparation, training, evaluation, geometry, inference and benchmarking into individual modules. This lets the trained detector be reused without running the preparation or training stages again.

| Module | Responsibility |
|---|---|
| `prepare_archives.py` | Select and extract the bounded nuImages subset |
| `prepare_data.py` | Extract BDD candidates and split grouped annotations |
| `nuimages_to_yolo.py` | Convert boxes and retain camera metadata |
| `build_dataset.py` | Merge reviewed annotations and produce the final splits |
| `inspect_data.py` | Validate images, labels and recording-group separation |
| `train.py`, `evaluate.py` | Fine-tune the detector and report held-out accuracy |
| `distance.py`, `calibrate.py` | Load camera data and calculate geometric depth |
| `infer.py` | Draw detections, save JSON and produce the demonstration |
| `export.py`, `benchmark.py` | Export the models and measure latency |
| `bev.py`, `make_report.py` | Produce the optional projection and measured report |

The project passed 17 automated tests. These cover archive path handling and disk reserve checks, annotation conversion, normalized-label validation, grouped splitting, training-only quantization calibration, depth calculations and the road-plane homography. The geometry checks include a known 3 m size-based case, a known 5 m ground-plane case, image-resizing consistency and rejection of unsuitable rays or clipped boxes.

Final artifact checks confirmed that the five runtime configurations had saved metrics, the quantization images belonged to training, recording groups did not cross splits, and the demo video contained the expected 64 frames at 2 FPS. A checksum of the final checkpoint is recorded in [final_verification.json](outputs/metrics/final_verification.json).

The original downloaded archives stayed outside Git. At the end of the run, about 22.2 GiB of disk space remained free. The uploaded repository contains the code, dependency versions, reviewed annotation metadata, trained and exported models, measured results and demonstration artifacts.

## What I would improve next

My first priority would be the data. I would add more unique cone and barrier examples, more independently annotated stop signs, and scenes containing no target objects. The current recall and small stop-sign holdout make those additions more useful than changing the model based only on the overall mAP.

For distance estimation, I would collect matched images and reference distances, measure the deployment camera's calibration, and replace the illustrative object-size assumptions where possible. That would allow actual distance-error measurements and a check of how road slopes and occlusion affect the ground-plane method.

I would then repeat the latency and accuracy tests on the intended edge device. The laptop measurements show the effect of the tested options here, but do not establish their performance on different hardware. Stereo depth, optical flow and pruning were not implemented in this submission.

## Running the project

The [README](README.md) contains environment setup and the full data-reproduction commands. After preparing the dataset and environment, the main stages can be run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.train --epochs 30 --batch 4 --imgsz 640 --name baseline --device 0 --workers 2
Copy-Item runs/baseline/weights/best.pt models/baseline.pt
.\.venv\Scripts\python.exe -m src.evaluate --device 0 --precision 32 --name baseline_test
.\.venv\Scripts\python.exe -m src.infer --device 0 --max-images 16
.\.venv\Scripts\python.exe -m src.export
.\.venv\Scripts\python.exe -m src.bev
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The saved models can be used directly for inference. Re-running training is optional; when a run folder already exists, Ultralytics creates a suffixed folder, so a new checkpoint should be copied from that new folder. The complete CPU/GPU evaluation and benchmark commands are in the README.

Dataset and model attribution, including the source terms for the included examples, is recorded in [DATASET_NOTICES.md](DATASET_NOTICES.md).

**Contact:** Kartik Sakhuja · 9325180298 · kartik.sakhuja2004@gmail.com

**Repository:** [KartikSakhuja02/eric-robotics-assignment](https://github.com/KartikSakhuja02/eric-robotics-assignment)
