# Dataset and model attribution

Annotated examples, the slideshow, the road projection and calibration metadata derive from their source datasets. Overlays, class-subtype review and geometric projections were added for this assignment. Source terms continue to apply to those artifacts.

- **BDD100K:** Fisher Yu and collaborators, Berkeley DeepDrive. [Official project](https://bdd-data.berkeley.edu/), [dataset terms](https://doc.bdd100k.com/license.html), [paper](https://arxiv.org/abs/1805.04687). Input was the user-downloaded `solesensei_bdd100k.zip` mirror; its bytes were not compared against an official release checksum. Image filenames and original splits are retained in the manifests.
- **nuImages:** Motional. [Official dataset](https://www.nuscenes.org/nuimages). The license and terms supplied with the downloaded archive are reproduced in [third_party/nuimages_LICENSE.txt](third_party/nuimages_LICENSE.txt). Dataset-derived media and metadata remain under those terms, including the stated CC BY-NC-SA 4.0 license and additional terms.
- **YOLO11n:** Ultralytics. [Official model documentation](https://docs.ultralytics.com/models/yolo11/) and [licensing](https://www.ultralytics.com/license). Fine-tuned checkpoints and optimized exports derive from its pretrained model. Source weights were initialized from the official Ultralytics release asset.

  The upstream license distributed with the installed Ultralytics package is retained in [third_party/ultralytics_LICENSE.txt](third_party/ultralytics_LICENSE.txt).
- **OpenVINO / NNCF:** Intel and contributors; dependency licenses apply to the installed libraries. [OpenVINO](https://github.com/openvinotoolkit/openvino), [NNCF](https://github.com/openvinotoolkit/nncf).

The raw datasets are not included in this repository. Demo filenames, dataset selection manifests and calibration tokens preserve provenance.
