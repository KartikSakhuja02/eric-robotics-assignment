# Submission steps

Author: Kartik Sakhuja

Deadline: **15 October 2026 at 15:14 IST**.

1. Open `README.md` and `outputs/report.md`. Understand the model, the dataset labels, the measured results and the geometric assumptions. The central limitation is that distances have no ground-truth accuracy score; BDD estimates assume calibration and sign/object heights are unmeasured priors.
2. Check `outputs/demo/gallery.jpg` and play `outputs/demo/dataset_slideshow.mp4`. These show held-out dataset images with detections and depth overlays. The slideshow is not a video-tracking or real-time capture experiment.
3. Review `outputs/bev/road_bev.jpg` for the optional calibrated road-plane homography. Raised objects stretch because the transformation assumes ground-plane points.
4. Check `outputs/metrics/` for detector accuracy, per-class scores, CPU/GPU latency, FP32/INT8 changes, quantization calibration and dataset provenance. Keep the small stop-sign holdout and teacher-assisted label limitations in mind when discussing results.
5. Run `.\.venv\Scripts\python.exe -m unittest discover -s tests -v` and `.\.venv\Scripts\python.exe -m src.inspect_data --root data/processed_v2 --output outputs/dataset_check_v2` if you need to verify the local workspace again.
6. Confirm the GitHub repository contains the code, README, dependency versions, final checkpoint, optimized exports, review manifests, measured report and demo. Downloaded archives, raw datasets, the virtual environment and caches must remain excluded.
7. Open `https://github.com/KartikSakhuja02/eric-robotics-assignment` after publishing and check that README images/links open and the final files appear. The recruitment email asks for a personal repository, despite the older contradictory sentence in the attached assignment.
8. Reply in the original recruitment email thread to `hr@ericrobotics.com`. Include full name **Kartik Sakhuja**, GitHub username **KartikSakhuja02**, and repository link **https://github.com/KartikSakhuja02/eric-robotics-assignment**.
9. Include these four addresses in CC: `parmeet.software@ericrobotics.com`, `shubham.petkar@ericrobotics.com`, `abhishek.singh@ericrobotics.com`, `muskan.sohaney@ericrobotics.com`. Send before the deadline. The assistant has not sent any email.

## Suggested email body

Dear ERIC Robotics Talent Team,

I have completed the Machine Learning technical assignment. My submission details are:

Full name: Kartik Sakhuja
GitHub username: KartikSakhuja02
Assignment repository: https://github.com/KartikSakhuja02/eric-robotics-assignment

The repository includes the implementation, setup and reproduction instructions, trained model, dataset-based demonstration, geometric distance estimation, and measured optimization comparisons. My contact details are included in the README.

Regards,
Kartik Sakhuja
9325180298
kartik.sakhuja2004@gmail.com
