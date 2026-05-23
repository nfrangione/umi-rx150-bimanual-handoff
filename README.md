# ISL Training Bimanual Handoff of Cube using UMI on RX150: Collection → SLAM → Training -> Evaluation

Project Team:
- Dr. Sudhir Shrestha
- Kai Nunnemaker
- Nick Frangione
- Antonio Lozoida Cigala

This README documents the end-to-end process used for the bimanual RX150 cube hand-off project:

1. Collect UMI/GoPro demonstration data.
2. Run the UMI SLAM/data pipeline.
3. Train a bimanual diffusion policy on NRP.
4. Evaluate the trained checkpoint offline against the zarr dataset.
5. Copy results/checkpoints back locally.

The project is based on Stanford's Universal Manipulation Interface (UMI) pipeline and the ISL cube hand-off project wrapper. The UMI framework is designed for data collection from human demonstrations and policy learning using image observations, robot/gripper state, and relative action representations.

### OUR REPLAY BUFFER:
- https://drive.google.com/file/d/1eWRg2ABzHR9hCOEbZNDyeoMLKhq25S_-/view?usp=sharing

### OUR LATEST CHECKPOINT: 
- https://drive.google.com/file/d/1KjVi1yoh-ULo72daNIsR49pvvbQyotnJ/view?usp=sharing

---

## 0. Repositories Used

### Stanford UMI repository
Main upstream repository:
https://github.com/real-stanford/universal_manipulation_interface


Important files/directories from UMI:
run_slam_pipeline.py
scripts_slam_pipeline/
diffusion_policy/
example/calibration/

Important pipeline scripts:
scripts_slam_pipeline/00_process_videos.py
scripts_slam_pipeline/01_extract_gopro_imu.py
scripts_slam_pipeline/02_create_map.py
scripts_slam_pipeline/03_batch_slam.py
scripts_slam_pipeline/04_detect_aruco.py
scripts_slam_pipeline/05_run_calibrations.py
scripts_slam_pipeline/06_generate_dataset_plan.py

### ISL cube hand-off repository

Project wrapper / lab-specific repository:

https://github.com/profshrestha/isl_umi_cube_hand_off

This repository contains lab-specific job files, Docker image references, NRP instructions, and project organization for the RX150 bimanual hand-off task.

---

## 0a. Clone Stanford Repository and Make Sure Environment Set-up is According to their Instructions

---

## 1. Data Collection

### 1.1 Hardware setup

The project uses:
- Two RX150 robot arms
- Two GoPros
- Bimanual cube/cup/object hand-off task
- UMI-style visual observations

The dataset used for the reported training of our model was:
- /workspace/data/may326.zarr.zip

The trained checkpoint was:
- /workspace/checkpoints/may326_002/checkpoints/latest.ckpt


### 1.2 Collect raw GoPro videos
1. Scan Hmdi output qrcode
2. Time synce using this qr at beginning and then every 10 videos filmed 
3. For each session, collect:
    a. Mapping video
    b. Gripper/ArUco calibration video(s)
    c. Demonstration Videos on UMI Grippers (50-100)

Recommended raw folder layout for future run of SLAM Pipeline:

data/raw/session1/
  raw_videos/
    mapping.mp4
    gp1_*.MP4
    gp2_*.MP4


### 1.3 Important collection notes
- The mapping video is critical. It should cover the full workspace, not only half of it.
- A good mapping video should include:
    - Full left robot workspace
    - Full right robot workspace
    - Center hand-off area
    - All object pick/place regions
    - Slow camera motion
    - Good lighting
    - Low motion blur
    - Static background features
- If the mapping video only covers part of the workspace, and the demo frames do not match the map well enough, later SLAM may fail with messages like:

Relocalization() failed.
Fail to track local map!
n_lost_frames=...

---

## 2. Local Video Preparation

### 2.1 Activate UMI environment
- Only tested on Ubuntu 22.04
- Install docker following the official documentation and finish linux-postinstall.
- Install system-level dependencies:
    - $ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
    - $ mamba env create -f conda_environment.yaml
- Clone Stanford Repository: 
    - $ git clone https://github.com/real-stanford/universal_manipulation_interface.git
- Then inside universal_manipulation_interface Directory:
    - $ conda activate umi


### 2.2 Step 00: process raw videos

- python scripts_slam_pipeline/00_process_videos.py ../data/raw/session1
- Purpose:
    - Reads raw GoPro videos
    - Creates structured demo folders
    - Moves/renames videos into UMI expected layout


Expected output:

data/raw/session1/demos/
  mapping/
    raw_video.mp4
  demo_*/
    raw_video.mp4

### 2.3 Step 01: extract GoPro IMU

python scripts_slam_pipeline/01_extract_gopro_imu.py ../data/raw/session1

Purpose:

Extracts GoPro IMU data from videos
Writes imu_data.json for mapping and each demo

Expected files:

data/raw/session1/demos/mapping/imu_data.json
data/raw/session1/demos/demo_*/imu_data.json

Common issue:

FileNotFoundError: docker

The upstream UMI scripts may call Docker internally for GoPro IMU extraction. On local Ubuntu, Docker should be installed and running.

Check Docker:

docker run hello-world

---

## 3. Run the SLAM Pipeline

There are two ways to run SLAM:

1. Full upstream UMI pipeline.
2. Manually run individual stages for debugging.

The upstream entry point is:

python run_slam_pipeline.py ../data/raw/session1


The upstream script is intended to run the SLAM pipeline stages in order.

### 3.1 Step 02: create map

Mapping builds a reusable visual/visual-inertial map from the mapping video.

Equivalent operation:

```bash
python scripts_slam_pipeline/02_create_map.py \
  --input_dir ../data/raw/session1/demos/mapping \
  --map_path ../data/raw/session1/demos/mapping/map_atlas.osa
```

Purpose:

```text
Uses mapping/raw_video.mp4 and mapping/imu_data.json
Runs ORB-SLAM3
Creates map_atlas.osa
Creates mapping_camera_trajectory.csv
```

Expected output:

```text
data/raw/session1/demos/mapping/map_atlas.osa
data/raw/session1/demos/mapping/mapping_camera_trajectory.csv
```

Important file:

```text
map_atlas.osa
```

This is the map used by batch SLAM for every demo.

### 3.2 Step 03: batch SLAM

Batch SLAM localizes every demonstration video against the map from Step 02.

Purpose:

For each demo_*/raw_video.mp4:
  load the mapping map_atlas.osa
  use the demo's imu_data.json
  generate camera_trajectory.csv

Expected output per demo:

data/raw/session1/demos/demo_*/camera_trajectory.csv

This file is essential. It contains the camera pose trajectory used downstream for calibration and dataset generation.

Check success count:

ls ../data/raw/session1/demos/demo_*/camera_trajectory.csv | wc -l

Common SLAM issues:

Relocalization() failed.
Fail to track local map!


Likely causes:

Mapping video did not cover full workspace
Demo video moves outside mapped region
Motion blur
Lighting changes
Weak visual features
Bad IMU/video timing


### 3.3 Step 04: detect ArUco / gripper markers

Run only Step 04:

```bash
python scripts_slam_pipeline/04_detect_aruco.py \
  --input_dir ../data/raw/session1/demos \
  --camera_intrinsics example/calibration/gopro_intrinsics_2_7k.json \
  --aruco_yaml example/calibration/aruco_config.yaml
```

Purpose:

```text
Detects ArUco/QR markers in demo videos
Produces marker detections used for gripper/camera calibration
```

Common issue:

```text
cv2.aruco missing
```

Fix:

```bash
pip uninstall -y opencv-python
pip install opencv-contrib-python
```

### 3.4 Step 05: run calibrations

Run only Step 05:

```bash
python scripts_slam_pipeline/05_run_calibrations.py ../data/raw/session1
```

Purpose:

```text
Uses camera trajectories + marker detections
Computes transformations needed to align camera, gripper, and robot frames
```

### 3.5 Step 06: generate dataset plan

Run:

```bash
python scripts_slam_pipeline/06_generate_dataset_plan.py --input ../data/raw/session1
```

Purpose:

```text
Pairs demos
Builds metadata for dataset generation
Plans which segments become training episodes
```

Common issue:

```text
ZeroDivisionError: float division by zero
```

Likely cause:

```text
No valid paired demos
GoPro timestamps do not align
Folder names/timestamps mismatch
```

For bimanual GoPros, exact timestamp matching may be too strict if cameras started a fraction of a second apart.

### 3.6 Step 07: generate replay buffer / zarr dataset

The final pipeline stage creates the training dataset:

```text
may326.zarr.zip
```

Expected final output:

```text
/workspace/data/may326.zarr.zip
```

The zarr contains processed demonstrations and arrays used by the UMI dataset class.

---

### OUR REPLAY BUFFER:
- https://drive.google.com/file/d/1KjVi1yoh-ULo72daNIsR49pvvbQyotnJ/view?usp=sharing

## 4. Upload / Store Dataset on NRP

The NRP namespace used:

```text
ssu-isl-bimanual-dexterity
```

PVC used:

```text
umi-data-pvc
```

Inside NRP pods, the PVC is mounted at:

```text
/workspace
```

Dataset location:

```text
/workspace/data/may326.zarr.zip
```

Use a staging pod to inspect/copy files:

```bash
kubectl apply -f pvc-stage.yaml -n ssu-isl-bimanual-dexterity
kubectl wait --for=condition=Ready pod/pvc-stage -n ssu-isl-bimanual-dexterity --timeout=120s
kubectl exec -it pvc-stage -n ssu-isl-bimanual-dexterity -- bash
```

Important: the PVC is ReadWriteOnce, so do not keep `pvc-stage` running while training/eval jobs run.

Delete it before training/eval:

```bash
kubectl delete pod pvc-stage -n ssu-isl-bimanual-dexterity --ignore-not-found
```

---

## 5. Training on NRP

### 5.1 Training job file

Key file:

```text
train-job.yaml
```

Important fields:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: umi-train-may326-v2
  namespace: ssu-isl-bimanual-dexterity
```

Image:

```yaml
image: profshrestha/umi-rx150:latest
```

Resources:

```yaml
resources:
  requests:
    cpu: "8"
    memory: "48Gi"
    nvidia.com/gpu: "1"
  limits:
    cpu: "8"
    memory: "48Gi"
    nvidia.com/gpu: "1"
```

PVC mount:

```yaml
volumeMounts:
  - name: workspace
    mountPath: /workspace

volumes:
  - name: workspace
    persistentVolumeClaim:
      claimName: umi-data-pvc
```

### 5.2 Training command

Inside the job:

```bash
DATASET=/workspace/data/may326.zarr.zip

cd /opt/umi

python train.py \
  --config-name=train_diffusion_unet_umi_bimanual_workspace \
  task.dataset.dataset_path=$DATASET \
  hydra.run.dir=/workspace/checkpoints/may326_002 \
  dataloader.batch_size=4 \
  val_dataloader.batch_size=1 \
  dataloader.num_workers=1 \
  val_dataloader.num_workers=1
```

Critical override:

```bash
task.dataset.dataset_path=$DATASET
```

Do not use:

```bash
task.dataset_path=$DATASET
```

The wrong override causes Hydra to keep the default dataset path.

### 5.3 Training output

Checkpoint directory:

```text
/workspace/checkpoints/may326_002
```

Latest checkpoint:

```text
/workspace/checkpoints/may326_002/checkpoints/latest.ckpt
```

Check checkpoint:

```bash
kubectl apply -f pvc-stage.yaml -n ssu-isl-bimanual-dexterity
kubectl wait --for=condition=Ready pod/pvc-stage -n ssu-isl-bimanual-dexterity --timeout=120s

kubectl exec -it pvc-stage -n ssu-isl-bimanual-dexterity -- bash -lc '
ls -lh /workspace/checkpoints/may326_002/checkpoints/latest.ckpt
'
```

Copy checkpoint to Mac:

```bash
mkdir -p ./checkpoints/may326_002

kubectl cp \
  ssu-isl-bimanual-dexterity/pvc-stage:/workspace/checkpoints/may326_002/checkpoints/latest.ckpt \
  ./checkpoints/may326_002/latest.ckpt
```

This warning is normal:

```text
tar: Removing leading '/' from member names
```

### 5.4 Training notes

The config had:

```yaml
checkpoint_every: 10
val_every: 5
rollout_every: 10
sample_every: 5
```

So checkpoints may not appear until epoch 10 unless the config is changed.

The training run used safer dataloader settings because previous settings created too many workers:

```text
This DataLoader will create 32 worker processes in total.
suggested max number of worker in current system is 8.
```

---

### OUR LATEST CHECKPOINT: 
- https://drive.google.com/file/d/1KjVi1yoh-ULo72daNIsR49pvvbQyotnJ/view?usp=sharing

## 6. Offline Evaluation Against Zarr

The requested evaluation was:

```text
Feed observations from the zarr file into the trained policy.
Compare policy predictions with ground-truth recorded actions.
Report the result.
```

### 6.1 Why the fixed evaluator uses UMI's dataset class

A manual evaluator failed because it read raw zarr keys directly.

Raw zarr keys included:

```text
robot0_eef_pos
robot0_eef_rot_axis_angle
robot0_gripper_width
robot1_eef_pos
robot1_eef_rot_axis_angle
robot1_gripper_width
camera0_rgb
camera1_rgb
```

But the trained model expects processed observation keys including:

```text
robot0_eef_pos_wrt1
robot0_eef_rot_axis_angle_wrt1
robot1_eef_pos_wrt0
robot1_eef_rot_axis_angle_wrt0
```

These are produced by the UMI dataset/preprocessing code.

Therefore the fixed evaluator uses:

```python
cfg.task.dataset.dataset_path = args.zarr
dataset = hydra.utils.instantiate(cfg.task.dataset)
sample = dataset[idx]
obs = sample["obs"]
gt_action = sample["action"]
```

This is the key reason the fixed evaluation works.

### 6.2 Key eval file

```text
eval_from_zarr.py
```

### 6.3 Important eval functions

#### `load_workspace_policy(checkpoint_path, dataset_path, device)`

Purpose:

```text
Load checkpoint with dill
Rebuild the Hydra workspace
Override dataset path
Load model weights
Use EMA model if available
Return policy
```

Key implementation idea:

```python
with open(checkpoint_path, "rb") as f:
    payload = torch.load(f, pickle_module=dill, map_location=device)

cfg = payload["cfg"]
cfg.task.dataset.dataset_path = dataset_path

workspace_cls = hydra.utils.get_class(cfg._target_)
workspace = workspace_cls(cfg)
workspace.load_payload(payload)

policy = workspace.ema_model if workspace.ema_model is not None else workspace.model
```

#### `load_training_dataset(cfg)`

Purpose:

```text
Instantiate the same UMI dataset object used during training
Verify sample contains obs and action
Print obs/action shapes
```

Important output from successful run:

```text
Dataset length: 44900
Sample keys: ['obs', 'action']
Obs keys include robot0_eef_pos_wrt1 and robot1_eef_pos_wrt0
Action shape: (16, 20)
```

#### `to_device_batch(x, device)`

Purpose:

```text
Convert numpy arrays/tensors from dataset sample into batched tensors
Move them to GPU
```

#### `extract_policy_action(policy_output)`

Purpose:

```text
Extract the first predicted action vector from policy output
```

The policy output had:

```text
Policy output keys: ['action', 'action_pred']
```

#### `extract_ground_truth_action(sample)`

Purpose:

```text
Extract the first ground-truth action vector from sample["action"]
```

The dataset action shape was:

```text
(16, 20)
```

Meaning:

```text
16-step action horizon
20-dimensional action vector
```

#### `evaluate(policy, dataset, device, num_samples, seed)`

Purpose:

```text
Sample dataset indices
Run policy.predict_action(obs)
Compare predicted action to ground-truth action
Compute MAE, MSE, RMSE, L2, and max absolute error
```

#### `save_outputs(summary, output_dir)`

Purpose:

```text
Save JSON metrics
Save CSV metrics
Save MAE plot
Save L2 plot
```

---

## 7. Run Evaluation on NRP

### 7.1 Eval job file

Key file:

```text
nrp_eval_job.yaml
```

Make sure the eval job:

1. Uses the training image.
2. Mounts the same PVC.
3. Runs from `/opt/umi`.
4. Sets `PYTHONPATH=/opt/umi:$PYTHONPATH`.
5. Installs/registers image codecs if needed.
6. Runs Python with `-u` for unbuffered logs.

Example command inside the job:

```bash
cd /opt/umi
export PYTHONPATH=/opt/umi:$PYTHONPATH

conda run -n umi python -m pip install imagecodecs imagecodecs-numcodecs

conda run -n umi python -u /workspace/eval_from_zarr.py \
  --checkpoint /workspace/checkpoints/may326_002/checkpoints/latest.ckpt \
  --zarr /workspace/data/may326.zarr.zip \
  --num_samples 200 \
  --output_dir /workspace/eval_results
```

### 7.2 Submit eval job

Delete staging pod and old eval job first:

```bash
kubectl delete pod pvc-stage -n ssu-isl-bimanual-dexterity --ignore-not-found
kubectl delete job bimanual-eval -n ssu-isl-bimanual-dexterity --ignore-not-found
```

Apply eval job:

```bash
kubectl apply -f nrp_eval_job.yaml -n ssu-isl-bimanual-dexterity
```

Stream logs:

```bash
kubectl logs -f job/bimanual-eval -n ssu-isl-bimanual-dexterity
```

### 7.3 Run on all samples

The fixed evaluator defaults to 200 samples.

To evaluate all samples:

```bash
--num_samples -1
```

Example:

```bash
conda run -n umi python -u /workspace/eval_from_zarr.py \
  --checkpoint /workspace/checkpoints/may326_002/checkpoints/latest.ckpt \
  --zarr /workspace/data/may326.zarr.zip \
  --num_samples -1 \
  --output_dir /workspace/eval_results_all
```

---

## 8. Evaluation Results

Successful eval output:

```text
GPU: NVIDIA RTX 5000 Ada Generation
Using EMA model
Policy loaded successfully
Dataset length: 44900
Action shape: (16, 20)
Evaluating 200 samples from dataset of length 44900
Policy output keys: ['action', 'action_pred']
```

Reported metrics:

```text
num_samples: 200
mean_mae: 0.0013309495017165317
median_mae: 0.00122255424503237
mean_mse: 0.000004004137602784397
mean_rmse: 0.0020010341333381587
mean_l2: 0.008128456201520749
median_l2: 0.007366632344201207
mean_max_abs: 0.00458133560285205
```

Metric meanings:

```text
num_samples:
  Number of processed dataset samples evaluated.

mean_mae:
  Average absolute per-dimension action error.

median_mae:
  Median per-sample MAE.

mean_mse:
  Mean squared action error.

mean_rmse:
  Square root of MSE, in the original action numeric scale.

mean_l2:
  Average Euclidean distance between predicted and ground-truth action vector.

median_l2:
  Median per-sample L2 action-vector error.

mean_max_abs:
  Average maximum absolute error across action dimensions for each sample.
```

These are not percentages. They are errors in the numeric scale of the model action vector.

---

## 9. Evaluation Output Files

Expected output directory:

```text
/workspace/eval_results
```

Expected fixed evaluator files:

```text
policy_vs_zarr_eval.json
policy_vs_zarr_eval.csv
mae_by_sample.png
l2_by_sample.png
```

Copy results locally:

```bash
kubectl apply -f pvc-stage.yaml -n ssu-isl-bimanual-dexterity
kubectl wait --for=condition=Ready pod/pvc-stage -n ssu-isl-bimanual-dexterity --timeout=120s

mkdir -p ./eval_results

kubectl cp \
  ssu-isl-bimanual-dexterity/pvc-stage:/workspace/eval_results/. \
  ./eval_results
```

Open locally on Mac:

```bash
open ./eval_results
```

Old broken evaluator files looked like:

```text
episode_000_eval.png
episode_000_trajectory.png
```

Do not use those as final results if they show zero error everywhere. That was caused by swallowed inference failures and zero-filled predictions.

---

## 10. Common Errors and Fixes

### Error: `codec not available: 'imagecodecs_jpegxl'`

Fix:

```bash
conda run -n umi python -m pip install imagecodecs imagecodecs-numcodecs
```

And in Python:

```python
import imagecodecs_numcodecs
imagecodecs_numcodecs.register_codecs()
```

### Error: `No module named diffusion_policy`

Fix job command:

```bash
cd /opt/umi
export PYTHONPATH=/opt/umi:$PYTHONPATH
```

### Error: missing `robot0_eef_pos_wrt1`

Cause:

```text
Manual raw zarr observation construction
```

Fix:

```text
Use hydra.utils.instantiate(cfg.task.dataset)
Use sample["obs"] directly
```

### Error: fake zero evaluation error

Cause:

```text
Inference failures were caught and replaced with zeros
```

Fix:

```text
Fail loudly on inference errors
Compare policy_output["action"] or ["action_pred"] to sample["action"]
```

### Error: PVC Multi-Attach

Cause:

```text
pvc-stage or another pod is already using umi-data-pvc
```

Fix:

```bash
kubectl delete pod pvc-stage -n ssu-isl-bimanual-dexterity --ignore-not-found
kubectl delete job bimanual-eval -n ssu-isl-bimanual-dexterity --ignore-not-found
```

### Error: pod stuck in `ContainerCreating`

Check:

```bash
kubectl describe pod <pod-name> -n ssu-isl-bimanual-dexterity
```

Common causes:

```text
Image pulling
PVC attaching
Multi-Attach
```

---

## 11. Minimal End-to-End Command Summary

### Local/Ubuntu SLAM

```bash
cd universal_manipulation_interface
conda activate umi

python scripts_slam_pipeline/00_process_videos.py ../data/raw/session1
python scripts_slam_pipeline/01_extract_gopro_imu.py ../data/raw/session1
python run_slam_pipeline.py ../data/raw/session1
```

### NRP Training

```bash
kubectl delete pod pvc-stage -n ssu-isl-bimanual-dexterity --ignore-not-found
kubectl apply -f train-job.yaml -n ssu-isl-bimanual-dexterity
kubectl logs -f job/umi-train-may326-v2 -n ssu-isl-bimanual-dexterity
```

### NRP Evaluation

```bash
kubectl delete pod pvc-stage -n ssu-isl-bimanual-dexterity --ignore-not-found
kubectl delete job bimanual-eval -n ssu-isl-bimanual-dexterity --ignore-not-found
kubectl apply -f nrp_eval_job.yaml -n ssu-isl-bimanual-dexterity
kubectl logs -f job/bimanual-eval -n ssu-isl-bimanual-dexterity
```

### Copy Results

```bash
kubectl apply -f pvc-stage.yaml -n ssu-isl-bimanual-dexterity
kubectl wait --for=condition=Ready pod/pvc-stage -n ssu-isl-bimanual-dexterity --timeout=120s

kubectl cp \
  ssu-isl-bimanual-dexterity/pvc-stage:/workspace/eval_results/. \
  ./eval_results
```
