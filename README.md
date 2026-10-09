# Multi-view dynastic classification and visual interpretation of blue-and-white porcelain

Code accompanying the paper:

> Yapp, E. K. Y., Zhuang, W., Chen, C., Wang, X., Wen, Y., & Yeh, H.-Y. (2026).
> Multi-view dynastic classification and visual interpretation of blue-and-white porcelain.
> *Social Sciences & Humanities Open*, 13, 102768.
> https://doi.org/10.1016/j.ssaho.2026.102768

The code trains image classifiers that assign Ming or Qing dynasty to
blue-and-white porcelain objects from the Metropolitan Museum of Art Open
Access collection. Each object is photographed from several views. A
pretrained ResNet with a frozen backbone and a new linear head is trained on
individual images, and its predictions are combined per object by averaging
softmax probabilities. Each configuration is trained 30 times with different
seeds, and results are reported as mean ± 95% confidence interval. The
repository also contains the scripts for the learning curves, the
Grad-CAM / Ablation-CAM / Score-CAM visualisations and the t-SNE plot.

This repository contains the code exactly as used for the published results.
For readability, exploratory scripts have since been moved to `archive/` and
data-preparation scripts to `data_preparation/`; no file content has changed,
and the exact original layout is at tag `paper-original`. See
[Implementation notes](#implementation-notes) for details that differ from the
paper's description.

---

## Contents

### Final pipeline (used for the published results)

| File | Purpose |
|---|---|
| `main.py` | Trains and evaluates one configuration over N seeds and writes per-seed logs, checkpoints and a summary (mean ± 95% CI). |
| `run_experiments.sh` | Runs the `main.py` configurations behind Tables 2 and 3 and saves console output to `bash_logs/`. |
| `learning_curves2.py` | Per-epoch mean training/validation loss and accuracy with 95% CI for every experiment (Fig. 1 and supplementary curves). |
| `visualExplanation.py` | Grad-CAM, Ablation-CAM and Score-CAM for four selected images (Fig. 2). |
| `tSNE.py` | t-SNE of backbone features of the test images (Fig. 3). |
| `dataset_summary_statistics.py` | Image count and size range of the dataset. |
| `filtered_chinese_porcelain.csv` | Met Open Access metadata for the Chinese porcelain objects; dynasty labels are derived from it. |
| `object-IDs.csv` | Default object-ID list for the optional ID filter in `main.py` (`filter_by_object_id_file`, off in the published runs). |
| `multi_seed_results.csv` | Written by `main.py` and overwritten by every run; the per-experiment `summary.txt` files are authoritative. |
| `blue-object-ids.csv` | IDs of the blue-and-white objects selected from the collection; read by `data_preparation/rename-files.py`. |
| `visual_explanations_seed50.pdf` | Fig. 2 as produced by `visualExplanation.py`. |
| `tsne_perp30_iter1000.pdf` | Fig. 3 as produced by `tSNE.py`. |
| `requirements-original.txt` | Exact package versions of the environment used. |
| `bash_logs/` | Console output of six of the seven published runs, including the arguments used, and of two runs not in the paper (see [Tables 1–3](#tables-13)). |
| `paper/paper.pdf` | The published paper. |

### Data preparation (`data_preparation/`)

Run these from the repository root (e.g. `python
data_preparation/rename-files.py`); they read and write paths relative to the
root, so `generate_period_visual_audit.py` writes `period_visual_audit.html`
to the root.

| File | Purpose |
|---|---|
| `find_corrupted_images.py` | Lists images that fail to decode. |
| `checkDynastyAndDates.py` | Compares the dynasty in the `Period` text with the dynasty implied by the object dates. |
| `rename-files.py`, `recreateImageFolderHierarchy.py` | Flatten object folders for manual review and rebuild them afterwards. |
| `generate_period_visual_audit.py` | HTML gallery of objects grouped by reign period, used for visual review (`period_visual_audit.html`). The gallery references images in an earlier folder (`images_clean/`), which are not distributed, so its images do not display. |

The final image set was curated by hand: objects and images were removed after
visual review, and views were renamed by type. No script records this
curation. The image-order manifest (see [Data](#data)) records the result.

### Archive (`archive/`; not used for the published results)

Kept for reference. The scripts use paths relative to the repository root and
are not maintained.

| File | Notes |
|---|---|
| `main-commandLine.py`, `main-object-level.py`, `main-multi-task.py` | Earlier training scripts; `main-multi-task.py` does not run as is. |
| `learning_curves.py` | Earlier version of `learning_curves2.py`. |
| `augmentImages.py` | Offline augmentation (not used; augmentation is done on the fly in `main.py`). |
| `auto_crop_objects.py`, `auto_crop_objects2.py`, `auto_crop_objects3.py`, `sam_auto_crop.py`, `fasterrcnn_coco_crop.py`, `faster_rcnn_auto_crop.py`, `generate_pseudo_boxes.py`, `train_ceramic_detector.py` | Automatic cropping experiments; the published results use uncropped images. |
| `preview_yolo_boxes.py` | Viewer for an unrelated detection dataset. |
| `unique_object_ids.csv`, `pseudo_boxes.csv`, `learning_curves_gold_standard.csv` | Intermediate or scratch files. |
| `tsne_perp*_iter*.pdf` | t-SNE parameter sweep (24 settings). Fig. 3 (`tsne_perp30_iter1000.pdf`) is in the root. |
| `tsne_dynasty_correctness.pdf` | Earlier t-SNE output; its name is the default file name in `tSNE.py`, which the current script no longer uses. |

---

## Requirements

- Python 3.11.15
- Packages: `pip install -r requirements-original.txt`. This installs PyTorch
  2.9.1 and torchvision 0.24.1 built for CUDA 13.0, plus scikit-learn 1.8.0,
  pandas 2.3.3, numpy 2.2.6, pillow 12.0.0, matplotlib 3.10.8, opencv-python
  4.12.0.88 and grad-cam 1.5.5.
- Hardware used for the published results: Ubuntu 24.04 with two GPUs, an
  NVIDIA GeForce RTX 4090 (24 GB, `cuda:0`) and an RTX 5090 (32 GB, `cuda:1`).
  `main.py` runs two seeds at a time, one per GPU.
- Pretrained ImageNet weights are downloaded by torchvision on first use.

---

## Data

The images, experiment logs and trained models are available from Zenodo:
https://doi.org/10.5281/zenodo.23254363

The images come from the Metropolitan Museum of Art Open Access collection
and are in the public domain (CC0). Unzip every zip file in the repository
root; each one extracts to the path the code expects. Put the checkpoint files
in the folder given in the table.

| File | Contents | Extracts to |
|---|---|---|
| `blue-and-white-porcelain-images_part1of9_<first>-<last>.zip` … `_part9of9_<first>-<last>.zip` (9 files) | The final image set, split by Met Object ID range into independent zips of about 150 MB; the first and last object IDs are in each file name. Together: 297 objects, 986 JPEG images (1.3 GB). | `images_clean_4/<object_id>/` |
| `experiment_logs.zip` | Per-seed logs (seeds 42–71) and `summary.txt` of the seven published configurations. | `experiments/<name>/logs/`, `experiments/<name>/summary.txt` |
| `model_seed42.pth` … `model_seed71.pth` (30 files) | Checkpoints of Table 2 ResNet-50 (V2) and Table 3 All, one per seed (about 94 MB each, 2.8 GB in total). | Not zipped; place in `experiments/resnet50_frozen_all_baseaug_img448_ptV2_softmax_seed42_runs30/models/` |
| `filtered_chinese_porcelain.csv` | Object metadata; dynasty labels are derived from it. Also in this repository. | Not zipped; repository root |
| `image_order_manifest.csv` | For every object, the order in which its images were read when the published results were produced, with file sizes and MD5 checksums. See [Exact reproduction](#exact-reproduction). Not read by the code. | Not zipped; repository root (optional) |

Checkpoints for the other six configurations are available from the authors on
request.

Image file names give the view: `primary.jpg`, `extra_NN.jpg`,
`bottom_NN.jpg`, `top_NN.jpg`, `closeup_NN.jpg`, `inside_NN.jpg`,
`inside_bottom_NN.jpg`, `top_bottom_NN.jpg` and `top_inside_NN.jpg`. The
image-selection variants in Table 3 rely on these names.

Each checkpoint `model_seed<N>.pth` is a dictionary whose `model_state_dict`
holds the full network, including the BatchNorm statistics estimated during
training (see [Implementation notes](#implementation-notes)). Only the images
are needed to retrain. Without retraining, Fig. 1 can be produced from the
logs, and Figs 2 and 3 from the images plus the ResNet-50 (V2) / All logs and
checkpoints.

All scripts use paths relative to the repository root and must be run from
there.

**Labels.** `main.py` derives each object's dynasty from the `Period` field of
the metadata. If `Period` names no dynasty, it uses `Object Begin Date` (or
`Object End Date`). After keeping only Ming and Qing (13 Yuan objects are
excluded), the dataset has 284 objects and 963 images:

| | Objects | Images |
|---|---|---|
| Ming | 98 | 358 |
| Qing | 186 | 605 |

The split is by object and stratified by dynasty. A fixed 30% of objects
(86) form the test set for all runs. The remaining objects are split evenly
into training and validation (99 each), with a different split for each seed.

---

## Reproducing the results

Run all commands from the repository root. Each `main.py` call writes to
`experiments/<configuration name>/`:

- `logs/results_seed<N>.txt`: per-seed training and test log;
- `models/model_seed<N>.pth`: checkpoints;
- `summary.txt`: mean ± 95% CI over the 30 seeds.

Seeds are 42–71. A full 30-seed run takes about 8–11 hours on the hardware
above.

### Tables 1–3

Common arguments:

```bash
COMMON="--freeze_backbone True --gpus 0 1 --n_runs 30 --base_seed 42 \
        --image_root images_clean_4 --alt_augmentation False --image_size 448 \
        --softmax_voting True"
```

| Result | Command | Output (`experiments/…`) |
|---|---|---|
| Table 2, ResNet-50 (V2); Table 3, All | `python main.py --resnet 50 --pretrained_version V2 $COMMON` | `resnet50_frozen_all_baseaug_img448_ptV2_softmax_seed42_runs30` |
| Table 2, ResNet-50 (V1) | `python main.py --resnet 50 --pretrained_version V1 $COMMON` | `resnet50_frozen_all_baseaug_img448_ptV1_softmax_seed42_runs30` |
| Table 2, ResNet-34 (V1) | `python main.py --resnet 34 --pretrained_version V1 $COMMON` | `resnet34_frozen_all_baseaug_img448_ptV1_softmax_seed42_runs30` |
| Table 2, ResNet-18 (V1) | `python main.py --resnet 18 --pretrained_version V1 $COMMON` | `resnet18_frozen_all_baseaug_img448_ptV1_softmax_seed42_runs30` |
| Table 3, All but Bottom | `python main.py --resnet 50 --pretrained_version V2 --image_selection no_bottom $COMMON` | `resnet50_frozen_no_bottom_…` |
| Table 3, Primary + Bottom | `python main.py --resnet 50 --pretrained_version V2 --image_selection primary_bottom $COMMON` | `resnet50_frozen_primary_bottom_…` |
| Table 3, Primary | `python main.py --resnet 50 --pretrained_version V2 --image_selection primary $COMMON` | `resnet50_frozen_primary_…` |

`run_experiments.sh` contains the same commands; its blocks were enabled one
at a time. It contains absolute paths from the original machine (`PYTHON`
and `SCRIPT`), so use the `main.py` commands in the table instead.
`bash_logs/` holds the console output, including all arguments, of six of
the seven published runs: the three V1 models of Table 2 (`resnet18_V1.log`,
`resnet34_V1.log`, `resnet50_V1.log`) and three image selections of Table 3
(`resnet50_V2_no_bottom.log`, `resnet50_V2_primary_bottom.log`,
`resnet50_V2_primary.log`). The headline run (Table 2 ResNet-50 (V2), Table 3
All) has no log; its arguments are those in the table above.
`resnet50.log` (ResNet-50 V2 with argmax instead of softmax voting) and
`resnet34.log` (an aborted ResNet-34 V2 run) are runs not in the paper.

The table values (accuracy, macro precision, macro recall) are lines 2–4 of
each `summary.txt`; macro F1 is on line 5. The Table 1 counts are printed at
the top of every per-seed log.

### Figures

| Figure | Command | Output |
|---|---|---|
| Fig. 1 (learning curves) | `python learning_curves2.py` | `experiments/<name>/<name>_learning_curves.pdf` for every experiment (the paper uses the ResNet-50 V2 one) |
| Fig. 2 (visual explanations) | `MPLBACKEND=Agg python visualExplanation.py` | `visual_explanations_seed<N>.pdf` |
| Fig. 3 (t-SNE) | `python tSNE.py` | `tsne_perp30_iter1000.pdf` |
| Dataset statistics (Section 3) | `python dataset_summary_statistics.py` | printed |

Figs 2 and 3 need the ResNet-50 V2 experiment. Both scripts pick the seed with
the highest object-level test macro F1, which is seed 50 (see
[Implementation notes](#implementation-notes)).

### Run order

1. Obtain `images_clean_4/` (see [Data](#data)).
2. Run the seven `main.py` configurations (any order).
3. Run `learning_curves2.py`, then `visualExplanation.py` and `tSNE.py`.

### Exact reproduction

Training on GPUs is not bit-for-bit deterministic in general. Under the
following conditions, however, published runs on GPU 1 were reproduced
exactly: identical per-epoch logs, test results and checkpoint tensors.

1. **Image order.** `main.py` reads each object folder with `os.listdir`,
   whose order depends on the filesystem, not on file names. This order
   determines the order of training samples and therefore the batches.
   Copying the image folder normally changes it. The order used for the
   published results is recorded in `image_order_manifest.csv`. The released
   code does not read the manifest: exact reproduction requires a folder
   whose `os.listdir` order matches it. Otherwise results reproduce only up
   to normal training variation.
2. **GPU assignment.** Run with `--gpus 0 1`, which places even seeds on GPU 0
   and odd seeds on GPU 1, two at a time.
3. **Numerical settings.** Leave PyTorch's defaults unchanged.
   Convolutions use TF32 on these GPUs, and `main.py` sets
   `cudnn.benchmark = True`. Evaluating the published checkpoints with TF32
   disabled, or on CPU, changes predicted probabilities by up to about 0.003.
4. **Software.** Use the package versions in `requirements-original.txt`.

Even under these conditions, runs on GPU 0, which also drove the display, can
differ slightly between repetitions (third decimal of the logged losses),
because `cudnn.benchmark` may select different convolution algorithms.

Re-evaluating the published checkpoints on the test set, with the same
test-set construction, preprocessing, softmax voting and metrics as main.py,
reproduces each experiment's summary.txt exactly for six of the seven
configurations, and within 0.0008 for "All but Bottom", where one object's
averaged probability lies at 0.5001 / 0.4999. Only the ResNet-50 (V2) / All
checkpoints are in the Zenodo deposit; see [Data](#data).

---

## Implementation notes

The code is unchanged from the version used for the paper. Some
implementation details differ from, or are not described in, the paper:

- **BatchNorm statistics.** The backbone weights are frozen and only the
  classifier head is optimised, but training runs with the model in training
  mode, so the backbone's BatchNorm running statistics are re-estimated on the
  training images (similar to adaptive batch normalisation; no test images are
  involved). With the statistics kept at their pretrained values, ResNet-50
  (V2) reaches 0.809 ± 0.010 accuracy, compared with 0.853 ± 0.011 published.
  All configurations were trained the same way. Retraining all configurations
  with the statistics fixed lowers every result by about 0.02–0.08, but the
  paper's comparisons are unchanged: ResNet-50 (V2) remains best in Table 2
  and using all views remains best in Table 3, both with statistically
  significant leads (paired tests across seeds). Elsewhere, only neighbouring
  configurations whose results are statistically indistinguishable change
  order.
- **Labels.** Dynasty keywords are matched as substrings, so one Ming object
  (75721, Longqing reign) is labelled Qing. It was never in the test set.
- **Class weights** are computed from all images rather than the training
  split.
- **Checkpoint selection.** The saved checkpoint's validation loss is within
  the early-stopping tolerance (0.01) of the minimum.
- **Figs 2 and 3** use the model with the highest test macro F1 (seed 50)
  rather than the lowest validation loss.

---

## License

Code: MIT (see [LICENSE](LICENSE)). Data and trained models on Zenodo: CC BY
4.0; the Met images are public domain (CC0).

## Contact

Please open a GitHub issue, or email Edward Yapp at edwardyappky@suss.edu.sg.
