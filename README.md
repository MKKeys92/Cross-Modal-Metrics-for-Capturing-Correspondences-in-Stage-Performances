# Cross-Modal Metrics for Capturing Correspondences in Stage Performances

This is the official repository accompanying the paper:  

**Cross-Modal Metrics for Capturing Correspondences Between Music Audio and Stage Lighting Signals**  
*Proceedings of the 33rd ACM International Conference on Multimedia (MM ’25), October 27–31, 2025, Dublin, Ireland.*  
ACM, New York, NY, USA, 7 pages.  
[https://doi.org/10.1145/3746027.3755488](https://doi.org/10.1145/3746027.3755488)

---

## Overview

This repository provides the code and documentation for:

- **Metrics for measuring multimodal correspondences** in stage performances:
  - Beat-align metric
  - Intensity correlation metric
  - Structure correlation metric
  - Novelty correlation metric
- **Visualization tools** for all metrics (including the graphs from the paper)
- **Lighting feature abstraction** code and examples
- **Guide to training a generative system for stage lighting**, based on  
  *Alexanderson et al., “Listen, Denoise, Action! Audio-Driven Motion Synthesis with Diffusion Models,” 2023*

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Creating a Dataset](#creating-a-dataset)
  - [Evaluation Scripts](#evaluation-scripts)
  - [Additional Usages](#additional-usages)
    - [Extracting Audio Features](#extracting-audio-features)
    - [Extracting Lighting Features](#extracting-lighting-features)
    - [Training a Generative System](#training-a-generative-system)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Installation

1. Install **Python 3.9**
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Creating a Dataset

Run:

```bash
python RunScriptsAndConfigs/Preprocessing/utils/create_dataset.py
```

- **`--config`**: select which config to use from `dataset.conf`, which defines generators, filters, feature selection, and train/test splits.  Also you can filter data subsets, only use parts of the audio features and set splits for possible use in training a neural network. 
- **Optional overrides**:
  - `--light_input_dir`
  - `--audio_input_dir`
  - `--dataset_output_dir`

**Default input paths:**
- Lighting: `./Data/Input/Light/AbstractedData/PASv02/`
- Audio: `./Data/Input/Audio/ExtractedData/standard/`

**Default output path:**
- `./Data/Output/CombinedDataSets/`

The Script will iterate over all directories (one directory per subset) of the input directory.

---

### Evaluation Scripts

Located in: `RunScriptsAndConfigs/Evaluation/`

- **`dataset_comparison.py`**  
  Generate comparison graphs between datasets (as in the paper).  
  Configure visualization offsets and dataset selection directly in the script.

- **`find_params.py`**  
  Utility to decide suitable hyperparameters for the defined metrics.

- **`test_dataset.py`**  
  Run metrics and visualizations on a dataset.  
  ```bash
  python test_dataset.py --dataset NAME_OF_YOUR_DATASET.pkl
  ```
  - `--dataset`: dataset file (default search path: `./Data/Output/CombinedDataSets/`)
  - Other overrides: dataset directory, create plots, etc.  

  **Outputs**:  
  Results and visualizations (e.g., SSM matrices, novelty plots) are stored in:  
  ```
  ./Data/Output/Evaluation/<script_name>/<date_and_time>/
  ```
  including a logfile.

---

### Additional Usages

#### Extracting Audio Features


```bash
python RunScriptsAndConfigs/Preprocessing/Audio/extract_audio_features.py   --input_path <path_to_raw_audio>   --output_path <path_to_save_features>
```

- The script iterates over all subdirectories in `--input_path` (each subdirectory corresponds to a dataset subset).  
- Extraction behavior is controlled by the configuration file:  
  `RunScriptsAndConfigs/Preprocessing/Audio/audio_extraction.conf`  
- Extracted features are saved to:  
  ```
  <output_path>/<config_name>/<subset_name>/
  ```
- Currently, only **.wav** audio files are supported.

#### Extracting Lighting Features

Detailed documentation and examples can be found here:  
[Lighting Feature Extraction](Further%20Documentation/Lighting%20Feature%20Extraction/README.md)

#### Training a Generative System

Step-by-step guide:  
[Training a Generative Model](Further%20Documentation/Training%20a%20Generative%20Model/README.md)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{Kohl2025,
  title     = {Cross-Modal Metrics for Capturing Correspondences Between Music Audio and Stage Lighting Signals},
  author    = {Michael Kohl and Tobias Wursthorn and Christof Weiß},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia (MM '25)},
  year      = {2025},
  location  = {Dublin, Ireland},
  isbn      = {979-8-4007-2035-2/2025/10},
  doi       = {10.1145/3746027.3755488}
}
```

---

## Acknowledgments

This research and code were developed as part of the **Computational Humanities** research group at the **University of Würzburg**, within the  
- Center for Artificial Intelligence and Data Science (CAIDAS)  
- Center for Philology and Digitality (ZPD)  

More information: [Computational Humanities Research Group](https://www.caidas.uni-wuerzburg.de/research-groups/computational-humanities/)

Funding:  
This work was funded by the **German Research Foundation (DFG)** within the Emmy Noether Junior Research Group *Computational Analysis of Music Audio Recordings: A Cross-Version Approach* (DFG WE 6611/3-1, Grant No. 531250483).

---
