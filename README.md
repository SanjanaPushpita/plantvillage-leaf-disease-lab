# PlantVillage Leaf Disease Lab

An end-to-end Jupyter notebook for training and comparing multiple transfer-learning models on the Kaggle PlantVillage leaf disease dataset. The workflow is designed and authored by Maliha Sanjana and fine-tunes several torchvision backbones to report accuracy, precision/recall/F1, confusion matrices, and learning curves.

## Highlights
- **ResNet-18 baseline** with straightforward training loop and artifact export.
- **Optional model zoo** that currently includes ResNet-34, DenseNet-121, and EfficientNet-B0 with identical evaluation tooling.
- **Visual diagnostics** for per-epoch loss/accuracy, confusion matrices, and sample predictions.
- **Self-contained workflow** that downloads the dataset directly via `kagglehub` without Google Drive dependencies.

## Repository Contents
```
├── simple_leaf_disease_trainer.ipynb  # Main notebook workflow
├── README.md                          # Project overview and usage notes
├── LICENSE                            # Project license (MIT)
└── .gitignore                         # Keeps large/raw data out of version control
```

## Prerequisites
1. Python 3.9 or newer.
2. pip (or conda) for dependency management.
3. CUDA-capable GPU strongly recommended for practical training times, though CPU will work for quick dry runs.
4. Kaggle credentials configured for `kagglehub` (set the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables or login via the library prompt).

Install the runtime dependencies:
```bash
pip install torch torchvision kagglehub pandas scikit-learn tqdm matplotlib
```

## Usage
1. **Clone the repository** (see instructions below).
2. Launch Jupyter Lab/Notebook or VS Code and open `fixed_Leaf_disease_trainer.ipynb`.
3. Execute the cells in order:
   - Section 1 installs packages if needed.
   - Section 2 sets hyperparameters and device configuration.
   - Section 3 downloads the dataset and prepares dataloaders.
   - Section 5 trains/evaluates the ResNet-18 baseline and writes summary metrics.
   - Section 6 visualizes metrics, confusion matrix, and sample predictions.
   - Section 7 (optional) iterates through additional architectures and surfaces comparative diagnostics.
   - Section 8 saves model weights and reports.

The notebook stores artifacts under `plantvillage_data/simple_runs/` by default; adjust paths to taste.

### If a Notebook Upload Looks Corrupted
Occasionally GitHub (or another platform) may report that the notebook is invalid because it still contains widget metadata. If that happens, open Google Colab and run the following cell to sanitize the file, then re-upload the generated `fixed_*.ipynb`:

```python
# Paste and run this in a Colab code cell
!pip install -q nbformat
from google.colab import files
import nbformat, io

print("Upload the broken .ipynb file now (choose the file from your computer):")
uploaded = files.upload()  # file picker

fname = list(uploaded.keys())[0]
print("Sanitizing:", fname)

# read uploaded bytes into a NotebookNode
nb = nbformat.reads(uploaded[fname].decode('utf-8'), as_version=4)

# remove top-level widgets metadata if present
nb.get('metadata', {})
if 'widgets' in nb['metadata']:
   nb['metadata'].pop('widgets', None)

# remove widgets metadata from each cell
for cell in nb.get('cells', []):
   if 'widgets' in cell.get('metadata', {}):
      cell['metadata'].pop('widgets', None)

outname = 'fixed_' + fname
nbformat.write(nb, outname)
print("Sanitized file written to:", outname)

# trigger browser download
files.download(outname)
```

Recommit the cleaned notebook and push again; it should render correctly afterward.

## Suggested Repository Name
- **`plantvillage-leaf-disease-lab`** — concise, descriptive, and highlights the comparative nature of the notebook.

## How to Publish on GitHub
1. Create a new repository on GitHub named `plantvillage-leaf-disease-lab` (or your preferred variant).
2. In your local project directory:
   ```bash
   git init
   git add README.md LICENSE .gitignore simple_leaf_disease_trainer.ipynb
   git commit -m "Initial commit: PlantVillage notebook by Maliha Sanjana"
   git branch -M main
   git remote add origin https://github.com/<your-user>/plantvillage-leaf-disease-lab.git
   git push -u origin main
   ```
3. Confirm the notebook renders correctly on GitHub and add repository topics such as `plant-disease`, `deep-learning`, `pytorch`, `transfer-learning`, and `notebook` to improve discoverability.

## Customizing the Notebook
- Update `models_to_run` in Section 7 to include or exclude architectures.
- Modify hyperparameters (`EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`) in Section 2 for experiments.
- Extend the helper registry with more models (e.g., MobileNetV3, ConvNeXt) to grow the comparison suite.

## License
Distributed under the MIT License. See [`LICENSE`](./LICENSE) for details.

---
Crafted with ❤️ by **Maliha Sanjana**.

