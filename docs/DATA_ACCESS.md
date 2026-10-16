# Obtaining the data

The images are **not** redistributed. Obtain each source from its provider and respect its terms.

| Source | Where | Access / terms |
|---|---|---|
| Chest X-Ray (Pneumonia, COVID-19, Tuberculosis) | Kaggle: https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis | Open (Kaggle account); composite of third-party datasets: check the terms of each |
| COVID-QU-Ex | Kaggle: https://www.kaggle.com/dsv/3122898 (doi:10.34740/KAGGLE/DSV/3122898); Tahir et al. | Open |
| RSNA Pneumonia Detection Challenge | Kaggle competition `rsna-pneumonia-detection-challenge` (Shih et al., doi:10.1148/ryai.2019180041) | Accept the competition rules |
| TBX11K (simplified) | Kaggle: https://www.kaggle.com/datasets/vbookshelf/tbx11k-simplified (Liu et al., CVPR 2020) | Open (Kaggle account) |
| Shenzhen and Montgomery | Jaeger et al., doi:10.3978/j.issn.2223-4292.2014.11.20 (as redistributed on Kaggle: *Pulmonary Chest X-Ray Abnormalities*) | Open; the providers ask not to redistribute the dataset |
| SilicoData (Set A) | https://doi.org/10.6084/m9.figshare.25383697.v13 (Akhter et al.) | Data use agreement; **no redistribution** of images or derived material |
| VinDr-CXR (tuberculosis subset) | https://physionet.org/content/vindr-cxr/1.0.0/ (doi:10.13026/3AKN-B287) | PhysioNet credentialed access; **no redistribution** |

Expected layout under `Datasets/` (as read by `notebooks_eda/`): `Chest_x_ray`, `COVID-QU-Ex-Dataset`, `RSNA_Pneumonia`,
`tbx11k-simplified`, `Pulmonary_chest_x_ray_adnomalities`, `Silicodata_Updated_Feb2025`, `pyshionet`.

The per-image manifests that define the train/validation/test partition (seed 42) are rebuilt by `notebooks_eda/08_consolidacion.ipynb`.
They are not published because they contain perceptual hashes, file names and patient identifiers of the restricted sources.

The three example radiographs used in Figures 1–2 (`figuras/_recursos_cxr/`) come from Kermany et al. (Mendeley Data, CC BY 4.0).
