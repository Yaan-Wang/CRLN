## CRLN

## Clone the Git repo

```bash
$ git clone https://github.com/Yaan-Wang/CRLN.git
$ cd CRLN
```
## Prepare dataset

1. Please download the LA, Pancreas and BRaTs19 datasets from the Google Drive link:[Google Drive - Datasets](https://drive.google.com/drive/folders/1_cOBDNlMYjNG-CJFzbmWXXGPj2vvKESE?usp=drive_link)
2. After downloading the datasets, please place them in the `./Datasets` directory and organize the folder structure as follows:
```bash
Datasets/
│
├── BRATS19/
│   ├── data/
│   └── datalist/
│
├── Left_Atrium/
│   ├── data/
│   └── datalist/
│
└── Pancreas/
    ├── data/
    └── datalist/
```
## Train and test the model

```bash
# e.g., for 10% on LA dataset
cd VnetLA
./la.sh
```
## Questions and Suggestions

If you have any questions or suggestions about this project, please contact us via email at wangyanyan_neu@163.com

