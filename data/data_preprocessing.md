# Data PrePrecessing

## Download

download the dataset from the link below or select another traffic dataset

- [X-APP](https://drive.google.com/file/d/1C-K9V03plCPrv5k3lvrwVLxCkm-WPlk5/view?usp=drive_link)
- [X-WEB](https://drive.google.com/file/d/1S_4Z1i5vwU3nFya08UYlrvpUFFkl-r1f/view?usp=drive_link)
- [USTC-TFC](https://github.com/yungshenglu/USTC-TFC2016)
- [ISCX-VPN](https://www.unb.ca/cic/datasets/vpn.html)

## Organize File directory

* extract the data into a folder (e.g., `X-APP`) under `./data/`
* the subfolders of the `./data/X-APP/` are all named candidate labels (e.g., `./data/X-APP/Viemo/`)
* `./data/X-APP/Viemo/` contains all PCAP files of this label

```python
data/
  ├── dataset1
  │      ├── label_1
  │      │    ├── f1.pcap
  │      │    ├── ...
  │      │    └── fn.pcap
  │      ├── label_2
  │      │    └── f.pcap
  │      ├── ...
  │      └── label_n
  │           ├── f1.pcap
  │           ├── ...
  │           └── fn.pcap
  ├── dataset2
  │      ├── label_1
  │      │    ├── f1.pcap
  │      ...
  │   ...   
  └── data_preprocessing.md
```

## Run

* set the parameter `CLASSES` in `utils.py` according to your requirement. If you choose an other dataset, modify the parameter `DATA_MAP` to specify the file directory
* run `data_preprocessing.py` to process the data into `.pkl` files
* The preprocessed traffic files are output at `./data/{dataset}_pkt/{label}.pkl`
