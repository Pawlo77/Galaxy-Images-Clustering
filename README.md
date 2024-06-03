# Galaxy-Images-Clustering
CV project - clustering low-res images, part of IAD at WUT


## Installation step

1. ```bash
      pip install -r requirements.txt
   ```
2. Run [eda](./notebooks/eda.ipynb) notebook to see the data distribution and some examples of images.3
3. Data preprocessing and features extraction using EfficientNet:
    ```bash
      python src/preprocessing.py --name train_mapping.csv
      python src/preprocessing.py --name validation_mapping.csv
      python src/preprocessing.py --name test_mapping.csv
      python src/features.py --name train_mapping.csv
      python src/features.py --name validation_mapping.csv
      python src/features.py --name test_mapping.csv
   ```
4. Run [train](./notebooks/features.ipynb) notebook to train feature extractor architecture.

## Sources
- [paper](https://arxiv.org/pdf/2311.14157)
- [paper](https://arxiv.org/pdf/2103.09382)
- [paper](https://arxiv.org/pdf/2304.12210)
- [paper](https://arxiv.org/pdf/1906.02864v1)
- [video](https://www.youtube.com/watch?feature=shared&fbclid=IwAR31M6TBsaNHr5Rn3Gqa22CCw5dr72F6hhIL1loLrt14kLzjpTq2XQdhq8M&v=TI0-S-Nco_A)