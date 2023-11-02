# Change detection from satellite images
Implementation of "End-to-End Change Detection for High-Resolution Satellite Images Using Improved UNet++" (Peng et al., 2019).
Model converted to Pytorch from the TensorFlow implementation at https://github.com/daifeng2016/End-to-end-CD-for-VHR-satellite-image

### Data loading 
```
python create_csv_file.py
```

### Model training
```
python train.py
```

### Model evaluation
```
python evaluate.py
```

### Prediction visualization
```
python visualize_predictions.py
```

