# stop the excecution if there is error
set -e

echo "Running Annotations.py - Apply instances over the images"
python3 annotations.py

echo "Running Dataset.py - Splitting dataset into train and val"
python3 dataset.py

echo "Running yolo_labels.py - Creating COCO labels files for tain and val images"
python3 yolo_labels.py

echo "Running model_train.py - Training Selected Yolo Model"
python3 model_train.py

echo "Running test.py - To see the best results of the selected model"
python3 test.py
