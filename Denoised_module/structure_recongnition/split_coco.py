import json
import os


def filter_annotations_by_image_ids(annotation_file, image_ids):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    filtered_annotations = {
        'images': [img for img in annotations['images'] if img['id'] in image_ids],
        'annotations': [ann for ann in annotations['annotations'] if ann['image_id'] in image_ids],
        'categories': annotations['categories']
    }

    return filtered_annotations


def save_to_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

# Paths to the original annotation file and the trainval and test_ori image files
annotation_file = 'D:\\project\\phase_structure\\phase_structure_recognition\\faster_rcnn_stem_dataset\\annotations\\test_annotation.json'
trainval_image_files = os.listdir('F:\\different dose img\\500dose')  # List of trainval image file names
test_image_files = os.listdir('F:\\different dose img\\1000dose')  # List of test_ori image file names

with open(annotation_file, 'r') as f:
    annotations = json.load(f)
# Extract image IDs for trainval and test_ori images
trainval_image_ids = [img['id'] for img in annotations['images'] if img['file_name'] in trainval_image_files]
test_image_ids = [img['id'] for img in annotations['images'] if img['file_name'] in test_image_files]

# Filter and save annotations for trainval and test_ori sets
trainval_annotations = filter_annotations_by_image_ids(annotation_file, trainval_image_ids)
test_annotations = filter_annotations_by_image_ids(annotation_file, test_image_ids)

save_to_json(trainval_annotations, '../faster_rcnn_stem_dataset/annotations/test_annotation.json')
save_to_json(test_annotations, '../faster_rcnn_stem_dataset/annotations/1000dose_annotation.json')
