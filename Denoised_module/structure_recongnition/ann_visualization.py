# import torch
# import torchvision
# import torchvision.transforms as T
# import torchvision.datasets as datasets
# from torchvision.transforms import functional as F
# import cv2
# import random
#
# font = cv2.FONT_HERSHEY_SIMPLEX
#
# root = 'D:\\project\\phase_structure\\phase_structure_recognition\\faster_rcnn_stem_dataset\\hr_labels'
# annFile = 'D:\\project\\phase_structure\\phase_structure_recognition\\faster_rcnn_stem_dataset\\annotations\\annotation.json'
#
#
# # 定义 coco collate_fn
# def collate_fn_coco(batch):
#     return tuple(zip(*batch))
#
#
# # 创建 coco dataset
# coco_det = datasets.CocoDetection(root, annFile, transform=T.ToTensor())
# # 创建 Coco sampler
# sampler = torch.utils.data.RandomSampler(coco_det)
# batch_sampler = torch.utils.data.BatchSampler(sampler, 8, drop_last=True)
#
# # 创建 dataloader
# data_loader = torch.utils.data.DataLoader(coco_det, batch_sampler=batch_sampler, collate_fn=collate_fn_coco)
#
# # 可视化
# for imgs, labels in data_loader:
#     for i in range(len(imgs)):
#         bboxes = []
#         ids = []
#         img = imgs[i]
#         labels_ = labels[i]
#         for label in labels_:
#             bboxes.append([label['bbox'][0],
#                            label['bbox'][1],
#                            label['bbox'][0] + label['bbox'][2],
#                            label['bbox'][1] + label['bbox'][3]
#                            ])
#             ids.append(label['category_id'])
#
#         img = img.permute(1, 2, 0).numpy()
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         for box, id_ in zip(bboxes, ids):
#             x1 = int(box[0])
#             y1 = int(box[1])
#             x2 = int(box[2])
#             y2 = int(box[3])
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
#             cv2.putText(img, text=str(id_), org=(x1 + 5, y1 + 5), fontFace=font, fontScale=1,
#                         thickness=2, lineType=cv2.LINE_AA, color=(0, 255, 0))
#         cv2.imshow('test_ori', img)
#         cv2.waitKey()


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pycocotools.coco import COCO
from PIL import Image, ImageOps
import os
import shutil


def visualize_labels(coco, img_id, output_dir, image_dir, hr_label_dir):
    # Load image information
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(image_dir, img_info['file_name'])
    shutil.copy(img_path, hr_label_dir)

    # Load the image
    image = Image.open(img_path).convert('L')  # Convert to grayscale

    # Load annotations for the image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(ann_ids)

    # Create a figure and axis to plot the image
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')  # Display the image in grayscale
    plt.axis('off')

    # Plot bounding boxes and labels
    for ann in annotations:
        bbox = ann['bbox']
        label = coco.loadCats(ann['category_id'])[0]['name']
        rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=4, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
        # ax.text(bbox[0], bbox[1] - 5, label, color='r', fontsize=8, backgroundcolor='white')

    # Save the image with annotations
    output_path = os.path.join(output_dir, f"annotated_label_{img_info['file_name']}")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def visualize_all_labels(coco, output_dir, image_dir, hr_label_dir):
    # Retrieve all image IDs
    img_ids = coco.getImgIds()

    # Loop through all image IDs and visualize labels
    for img_id in img_ids:
        visualize_labels(coco, img_id, output_dir, image_dir, hr_label_dir)


# Path to the COCO annotation file and image directory
annotation_file = r'D:\project\phase_structure\phase_structure_recognition\faster_rcnn_stem_dataset\annotations\trainval_annotation.json'
image_dir = r'F:\final_version_stem_project\phase_structure_recognition\faster_rcnn_stem_dataset\hr_labels'
output_directory = r'C:\Users\yyt70\Desktop\paper_data\Fig2_files\hr_labels'  # Directory to save annotated images
output_directory_coco = r'C:\Users\yyt70\Desktop\paper_data\Fig2_files\hr_labels_coco'

# Ensure output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# COCO instance for loading annotations
coco = COCO(annotation_file)

# Visualize and save labels for all images
visualize_all_labels(coco, output_directory_coco, image_dir, output_directory)

