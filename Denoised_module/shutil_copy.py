# from PIL import Image
# import os
#
#
# def resize_and_save_image(input_path, output_path, scale_factor=2.51851):
#     """
#     Resizes an image to a given scale factor and saves it to the output path.
#
#     Parameters:
#         input_path (str): Path to the input image file.
#         output_path (str): Path where the resized image will be saved.
#         scale_factor (int, optional): Factor by which the image will be scaled. Default is 2.
#     """
#     # 打开图像
#     with Image.open(input_path) as img:
#         # 计算新的尺寸
#         new_size = tuple([int(dim * scale_factor) for dim in img.size])
#
#         # 放大图像
#         resized_img = img.resize(new_size, Image.NEAREST)
#
#         # 保存图像
#         resized_img.save(output_path)
#
# # 示例使用
# input_image_path = 'F:\\rewrite_test_gauss\\large_area_dose_3000.0_1_gauss_cell.jpg'  # 将此路径替换为您的图片文件路径
# output_image_path = 'F:\\rewrite_test_gauss\\test_scale_3000dose.jpg'  # 将此路径替换为希望保存的新图片的路径
#
# resize_and_save_image(input_image_path, output_image_path)


# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
#
#
# def linear_normalization(image):
#     return (image - np.min(image)) / (np.max(image) - np.min(image))
#
#
# def logarithmic_transformation(image, constant=1):
#     return np.log1p(constant * image) / np.log1p(constant * np.max(image))
#
#
# def gamma_correction(image, gamma=1.0):
#     return np.power(image / np.max(image), gamma)
#
#
# # 假设image是你的原始图像
# image = Image.open('F:\\large_scale\\molecule_84_r30_blur.png').convert('L')
# image = np.array(image)
#
# # 应用归一化方法
# image_normalized = linear_normalization(image)
# image_log_transformed = logarithmic_transformation(image)
# image_gamma_corrected = gamma_correction(image, gamma=4)
#
# # 可视化结果
# plt.figure(figsize=(12, 4))
# plt.subplot(131)
# plt.imshow(image_normalized, cmap='gray')
# plt.title("Linear Normalization")
# plt.subplot(132)
# plt.imshow(image_log_transformed, cmap='gray')
# plt.title("Logarithmic Transformation")
# plt.subplot(133)
# plt.imshow(image_gamma_corrected, cmap='gray')
# plt.title("Gamma Correction")
# plt.show()


import os
import shutil

source_path = 'F:\\determined_best_model\\test_ori'
target_folder = 'F:\\determined_best_model\\labels'
source_folder = 'F:\\phase_structure_recognition_new\\faster_rcnn_stem_dataset\\hr_labels'

for img_i in os.listdir(source_path):
    target_img = os.path.join(source_folder, img_i)
    shutil.copy(target_img, target_folder)








