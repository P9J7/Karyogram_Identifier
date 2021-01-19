from PIL import Image
import numpy as np
import os
import shutil


img_size = 300
img_read_path = 'path of karyogram'
img_save_people_path = 'path of saved by peopleId'
img_save_class_path = 'path of saved by class'


def extract_chromosome(file_path, save_by_people=False):
    if save_by_people:
        create_floder(img_save_people_path, os.path.splitext(file_path)[0].split('/')[-1])
    img = Image.open(file_path)
    array = np.array(img)
    if len(array.shape) == 3:
        array = array[:, :, 0]
    height, width = array.shape
    row = []
    for i in range(height):
        sub_array = list(array[i])
        if np.sum(list(map(lambda x: x < 5, sub_array))) > 200:
            row.append(i)
    count = 0
    for row_index in range(len(row)):
        col = []
        for j in range(width):
            if np.sum(list(map(lambda x: x < 10, array[row[row_index] - 3:row[row_index], j]))) >= 1:
                col.append(j)
                if len(col) % 2 == 1:
                    continue
                start_row = 20
                if row_index != 0:
                    start_row = row[row_index - 1] + 10 + 10 * row_index
                end_row = row[row_index]
                left_col_index = col[-2]
                right_col_index = col[-1]
                bias = 10
                while end_row >= start_row and \
                        np.sum(list(map(lambda x: x > 240, array[start_row, left_col_index:right_col_index]))) \
                        > right_col_index - left_col_index - bias:
                    start_row += 1
                while right_col_index >= left_col_index and \
                        np.sum(list(map(lambda x: x > 245, array[start_row:end_row, left_col_index]))) \
                        > end_row - start_row - bias:
                    left_col_index += 1
                while right_col_index >= left_col_index and \
                        np.sum(list(map(lambda x: x > 245, array[start_row:end_row, right_col_index]))) \
                        > end_row - start_row - bias:
                    right_col_index -= 1
                if start_row >= end_row:
                    count += 1
                    continue
                while right_col_index >= start_row and \
                        np.sum(list(map(lambda x: x > 245, array[end_row, left_col_index:right_col_index]))) \
                        > right_col_index - left_col_index - bias:
                    end_row -= 1
                middle_col = left_col_index + 3
                while np.sum(list(map(lambda x: x > 245, array[start_row:end_row, middle_col]))) \
                        < end_row - start_row - bias and middle_col < right_col_index:
                    middle_col += 1
                temp_array = np.zeros((img_size, img_size), dtype=np.uint8)
                temp_higth = end_row - start_row
                x = int((img_size - temp_higth) / 2)
                temp_width = middle_col - left_col_index
                y = int((img_size - temp_width) / 2)
                temp_array[x:x + temp_higth, y:y + temp_width] = \
                    255 - array[start_row:end_row, left_col_index:middle_col]
                if save_by_people:
                    temp_array_save_path = os.path.join(img_save_people_path,
                                                        os.path.splitext(file_path)[0].split('/')[-1],
                                                        str(count),
                                                        os.path.splitext(file_path)[0].split('/')[-1] + '_l.png')
                else:
                    temp_array_save_path = os.path.join(img_save_class_path,
                                                        str(count),
                                                        os.path.splitext(file_path)[0].split('/')[-1] + '_l.png')
                Image.fromarray(temp_array).save(temp_array_save_path)
                if middle_col == right_col_index:
                    count += 1
                    continue
                temp_array = np.zeros((img_size, img_size), dtype=np.uint8)
                temp_width = right_col_index - middle_col
                y = int((img_size - temp_width) / 2)
                temp_array[x:x + temp_higth, y:y + temp_width] = \
                    255 - array[start_row:end_row, middle_col:right_col_index]
                if save_by_people:
                    temp_array_save_path = os.path.join(img_save_people_path,
                                                        os.path.splitext(file_path)[0].split('/')[-1],
                                                        str(count),
                                                        os.path.splitext(file_path)[0].split('/')[-1] + '_r.png')
                else:
                    temp_array_save_path = os.path.join(img_save_class_path,
                                                        str(count),
                                                        os.path.splitext(file_path)[0].split('/')[-1] + '_r.png')
                Image.fromarray(temp_array).save(temp_array_save_path)
                count += 1


def create_floder(save_path, inner_path=''):
    for index in range(24):
        if os.path.exists(os.path.join(save_path, inner_path, str(index))):
            shutil.rmtree(os.path.join(save_path, inner_path, str(index)))
        os.makedirs(os.path.join(save_path, inner_path, str(index)))


if __name__ == '__main__':
    # default: saved by peopleId
    save_by_people = False
    if not save_by_people:
        create_floder(img_save_class_path)
    for path, dir_list, file_list in os.walk(img_read_path):
        for file_name in file_list:
            file_path = os.path.join(path, file_name)
            try:
                extract_chromosome(file_path, save_by_people)
            except RuntimeError:
                print('Found Exception:', file_path)
