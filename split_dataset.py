import os
import shutil
import random
import argparse


def create_folders(output_dir):
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    return train_dir, test_dir


def copy_files(input_subfolder, output_subfolder, files):
    os.makedirs(output_subfolder, exist_ok=True)
    for file in files:
        shutil.copy(os.path.join(input_subfolder, file), os.path.join(output_subfolder, file))


def split_files(input_dir, ratio, output_dir):
    train_dir, test_dir = create_folders(output_dir)

    for subfolder in os.listdir(input_dir):
        input_subfolder = os.path.join(input_dir, subfolder)
        if os.path.isdir(input_subfolder):
            files = os.listdir(input_subfolder)
            random.shuffle(files)
            split_index = int(len(files) * ratio)
            train_files = files[:split_index]
            test_files = files[split_index:]

            train_output_subfolder = os.path.join(train_dir, subfolder)
            test_output_subfolder = os.path.join(test_dir, subfolder)

            copy_files(input_subfolder, train_output_subfolder, train_files)
            copy_files(input_subfolder, test_output_subfolder, test_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split dataset into train and test sets.')
    parser.add_argument('input_dir', type=str, help='Input directory containing subfolders')
    parser.add_argument('ratio', type=float, help='Ratio of files to be used for training')
    parser.add_argument('output_dir', type=str, help='Output directory to save train and test sets')

    args = parser.parse_args()

    split_files(args.input_dir, args.ratio, args.output_dir)