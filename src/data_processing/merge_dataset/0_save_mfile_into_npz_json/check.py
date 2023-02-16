''' check what happened to the merging process

author: 
    Zhangsihao Yang

logs:
    2023-02-15: init
'''
import os


def check():
    for i in range(2296):
        file_path = f'/scratch/zyang195/projects/brain/data/merged/raw/{i:04d}.npz'
        if not os.path.exists(file_path):
            print(f'file {i} does not exist')


if __name__ == '__main__':
    check()
