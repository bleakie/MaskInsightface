# -*- coding: utf-8 -*
# clean unuseful imgs in datasets(one id imgs low 10)
import os, shutil
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'remove low-shot classes')
    parser.add_argument("-root", "--root", help = "specify your dir",default = '/data1t/GateID/gate_aligned_retina_crop', type = str)
    parser.add_argument("-min_num", "--min_num", help = "remove the classes with less than min_num samples", default = 3, type = int)
    args = parser.parse_args()

    root = args.root # specify your dir
    min_num = args.min_num # remove the classes with less than min_num samples

    cwd = os.getcwd()  # delete '.DS_Store' existed in the source_root
    os.chdir(root)
    os.system("find . -name '*.DS_Store' -type f -delete")
    os.chdir(cwd)
    import re
    for subfolder in os.listdir(root):
        if re.search('.jpg', subfolder):
            os.remove(os.path.join(root, subfolder))
            continue
        file_num = len(os.listdir(os.path.join(root, subfolder)))
        if file_num <= min_num:
            print("Class {} has less than {} samples, removed!".format(subfolder, min_num))
            shutil.rmtree(os.path.join(root, subfolder))