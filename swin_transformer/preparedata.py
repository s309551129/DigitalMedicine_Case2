import os

def movefile(mode):
    mode_path = os.path.join('data', mode)
    dirs = os.listdir(os.path.join('data', mode))

    for dir1 in dirs:
        dir1_path = os.path.join(mode_path, dir1)
        dir2 = os.listdir(dir1_path)

        dir2_path = os.path.join(dir1_path, dir2[0])
        file = os.listdir(dir2_path)

        origin_path = os.path.join(dir2_path, file[0])
        new_path = os.path.join(mode_path, file[0])
        os.rename(origin_path, new_path)

movefile('train')
movefile('valid')