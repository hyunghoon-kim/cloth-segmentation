import os
from glob import glob


target_dirs = ["captions_woman", "conditions_woman", "images_woman"]

def getfilenames(filepaths):
    filenames = []
    for filepath in filepaths:
        _, filename = os.path.split(filepath)
        filename = filename.split(".")[0]
        filenames.append(filename)
    return filenames

if __name__ == "__main__":
    filepaths = sorted(glob("chunk_woman_clean/*.png"))
    filenames = getfilenames(filepaths)

    remove_path_list = []

    for target_dir in target_dirs:
        fpaths = glob(f"{target_dir}/*")
        for fpath in fpaths:
            _, fname = os.path.split(fpath)
            fname = fname.split(".")[0]
            if fname not in filenames:
                remove_path_list.append(fpath)

    # 이제 remove_path_list에 목록이 쌓였다면, 실제로 제거:
    for path_to_remove in remove_path_list:
        if os.path.isfile(path_to_remove):
            os.remove(path_to_remove)
            print(f"Removed: {path_to_remove}")
        else:
            print(f"Skipping (not a file): {path_to_remove}")
