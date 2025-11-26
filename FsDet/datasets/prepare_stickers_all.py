import argparse
import json
import os
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 10],
                        help="Range of seeds")
    args = parser.parse_args()
    return args


def generate_seeds(args):
    data_path = 'stickers_all_28shot.json'
    data = json.load(open(data_path))

    specified_classes = [1, 8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35,
    36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 84, 85, 86, 87, 88, 89, 90]
    
    new_all_cats = []
    for cat in data['categories']:
        new_all_cats.append(cat)

    firstIter = True
    
        
    all_data = {}
    id2img = {}
    for i in data['images']:
        id2img[i['id']] = i

    anno = {c: [] for c in specified_classes}
    for a in data['annotations']:
        if a['category_id'] in specified_classes and a['iscrowd'] != 1:
            anno[a['category_id']].append(a)

    #for i in range(args.seeds[0], args.seeds[1]):
    random.seed(1)
    for c in specified_classes:
        img_ids = {}
        for a in anno[c]:
            if a['image_id'] in img_ids:
                img_ids[a['image_id']].append(a)
            else:
                img_ids[a['image_id']] = [a]

        sample_shots = []
        sample_imgs = []
           # for shots in [1, 2, 3, 5, 10, 30]:
        for shots in [28]:
            print("Class:", c)
            print("Number of Annotations for this Class:", len(anno[c]))
            print("Number of Images for this Class:", len(img_ids))
            print("Shots:", shots)
            while True:
                if shots > len(img_ids):
                    imgs = list(img_ids.keys())
                else:
                    imgs = random.sample(list(img_ids.keys()), shots)
                  #  break
                for img in imgs:
                    skip = False
                    for s in sample_shots:
                        if img == s['image_id']:
                            skip = True
                            break
                    if skip:
                        continue
                    if len(img_ids[img]) + len(sample_shots) > shots:
                        continue
                    sample_shots.extend(img_ids[img])
                    sample_imgs.append(id2img[img])
                    if len(sample_shots) == shots:
                        break
                if len(sample_shots) == shots:
                    break
            if firstIter == True:
                firstIter = False
                new_data = {
                    'info': data['info'],
                    'licenses': data['licenses'],
                    'images': sample_imgs,
                    'annotations': sample_shots,
                }
            else:
                new_data['images'].extend(sample_imgs)
                new_data['annotations'].extend(sample_shots)

        
    filename = "full_box_stickers_28shot_all.json"
        
        #save_filename = get_save_path_seeds(ID2CLASS[c], shots, i)
        #save_path = os.path.join(os.getcwd(), save_filename)
    new_data['categories'] = new_all_cats
    with open(filename, 'w') as f:
        json.dump(new_data, f)




def get_save_path_seeds(cls, shots, seed):
    prefix = 'full_box_{}shot_{}_stickers_trainval'.format(shots, cls)
    filename = prefix + '_seed{}.json'.format(seed)
    return filename


if __name__ == '__main__':
    ID2CLASS = {
        1: "car-sticker",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bus",
        7: "train",
        8: "truck",
        9: "boat",
        10: "traffic light",
        11: "fire hydrant",
        13: "stop sign",
        14: "parking meter",
        15: "bench",
        16: "bird",
        17: "cat",
        18: "dog",
        19: "horse",
        20: "sheep",
        21: "cow",
        22: "elephant",
        23: "bear",
        24: "zebra",
        25: "giraffe",
        27: "backpack",
        28: "umbrella",
        31: "handbag",
        32: "tie",
        33: "suitcase",
        34: "frisbee",
        35: "skis",
        36: "snowboard",
        37: "sports ball",
        38: "kite",
        39: "baseball bat",
        40: "baseball glove",
        41: "skateboard",
        42: "surfboard",
        43: "tennis racket",
        44: "bottle",
        46: "wine glass",
        47: "cup",
        48: "fork",
        49: "knife",
        50: "spoon",
        51: "bowl",
        52: "banana",
        53: "apple",
        54: "sandwich",
        55: "orange",
        56: "broccoli",
        57: "carrot",
        58: "hot dog",
        59: "pizza",
        60: "donut",
        61: "cake",
        62: "chair",
        63: "couch",
        64: "potted plant",
        65: "bed",
        67: "dining table",
        70: "toilet",
        72: "tv",
        73: "laptop",
        74: "mouse",
        75: "remote",
        76: "keyboard",
        77: "cell phone",
        78: "microwave",
        79: "oven",
        80: "toaster",
        81: "sink",
        82: "refrigerator",
        84: "book",
        85: "clock",
        86: "vase",
        87: "scissors",
        88: "teddy bear",
        89: "hair drier",
        90: "toothbrush",
    
    }
    CLASS2ID = {v: k for k, v in ID2CLASS.items()}

    args = parse_args()
    generate_seeds(args)
