import json

# orig_annot_path = "datasets/stickers_split/stickers_ws_train_31shot_1280.json"
orig_annot_path = "datasets/stickers/annotations/stickers_ws_31shot_test_1280.json"
tags_annot_path = "datasets/all_view_tags.json"

with open(orig_annot_path, "r") as f:
    orig_annot = json.load(f)
with open(tags_annot_path, "r") as f:
    tags_annot = json.load(f)


# get the original filename w/o roboflow's augmentation
def get_name(filename):
    return filename.split(".rf")[0]

# list of all image file names and their tags (i.e. "extra")
images_lookup = {
    get_name(image["file_name"]) : 
    image.get("extra", []) for image in tags_annot.get("images", [])
}

for image in orig_annot.get("images", []):
    orig_name = get_name(image["file_name"])
    if orig_name in images_lookup:
        # add images_lookup[orig_name] as "extra" to the orig_annot's image
        image["extra"] = images_lookup[orig_name]
    else:
        print(f"Error: {orig_name} is not found within the tagged dataset.")


output_path = orig_annot_path
with open(output_path, "w") as f:
    json.dump(orig_annot, f)
    
print(f"View tags added and saved to {output_path}")