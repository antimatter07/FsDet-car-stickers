import json

# Open and read the JSON file
with open('car_sticker_predictions.json', 'r') as file:
    data = json.load(file)

# Print the data
ctr = 0
for prediction in data:
    if prediction['category_id'] == 92:
        ctr += 1


print('**CAT ID COUNT: **' + str(ctr))
        
