import zipfile
import os
import csv
import json



current_directory = os.getcwd()
extract_directory = os.path.join(current_directory, '..', '..', 'data', 'parent_mapping.json')
metadata_directory = os.path.join(current_directory, '..', '..','data', 'fma_metadata.zip')
genres = ''
index_mapping = {}
parent_mapping = {}      

with zipfile.ZipFile(metadata_directory, 'r') as zip_ref:
    genres = [file_name for file_name in zip_ref.namelist() if file_name == 'fma_metadata/genres.csv']

    with zip_ref.open(genres[0]) as genre_file:
        content = genre_file.read().decode('utf-8').splitlines()

    # One pass for the mapping
    genre_reader = csv.reader(content, delimiter=',', quotechar='"')
    # skip the feature names row
    next(genre_reader)
    for row in genre_reader:
        own_idx = row[0]
        child_genre = row[3]
        parent_genre_idx = row[2]
        genre_dict = {"name": child_genre, "parent_id": parent_genre_idx }
        index_mapping[own_idx] = genre_dict


for genre_id in index_mapping.keys():
    genre_name = index_mapping[genre_id]["name"]
    parent_genre_idx = index_mapping[genre_id]["parent_id"]
    try:
        parent_genre_name = index_mapping[parent_genre_idx]['name']
        parent_mapping[genre_name] = parent_genre_name
        print(genre_name, "----->", parent_genre_name)
    except:
        print(genre_name, "----->", genre_name)
        parent_mapping[genre_name] = genre_name



missing_genres = {"chamber music [delete]": "Classical", "Psych REPLACE w/ Psych-Rock": "Rock", "Electronica": "Electronic", "Club": "Electronic", 
 "Folk/Rock": "Folk", "Acid": "Electronic", "Alternative & Punk": "Rock", "Ethnic": "International", "Ambient Electronica": "Electronic",
 "avant-garage, experimental; soundcollage; free jazz; post-funk; minimalist; other": "Experimental",
 "Electronica/Dance": "Electronic", "Pop: Experimental": "Pop", "Audio-Collage": "Experimental", "Acoustic": "Folk",
 "Field Recording": "Experimental", "experimental; acoustic; electroacoustic; gahlism; other": "Experimental", "Electro-punk": "Electronic",
 "Revival": "Rock", "Freak-folk": "Folk", "Brazil": "International", "Free Folk": "Folk",
 "Misc. International": "International", "Avante pop": "Pop", "Drum N Bass": "Electronic", "Plunderphonic": "Experimental",
 "World": "International", "experimental pop; electronic pop; improvisation; live; other": "Pop", "Drum and Bass": "Electronic"}

for missing_genre in missing_genres.keys():
    parent_mapping[missing_genre] = missing_genres[missing_genre]

# Save parent_mapping to a JSON file
with open(extract_directory, 'w') as json_file:
    json.dump(parent_mapping, json_file, indent=4)