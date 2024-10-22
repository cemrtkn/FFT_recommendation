import zipfile
import os
import csv
import json


current_directory = os.getcwd()
extract_directory = os.path.join(current_directory, '..', '..', 'data', 'parent_mapping.json')
metadata_directory = os.path.join(current_directory, '..', '..','data', 'fma_metadata.zip')
tracks_extract_directory = os.path.join(current_directory, '..', '..', 'data', 'tracks.csv')
genres = ''
index_mapping = {}
parent_mapping = {}

def rec_parent_getter(genre_id, index_mapping):
    parent_genre_idx = index_mapping[genre_id]["parent_id"]
    if parent_genre_idx == '0':
        return index_mapping[genre_id]["name"]
    
    return rec_parent_getter(parent_genre_idx, index_mapping)

def metad_data_extractor(metadata_directory):
    with zipfile.ZipFile(metadata_directory, 'r') as zip_ref:
        # Find the tracks.csv file inside the ZIP archive
        tracks = [file_name for file_name in zip_ref.namelist() if file_name == 'fma_metadata/tracks.csv']
        
        if tracks:
            with zip_ref.open(tracks[0]) as input_file:
                with open(tracks_extract_directory, 'wb') as output_file:
                    output_file.write(input_file.read())  

            print(f"File 'tracks.csv' extracted to {tracks_extract_directory}")
        else:
            print("No tracks.csv file found in the archive.")


with zipfile.ZipFile(metadata_directory, 'r') as zip_ref:
    genres = [file_name for file_name in zip_ref.namelist() if file_name == 'fma_metadata/genres.csv']

    with zip_ref.open(genres[0]) as genre_file:
        content = genre_file.read().decode('utf-8').splitlines()

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
    parent_genre_name = rec_parent_getter(genre_id, index_mapping)
    print(genre_name, " ----> ",parent_genre_name )
    parent_mapping[genre_name] = parent_genre_name



missing_genres = {"chamber music [delete]": "Classical", "Psych REPLACE w/ Psych-Rock": "Rock", "Electronica": "Electronic", "Club": "Electronic", 
 "Folk/Rock": "Folk", "Acid": "Electronic", "Alternative & Punk": "Rock", "Ethnic": "International", "Ambient Electronica": "Electronic",
 "avant-garage, experimental; soundcollage; free jazz; post-funk; minimalist; other": "Experimental",
 "Electronica/Dance": "Electronic", "Pop: Experimental": "Pop", "Audio-Collage": "Experimental", "Acoustic": "Folk",
 "Field Recording": "Experimental", "experimental; acoustic; electroacoustic; gahlism; other": "Experimental", "Electro-punk": "Electronic",
 "Revival": "Rock", "Freak-folk": "Folk", "Brazil": "International", "Free Folk": "Folk",
 "Misc. International": "International", "Avante pop": "Pop", "Drum N Bass": "Electronic", "Plunderphonic": "Experimental",
 "World": "International", "experimental pop; electronic pop; improvisation; live; other": "Pop", "Drum and Bass": "Electronic",
 "dub": "Electronic", "Dub": "Electronic", "Soul-RandB": "Soul-RnB", "Hip Hop": "Hip-Hop"}

for missing_genre in missing_genres.keys():
    parent_mapping[missing_genre] = missing_genres[missing_genre]
print(set(list(parent_mapping.values())),len(set(list(parent_mapping.values()))) )
# Save parent_mapping to a JSON file
with open(extract_directory, 'w') as json_file:
    json.dump(parent_mapping, json_file, indent=4)