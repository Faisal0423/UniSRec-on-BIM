# Dataset Preprocessing

If you plan to implement the proposed UniSRec framework to your custom dataset, the first step is to preprocess the data. Below, I outline the specific modifications I made to preprocess my BIM software dataset, called 'Anonymous Allplan data.'


## My dataset 'Anonymous Allplan data'

### 1. Convert raw datasets 

I transformed the raw 'Anonymous Allplan dataset' from its original JSON format, which consisted of multiple files—each representing a single user's data—into a CSV format with desired columns using a Python script `JSON to CSV code.py`.

After converting the data, I organized the dataset into two formats: a single zipped JSON file containing all the original JSON files and the converted CSV file. I created two subfolders within the dataset directory, named 'Metadata' and 'Ratings,' to store the JSON and CSV files, respectively. These folder names follow the convention proposed in the original Git repository.

Here, we take `Anonymous Allplan_data` for example.

```
dataset/
  raw/
    Metadata/
      meta_Allplan_data.zip
    Ratings/
      Allplan_data.csv
```

### 2. Process downstream datasets

The structure of preprocessing file is as follows:

  - Load interaction data from the CSV file in the "rating" folder.
  - Load item text data from the JSON file in the "meta" folder.
  - Split the dataset into training, validation, and test sets.
  - Initialize the device and pre-trained language model (PLM), setting up the computational environment.
  - Create the output directory.
  - Generate PLM embeddings (representing semantic information) and save interaction sequences into atomic files.
  - Store word-dropped PLM embeddings for future use.
  - Save essential data, including Text, User, and Item index files.

#### 2.1 Processing File's Modifications 

The specific modifications I made to the original processing file stated as follows

- I modified the 'load_ratings' method, which generates the User-Item Interaction list from a CSV file, to align with the structure of my specific dataset. The changes were made to extract the necessary columns: visitorId, command_ID, command_name, and timestamp from the CSV file.

- I also modified the original code to accommodate the structure of my specific JSON file. Specifically, I wrote a new method, 'extract_zip,' to extract JSON files from a zipped folder. This method is introduced before the 'load_meta_items method,' which extracts text related to each command from the JSON files.

- In my implementation, I modified the 'preprocess_rating' method, which is used to provide paths for the rating and metadata files. Since my data is stored across multiple JSON files in a zipped format, I adjusted the process to first create a temporary directory for extraction. The zip file is then extracted into this directory, from which the item data is loaded. Additionally, I removed the 'can_items=meta_items' filter from the 'filter_inter' list. This change was necessary because, unlike the original repository, where data was identical in both JSON and CSV formats, my CSV file was only a subset of the JSON data, and only certain columns were initially required.

- The 'generate_text' method, originally designed to generate an item text list from extracted item data, included a check using the 'already_items' list to avoid duplication between JSON and CSV sources. However, this duplication check is unnecessary in my case due to the revised data extraction approach, as previously mentioned. Additionally, I modified the extraction format to accommodate the structure of my JSON data, specifically to target the extraction of features related to command IDs.

- In the method 'generate_item_embedding', which generates text embeddings (numerical representations) for a list of items, I introduced additional 'if-else' conditions and implemented debugging statements for validation purposes. These debugging lines can be disregarded for general usage.


```bash
cd dataset/preprocessing/
python process_allplan.py --dataset Allplan
```

### 3. Process pretrain datasets
In this project, we didn't do pretraining. Please refer to https://github.com/RUCAIBox/UniSRec/tree/master/dataset for preparing the pretrain dataset.


# Useful Files
You may find some files useful for your research, including:
  1. Clean item text (`*.text`);
  2. Index mapping between raw IDs and remapped IDs (`*.user2index`, `*.item2index`);