import argparse
import collections
import gzip
import zipfile
import html
import json
import os
import random
import re
import torch
from tqdm import tqdm
import csv
import tempfile
from utils import check_path, set_device, load_plm, amazon_dataset2fullname


def load_ratings(file):
    users, items, inters = set(), set(), set()
    with open(file, 'r', encoding='utf-8') as fp:
        reader = csv.reader(fp)
        header = next(reader)
        command_id_idx = 3
        visitor_id_idx = 0
        command_name_idx = 2
        timestamp_idx = 11
        for line in tqdm(reader, desc='Load ratings'):
            try:
                # Extract selected columns based on indices
                command_ID = line[command_id_idx]
                visitorId = line[visitor_id_idx]
                command_name = line[command_name_idx]
                timestamp = int(line[timestamp_idx])  # Convert timestamp to int
                # Add to sets and create the interaction tuple
                users.add(visitorId)
                items.add(command_ID)
                inters.add((visitorId, command_ID, command_name, timestamp))
            except (ValueError) as e:
                print(f"Error processing row {line}: {e}")
    print(f"Number of users: {len(users)}")
    print(f"Number of items: {len(items)}")
    print(f"Number of interactions: {len(inters)}")
    print("Sample interactions:", list(inters)[:5])
    return users, items, inters #(inters = A list of interactions. Each interaction includes a user ID and an command ID) 

def extract_zip(zip_path, extract_to='extracted_files'):
    """Extract all files from a zip archive."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return extract_to

def load_meta_items(directory):
    items = set()
    for filename in os.listdir(directory): 
        if filename.endswith ('.json'):
            meta_file_path =os.path.join(directory, filename)
            with open(meta_file_path, 'r', encoding= 'utf-8') as fp:
                data = json.load(fp)  # Load the entire JSON content
                # Navigate through the JSON structure to get to 'lastVisits'
                last_visits = data.get('lastVisits', [])
                for visit in last_visits:
                    action_details = visit.get('actionDetails', [])
                    for action in tqdm(action_details, desc='Load metas'):
                        try:
                            items.add(action['pageIdAction'])  # 'pageIdAction' is the identifier we need
                        except KeyError as e:
                            print(f"KeyError: {e}. Missing key in: {action}")
                            continue     
    return items



def get_user2count(inters):  #counts the interactions of each user across all items they have interacted
    user2count = collections.defaultdict(int)
    for unit in inters:
        user2count[unit[0]] += 1
    return user2count


def get_item2count(inters): #counts how many items got interacted in a tuples of inters
    item2count = collections.defaultdict(int)
    for unit in inters:
        item2count[unit[1]] += 1
    print(f"item2count: {item2count}")
    return item2count

def generate_candidates(unit2count, threshold): #It returns the set of candidates and the count of units..
                                                # that do not meet the threshold.
                                                
                                                #A threshold is a value that dictates the minimum number of interactions..
                                                #a user must have for their interactions to be included
                                                
                                                #A threshold value that dictates the minimum number of interactions 
                                                #an item must have for it to be included
    cans = set()
    for unit, count in unit2count.items():
        if count >= threshold:
            cans.add(unit)
    return cans, len(unit2count) - len(cans)

def filter_inters(inters, user_k_core_threshold=0, item_k_core_threshold=0):
    new_inters = []  #this fn filters interactions based a Set of candidate items (can_items) if provided, 
                    #user and item k-core threshold 
    

    # filter by k-core (Filter interactions based on user and item k-core thresholds)

    if user_k_core_threshold or item_k_core_threshold:
        print('\nFiltering by k-core:')
        idx = 0
        user2count = get_user2count(inters)
        item2count = get_item2count(inters)
        while True:
            new_user2count = collections.defaultdict(int)
            new_item2count = collections.defaultdict(int)
            users, n_filtered_users = generate_candidates(user2count, user_k_core_threshold)
            items, n_filtered_items = generate_candidates(item2count, item_k_core_threshold)
            if n_filtered_users == 0 and n_filtered_items == 0:
                break
            for unit in inters:
                if unit[0] in users and unit[1] in items:
                    new_inters.append(unit)
                    new_user2count[unit[0]] += 1
                    new_item2count[unit[1]] += 1
            idx += 1
            inters, new_inters = new_inters, []
            user2count, item2count = new_user2count, new_item2count
            print('    Epoch %d The number of inters: %d, users: %d, items: %d'
                    % (idx, len(inters), len(user2count), len(item2count)))
    return inters


def make_inters_in_order(inters): #Sort Interactions by Timestamp ensuring that each user's interactions are ordered chronologically.
    user2inters, new_inters = collections.defaultdict(list), list() #A defaultdict where each key is a user ID, and the value is a list of that user's interactions.
    for inter in inters:
        user, command_ID, command_name, timestamp = inter
        user2inters[user].append((user, command_ID, command_name, timestamp))
    for user in user2inters:
        user_inters = user2inters[user]
        user_inters.sort(key=lambda d: d[3]) #sort the interactions for the user by the time (the 4th element in tuple) using a lambda fn
        for inter in user_inters:
            new_inters.append(inter)
    return new_inters


def preprocess_rating(args):
    dataset_full_name = amazon_dataset2fullname[args.dataset]

    print('Process rating data: ')
    print(' Dataset: ', dataset_full_name)

    # load ratings
    rating_file_path = os.path.join(args.input_path, 'Ratings', dataset_full_name + '.csv')
    rating_users, rating_items, rating_inters = load_ratings(rating_file_path)

    # load item IDs with meta data
    zip_file_path = os.path.join(args.input_path, 'Metadata', f'meta_{dataset_full_name}.zip')
    # Use a temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        # Step 1: Extract the zip file into the temporary directory
        extract_zip(zip_file_path, temp_dir)

        # Step 2: Load items from the extracted files
        meta_items = load_meta_items(temp_dir)

    # 1. Filter items w/o meta data;
    # 2. K-core filtering;
    print('The number of raw inters: ', len(rating_inters))
    print(f"user_k: {args.user_k}")
    print(f"item_k: {args.item_k}")
    rating_inters = filter_inters(rating_inters,  user_k_core_threshold=args.user_k,
                                  item_k_core_threshold=args.item_k)

    # sort interactions chronologically for each user
    rating_inters = make_inters_in_order(rating_inters)
    print('\n')

    # return: list of (visitorID, command_ID, command_name, timestamp)
    return rating_inters


def get_visitorId_command_Id_from_ratings(ratings): #function processes a list of ratings and extracts the unique users and items from it
    print(f"ratings is this: {ratings[:4]}")
    users, items = set(), set()
    for line in ratings:
        visitorId, command_ID, command_name, timestamp = line
        users.add(visitorId)
        items.add(command_ID)
    return users, items


def clean_text(raw_text):
    if raw_text is None:
        return ""  # Return an empty string if the input is None
    if isinstance(raw_text, list):
        cleaned_text = ' '.join(raw_text)
    elif isinstance(raw_text, dict):
        cleaned_text = str(raw_text)
    else:
        cleaned_text = raw_text
    cleaned_text = html.unescape(cleaned_text)
    cleaned_text = re.sub(r'["\n\r]*', '', cleaned_text)
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == '.':
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + '.'
    else:
        cleaned_text = cleaned_text[:index] + '.'
    if len(cleaned_text) >= 2000:
        cleaned_text = ''
    return cleaned_text

import sys

def generate_text(args, item, features):
    item_text_list = []
    #already_items = set()
    dataset_full_name = amazon_dataset2fullname[args.dataset]
    zip_file_path = os.path.join(args.input_path, 'Metadata', f'meta_{dataset_full_name}.zip')
    with tempfile.TemporaryDirectory() as temp_dir:
        extract_zip(zip_file_path, temp_dir)
        # Iterate over all extracted JSON files in the temporary directory
        for filename in os.listdir(temp_dir):
            if filename.endswith('.json'):
                meta_file_path = os.path.join(temp_dir, filename)
    
                with open(meta_file_path, 'r', encoding='utf-8') as fp:
                    try:
                        data = json.load(fp)
                    except json.JSONDecodeError as e:
                        print(f"Error reading {meta_file_path}: {e}")
                        continue
                    # Navigate the JSON structure carefully
                    last_visits = data.get("lastVisits", [])
                    if not last_visits:
                        print(f"No lastVisits found in {filename}")
                        continue
                    for visit in last_visits:
                        action_details = visit.get("actionDetails", [])
                        if not action_details:
                            print(f"No actionDetails found for visit in {filename}")
                            continue
                        for action in action_details:
                            item = action.get('pageIdAction')
                            if item is None:
                                print(f"No pageIdAction found in action: {action}")
                                continue
                            text = ''
                            for meta_key in features:
                                if meta_key in action:
                                    meta_value = clean_text(action[meta_key])
                                    text += meta_value + ' '        
                            item_text_list.append([item, text])
    return item_text_list

def load_text(file):
    item_text_list = []
    with open(file, 'r') as fp:
        fp.readline()
        for line in fp:
            try:
                item, text = line.strip().split('\t', 1)
            except ValueError:
                item = line.strip()
                text = '.'
            item_text_list.append([item, text])
    return item_text_list


def write_text_file(item_text_list, file):
    print('Writing text file: ')
    with open(file, 'w', encoding= 'utf-8') as fp:
        fp.write('command_ID:token\ttext:token_seq\n')
        for command_ID, text in item_text_list:
            fp.write(str(command_ID) + '\t' + text + '\n')


def preprocess_text(args, rating_inters): #preprocesses text data related to items based on ratings interactions
    print('Process text data: ')
    print(' Dataset: ', args.dataset)
    rating_users, rating_items = get_visitorId_command_Id_from_ratings(rating_inters)

    #load item text and clean                            #command_title, command_subtitle, URl 
    item_text_list = generate_text(args, rating_items, ['pageTitle', 'dimension5', 'subtitle'])
    print('\n')                                       #A list of metadata features to be included in the generated text

    #return: list of (item_ID, cleaned_item_text)
    
    return item_text_list
    


def convert_inters2dict(inters):
    user2items = collections.defaultdict(list)
    user2index, item2index = dict(), dict()
    for inter in inters:
        user, item, command_name, timestamp = inter
        # Debugging statements to inspect values before insertion
        #print(f"Processing row - User: {user}, Item: {item}, Command: {command_name}, Timestamp: {timestamp}")
        if user not in user2index:
            user2index[user] = len(user2index)
        if item not in item2index:
            item2index[item] = len(item2index)
        user2items[user2index[user]].append(item2index[item])
    #debugging statement
    print(f"following item2index: {item2index}")
    return user2items, user2index, item2index


def generate_training_data(args, rating_inters):
    print('Split dataset: ')
    print(' Dataset: ', args.dataset)
    
    # generate train valid test
    user2items, user2index, item2index = convert_inters2dict(rating_inters)
    train_inters, valid_inters, test_inters = dict(), dict(), dict()
    for u_index in range(len(user2index)):
        inters = user2items[u_index]
        # leave one out
        train_inters[u_index] = [str(i_index) for i_index in inters[:-2]]
        valid_inters[u_index] = [str(inters[-2])]
        test_inters[u_index] = [str(inters[-1])]
        assert len(user2items[u_index]) == len(train_inters[u_index]) + \
               len(valid_inters[u_index]) + len(test_inters[u_index])
    return train_inters, valid_inters, test_inters, user2index, item2index


def load_unit2index(file):
    unit2index = dict()
    with open(file, 'r') as fp:
        for line in fp: 
            unit, index = line.strip().split('\t')
            unit2index[unit] = int(index)
    return unit2index


def write_remap_index(unit2index, file): #To write a dict mapping users or items to indices to a file in a tab-sep format
    with open(file, 'w') as fp:
        for unit in unit2index:
            fp.write(unit + '\t' + str(unit2index[unit]) + '\n')

#function generates text embeddings (numerical representation) for a list of items using a specified tokenizer and model
def generate_item_embedding(args, item_text_list, item2index, tokenizer, model, word_drop_ratio=-1):
    print(f'Generate Text Embedding by {args.emb_type}: ')
    print(' Dataset: ', args.dataset)
    unique_items = sorted(set(str(item) for item, text in item_text_list))
    print(f"Total unique items: {len(unique_items)}")
    print(f"Total items in item2index after construction: {len(item2index)}")
    items, texts = zip(*item_text_list)
    order_texts = [[0]] * len(item2index)
    for item, text in zip(items, texts):
        if str(item) not in item2index:
            print(f"Warning: '{(item)}' in item2index is missing")
        else:
            order_texts[item2index[str(item)]] = text
    print("Order texts after filling:")
    for text in order_texts:
        assert text != [0]
    embeddings = []
    start, batch_size = 0, 4
    while start < len(order_texts):
        sentences = order_texts[start: start + batch_size]
        if word_drop_ratio > 0:
            print(f'Word drop with p={word_drop_ratio}')
            new_sentences = []
            for sent in sentences:
                new_sent = []
                sent = sent.split(' ')
                for wd in sent:
                    rd = random.random()
                    if rd > word_drop_ratio:
                        new_sent.append(wd)
                new_sent = ' '.join(new_sent)
                new_sentences.append(new_sent)
            sentences = new_sentences
        encoded_sentences = tokenizer(sentences, padding=True, max_length=512,
                                      truncation=True, return_tensors='pt').to(args.device)
        outputs = model(**encoded_sentences)
        if args.emb_type == 'CLS':
            cls_output = outputs.last_hidden_state[:, 0, ].detach().cpu()
            embeddings.append(cls_output)
        elif args.emb_type == 'Mean':
            masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
            mean_output = masked_output[:,1:,:].sum(dim=1) / \
                encoded_sentences['attention_mask'][:,1:].sum(dim=-1, keepdim=True)
            mean_output = mean_output.detach().cpu()
            embeddings.append(mean_output)
        start += batch_size
    embeddings = torch.cat(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape)

    # suffix=1, output DATASET.feat1CLS, with word drop ratio 0;
    # suffix=2, output DATASET.feat2CLS, with word drop ratio > 0;
    if word_drop_ratio > 0:
        suffix = '2'
    else:
        suffix = '1'

    file = os.path.join(args.output_path, args.dataset,
                        args.dataset + '.feat' + suffix + args.emb_type)
    embeddings.tofile(file)


def convert_to_atomic_files(args, train_data, valid_data, test_data):
    print('Convert dataset: ')
    print(' Dataset: ', args.dataset)
    uid_list = list(train_data.keys())
    uid_list.sort(key=lambda t: int(t))

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.train.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid]
            seq_len = len(item_seq)
            for target_idx in range(1, seq_len):
                target_item = item_seq[-target_idx]
                seq = item_seq[:-target_idx][-50:]
                file.write(f'{uid}\t{" ".join(seq)}\t{target_item}\n')

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.valid.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = train_data[uid][-50:]
            target_item = valid_data[uid][0]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.test.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
        for uid in uid_list:
            item_seq = (train_data[uid] + valid_data[uid])[-50:]
            target_item = test_data[uid][0]
            file.write(f'{uid}\t{" ".join(item_seq)}\t{target_item}\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Allplan', help='Pantry / Scientific / Instruments / Arts / Office / Allplan')
    parser.add_argument('--user_k', type=int, default=5, help='user k-core filtering')
    parser.add_argument('--item_k', type=int, default=5, help='item k-core filtering')
    parser.add_argument('--input_path', type=str, default=r'dataset\raw')
    parser.add_argument('--output_path', type=str, default=r'dataset\downstream')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--plm_name', type=str, default='bert-base-uncased')
    parser.add_argument('--emb_type', type=str, default='CLS', help='item text emb type, can be CLS or Mean')
    parser.add_argument('--word_drop_ratio', type=float, default=-1, help='word drop ratio, do not drop by default')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # load interactions from raw rating file
    rating_inters = preprocess_rating(args)

    # load item text from raw meta data file
    item_text_list = preprocess_text(args, rating_inters)

    # split train/valid/test
    train_inters, valid_inters, test_inters, user2index, item2index = \
        generate_training_data(args, rating_inters)

    # device & plm initialization
    device = set_device(args.gpu_id)
    args.device = device
    plm_tokenizer, plm_model = load_plm(args.plm_name)
    plm_model = plm_model.to(device)

    # create output dir
    check_path(os.path.join(args.output_path, args.dataset))

    # generate PLM emb and save interaction sequences into atomic files
    generate_item_embedding(args, item_text_list, item2index, 
                            plm_tokenizer, plm_model, word_drop_ratio=-1)
    # pre-stored word drop PLM embs
    if args.word_drop_ratio > 0:
        generate_item_embedding(args, item_text_list, item2index, 
                                plm_tokenizer, plm_model, word_drop_ratio=args.word_drop_ratio)

    # 
    convert_to_atomic_files(args, train_inters, valid_inters, test_inters)

    # save useful data
    write_text_file(item_text_list, os.path.join(args.output_path, args.dataset, f'{args.dataset}.text'))
    write_remap_index(user2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.user2index'))
    write_remap_index(item2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item2index'))
