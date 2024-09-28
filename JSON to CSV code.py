#import modules
import csv
import json 
import pandas as pd
import os
import re

#lets open the json file & load it
def accessing_json_file(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        rows = []
        #The r before the string indicates a raw string, which helps avoid interpreting backslashes (\) within the pattern itself
    extracting_text = re.compile(r'/(\d{4})/([^/]+)/([^/]+)/?') 
        #above line of code has 4 main parts that mean in the URL 
        #Starts and ends with forward slashes (/).
        #Contains four digits (\d{4}) in the middle, which are captured in the first group
        #Contain any characters other than forward slashes ([^/]+) after the year, captured in the second group.
        #Contain another sequence of characters other than forward slashes ([^/]+) after the second group, captured in the third group
    
    for x in data.get('lastVisits', []):
        visitor_id = x.get('visitorId')
        session_id = x.get('idVisit')
        for action in x.get('actionDetails', []):
            url = action.get('url')         #retrieves the value extracting from URL key in JSON file
            match = extracting_text.search(url) # will search on pattern defined in extracting_year within URL string along with its position
            year = match.group(1) #exactly capture the 'year' from the match object    
            text1 = match.group(2) #exactly capture the 'text' from the match object
            text2 = match.group(3) #exactly capture the 'text' from the match object
            last_field = url.split('/')[-1] #split the URL string by '/' and get the last field
            row = {
                    "visitorId": visitor_id,
                    "session_Id": session_id,
                    "command_name": last_field if action.get('pageTitle') is None else action.get('pageTitle'),
                    "command_Id":action.get('pageIdAction'),
                    "command_category":action.get('dimension5'),
                    "command_subcategory":action.get('dimension6'),
                    "command_subcategory 2": action.get('dimension7'),
                    "allpaln_version":year,
                    "category 1":text1,
                    "category 2":text2,
                    "URL": url,
                    "timestamp": action.get('timestamp'),
                    "time_spent":action.get('timeSpent')

                }
            rows.append(row)   
        # Debugging: Print the number of rows processed
        print(f"Number of rows processed for file {file_path}: {len(rows)}")

    return rows
             
all_rows = []
#data_file.close()
json_directory = r"C:\Users\dell\Downloads\Allplan_data" #to find where the JSON files are

fieldnames= ("visitorId", "session_Id", "command_name", "command_Id", "command_category", "command_subcategory", "command_subcategory 2", "allpaln_version", "category 1", "category 2", "URL", "timestamp", "time_spent")
 
for file in os.listdir(json_directory):
    if file.endswith('.json'):
        file_path = os.path.join(json_directory, file)
        all_rows.extend(accessing_json_file(file_path))


#now open the file for writing
csv_path = os.path.join('dataset', 'raw', 'Ratings', 'Allplan_data.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file: #newline tells Python not to add any extra newline characters when writing to the file
                                              #specifies the character encoding   
    writer = csv.DictWriter(csv_file,fieldnames = fieldnames) #dicwriter a specialized writer class that helps to write data from dictionaries 
    writer.writeheader() 
    writer.writerows(all_rows) 
    
    

        

print(f"Data from JSON files has been written to output_file.csv")
