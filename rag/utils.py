import os
import pandas as pd
import ast


def write_to_txt_file(filename, text):
    try:
        with open(filename,'a') as file:
            file.write(text)
    except Exception as e:
        print("Error:", e)

def get_paths(folder_path, extension):
    paths = []
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                paths.append(os.path.join(root, file))
    
    return paths


df = pd.read_csv("output/generated_QAs/by_pages/hDVS_Internal_use_FAQ_V2-5_June-2016_by_pages.csv")
def refine_csv(path):
    dic = []
    for each in df["QA-pairs"]:
        
        if not (each.endswith("]}") or each.endswith("]\n}")):
            if each[-1] == "]":
                each+="}"
                # print("add }")
            elif each.endswith("}") and not each.endswith("]}"):
                each = each[:-1] + "]"
                # print("add ]")
            # else: each+="}"
            
        try:
            dic.append(ast.literal_eval(each))
        except Exception as e:
            print(each)
            print(each[-1])
            print(e)
            print("\n\n\n")
            pass

    dfs = pd.DataFrame(dic)
    # print(dfs)
    return dfs
