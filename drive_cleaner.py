import os
import shutil
import glob
import json
import numpy as np
from datetime import datetime, timedelta



class DriveCleaner:
    def __init__(self, paths_to_clean = None, data_path = None, last_edit = 24, config_path = None,
                 current_path = None, extensions = ['wav']):
        
        self.paths_to_clean = paths_to_clean
        self.data_path = data_path
        self.last_edit =timedelta(hours = last_edit)
        self.config_path = config_path
        self.extensions = extensions
        self.all_files = None
        if not current_path:
            self.current_path  = os.getcwd()
        else:
            self.current_path = current_path

    def is_digit(self, digit_str):
        try:
            return np.float(digit_str)
        except:
            return False
    def get_last_edit_value(self):
            # return last_edit if is exist, else return default 
            # remove config file if exist
            if not self.config_path:
                print("no config_path")
                return self.last_edit
            try:
                os.remove(os.path.join(self.current_path, "config.txt"))
            except:
                print("no config to remove")

            try:
                shutil.copy(self.config_path, self.current_path)
                config = json.load(open(os.path.join(self.current_path, "config.txt")))   # audio_legnth  queries    score_threshold
                last_edit_val = self.is_digit(config['last_edit'])
                self.extensions = config["extensions"]
                if last_edit_val:
                    self.last_edit = timedelta(hours = last_edit_val)
                    print(f"self.last_edit from config: {self.last_edit}")
                return self.last_edit
            except:
                print("couldn't read  last_edit from config, make default: 24")
                return self.last_edit
                

    def get_all_files(self):
        all_files = []
        for base_path in self.paths_to_clean:
            files = os.listdir(base_path)
            print(len(files))
            files= [os.path.join(base_path, file) for file in files if (file[-3:] in self.extensions)]
            print(len(files))
            all_files.extend(files)

        return all_files

    def get_last_edit(self, file_path):
        time_stamp =  os.path.getmtime(file_path)
        return datetime.fromtimestamp(time_stamp)

    def remove_files_exceed_limit(self):
        for file in self.all_files:
            editing_date = self.get_last_edit(file)
            time_delta = datetime.now() - editing_date
            if time_delta > self.last_edit:
                try:
                    os.remove(file)
                    print(f"deleting {file}")
                except:
                    print(f"couldn't delete {file}")

    def clean_data_path(self):
        files = os.listdir(self.data_path)  
        print(self.extensions)
        files= [os.path.join(self.data_path, file) for file in files if (file[-3:] not in self.extensions)]
        for file in files:
            try:
                os.remove(file)
                print(f"removing:  {file}")
            except:
                print(f"can't remove:  {file}")

    def pre_process(self):
        # adjust paths_to_clean and t from config file
        self.last_edit = self.get_last_edit_value()
        print(self.last_edit)
        # get all files complete path in paths_to_clean
        self.all_files = self.get_all_files()

    def process(self):
        print("+++++++++++++++++++++++++ run pre_process +++++++++++++++++++++++++")
        self.pre_process()
        print("************************* run remove_files_exceed_limit *************************")
        self.remove_files_exceed_limit()
        print("------------------------- run clean_data_path -------------------------")
        self.clean_data_path()
