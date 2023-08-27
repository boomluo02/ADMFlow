import os,sys
current_path = os.path.dirname(os.path.abspath(__file__))
proc_path = current_path.rsplit("/",1)[0]
sys.path.append(current_path)
sys.path.append(proc_path)

import shutil
import argparse

TRAIN_DATATYPE = ["events1", "events2", "best_density_events1", "best_density_events2", "flow"]
TEST_DATATYPE = ["events1", "events2", "flow"]

def manage_trainset(data_root, out_dir, dt=1):

    trainset_path = os.path.join(data_root, "dt"+str(dt), "train")
    out_trainset_path =  os.path.join(out_dir, "dt"+str(dt), "train")
    if(os.path.exists(trainset_path)):
        batch_dir_list = os.listdir(trainset_path)

        for batch_dir in batch_dir_list:
            source_folder = os.path.join(trainset_path, batch_dir)

            for datatype in TRAIN_DATATYPE:
                source_subdir = os.path.join(source_folder, datatype)
                if(os.path.isdir(source_subdir)):
                    print("Begin to copy Path {:s}".format(source_subdir))
                    target_subdir = os.path.join(trainset_path, datatype)
                    shutil.copytree(source_subdir, target_subdir, dirs_exist_ok=True)
            
            print("Path {:s} has been menaged!".format(source_folder))
            # remove source_subdir
            shutil.rmtree(source_folder) 
    else:
        raise Exception('Please provide a valid Dataset Path!')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MDR dataset Menaging')
    parser.add_argument('--save-dir', '-sd', type=str, default='/home/luoxinglong/ADMFlow/dataset/MDR', metavar='PARAMS',
                        help='Main Directory to save all encoding results')
    parser.add_argument('--out-dir', '-od', type=str, default='dataset/MDR', metavar='PARAMS')
    parser.add_argument('--dt', '-dt', type=int, default=1, help='time interval')
    args = parser.parse_args()

    manage_trainset(args.save_dir, args.out_dir, args.dt)
