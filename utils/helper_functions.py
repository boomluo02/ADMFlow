import torch
import numpy
import os
import smtplib
import json
import shutil

def move_list_to_cuda(list_of_dicts, gpu):
    for i in range(len(list_of_dicts)):
        list_of_dicts[i] = move_dict_to_cuda(list_of_dicts[i], gpu)
    return list_of_dicts

def move_dict_to_cuda(dictionary_of_tensors, gpu):
    if isinstance(dictionary_of_tensors, dict):
        dict_cuda = {}
        for key, value in dictionary_of_tensors.items():
            if isinstance(value, list):
                value = move_list_to_cuda(value, gpu)
            else:
                value = move_dict_to_cuda(value, gpu)
            dict_cuda[key] = value
        return dict_cuda
    elif isinstance(dictionary_of_tensors, str):
        return dictionary_of_tensors
    else:
        dictionary_of_tensors = dictionary_of_tensors.to(gpu, dtype=torch.float)
        return dictionary_of_tensors        

def get_values_from_key(input_list, key):
    # Returns all the values with the same key from
    # a list filled with dicts of the same kind
    out = []
    for i in input_list:
        out.append(i[key])
    return out

def create_save_path(subdir, name, restart=False):
    # Check if sub-folder exists, and create if necessary
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    # Create a new folder (named after the name defined in the config file)
    path = os.path.join(subdir, name)
    # Check if path already exists. if yes -> append a number
    if os.path.exists(path):
        if not restart:
            i = 1
            while os.path.exists(path + "_" + str(i)):
                i += 1
            path = path + '_' + str(i)
            os.mkdir(path)
    else:
        os.makedirs(path)
    return path

def get_nth_element_of_all_dict_keys(dict, idx):
    out_dict = {}
    for k in dict.keys():
        d = dict[k][idx]
        if isinstance(d,torch.Tensor):
            out_dict[k]=d.detach().cpu().item()
        else:
            out_dict[k]=d
    return out_dict

def get_number_of_saved_elements(path, template, first=1):
    i = first
    while True:
        if os.path.exists(os.path.join(path,template.format(i))):
            i+=1
        else:
            break
    return range(first, i)

def create_file_path(subdir, name):
    # Check if sub-folder exists, else raise exception
    if not os.path.exists(subdir):
        raise Exception("Path {} does not exist!".format(subdir))
    # Check if file already exists, else create path
    if not os.path.exists(os.path.join(subdir,name)):
        return os.path.join(subdir,name)
    else:
        path = os.path.join(subdir,name)
        prefix,suffix = path.split('.')
        i = 1
        while os.path.exists("{}_{}.{}".format(prefix,i,suffix)):
            i += 1
        return "{}_{}.{}".format(prefix,i,suffix)

def update_dict(dict_old, dict_new):
    # Update all the entries of dict_old with the new values(that have the identical keys) of dict_new
    for k in dict_new.keys():
        if k in dict_old.keys():
            # Replace the entry
            if isinstance(dict_new[k], dict):
                update_dict(dict_old[k], dict_new[k])
            else:
                dict_old[k] = dict_new[k]
    return dict_old

class ImagePadder(object):
    # =================================================================== #
    # In some networks, the image gets downsized. This is a problem, if   #
    # the to-be-downsized image has odd dimensions ([15x20]->[7.5x10]).   #
    # To prevent this, the input image of the network needs to be a       #
    # multiple of a minimum size (min_size)                               #
    # The ImagePadder makes sure, that the input image is of such a size, #
    # and if not, it pads the image accordingly.                          #
    # =================================================================== #

    def __init__(self, min_size=64):
        # --------------------------------------------------------------- #
        # The min_size additionally ensures, that the smallest image      #
        # does not get too small                                          #
        # --------------------------------------------------------------- #
        self.min_size = min_size
        self.pad_height = None
        self.pad_width = None

    def pad(self, image):
        # --------------------------------------------------------------- #
        # If necessary, this function pads the image on the left & top    #
        # --------------------------------------------------------------- #
        height, width = image.shape[-2:]
        if self.pad_width is None:
            self.pad_height = (self.min_size - height % self.min_size)%self.min_size
            self.pad_width = (self.min_size - width % self.min_size)%self.min_size
        # else:
        #     pad_height = (self.min_size - height % self.min_size)%self.min_size
        #     pad_width = (self.min_size - width % self.min_size)%self.min_size
        #     if pad_height != self.pad_height or pad_width != self.pad_width:
        #         raise
        return nn.ZeroPad2d((self.pad_width, 0, self.pad_height, 0))(image)

    def unpad(self, image):
        # --------------------------------------------------------------- #
        # Removes the padded rows & columns                               #
        # --------------------------------------------------------------- #
        return image[..., self.pad_height:, self.pad_width:]

# class InputPadder:
#     """ Pads images such that dimensions are divisible by 8 """

#     def __init__(self, dims, mode='sintel', eval_pad_rate=32):
#         self.eval_pad_rate = eval_pad_rate
#         self.ht, self.wd = dims[-2:]
#         pad_ht = (((self.ht // eval_pad_rate) + 1) * eval_pad_rate - self.ht) % eval_pad_rate
#         pad_wd = (((self.wd // eval_pad_rate) + 1) * eval_pad_rate - self.wd) % eval_pad_rate
#         if mode == 'sintel':
#             self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
#         else:
#             self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

#     def pad(self, *inputs):
#         return [F.pad(x, self._pad, mode='replicate') for x in inputs]

#     def unpad(self, x):
#         ht, wd = x.shape[-2:]
#         c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
#         return x[..., c[0]:c[1], c[2]:c[3]]

class Logger:
    # Logger of the Training/Testing Process
    def __init__(self, save_path, custom_name='log.txt'):
        self.toWrite = {}
        self.signalization = "========================================"
        self.path = os.path.join(save_path,custom_name)

    def initialize_file(self, mode):
        # Mode : "Training" or "Testing"
        with open(self.path, 'a') as file:
            file.write(self.signalization + " " + mode + " " + self.signalization + "\n")

    def write_as_list(self, dict_to_write, overwrite=False):
        if overwrite:
            if os.path.exists(self.path):
                os.remove(self.path)
        with open(self.path, 'a') as file:
            for entry in dict_to_write.keys():
                file.write(entry+"="+json.dumps(dict_to_write[entry])+"\n")

    def write_dict(self, dict_to_write, array_names=None, overwrite=False, as_list=False):
        if overwrite:
            open_type = 'w'
        else:
            open_type = 'a'
        dict_to_write = self.check_for_arrays(dict_to_write, array_names)
        if as_list:
            self.write_as_list(dict_to_write, overwrite)
        else:
            with open(self.path, open_type) as file:
                #if "epoch" in dict_to_write:
                 #   file.write("Epoch")
                file.write(json.dumps(dict_to_write) + "\n")

    def write_line(self,line, verbose=False):
        with open(self.path, 'a') as file:
            file.write(line + "\n")
        if verbose:
            print(line)

    def arrays_to_dicts(self, list_of_arrays, array_name, entry_name):
        list_of_arrays = numpy.array(list_of_arrays).T
        out = {}
        for i in range(list_of_arrays.shape[0]):
            out[array_name+'_'+entry_name[i]] = list(list_of_arrays[i])
        return out


    def check_for_arrays(self, dict_to_write, array_names):
        if array_names is not None:
            names = []
            for n in range(len(array_names)):
                if hasattr(array_names[n], 'name'):
                    names.append(array_names[n].name)
                elif hasattr(array_names[n],'__name__'):
                    names.append(array_names[n].__name__)
                elif hasattr(array_names[n],'__class__'):
                    names.append(array_names[n].__class__.__name__)
                else:
                    names.append(array_names[n])

        keys = dict_to_write.keys()
        out = {}
        for entry in keys:
            if hasattr(dict_to_write[entry], '__len__') and len(dict_to_write[entry])>0:
                if isinstance(dict_to_write[entry][0], numpy.ndarray) or isinstance(dict_to_write[entry][0], list):
                    out.update(self.arrays_to_dicts(dict_to_write[entry], entry, names))
                else:
                    out.update({entry:dict_to_write[entry]})
            else:
                out.update({entry: dict_to_write[entry]})
        return out
