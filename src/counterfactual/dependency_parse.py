# Generate dependency parses in CoNNL-U format for a given dataset
# Author: Thomas Hikaru Clark (thclark@mit.edu)
# Example Usage:
# python dep_parse.py --lang en --udpipe_model_path models/model.udpipe \
#   --data_dir wiki40b-txt-final --parse_dir wiki40b-txt-parsed \
#   --partitions "train,test,valid"