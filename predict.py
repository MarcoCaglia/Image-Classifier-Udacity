from utility import load_model, predict
from transform import process_image
import torch as tf
import numpy as np
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('input',type=str)
parser.add_argument('--checkpoint',default='icp.pth',type=str)
parser.add_argument('--top_k',default=5,type=int)
parser.add_argument('--mapping',default='cat_to_name.json',type=str)
parser.add_argument('--gpu',action='store_true',default=False)
args = parser.parse_args()

# Load trained model
model = load_model(args.checkpoint)
with open(args.mapping,'r') as mapping_file:
    model.class_to_idx = json.load(mapping_file)

# Convert picture to model input
picture_path = args.input
model_input = tf.from_numpy(process_image(picture_path))

# Get prediction from model
if args.gpu:
    device = 'cuda'
else:
    device = 'cpu'
model.to(device)
results = predict(model,model_input,top_k=args.top_k,device=device)

# Display results
for name, probability in results:
    print('{}: {}%'.format(name,round(float(probability)*100,2)))
    