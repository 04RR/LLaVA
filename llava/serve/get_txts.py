import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def generate_responses_for_inputs(text_strs, image_paths, args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    responses = []
    
    for text, image_path in zip(text_strs, image_paths):
        print(f"User: {text}")
        inp = text
        if image_path is not None:
            image = load_image(image_path)
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
            
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            conv.append_message(conv.roles[0], inp)
            
        # Rest of the code remains the same until text generation
        
        responses.append(outputs)
    
    return responses

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_strs", type=str, required=True)
    parser.add_argument("--image_paths", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    args = parser.parse_args()
    
    model_args = argparse.Namespace(
    model_path=args.model_path,  # Specify the correct model path
    model_base=args.model_base,
    image_file=None,  # Not needed since we're passing image paths separately
    num_gpus=1,
    conv_mode=None,
    temperature=0.2,
    max_new_tokens=512,
    load_8bit=False,
    load_4bit=False,
    debug=False
    )

    responses = generate_responses_for_inputs([args.text_strs], [args.image_paths], model_args)
    for text, response in zip(args.image_paths, responses):
        print(f"Image: {text}")
        print(f"Description: {response}\n")
