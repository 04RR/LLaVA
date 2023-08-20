# import argparse
# import torch
# from PIL import Image
# from io import BytesIO
# import requests
# from transformers import TextStreamer

# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

# def load_image(image_file):
#     if image_file.startswith('http') or image_file.startswith('https'):
#         response = requests.get(image_file)
#         image = Image.open(BytesIO(response.content)).convert('RGB')
#     else:
#         image = Image.open(image_file).convert('RGB')
#     return image

# def generate_outputs(texts, image_paths, model_path, model_base="gpt2"):
#     # Load model and tokenizer
#     model_name = get_model_name_from_path(model_path)
#     # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base)
#     tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, False, False)
    
#     outputs = []

#     for text, image_path in zip(texts, image_paths):
#         image = load_image(image_path)
#         image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

#         input_text = text
#         conv_prompt = input_text
#         input_ids = tokenizer_image_token(conv_prompt, tokenizer, return_tensors='pt').unsqueeze(0).cuda()
        
#         streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
#         keywords = ["<|endoftext|>"]
#         stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

#         with torch.inference_mode():
#             output_ids = model.generate(
#                 input_ids,
#                 images=image_tensor,
#                 do_sample=True,
#                 temperature=0.2,
#                 max_new_tokens=1024,
#                 streamer=streamer,
#                 use_cache=True,
#                 stopping_criteria=[stopping_criteria])

#         generated_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
#         outputs.append(generated_text)
    
#     return outputs

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model-path", type=str, default="facebook/opt-350m", help="Path to the pretrained model")
#     parser.add_argument("--model-base", type=str, default="gpt2", help="Base model type")
#     parser.add_argument("--texts", nargs="+", help="List of texts for which to generate outputs")
#     parser.add_argument("--image-paths", nargs="+", help="List of image paths corresponding to the texts")
#     args = parser.parse_args()

#     generated_outputs = generate_outputs(args.texts, args.image_paths, args.model_path, args.model_base)
    
#     for input_text, output_text in zip(args.texts, generated_outputs):
#         print("Input:", input_text)
#         print("Generated Output:", output_text)
#         print("=" * 40)

import argparse
import torch
from PIL import Image
from io import BytesIO
import requests
from transformers import TextStreamer
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def generate_responses(model, tokenizer, image_processor, text_inputs, image_paths):
    responses = []

    for text_input, image_path in zip(text_inputs, image_paths):
        image = load_image(image_path)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

        # Prepare input prompt for conversation
        prompt = text_input

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        responses.append(outputs)

    return responses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--text-inputs", nargs='+', required=True, help="List of text inputs")
    parser.add_argument("--image-paths", nargs='+', required=True, help="List of image file paths")
    args = parser.parse_args()

    disable_torch_init()
    model_name = "opt-350m"
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_name, args.load_8bit, args.load_4bit)

    conv_mode = "llava_v0"  # Update this based on your requirements
    conv = conv_templates[conv_mode].copy()

    responses = generate_responses(model, tokenizer, image_processor, args.text_inputs, args.image_paths)
    for text_input, response in zip(args.text_inputs, responses):
        print(f"Input: {text_input}")
        print(f"Response: {response}")
        print("=" * 40)

