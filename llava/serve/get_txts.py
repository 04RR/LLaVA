import argparse
import torch
from PIL import Image
from io import BytesIO
import requests
from transformers import TextStreamer

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def generate_outputs(texts, image_paths, model_path, model_base="gpt2"):
    # Load model and tokenizer
    model_name = get_model_name_from_path(model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, False, False)
    
    outputs = []

    for text, image_path in zip(texts, image_paths):
        image = load_image(image_path)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

        input_text = text
        conv_prompt = input_text
        input_ids = tokenizer_image_token(conv_prompt, tokenizer, return_tensors='pt').unsqueeze(0).cuda()
        
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        keywords = ["<|endoftext|>"]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

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

        generated_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        outputs.append(generated_text)
    
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m", help="Path to the pretrained model")
    parser.add_argument("--model-base", type=str, default="gpt2", help="Base model type")
    parser.add_argument("--texts", nargs="+", help="List of texts for which to generate outputs")
    parser.add_argument("--image-paths", nargs="+", help="List of image paths corresponding to the texts")
    args = parser.parse_args()

    generated_outputs = generate_outputs(args.texts, args.image_paths, args.model_path, args.model_base)
    
    for input_text, output_text in zip(args.texts, generated_outputs):
        print("Input:", input_text)
        print("Generated Output:", output_text)
        print("=" * 40)
