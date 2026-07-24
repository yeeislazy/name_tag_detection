import os
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from argparse import ArgumentParser
from tqdm import tqdm
from random import choice
import pandas as pd
import random

from configuration.config import train_img_prompt, train_img_params, data_path , gen_model


parser = ArgumentParser()
parser.add_argument("--categories", type=str, default="true", help="Specify the category of images to generate: 'true' or 'fake'.")
parser.add_argument("--num_samples", type=int, default=1, help="Number of how many sets of images to generate, 1 set = 3(front, side, high_angle).")

args = parser.parse_args()
num_samples = args.num_samples

categories = args.categories.split(",")  # Split the input string into a list of categories

load_dotenv()   # import .env file to load GEMINI_API key
client = genai.Client(api_key=os.getenv("GEMINI_API")) 

# load the reference image (name tag)
print(f"Loading reference image from: {data_path / 'reference1.png'}")
reference_images = [
    #Image.open(data_path / "reference1.png"),
    Image.open(data_path / "reference2.png"),
    Image.open(data_path / "reference3.png"),
    Image.open(data_path / "reference4.png")
]
print(f"Loading reference image from: {data_path / 'reference_name_tag.png'}")
reference_image_name_tag = Image.open(data_path / "reference_name_tag.png")

# raw csv file path
csv_file_path = data_path / "raw.csv"
if not csv_file_path.exists():
    # create a new DataFrame and save it as CSV
    df = pd.DataFrame(columns=["category","orientation", "color", "setting", "style", "light", "file_path"])
    df.to_csv(csv_file_path, index=False)
else:
    # load the existing CSV file into a DataFrame
    df = pd.read_csv(csv_file_path, dtype={"category": str})

# generate num_samples images for each prompt in train_img_prompt
for num in tqdm(range(num_samples), desc="Generating images"):
        for category in categories:
            
            # randomly select parameters for the prompt
            orientation = choice(train_img_params["orientation"])
            color = choice(train_img_params["color"])
            setting = choice(train_img_params["setting"])
            style = choice(train_img_params["style"])
            light = choice(train_img_params["light"])

            # format the prompt with the selected parameters
            prompt = train_img_prompt[category].format(orientation=orientation, color=color, setting=setting, style=style, light=light)

            print(f"Generating image sample: {num+1}")
            ratio = ['1:1', '1:4', '1:8', '2:3', '3:2', '3:4', '4:1', '4:3', '4:5', '5:4', '9:16']
            try:
                response = client.models.generate_content(
                    model=gen_model,
                    contents=[
                        prompt, 
                        random.choice(reference_images) , 
                        reference_image_name_tag
                    ],
                    config=types.GenerateContentConfig(
                        response_modalities=['TEXT', 'IMAGE'],
                        image_config=types.ImageConfig(
                            aspect_ratio=random.choice(ratio),
                            image_size="1K"
                        )
                    )
                )


                
                
                # save the generated image
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        # decode the image data and save it as a PNG file
                        image = Image.open(BytesIO(part.inline_data.data))
                        
                        # get the max num in folder to create a new count
                        save_path = data_path / category
                        save_path.mkdir(parents=True, exist_ok=True)
                        existing_files = list(save_path.glob("generated_*.png"))
                        if existing_files:
                            # extract the numeric part from the filenames and find the maximum
                            existing_nums = [int(f.stem.split("_")[1]) for f in existing_files if f.stem.split("_")[1].isdigit()]
                            next_count = max(existing_nums) + 1 if existing_nums else 1
                        else:
                            next_count = 1
                        
                        path_to_save = save_path / f"generated_{next_count}.png"
                        image.save(path_to_save)
                        print(f"Image saved to: {path_to_save}")
                        # append the new row to the DataFrame
                        new_row = pd.DataFrame({
                            "category": ["true"],
                            "orientation": [orientation],
                            "color": [color],
                            "setting": [setting],
                            "style": [style],
                            "light": [light],
                            "file_path": [path_to_save]
                        })
                        df = pd.concat([df, new_row], ignore_index=True)
                        df.to_csv(csv_file_path, index=False)  # save the updated DataFrame back to CSV
                        
                        if part.text:
                            # print the text part of the response (if any)
                            print("Text part of the response:")
                            print(part.text.strip())
                
            except Exception as e:
                print(f"Error generating image for orientation: {orientation}, sample: {num+1}. Error: {e}")