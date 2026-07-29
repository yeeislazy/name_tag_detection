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

from configuration.config import train_img_prompt, train_params, data_dir, name_tag_data_dir,yolo_data_dir, gen_model


def load_reference_images(data_path):
    # load the reference image (name tag)
    if (data_path / "reference_name_tag.png").exists():
        reference_images = [
            Image.open(data_path / "reference_name_tag.png"),
        ]
    else:
        reference_images = []
        
    for path in data_path.glob("reference_*.png"):
        reference_images.append(Image.open(path))
    print(f"Loaded {len(reference_images)} reference images.")
    return reference_images


def generate_images( num_samples, target, categories, train_img_prompt, train_img_params, reference_images, data_dir = data_dir / "name_tag", gen_model = gen_model):
    client = genai.Client(api_key=os.getenv("GEMINI_API")) 
    
     # raw csv file path
    csv_file_path = data_dir / "raw.csv"
    if not csv_file_path.exists():
        # create a new DataFrame and save it as CSV
        df = pd.DataFrame(columns=["target","category","orientation", "color", "setting", "style", "light", "people_count", "badge_count", "file_path"])
        df.to_csv(csv_file_path, index=False)
    else:
        # load the existing CSV file into a DataFrame
        df = pd.read_csv(csv_file_path, dtype={"category": str})
    
    # generate num_samples images for each prompt in train_img_prompt
    for num in tqdm(range(num_samples), desc="Generating images"):
            for category in categories:
                
                if target == "name_tag":
                    # randomly select parameters for the prompt
                    orientation = choice(train_img_params["orientation"])
                    color = choice(train_img_params["color"])
                    #setting = choice(train_img_params["setting"])
                    style = choice(train_img_params["style"])
                    #light = choice(train_img_params["light"])

                    # format the prompt with the selected parameters
                    prompt = train_img_prompt[target][category].format(
                        orientation=orientation, 
                        color=color, 
                        # setting=setting, 
                        style=style, 
                        # light=light
                        )
                elif target == "yolo":
                    # randomly select parameters for the prompt
                    people_count = random.randint(train_img_params["people_count"][0], train_img_params["people_count"][1])
                    badge_count = random.randint(
                        train_img_params["badge_count"][0] if people_count >= train_img_params["badge_count"][0] else 0, 
                        people_count)
                    # format the prompt with the selected parameters
                    prompt = train_img_prompt[target].format(
                        people_count=people_count, 
                        badge_count=badge_count
                        )

                print(f"Generating image sample: {num+1}")
                if target == "name_tag":
                    ratio = [
                        "1:1",
                        "4:5",
                        "3:4",
                        "2:3",
                        "5:4",
                        "9:16",
                    ]
                elif target == "yolo":
                    ratio = [
                        "1:1",
                        "3:2",
                        "4:3",
                        "16:9",
                    ]
                
                retry_count = 0
                while retry_count < 5:
                    try:
                        response = client.models.generate_content(
                            model=gen_model,
                            contents=[
                                prompt, 
                                *reference_images
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
                                
                                
                                save_dir = data_dir
                                if target == "name_tag":
                                    save_dir = data_dir / category
                                elif target == "yolo":
                                    save_dir = data_dir / "raw"
                                                            
                                # get the max num in folder to create a new count
                                existing_files = list(save_dir.glob("generated_*.png"))
                                if existing_files:
                                    # extract the numeric part from the filenames and find the maximum
                                    existing_nums = [int(f.stem.split("_")[1]) for f in existing_files if f.stem.split("_")[1].isdigit()]
                                    next_count = max(existing_nums) + 1 if existing_nums else 1
                                else:
                                    next_count = 1
                                
                                path_to_save = save_dir / f"generated_{next_count}.png"
                                image.save(path_to_save)
                                print(f"Image saved to: {path_to_save}")
                                # append the new row to the DataFrame
                                new_row = pd.DataFrame({
                                    "target": [target],
                                    "category": [category if target == "name_tag" else "yolo"],
                                    "orientation": [orientation if target == "name_tag" else None],
                                    "color": [color if target == "name_tag" else None],
                                    # "setting": [setting],
                                    "style": [style if target == "name_tag" else None],
                                    # "light": [light],
                                    "people_count": [people_count if target == "yolo" else None],
                                    "badge_count": [badge_count if target == "yolo" else None],
                                    "file_path": [path_to_save if target == "name_tag" else None],
                                })
                                df = pd.concat([df, new_row], ignore_index=True)
                                df.to_csv(csv_file_path, index=False)  # save the updated DataFrame back to CSV
                                
                                if part.text:
                                    # print the text part of the response (if any)
                                    print("Text part of the response:")
                                    print(part.text.strip())
                        
                        break  # exit the retry loop if successful
                    
                    except Exception as e:
                        print(f"Error generating image for target: {target}, sample: {num+1}. Error: {e}")
                        retry_count += 1
                        print(f"Retrying... Attempt {retry_count}/5")
                else:
                    print(f"Failed to generate image for target: {target}, sample: {num+1} after 5 attempts.")
     
def main():
    parser = ArgumentParser()
    parser.add_argument("--target", type=str, default="name_tag", help="Specify the training target for image generation (name_tag or yolo).")
    parser.add_argument("--categories", type=str, default="true", help="Specify the name_tag category of images to generate: 'true' or 'fake'.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of how many sets of images to generate, 1 set = 3(front, side, high_angle).")

    args = parser.parse_args()
    num_samples = args.num_samples

    target = args.target
    categories = args.categories.split(",")  # Split the input string into a list of categories

    load_dotenv()   # import .env file to load GEMINI_API key
    
    if target == "name_tag":
        dir = name_tag_data_dir
    elif target == "yolo":
        dir = yolo_data_dir
    reference_images = load_reference_images(dir/"ref")
                   
   
        
    generate_images(num_samples, target, categories, train_img_prompt, train_params, reference_images, dir, gen_model)
    
if __name__ == "__main__":
    main()