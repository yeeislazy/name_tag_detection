from pathlib import Path

from torchvision.transforms import transforms

# cnn model
cnn_model = "resnet18"

# yolo model
yolo_model = "yolo11s"


# data home dir
data_dir = Path(__file__).parent.parent.parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)


# training data path
## name_tag training data path
name_tag_data_dir = data_dir / "name_tag"
name_tag_data_dir.mkdir(parents=True, exist_ok=True)
name_tag_ref_dir = name_tag_data_dir / "ref"
name_tag_true_dir = name_tag_data_dir / "true"
name_tag_false_dir = name_tag_data_dir / "false"
name_tag_fake_dir = name_tag_data_dir / "fake"
name_tag_true_dir.mkdir(parents=True, exist_ok=True)
name_tag_false_dir.mkdir(parents=True, exist_ok=True)
name_tag_fake_dir.mkdir(parents=True, exist_ok=True)

## yolo training data path
yolo_data_dir = data_dir / "yolo"
yolo_data_dir.mkdir(parents=True, exist_ok=True)
yolo_ref_dir = yolo_data_dir / "ref"
yolo_raw_dir = yolo_data_dir / "raw"
yolo_images_dir = yolo_data_dir / "images"
yolo_labels_dir = yolo_data_dir / "labels"
yolo_ref_dir.mkdir(parents=True, exist_ok=True)
yolo_raw_dir.mkdir(parents=True, exist_ok=True)
yolo_images_dir.mkdir(parents=True, exist_ok=True)
yolo_labels_dir.mkdir(parents=True, exist_ok=True)
for subdir in ["train", "val", "test"]:
    (yolo_images_dir / subdir).mkdir(parents=True, exist_ok=True)
    (yolo_labels_dir / subdir).mkdir(parents=True, exist_ok=True)

# model dir
model_dir = Path(__file__).parent.parent.parent / "models"
model_dir.mkdir(parents=True, exist_ok=True)
# cnn_model_path
cnn_model_path = model_dir / f"best_{cnn_model}_v6.pth"
# yolo_model_path
yolo_model_path = model_dir / f"{yolo_model}.pt"

# test videos dir
test_videos_dir = Path(__file__).parent.parent.parent / "test_videos"
test_videos_dir.mkdir(parents=True, exist_ok=True)
input_videos_dir = test_videos_dir / "input"
input_videos_dir.mkdir(parents=True, exist_ok=True)
output_videos_dir = test_videos_dir / "output"
output_videos_dir.mkdir(parents=True, exist_ok=True)

# debug dir
debug_dir = Path(__file__).parent.parent.parent / "debug"
debug_dir.mkdir(parents=True, exist_ok=True)

# person crop dir for debug
debug_crop_dir = debug_dir / "debug_crop"
debug_crop_dir.mkdir(parents=True, exist_ok=True)

train_img_prompt = {
    "name_tag": {
        "true" : """
        You are given several reference images.

        Reference image 1:
        A front-view image of a person wearing the target company name tag.
        Use this image only as the reference for the appearance, placement and design of the target name tag.

        Reference images 2 and onwards:
        Surveillance camera crops of people.
        Use these images as references for the camera angle, crop style, image quality and overall surveillance appearance.

        Generate ONE realistic surveillance-camera image of a person.

        Requirements:

        - Match the camera angle, crop style and image quality of the surveillance reference images.
        - The person should appear naturally captured by a surveillance camera.
        - Recreate the target company name tag based on Reference image 1.
        - Do NOT paste, overlay or directly copy the reference image.
        - The name tag must blend naturally with the clothing.
        - The name tag should have the same blur, lighting, perspective and resolution as the rest of the image.
        - The name tag should not appear sharper than the clothing.
        - Camera mounted high above, looking slightly downward.
        - Indoor environment.
        - The person occupies most of the frame, similar to a YOLO person crop.
        - Clothing: {color} {style}
        - Pose: {orientation}
        """,
        "fake": """
        You are given a surveillance image of a person.

        Generate another surveillance image with a similar camera angle and crop style.

        Requirements:
        - Match the reference person's pose and viewpoint.
        - The person is {orientation}.
        - Clothing is {color} {style}.
        - Camera is mounted high above, looking downward.
        - Indoor environment.
        - Wear a DIFFERENT badge or NO badge.
        - Do NOT use the company name tag from the reference image.
        - Slight motion blur and low-resolution surveillance camera appearance.
        """
    },
    "yolo": """
        You are given several reference images.

        Reference images:
        Frames captured from the target surveillance camera.

        Generate ONE new surveillance camera frame that closely matches the visual distribution of the reference surveillance images.

        The generated image will be used as training data for a YOLO person detection model.

        IMPORTANT:
        Do not create a normal office photograph.
        The output must look like a real fixed CCTV surveillance frame.

        Camera perspective:
        - Match the exact surveillance camera viewpoint from the reference images.
        - The camera is mounted on the ceiling and points almost vertically downward toward the floor.
        - The optical axis of the camera is close to perpendicular (90 degrees) to the ground plane.
        - Use a near top-down bird's-eye surveillance perspective.
        - People should be viewed mostly from above, with visible tops of heads and shoulders.
        - Faces should not be the main focus because of the high camera angle.
        - Avoid low-angle or eye-level viewpoints.
        - Avoid looking horizontally across the room.
        - Maintain the same high camera height, perspective distortion and person scale as the reference CCTV frames.
        
        Perspective constraints:
        - The image should resemble a ceiling-mounted security camera recording.
        - Human figures should appear smaller and flatter compared with normal photographs.
        - The scene should have reduced depth perspective due to the overhead viewpoint.

        Composition:
        - Match the same scene layout, camera height, perspective, lighting and image quality as the reference frames.
        - Keep similar empty spaces, walking paths and room coverage.
        - The frame should feel like a randomly captured surveillance moment, not a staged image.

        People:
        - Include {people_count} people.
        - People should have different clothing colors, body shapes, poses and walking directions.
        - People should be naturally distributed across the scene.
        - Some people should be close together.
        - Increase realistic human occlusion:
            - Most people (70-90%) in the frame should be partially hidden.
            - Many people should be blocked by office desks, chairs, cubicle partitions, glass dividers, monitors, and other employees.
            - A large number of people should not have a complete visible body.
            - Some people should only have their head, shoulders, or upper torso visible.
            - People walking behind others should be partially covered.
            - Overlapping groups of people should appear naturally.
            - The scene should resemble a crowded workplace surveillance recording, not a clean dataset

        Surveillance image characteristics:
        - Low-resolution CCTV appearance.
        - Slight compression artifacts.
        - Slight motion blur.
        - Realistic surveillance camera noise.
        - Uneven indoor lighting.
        - Slight distortion from a wide-angle security camera lens.
        - The image should look like a frame extracted from a security recording.

        Name tags:
        - Approximately {badge_count} people should wear company name tags.
        - Name tags should be small and naturally attached to clothing.
        - They should match the perspective, lighting, blur and resolution of the CCTV image.
        - Some name tags can be partially hidden by body pose or occlusion.
        - Do not make name tags large, clean or clearly visible.

        Environment:
        - Indoor office environment.
        - Realistic workplace layout.
        - Desks, corridors, meeting areas or office spaces.
        - No cinematic composition.

        Avoid:
        - eye-level camera angle
        - smartphone photo
        - DSLR photography
        - portrait photography
        - clean separated people
        - posed people looking at camera
        - large visible faces
        - perfectly sharp images
        - cinematic lighting
        - unrealistic oversized name tags
    """
}

# detail setting for generating images
train_params = {
    "orientation": [
        "facing the camera",
        "slightly facing left",
        "slightly facing right",
        "walking toward the camera",
        "walking away from the camera",
        "walking left",
        "walking right",
        "looking down",
        "standing naturally",
        "sitting infront of a desk",
    ],
    "style": [
        "polo shirt",
        "button-up shirt",
        "office shirt",
        "T-shirt",
    ],
    "color": [
        "white",
        "black",
        "blue",
        "green",
        "red",
        "gray",
        "yellow",
        "orange",
        "purple",
        "brown",
        "pink"
    ],
    "people_count": [2, 10],
    "badge_count": [0, 3],
}

# cnn val transformer
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# gen model to use
gen_model = "gemini-3.1-flash-lite-image"

# detect gap 
detect_gap = 5  # frames to skip before re-detecting persons in the frame