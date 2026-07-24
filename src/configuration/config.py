from pathlib import Path

from torchvision.transforms import transforms

# cnn model
cnn_model = "resnet18"

# yolo model
yolo_model = "yolo11s"


# data home dir
data_path = Path(__file__).parent.parent.parent / "data"
data_path.mkdir(parents=True, exist_ok=True)


# training data path
## raw data path
true_data_path = data_path / "true"
false_data_path = data_path / "false"
fake_data_path = data_path / "fake"
true_data_path.mkdir(parents=True, exist_ok=True)
false_data_path.mkdir(parents=True, exist_ok=True)
fake_data_path.mkdir(parents=True, exist_ok=True)

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
    "true" : """
    Based on the two reference images provided, generate a new image of a person wearing a name tag. 
    The first reference image shows a person wearing a name tag - use this as a guide for how the name tag should be positioned on the chest and the overall professional appearance.
    The second reference image is a close-up of the name tag itself - ensure the name tag design, colors, and text style match this reference exactly.

    IMAGE COMPOSITION: 
    Generate a **full-body or upper-body crop of a person**, as if extracted by a person detection algorithm (like YOLO). 
    - The person's **head should be near the top of the frame** and **feet near the bottom** (full-body), OR at least **waist-up** (upper-body).
    - The person should occupy approximately 80-95% of the image frame, with minimal background visible.
    - This simulates the output of a person detector that crops the bounding box around a detected person.

    CAMERA CHARACTERISTICS:
    - Simulated surveillance camera perspective (slightly elevated angle, looking down)
    - Cool/neutral color temperature typical of security cameras
    - Moderate contrast, slight vignetting at corners
    - The name tag must be clearly visible and readable on the person's chest

    PERSON'S ORIENTATION: The person is {orientation} relative to the camera.

    CLOTHING: The person is wearing {color} {style}. The name tag should contrast well with the clothing color to remain clearly visible.

    BACKGROUND: Minimal, showing {setting} in the background, mostly out of focus.

    LIGHTING: {light}, typical for indoor security camera footage.

    REQUIREMENTS:
    - **Full-body or at least waist-up crop** (YOLO-style person detection output)
    - Person fills most of the frame (80-95%)
    - Name tag visible and naturally positioned on chest
    - The clothing color {color} should be distinctly visible
    - Background minimal and slightly blurred
    - Realistic surveillance/camera aesthetic
    """,
    "fake": """
    Generate a new image of a person wearing a **DIFFERENT** name tag (NOT the one in the reference image).
    The name tag should have a different color, design, or format (e.g., red badge, vertical orientation, different logo).
    The person is {orientation}, dressed in {color} {style}, in {setting} with {light} lighting.
    Full-body or waist-up crop, YOLO-style person detection output.
    The name tag must be clearly visible on the chest.

    REQUIREMENTS:
    - **Full-body or at least waist-up crop** (YOLO-style person detection output)
    - Person fills most of the frame (80-95%)
    - Name tag visible and naturally positioned on chest
    - The clothing color {color} should be distinctly visible
    - Background minimal and slightly blurred
    - Realistic surveillance/camera aesthetic
    """
}

# detail setting for generating images
train_img_params = {
    # person orientation relative to the camera, make sure the name tag is visible
    "orientation": [
        "standing facing the camera directly (front view, badge fully visible)",
        "standing facing left (profile view, badge visible on the side)",
        "standing facing right (profile view, badge visible on the side)",
        "standing at 45° angle, facing left-forward (badge visible)",
        "standing at 45° angle, facing right-forward (badge visible)",
        "standing with body angled 45° but face turned toward camera (badge visible)",
        "walking directly toward the camera (front view, approaching, badge visible)",
        "walking from left to right across the field of view (profile, badge visible)",
        "walking from right to left across the field of view (profile, badge visible)",
        "walking diagonally from left-forward to right-backward (badge remains visible)",
        "walking diagonally from right-forward to left-backward (badge remains visible)",
        "approaching the camera at a slight angle (badge visible)",
        "standing with arms at sides, facing the camera (badge visible on chest)",
        "standing with arms crossed, facing the camera (badge visible on chest)",
        "standing with one hand in pocket, facing the camera (badge visible)",
        "looking down slightly but body still facing camera (badge visible)",
    ],
    
    "color": [
        "white",
        "black",
        "dark navy blue",
        "charcoal gray",
        "light gray",
        "beige",
        "cream",
        "brown",
        "dark green",
        "forest green",
        "light blue",
        "sky blue",
        "royal blue",
        "red",
        "burgundy",
        "dark red",
        "pink",
        "light pink",
        "orange",
        "mustard yellow",
        "yellow",
        "purple",
        "lavender",
        "teal",
        "turquoise",
        "mint green",
        "olive green",
        "maroon",
        "coral",
        "peach",
        "ivory",
        "tan",
        "khaki",
        "steel blue",
        "slate gray",
        "wine red",
        "emerald green",
        "sapphire blue",
        "golden yellow",
        "bronze",
    ],
        
    "setting": [
        "a blurred office corridor",
        "a blurred company lobby",
        "a blurred open-plan office",
        "a blurred office entrance with badge reader",
        "a blurred conference venue",
        "a blurred elevator lobby",
        "a blurred hallway with doors",
        "a blurred meeting room with glass panels",
        "a blurred break room with tables",
        "a blurred reception area with desk",
        "a blurred stairwell landing",
        "a blurred medical office reception",
        "a blurred hotel lobby",
        "a blurred university hallway",
        "a blurred airport terminal corridor",
        "a blurred shopping mall corridor",
        "a blurred indoor parking entrance",
        "a blurred museum gallery",
        "a blurred library with bookshelves",
        "a blurred gym entrance area",
    ],
    
    "style":[
        "business suit and tie",
        "business casual shirt and chinos",
        "professional dress",
        "blazer with jeans",
        "company-branded polo shirt with trousers",
        "dark suit with a name tag prominently displayed",
        "light-colored button-up shirt with dark trousers",
        "professional uniform",
        "sweater or cardigan with collared shirt underneath",
        "tailored skirt suit",
        "lab coat over professional clothing",
        "jacket with shirt and trousers",
        "vest and dress shirt",
        "turtleneck with blazer",
    ],
    
    "light": [
        "cool overhead fluorescent lights",
        "harsh ambient ceiling lights with some shadows",
        "mixed lighting (warm desk lamps and cool overhead)",
        "even ambient indoor lighting with moderate contrast",
        "bright white LED overhead lights",
        "standard office lighting with no dramatic shadows",
        "bright and evenly distributed lighting",
        "slightly dim lighting with visible shadows",
    ],
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