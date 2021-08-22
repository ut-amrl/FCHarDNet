import rospy
import torch
from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.utils import convert_state_dict
import cv2
import numpy as np
import argparse
from sensor_msgs.msg import CompressedImage

def init_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loader = get_loader(args.dataset)  
    loader = data_loader(
        root=args.input,
        is_transform=True,
        img_size=eval(args.size),
        test_mode=True
    )
    n_classes = loader.n_classes

    # Setup Model
    model = get_model({"arch": "hardnet"}, n_classes)
    state = convert_state_dict(torch.load(args.model_path, map_location=device)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    return device, model, loader

def process_img(img, size, device, model, loader):
    print("Processing Input Image")

    img_resized = cv2.resize(img, (size[1], size[0]))  # uint8 with RGB mode
    img = img_resized.astype(np.float16)

    # norm
    value_scale = 255
    mean = [0.406, 0.456, 0.485]
    mean = [item * value_scale for item in mean]
    std = [0.225, 0.224, 0.229]
    std = [item * value_scale for item in std]
    img = (img - mean) / std

    # NHWC -> NCHW
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    images = img.to(device)
    outputs = model(images)
    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
    decoded = loader.decode_segmap(pred)

    return img_resized, decoded


def create_msg_handler(model, device):
  def handle_img_msg(img_msg):
    np_arr = np.fromstring(img_msg.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
    _, decoded = process_img(image_np, args.size, device, model)

  return handle_img_msg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="jackal_seg",
        help="dataset type",
    )

    parser.add_argument(
        "--size",
        type=str,
        default="512,640",
        help="Inference size",
    )

    parser.add_argument(
        "--input", nargs="?", type=str, default=None, help="Path of the input image/ directory"
    )
    parser.add_argument(
        "--image_topic", type=str, default="/camera/rgb/image_raw/compressed", help="image topic"
    )
    parser.add_argument(
        "--seg_topic", type=str, default="/camera/segmentation", help="output segmentation topic"
    )
    args = parser.parse_args()

    device, model = init_model(args)

    n = rospy.init_node("seg_node")

    rospy.Subscriber(args.image_topic, CompressedImage, create_msg_handler(model, device))