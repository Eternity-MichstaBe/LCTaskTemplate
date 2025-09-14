import base64
import os

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


examples = [
    {
        "question": "你好", 
        "answer": "你好"
    },
    {
        "question": "你是谁？",
        "answer": "我是聊天助手。"
    },
    {
        "question": "你擅长什么？",
        "answer": "我擅长回答问题。"
    },
]


multi_model_examples = [
    {
        "question": "这张图片里有什么？",
        "image_data": encode_image_to_base64(os.path.join(root, "test/dog.jpg")),
        "answer": "这张图片展示的是一只黄色的小狗，可能是金毛犬或拉布拉多犬"
    }
]