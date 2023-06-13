"""
Image Captions processor
Author: Sheli Kohan
Date: 12th June 2023
"""
import requests
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from image_with_caption import ImageWithCaption


class ImageCaptionProcessor:
    """
    Class for generating captions for images using a pre-trained image captioning model.
    """

    def __init__(self, hf_model: str = "Salesforce/blip-image-captioning-large", max_new_tokens: int = 20,
                 skip_special_tokens: bool = True):
        """
        Initializes the ImageCaptionProcessor.

        Args:
            hf_model (str): The name or path of the Hugging Face model to use for captioning.
            max_new_tokens (int): The maximum number of new tokens to generate for the caption.
            skip_special_tokens (bool): Whether to skip special tokens when decoding the generated caption.
        """
        self.hf_model = hf_model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = BlipProcessor.from_pretrained(hf_model)
        self.model = BlipForConditionalGeneration.from_pretrained(hf_model).to(self.device)
        self.max_new_tokens = max_new_tokens
        self.skip_special_tokens = skip_special_tokens

    def __str__(self) -> str:
        return f'{self.hf_model},{self.device},{self.max_new_tokens},{self.skip_special_tokens}'

    def get_image_from_url(self, image_url: str, idx: int = -1, resize: bool = True,
                           resize_width: int = 512) -> ImageWithCaption:
        """
        Retrieves an image from the specified URL.

        Args:
            image_url (str): The URL of the image.
            idx (int, optional): The index of the image. Defaults to -1.
            resize (bool, optional): Whether to resize the image. Defaults to True.
            resize_width (int, optional): The width to resize the image. Defaults to 512.

        Returns:
            ImageWithCaption: The metadata of the retrieved image.
        """
        try:
            image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
            image_with_caption = ImageWithCaption(url=image_url, is_valid=True, idx=idx, image=image)
            if resize:
                image_with_caption.image = image_with_caption.get_resized_image(resize_width)
            return image_with_caption

        except Exception as e:
            print(str(e))
            return ImageWithCaption(url=image_url, is_valid=False, idx=idx, image=None)

    def get_caption_from_url(self, img_url: str) -> str:
        """
        Generates a caption for an image from the specified URL.

        Args:
            img_url (str): The URL of the image.

        Returns:
            str: The generated caption for the image.
        """
        image_info = self.get_image_from_url(img_url)
        return self.get_caption_from_image(image_info)

    def get_caption_from_image(self, image_info: ImageWithCaption) -> str:
        """
        Generates a caption for the given image.

        Args:
            image_info (ImageWithCaption): The metadata of the image.

        Returns:
            str: The generated caption for the image.
        """
        if image_info.is_valid:
            input = self.processor(image_info.image, return_tensors="pt").to(self.device)
            tokenized_sentences = self.model.generate(**input, max_new_tokens=self.max_new_tokens)
            caption = self.processor.decode(tokenized_sentences[0], skip_special_tokens=self.skip_special_tokens)
            return caption

    def get_image_with_caption_from_url(self, img_url: str) -> ImageWithCaption:
        """
        Retrieves an image from the specified URL and generates a caption for it.

        Args:
            img_url (str): The URL of the image.

        Returns:
            ImageWithCaption: The metadata of the retrieved image with the generated caption.
        """
        image_info = self.get_image_from_url(img_url)
        image_info.add_caption(self.get_caption_from_image(image_info))
        return image_info

    def add_image_captions(self, images: list):
        """
        Adds captions to a list of images.

        Args:
            images (list): A list of ImageWithCaption objects.

        Raises:
            AssertionError: If any image in the list is invalid.
        """
        assert all(image.is_valid for image in images)
        inputs = self.processor([image.image for image in images], return_tensors="pt").to(self.device)
        tokenized_sentences = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        captions = self.processor.batch_decode(tokenized_sentences, skip_special_tokens=self.skip_special_tokens)
        for image, caption in zip(images, captions):
            image.add_caption(caption)
