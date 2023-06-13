from dataclasses import dataclass
from PIL import Image


@dataclass
class ImageWithCaption:
    """
    Represents the metadata of an image.
    """

    show_width = 512
    url: str
    is_valid: bool
    idx: int = -1
    image: Image.Image = None
    caption: str = None

    def add_caption(self, caption: str) :
        """
        Adds a generated caption to the image metadata.

        Args:
            caption (str): The caption to add.
        """
        self.caption = caption

    def show(self) :
        """
        Displays the image with its caption if available, otherwise shows the image alone or an appropriate message
        if the image is not valid.
        """
        if not self.is_valid:
            print("Invalid image. Unable to display.")
            return
        resized_image = self.get_resized_image(self.show_width)
        if self.caption is None:
            resized_image.show()
        else:
            resized_image.show()
            print(f"Caption: {self.caption}")

    def get_resized_image(self, resize_width: int) -> Image.Image:
        """
        Resizes the image while maintaining its aspect ratio.

        Args:
            resize_width (int): The desired width of the resized image.

        Returns:
            Image.Image: The resized image.
        """
        if self.is_valid:
            original_width, original_height = self.image.size
            show_height = int((original_height / original_width) * resize_width)
            return self.image.resize((resize_width, show_height))
