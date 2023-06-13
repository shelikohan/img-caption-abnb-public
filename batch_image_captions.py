"""
Batch Image Captions
Author: Sheli Kohan
Date: 12th June 2023
"""
import torch
from multiprocessing import Queue, get_context
import threading
from image_caption_processor import ImageCaptionProcessor


class BatchImageCaptions:
    """
    Class for batch processing image captions using multiple threads.
    """

    def __init__(self, n_cpus: int = 4, q_len: int = 2, batch_size: int = 64, **kwargs):
        """
        Initializes the BatchImageCaptions.

        Args:
            n_cpus (int): The number of CPUs to use for parallel processing.
            q_len (int): The maximum length of the processing queue.
            batch_size (int): The number of images to process in each batch.
            **kwargs: Additional keyword arguments to be passed to the ImageCaptionProcessor.
        """
        self.n_cpus = n_cpus
        self.pool = None
        self.is_pool_closed = True
        self.processor = ImageCaptionProcessor(**kwargs)
        self.queue = Queue(q_len)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size

    def __add_images_to_queue(self, image_urls_ls: list):
        """
        Adds image URLs to the processing queue.

        Args:
            image_urls_ls (list): A list of image URLs.
        """
        for i in range(0, len(image_urls_ls), self.batch_size):
            image_urls_batch = image_urls_ls[i:i + self.batch_size]
            img_urls2idx = [(url, i) for i, url in enumerate(image_urls_batch)]
            processed_images = self.pool.starmap(self.processor.get_image_from_url, img_urls2idx)
            self.queue.put(processed_images)

    def __str__(self):
        return f'{self.n_cpus},{self.batch_size},{self.device},{str(self.processor)}'

    def get_image_captions_from_urls(self, image_urls_ls: list) -> list:
        """
        Generates captions for a list of image URLs.

        Args:
            image_urls_ls (list): A list of image URLs.

        Returns:
            list: A list of ImageWithCaption objects containing the generated captions.
        """
        if self.is_pool_closed:
            self.open_pool()
        thread = threading.Thread(target=self.__add_images_to_queue, args=(image_urls_ls,))
        thread.start()
        all_images = []
        while True:
            images = self.queue.get()
            valid_images = [image for image in images if image.is_valid]
            self.processor.add_image_captions(valid_images)
            all_images += images
            if not thread.is_alive() and self.queue.empty():
                break
        return all_images

    def open_pool(self):
        """
        Opens the thread pool for parallel processing.
        """
        ctx = get_context('spawn')
        self.pool = ctx.Pool(self.n_cpus)
        self.is_pool_closed = False

    def close_pool(self):
        """
        Closes the thread pool.
        """
        if not self.is_pool_closed:
            self.pool.close()
            self.pool.join()
            self.is_pool_closed = True
