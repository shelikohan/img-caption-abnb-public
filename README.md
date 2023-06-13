# Image Captioning on Large Datasets

![image](https://github.com/shelikohan/img-caption-abnb-public-/assets/50539882/e379a8b5-1b73-46c6-874a-51d8986c6556)

This repository serve two main purposes, the first is to extract image captions from image urls using the BLIP model.
Additionally, you can crawl [insideairbnb.com listings](insideairbnb.com).

### Image cations:

1. Download the repository and extract the ZIP file into your Google Drive:
   - Make sure to include all data and script from the repository
   -  If you prefer make it a temporary folder,
      you can first open Google Colab and then add the directory without mounting.

2. Open and run in [Google Colab Notebook](https://github.com/shelikohan/img-caption-abnb-public/blob/main/image_captions_on_large_datasets.ipynb)

### Crawl [insideairbnb.com listings](insideairbnb.com):
1. install package:
   ```shell
   pip install beautifulsoup4 file-logger
   ```  
2. run the shell:
   ```shell
   inside_abnb_crawler.py --parent-dir "airbnb-listings" --log-path crawler_log.log 
   ```
   - it will create nested directories by country of listings


5. Contact for further questions:
   - If you have any questions or need further assistance, please email me at sheli.kohan@gmail.com.
   - I will be happy to help you with any issues or queries you may have.




