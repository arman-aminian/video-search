<span align="center">
<a href="https://huggingface.co/arman-aminian/clip-farsi-text-0SQDA"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Text Models&color=blue"></a>
<a href="https://huggingface.co/arman-aminian/clip-farsi-vision-0SQDA"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Vision Models&color=blue"></a>
<a href="https://colab.research.google.com/drive/1UNzC_lrR0BiPcydvKvC2E6RiqQN6rVxr?usp=sharing"><img src="https://img.shields.io/static/v1?label=Colab&message=Training notebook&logo=Google%20Colab&color=purple"></a>
</span>

# Video Search

Video Search is an innovative project that enables users to search for specific movie sequences based on textual descriptions. Utilizing the power of the [CLIP](https://arxiv.org/abs/2103.00020) model by OpenAI, this project encodes movie frames and maps them to textual sequences for efficient search capabilities. Additionally, we have extended the capabilities of the original CLIP model by training it on the Persian language.

## Accessing the Product

Users can explore and use our product through the following link: [https://mlsd-video-search.darkube.app](https://mlsd-video-search.darkube.app).
<p float="left">
  <img src="./images/sc01.png" width="49%" />
  <img src="./images/sc02.png" width="49%" /> 
</p>

## CLIP

[CLIP](https://arxiv.org/abs/2103.00020) (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3.

![CLIP](https://github.com/openai/CLIP/blob/main/CLIP.png)

With CLIP, we can train any two image and text encoder models together to relate images and text. It gives a score for relatedness of any given text and image! We fine-tuned [Vision Transformer(ViT)](https://huggingface.co/openai/clip-vit-base-patch32) as the vision encoder and the [roberta-zwnj-wnli-mean-tokens](https://huggingface.co/m3hrdadfi/roberta-zwnj-wnli-mean-tokens) as the farsi text encoder.

You can find how to train the model in the [CLIP training notebook](https://colab.research.google.com/drive/1UNzC_lrR0BiPcydvKvC2E6RiqQN6rVxr?usp=sharing).

## Training Data

To train (fine-tune) this model, we need examples that are pairs of images and Persian text that are the text associated with the image.
Since Persian data in this field is not easily available and manual labeling of data is costly, we decided to translate the available English data and obtain the other part of the data from the web crawling method.

### Translation

There weren't datasets with Persian captioned images, so we translated datasets with English captions to Persian with Google Translate using [googletrans](https://pypi.org/project/googletrans/) python package.

Then we evaluated these translations with a [sentence-bert](https://www.sbert.net/) bilingual model named [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2) trained for sentence similarity.
We calculated cosine similarity for embeddings of English caption and its Persian translation. Finally, we filtered out top translations.

### Crawler

For improving our model performance we crawled divar posts with its API. We saved image-title pairs in Google Drive.

## Evaluation

### Accuracy @ k

This metric is used for evaluating how good an image search of a model is.

Acc@k definition: Is the best image (the most related to the text query), among the top-k outputs of the model?

We calculated this metric for both models (CLIP & baseline) on two datasets:
* [flickr30k](https://paperswithcode.com/dataset/flickr30k): some intersections with the training data.
* [nocaps](https://nocaps.org/): completely zero-shot for models!


## Deployment Tools

This project uses a variety of tools for deployment, database management, API implementation, documentation, and monitoring:

1. **Vector Database**: We utilize [Qdrant](https://github.com/qdrant/qdrant) as our vector database to store and manage the encoded movie frames.
2. **API Implementation**: Our API is implemented using [FastAPI](https://fastapi.tiangolo.com/), providing a robust and efficient way to handle requests. An example endpoint for querying is `/{video_name}/?search_entry={search_entry}`.
3. **Automated Documentation**: [Swagger](https://swagger.io/) is used for automated documentation, ensuring that all API endpoints are well-documented and easy to understand.
4. **Python Client**: A dedicated Python client is implemented which aids in rendering images from base64 strings.
5. **Monitoring**: We use [Prometheus](https://prometheus.io/) Instrumenter for real-time monitoring of our application.
6. **Dashboard**: For a visual representation of our monitoring data, [Grafana](https://grafana.com/) is used. The key metrics we track include:
   - Uptime
   - Response Time
   - Response Size

<p float="left">
  <img src="./images/components.png" width="49%" />
  <img src="./images/hamravesh.png" width="49%" /> 
</p>

## Encoding Frames and Storing in Vector DB

*Before proceeding, ensure you've set up the necessary dependencies and databases.*

To encode movie frames and store them in the Vector DB, use the `video_to_db.py` script. This script handles both the encoding of frames using the CLIP model and storing the resulting vectors in the database.

## Searching User's Query in Vector DB

1. **Text Encoding**: When a user provides a textual description, this text is first encoded using the CLIP text encoder.
2. **Vector Search**: The encoded text vector is then used to search within the Vector DB to find the closest matching movie frame vectors.
3. **Result Retrieval**: The top matching movie frames and their timestamps are retrieved and presented to the user.

Project Structure:
------------------

- **src**:
  - `train`: Contains scripts/modules related to training models.
  - `evaluation`: Contains scripts/modules for evaluating model performance.
  
- **data_pipelines**:
  - `dags`: Contains Directed Acyclic Graphs (DAGs) for Apache Airflow, orchestrating various data processing tasks.
  - `airflow_configs.env`: Configuration file for setting up Apache Airflow.

- **inference_client**: Client-side scripts/modules to interact with the inference server.
- **data**: Contains the data used for the project, such as movie frames or textual descriptions.
- **video_database**: Scripts/modules related to storing and retrieving video frame encodings.
- **inference_front**: Front-end interface allowing users to interact with the system.
- **mlflow_server**: Dedicated to model tracking and versioning using MLflow.
- **inference_server**: Contains server-side code/scripts for handling inference requests.

Usage:
------

*Note: Specific usage instructions will be based on the actual interface and API design. Below is a general overview.*

1. Users can access the system through the front-end interface provided in our [Website](https://mlsd-video-search.darkube.app).
2. Developers can access the system through the client interface provided in `inference_client.ipynb`.


License:
--------

This project is licensed under the terms mentioned in the `LICENSE` file.
