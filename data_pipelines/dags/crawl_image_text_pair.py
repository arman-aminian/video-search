from airflow import DAG
from airflow.decorators import task
from datetime import datetime
import requests


with DAG(
    dag_id="crawl_image_text_pair_dag",
    schedule="@hourly",
    start_date=datetime(2023, 5, 6),
):
    
    tasks = {}

    for category in ['tools-materials-equipment', 'electronic-devices', 'home-kitchen']:

        @task(task_id=f"crawl_divar_image_text_category_{category}")
        def crawl_divar_image_text(execution_date=None, **kwargs):
            
            def get_tokens(last_post_date, city_number, post_category, n_pages=10):
    
                url = 'https://api.divar.ir/v8/web-search/{city_number}/{post_category}'.format(
                    city_number=city_number,
                    post_category=post_category
                )

                headers = {
                    'Content-Type': 'application/json'
                }

                list_of_tokens = []
                for i in range(n_pages):
                    json = {"json_schema": {"category": {"value": post_category}},
                            "last-post-date": last_post_date}
                    res = requests.post(url, json=json, headers=headers)
                    
                    data = res.json()
                    last_post_date = data['last_post_date']

                    for widget in data['web_widgets']['post_list']:
                        token = widget['data']['token']
                        list_of_tokens.append(token)

                return list_of_tokens
            

            def save_image_title_pair(post_token):
                try:
                    response = requests.get(f"https://api.divar.ir/v8/posts/{post_token}", timeout=5)
                    post_data = response.json()

                    image_urls = post_data["widgets"]["images"]
                    title = post_data["widgets"]["header"]["title"]

                    # save images
                    for i, image_url in enumerate(image_urls):
                        with open(f'/home/divar/university/MLOps/Project/data/divar/{post_token}-{i}.jpg', 'wb') as handle:
                            response = requests.get(image_url, stream=True)
                            if not response.ok:
                                print(response)
                            for block in response.iter_content(1024):
                                if not block:
                                    break
                                handle.write(block)

                    # save text
                    with open(f'/home/divar/university/MLOps/Project/data/divar/{post_token}.txt', 'w') as file:
                        file.write(post_data["widgets"]["header"]["title"])

                except Exception as e:
                    print("skipping becuase of this error:")
                    print(e)

            tokens = get_tokens(execution_date.timestamp(), 1, category)
            for token in tokens:
                save_image_title_pair(token)

        
        tasks[category] = crawl_divar_image_text()

