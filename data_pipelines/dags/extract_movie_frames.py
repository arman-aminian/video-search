from airflow import DAG
from airflow.decorators import task
from datetime import datetime


with DAG(
    dag_id="extract_movie_frames_dag",
    schedule="@once",
    start_date=datetime(2023, 5, 1),
):

    @task(task_id="extract_movie_frames_task")
    def extract_movie_frames():
        import cv2

        cam = cv2.VideoCapture("/home/divar/university/MLOps/Project/data/movies/movie_1.mkv")
        currentframe = 0
        while(True):
            ret, frame = cam.read()
            if not ret: break
        
            if currentframe % 1000 == 0:
                name = '/home/divar/university/MLOps/Project/data/movies/movie_1_frames/' + str(currentframe) + '.jpg'
                cv2.imwrite(name, frame)
        
            currentframe += 1
        
        cam.release()
        cv2.destroyAllWindows()


    extract_movie_frames_task = extract_movie_frames()

