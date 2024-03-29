# docker run --rm -ti your-image

FROM python:latest

RUN pip install numpy pandas scikit-learn matplotlib

COPY main.py ./data/test_df.csv ./data/test_preds.csv ./data/train_df.csv model_rank_svm_without_ros.pk1 .


CMD [ "python", "main.py" ]