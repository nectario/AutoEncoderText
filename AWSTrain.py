import sagemaker
from pathlib import Path
from sagemaker.predictor import json_serializer, csv_serializer
import json
import boto3
from sagemaker import get_execution_role
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
import sys


hyperparameters = {
    "epochs": 7,
    "lr": 8e-5,
    "max_seq_length": 512,
    "train_batch_size": 25,
    "lr_schedule": "warmup_cosine",
    "warmup_steps": 1000,
    "optimizer_type": "adamw"
}

bucket = 'sagemaker-genreprediction'

prefix = 'input'
prefix_output = 'output'

image = "160609124801.dkr.ecr.us-east-1.amazonaws.com/fluent-fast-bert:1.0-gpu-py36"

output_path = "s3://{}/{}".format(bucket, prefix_output)

role = "arn:aws:iam::160609124801:role/AmazonSageMakerFullAccess"

estimator = sagemaker.estimator.Estimator(image,
                                          role,
                                          train_instance_count=1,
                                          train_instance_type='ml.p3.2xlarge',
                                          output_path=output_path,
                                          base_job_name='genre-prediction',
                                          hyperparameters=hyperparameters)


estimator.fit("s3://sagemaker-genreprediction/input")

predictor = estimator.deploy(1,
                             'ml.m5.large',
                             endpoint_name='genre-prediction',
                             update_endpoint=True,
                             serializer=json_serializer)