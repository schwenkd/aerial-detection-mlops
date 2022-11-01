import json
import urllib.parse
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.info('Loading function')


def lambda_handler(event, context):
    # Get the uploaded file info
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    input_filename = f's3://{bucket}/{key}'
    logger.info("input file received = {}".format(input_filename))
    ## Call detect function of the flask api based microservice to do inferencing (TODO)
    return {
        'statusCode': 200,
        'body': json.dumps('aerial-detection-mlops-lambda - called the detect api.')
    }
