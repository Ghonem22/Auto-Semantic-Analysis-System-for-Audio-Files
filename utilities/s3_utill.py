import boto3
import yaml
from botocore.exceptions import NoCredentialsError, ClientError


# Let's use Amazon S3
s3 = boto3.resource('s3')
s3_client = boto3.client('s3')
with open(r'utilities/configs.yml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    config = yaml.load(file, Loader=yaml.FullLoader)

def create_folder(folder_complete_path, bucket='audio-similarity'):
    object = s3.Object(bucket, folder_complete_path)
    object.put()

def upload_file(file_complete_path, content, bucket='audio-similarity'):
    object = s3.Object(bucket, file_complete_path)
    object.put(content)

def download_file(file_complete_path, saving_path, bucket='audio-similarity'):
    try:
        # add audio file
        s3.download_file(bucket, file_complete_path, saving_path)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("File Not Found....")
            return False
        else:
            print(e.response['Error'])
            return False


def get_sub_files(prefix, bucket='audio-similarity'):
    folders = s3_client.list_objects(Bucket=bucket, Prefix=prefix, Delimiter='/')

    folders_name = []
    for folder in folders['Contents']:
        folders_name.append(folder['Key'])
    return folders_name
