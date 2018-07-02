from google.cloud import bigquery as bq
import os

PROJECT_ID = 'linen-mapper-187215'

os.environ["http_proxy"] = "http://127.0.0.1:1087"
os.environ["https_proxy"] = "http://127.0.0.1:1087"

client = bq.Client(project=PROJECT_ID)

datasets = list(client.list_datasets())
project = client.project

if datasets:
    print('Datasets in project {}:'.format(project))
    for dataset in datasets:  # API request(s)
        print('\t{}'.format(dataset.dataset_id))
else:
    print('{} project does not contain any datasets.'.format(project))

# Make some query
query = (
    "select (mean-1.2)/3.4,(std-4.6)/1.9 from secu.fs_stats"
)

job_config = bq.QueryJobConfig()
job_config.dry_run = True
job_config.use_query_cache = False

query_job = client.query(
    query,
    # Location must match that of the dataset(s) referenced in the query.
    location='US',
    job_config=job_config)  # API request - starts the query

print("This query will process {} bytes.".format(
    query_job.total_bytes_processed))
