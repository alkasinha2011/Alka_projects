import base64
import json
from google.cloud import bigquery
from transformers import pipeline
from datetime import datetime

# Initialize BigQuery client
bq_client = bigquery.Client()
table_id = "autonomusvehicles.vehicle_diagnostics.nlp_logs"  
def process_nlp_log(event, context):
    try:
        if 'data' in event:
            # Decode the base64-encoded message
            pubsub_message = base64.b64decode(event['data']).decode('utf-8')
            print(f"Received message: {pubsub_message}")

            # Parse the JSON message
            message_data = json.loads(pubsub_message)

            # Extract the log entry from the JSON
            log_entry = message_data.get("log_entry", "")
            print(f"Log entry to process: {log_entry}")
            
            # Load the summarization model from Hugging Face
            summarizer = pipeline("summarization")

            # Perform summarization on the log entry
            summary = summarizer(log_entry, max_length=100, min_length=10, do_sample=False)
            log_summary = summary[0]['summary_text']
            print(f"Log summary: {log_summary}")

            # Insert the result into BigQuery
            rows_to_insert = [
                {
                    "log_entry": log_entry,
                    "summary": log_summary,
                    "timestamp": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                }
            ]

            # Insert rows into BigQuery table
            errors = bq_client.insert_rows_json(table_id, rows_to_insert)
            if errors == []:
                print("New rows have been added.")
            else:
                print(f"Encountered errors while inserting rows: {errors}")

        return "NLP processing completed"

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise e
