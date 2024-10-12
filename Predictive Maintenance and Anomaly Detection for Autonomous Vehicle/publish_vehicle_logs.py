import json
import base64
from google.cloud import storage, pubsub_v1

def publish_vehicle_logs(request):
    # Initialize Cloud Storage and Pub/Sub clients
    storage_client = storage.Client()
    pubsub_client = pubsub_v1.PublisherClient()
    
    # Specify bucket name and folder path
    bucket_name = 'bucket__data'  
    folder_path = 'vehicleSensorData/diagnosticLogs/' 
    bucket = storage_client.bucket(bucket_name)

    # List all files in the folder
    blobs = bucket.list_blobs(prefix=folder_path)

    # Pub/Sub topic path
    topic_path = pubsub_client.topic_path('autonomusvehicles', 'vehicle-sensor-stream')  
    
    for blob in blobs:
        if blob.name.endswith('.csv'):  # Only process CSV files
            # Read the content of each CSV file
            csv_content = blob.download_as_text()
            rows = csv_content.splitlines()

            for row in rows:
                # Wrap each row in a JSON structure
                message_data = {
                    "log_entry": row  # Assuming each row contains a log entry from your CSV
                }

                # Convert to JSON and base64 encode the message
                json_message = json.dumps(message_data)
                encoded_message = base64.b64encode(json_message.encode('utf-8')).decode('utf-8')

                # Publish the encoded message to Pub/Sub
                future = pubsub_client.publish(topic_path, encoded_message.encode('utf-8'))
                print(f"Published message ID: {future.result()}")

    return 'Data published to Pub/Sub', 200
