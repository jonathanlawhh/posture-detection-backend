gcloud builds submit --tag gcr.io/sitting-right/backend-api
gcloud run deploy --image gcr.io/sitting-right/backend-api --platform managed --max-instances 3
