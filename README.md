artifact_path="us-central1-docker.pkg.dev/technology-robot/aisha-for-product-recommendation/rag:latest"
docker build . -t $artifact_path
docker push $artifact_path

gcloud builds submit --tag=$artifact_path

gcloud run deploy aisha-for-product-recommendation --image=$artifact_path ...

---

gcloud compute networks subnets create egress \
--range=10.124.0.0/28 --network=default --region=us-central1

gcloud compute networks vpc-access connectors create egress \
  --region=us-central1 \
  --subnet-project=technology-robot \
  --subnet=egress

gcloud compute routers create egress \
  --network=default \
  --region=us-central1

gcloud compute addresses create egress --region=us-central1

gcloud compute routers nats create egress \
  --router=egress \
  --region=us-central1 \
  --nat-custom-subnet-ip-ranges=egress \
  --nat-external-ip-pool=egress


