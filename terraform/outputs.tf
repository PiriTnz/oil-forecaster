output "cluster_name" {
  value = module.eks.cluster_name
}
output "cluster_endpoint" {
  value     = module.eks.cluster_endpoint
  sensitive = true
}
output "ecr_api_url" {
  value = aws_ecr_repository.api.repository_url
}
output "ecr_trainer_url" {
  value = aws_ecr_repository.trainer.repository_url
}
output "s3_data_bucket" {
  value = aws_s3_bucket.data.bucket
}
output "configure_kubectl" {
  value = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}
