terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" { region = var.region }

variable "region" { default = "ap-south-1" }
variable "project" { default = "audiolens" }

# audio object storage
resource "aws_s3_bucket" "audio" {
  bucket = "${var.project}-audio"
}

resource "aws_s3_bucket_lifecycle_configuration" "audio" {
  bucket = aws_s3_bucket.audio.id
  rule {
    id     = "intelligent-tiering"
    status = "Enabled"
    transition {
      days          = 30
      storage_class = "INTELLIGENT_TIERING"
    }
  }
}

# postgres + pgvector
resource "aws_db_instance" "main" {
  identifier             = "${var.project}-pg"
  engine                 = "postgres"
  engine_version         = "16"
  instance_class         = "db.r6g.large"
  allocated_storage      = 100
  db_name                = "audiolens"
  username               = "audiolens"
  manage_master_user_password = true
  skip_final_snapshot    = true
  performance_insights_enabled = true
}

# kafka
resource "aws_msk_serverless_cluster" "main" {
  cluster_name = "${var.project}-kafka"
  vpc_config {
    subnet_ids         = var.subnet_ids
    security_group_ids = [aws_security_group.kafka.id]
  }
  client_authentication {
    sasl { iam { enabled = true } }
  }
}

variable "subnet_ids" { type = list(string) }

resource "aws_security_group" "kafka" {
  name = "${var.project}-kafka"
}

# EKS for api + workers (module keeps this short)
module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  version         = "~> 20.0"
  cluster_name    = "${var.project}-cluster"
  cluster_version = "1.29"
  subnet_ids      = var.subnet_ids
  vpc_id          = var.vpc_id

  eks_managed_node_groups = {
    general = {
      instance_types = ["m6i.large"]
      min_size       = 2
      max_size       = 6
      desired_size   = 2
    }
    extraction = {
      instance_types = ["c6i.2xlarge"]
      min_size       = 1
      max_size       = 10
      desired_size   = 2
      labels         = { workload = "extraction" }
    }
  }
}

variable "vpc_id" { type = string }

output "db_endpoint" { value = aws_db_instance.main.endpoint }
output "s3_bucket" { value = aws_s3_bucket.audio.bucket }
