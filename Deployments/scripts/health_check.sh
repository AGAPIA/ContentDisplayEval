#!/bin/sh
grpcurl -plaintext localhost:50051 inference.Inference/Predict
