import grpc
from concurrent import futures
import inference_pb2
import inference_pb2_grpc
import yaml
import torch
from model_utils import load_model
from data_utils import get_default_transforms
from PIL import Image
import io

class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    def __init__(self, config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = load_model(cfg['inference']['model_path'], device=device)
        self.model.eval()
        self.threshold = cfg['inference'].get('threshold', 0.5)
        self.transform = get_default_transforms()

    def Predict(self, request_iterator, context):
        for frame in request_iterator:
            img = Image.open(io.BytesIO(frame.image_data)).convert('RGB')
            tensor = self.transform(img).unsqueeze(0).to(self.model.device)
            with torch.no_grad():
                output = self.model(tensor)
                score = torch.sigmoid(output)[0].item()
                flag = score >= self.threshold
            yield inference_pb2.Prediction(anomaly_score=score, anomaly_flag=flag)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    inference_pb2_grpc.add_InferenceServicer_to_server(
        InferenceServicer('config/server_config.yaml'), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
