import torch
import torchvision.models as models
import onnx
import onnxruntime as ort
import time
from _make_testdata import _make_testdata
from _select_model import load_vit, load_vit_onnx

import numpy as np
# データの準備
test_loader = _make_testdata()

# .pthファイルで推論するとき
def predict_pth(model_name):
    
    # インスタンス化したモデルをロード
    model = load_vit(model_name)
    
    # 検証モード
    model.eval()
    
    # 初期化
    correct = 0
    total = 0
    inference_time = []
    
    # 推論
    with torch.no_grad():
        for images, labels in test_loader:
            
            # 時間の計測
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            batch_time = end_time - start_time
            inference_time.append(batch_time)
            
            # 予測ラベル取得
            _, predicted = torch.max(outputs,1)
            
            # 正解数のカウント
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    # 精度計算
    accuracy = 100 * correct / total
    
    # 推論時間の平均
    ave_inference_time = sum(inference_time) / len(inference_time)
    
    return accuracy, ave_inference_time
            

# onnxファイルでの推論
def predict_onnx(model):
    
    # インスタンス化したモデルをロード
    model, file_path = load_vit_onnx(model)
    
    # モデルの検証
    onnx.checker.check_model(model)
    print('ONNXモデルの検証完了')
    
    # ONNXセッションを作成
    ort_session = ort.InferenceSession(file_path, providers=["CPUExecutionProvider"])
    
    # 出力名を確認
    output_name = [output.name for output in ort_session.get_outputs()]
    
    correct = 0
    total = 0
    inference_time = []
    
    # 推論
    for images, labels in test_loader:
        # 入力名の追加
        ort_inputs = {ort_session.get_inputs()[0].name: images.numpy()}
        
        # 推論時間の計測
        start_time = time.time()
        ort_outputs = ort_session.run(output_name, ort_inputs)
        end_time = time.time()
        batch_time = end_time - start_time
        inference_time.append(batch_time)
        predicted = np.argmax(ort_outputs[0], axis = 1)
        labels = labels.numpy()
        
        total += len(labels)
        correct += (predicted == labels).sum()

    # 精度計算
    accuracy = 100 * correct / total
    
    # 推論時間の平均
    ave_inference_time = sum(inference_time) / len(inference_time)
    
    return accuracy, ave_inference_time