import onnx 
import argparse

def check_onnx_model(model_path):
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print(f"✅ ONNX モデルは正しく動作します: {model_path}")
    except Exception as e:
        print(f'エラー: {e}')
        
        
        
"""
使い方
python convert-to-onnx.py pytorch --output models/pytorch_model.onnx
python scripts/check_onnx.py --model models/tf_model.onnx

"""
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check ONNX model validity")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    
    args = parser.parse_args()
    
    check_onnx_model(args.model)
    
    model = onnx.load(args.model)
    print(onnx.helper.printable_graph(model.graph))