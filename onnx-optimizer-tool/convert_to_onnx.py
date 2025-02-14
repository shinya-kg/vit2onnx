import argparse
import os 
import sys
from scripts.convert_pytorch_to_onnx import convert_pytorch_to_onnx
from scripts.convert_tf_to_onnx import convert_tf_to_onnx


def detect_framework(model_path:str) -> Literal["pytorch","tensorflow"]:
    """ファイル拡張子からフレームワークを判別"""
    if model_path.endswith('.pth'):
        return "pytorch"
    elif model_path.endswith(".h5") or model_path.endswith('.pd'):
        return "tensorflow"
    else:
        raise ValueError(f'Unsupported model format: {model_path}')


def main()　-> None:
    """コマンドライン引数を受け取り、モデルを ONNX 形式に変換する。

    使用可能な引数:
    --input <モデルファイルのパス>: str
        変換したいモデルのパス (.pth, .h5, .pb)
    --output <出力ファイルのパス>: str
        変換後の ONNX モデルの保存パス (デフォルト: "models/model.onnx")
    --input-shape <入力テンソルの形状>: str
        モデルの入力形状をカンマ区切りで指定 (例: "1,3,224,224")

    変換処理:
    - モデルの形式を判定 (PyTorch または TensorFlow)
    - 指定された形式に応じて適切な変換関数を実行
    - 変換後の ONNX モデルを指定のパスに保存
    """
    parser = argparse.ArgumentParser(description="Convert models to ONNX format")
    parser.add_argument('--input', type=str, required=True, 
                        help="Path to the input model file (.pth, .h5, .pb)")
    parser.add_argument("--output", type=str, default="models/model.onnx",
                        help="Output ONNX model path")
    parser.add_argument('--input-shape', type=str, help="Input shape for the model (e.g., '1,3,224,224')")
    args = parser.parse_args()
    
    # モデルの種類を判定
    framework = detect_framework(args.input)

    # --input-shapeが指定されたときにタプルへ変換
    input_shape = tuple(map(int, args.input_shape.split(','))) if args.input_shape else None
    
    os.makedirs("models",exist_ok=True)
    
    if framework == "pytorch":
        convert_pytorch_to_onnx(args.input, args.output, input_shape)
    elif framework == "tensorflow":
        convert_tf_to_onnx(args.input, args.output, input_shape)
    else:
        print("❌ Unsupported model format")
        sys.exit(1)
        
        
if __name__ == "__main__":
    main()
