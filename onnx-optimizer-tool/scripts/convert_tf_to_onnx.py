import tensorflow as tf
import tf2onnx
import onnx
import argparse
import numpy as np


def create_dummy_input(model, input_shape):
    """指定されたinput_shapeでダミー入力を作成"""
    if input_shape is None:
        input_shape = model.input_shape
    return np.random.rand(*input_shape).astype(np.float32)



def convert_tf_to_onnx(model_path, output_path, input_shape = None):
    """TensorFlow/KerasモデルをONNXに変換"""
    if model_path.endswith('.h5'):
        model = tf.keras.models.load_model(model_path)
    elif model_path.endswith('.pb'):
        model = tf.saved_model.load(model_path)
    else:
        raise ValueError(f"Unsupported TensorFlow model format: {model_path}")    
    
    dummy_input = create_dummy_input(model, input_shape)
    # モデルの出力名を設定
    model.output_names = ['output']

    # TensorFlow モデルを ONNX に変換
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=[tf.TensorSpec(dummy_input.shape, tf.float32, name="input")], opset=13)

    # ONNX モデルを保存
    onnx.save(onnx_model, output_path)
    print(f"✅ TensorFlow モデルを ONNX に変換しました: {output_path}")
    
    # モデルの検証
    onnx.checker.check_model(output_path)

    print("✅ ONNXモデルの検証が成功しました！")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TensorFlow model to ONNX")
    parser.add_argument("--input", type=str, required=True, help="Path to the input TensorFlow model file (.h5 or .pb)")
    parser.add_argument("--output", type=str, default="models/tf_model.onnx", help="Output ONNX model path")
    parser.add_argument('--input-shape', type=str, help="Input shape for the model (e.g., '1,224,224,3')")
    args = parser.parse_args()

    input_shape = tuple(map(int, args.input_shape.split(','))) if args.input_shape else None
    convert_tf_to_onnx(args.input, args.output, input_shape)