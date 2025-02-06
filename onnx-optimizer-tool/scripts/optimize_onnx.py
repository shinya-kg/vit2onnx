import onnx, onnxoptimizer
import onnxruntime as ort
import time
import argparse
import torch
import numpy as np


# モデルの最適化
def optimize_onnx_model(input_model_path, output_model_path):
    # モデルのロード
    model = onnx.load(input_model_path)

    # 使用する最適化手法のリスト
    # passes = ["fuse_consecutive_transposes", "eliminate_deadend", "fuse_add_bias_into_conv", "eliminate_identity"]

    # 最適化
    optimized_model = onnxoptimizer.optimize(model)

    # 最適化後のモデルを保存
    onnx.save(optimized_model, output_model_path)
    print(f"✅ 最適化されたモデルを保存しました: {output_model_path}")


# Runtimeで推論時間を計測
def benchmark_onnx_model(model_path, input_data):
    # Runtimeセッションを作成
    session = ort.InferenceSession(model_path)

    # 入力名を取得
    input_name = session.get_inputs()[0].name

    # 推論時間のstart
    start_time = time.time()

    # 推論実行
    output = session.run(None, {input_name: input_data})

    # 推論のend
    end_time = time.time()

    # 実行時間
    inference_time = end_time - start_time
    print(f"✅ 推論時間: {inference_time:.6f}秒")

    return output, inference_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert models to ONNX format")
    parser.add_argument(
        "--framework", type=str, required=True, help="to detect framework"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input model file (.onnx)"
    )
    parser.add_argument(
        "--output", type=str, default="models/model.onnx", help="Output ONNX model path"
    )
    parser.add_argument(
        "--input-shape",
        type=str,
        help="Input shape for the model (e.g., '1,3,224,224')",
    )
    args = parser.parse_args()

    input_model_path = args.input
    optimized_model_path = args.output
    input_shape = (
        tuple(map(int, args.input_shape.split(","))) if args.input_shape else None
    )

    if args.framework == "pytorch":

        # モデルの最適化
        optimize_onnx_model(input_model_path, optimized_model_path)

        # ダミーデータを使用して推論ベンチマーク
        input_data = np.random.randn(*input_shape).astype(np.float32)

        # 最適化前の推論
        print("最適化前の推論時間:")
        _, time_before = benchmark_onnx_model(input_model_path, input_data)

        # 最適化後の推論
        print("\n最適化後の推論時間:")
        _, time_after = benchmark_onnx_model(optimized_model_path, input_data)

        # 推論時間の比較
        print(f"\n推論時間の差: {time_before - time_after:.6f}秒")

    elif args.framework == "tensorflow":

        # モデルの最適化
        optimize_onnx_model(input_model_path, optimized_model_path)

        # ダミーデータを使用して推論ベンチマーク
        input_data = np.random.rand(*input_shape).astype(np.float32)

        # 最適化前の推論
        print("最適化前の推論時間:")
        _, time_before = benchmark_onnx_model(input_model_path, input_data)

        # 最適化後の推論
        print("\n最適化後の推論時間:")
        _, time_after = benchmark_onnx_model(optimized_model_path, input_data)

        # 推論時間の比較
        print(f"\n推論時間の差: {time_before - time_after:.6f}秒")
