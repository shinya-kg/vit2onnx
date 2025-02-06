# ONNX 最適化パイプライン & モデルフォーマット変換ツール

## 概要

このプロジェクトは、TensorFlow や PyTorch のモデルを ONNX 形式に変換し、ONNX モデルを最適化するツールです。ONNX モデルの最適化には `onnxoptimizer` を使用し、推論速度を `ONNX Runtime` で比較することができます。

## 機能

- TensorFlow と PyTorch モデルを ONNX に変換
- ONNX モデルの最適化（冗長演算削除、精度低減など）
- 最適化前後の推論速度を比較

## インストール

依存ライブラリをインストールするには、以下のコマンドを実行してください。

```bash
pip install -r requirements.txt
```

## 使用方法
### 仮想環境の構築
```
# 仮想環境を作成
python3 -m venv venv

# 仮想環境を有効化
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# ライブラリのインストール
pip install -r requirement.txt
```

> [!WARNING]
> もしエラーが出る場合は`pip install cmake`解決できる可能性が高いです

### モデルの変換
TensorFlow または PyTorch モデルを ONNX に変換するには、以下のコマンドを使用します
```
# Pytorchの場合
python convert_to_onnx.py --input models/<your/model/path>.pth --output models/<output/model/path>.onnx --input-shape <your/model/input_shape>

# TensorFlowの場合
python convert_to_onnx.py --input models/<your/model/path>.h5 --output models/<output/model/path>.onnx --input-shape <your/model/input_shape>
```

### モデルの最適化
`fuse_consecutive_transposes`,`eliminate_deadend`,`fuse_add_bias_into_conv`,`eliminate_identity`の中からモデルを最適化します  
最適化されたモデルとベースモデルの推論速度を比較します


> [!WARNING]
> models内に`pt_model.onnx`または`tf_model.onnx`があることが前提になります


```
# Pytorchの場合
python scripts/optimize_onnx.py --framework pytorch --input models/<your/model/path>.onnx --output models/<output/model/path> --input-shape <your/model/input_shape>

# TensorFlowの場合
python scripts/optimize_onnx.py --framework tensorflow --input models/<your/model/path>.onnx --output models/<output/model/path>.onnx --input-shape <your/model/input_shape>

```

**【参考】**  
✅ 最適化されたモデル: models/pytorch_model_optimized.onnx  

最適化前の推論時間:  
✅ 推論時間: 0.000766秒

最適化後の推論時間:  
✅ 推論時間: 0.000022秒

**推論時間の差: 0.000744秒**

