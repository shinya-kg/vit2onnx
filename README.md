# VisionTransformer画像分類モデルをONNX形式へ変換

## 概要

このプロジェクトは、10種類の動物の画像を学習させたVisionTransformerの画像分類モデルを`.onnx`モデルへ変換し精度の変化を確かめることを目的としています。  
また、onnx-optimizer-tool内のメソッドを使用して`.onnx`ファイルに変換したり、最適化することができます。  さらに、作成したモデルを指定してimagesディレクトリ内の画像を推論させる事もできます。

## 機能

- 学習済みの`.pth`モデルを`.onnx`形式に変換
- ONNX モデルの最適化（冗長演算削除、精度低減など）
- imagesファイル内の画像を推論（犬、猫、象、馬、鶏、蝶、羊、蜘蛛、牛、リスの10種類、.jpegのみ）

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