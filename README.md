# VisionTransformer画像分類モデルをONNX形式へ変換

## 概要

このプロジェクトは、10種類の動物の画像を学習させたVisionTransformerの画像分類モデルを`.onnx`モデルへ変換し精度の変化を確かめることを目的としています。  
また、onnx-optimizer-tool内のメソッドを使用して`.onnx`ファイルに変換したり、最適化することができます。  さらに、作成したモデルを指定してimagesディレクトリ内の画像を推論させる事もできます。
> [!NOTE]
> models内の.pth,.onnxファイルはロードできないので、model_validation > learning_vit.ipynb内で学習させる必要があります。



## 機能

- 学習済みの`.pth`モデルを`.onnx`形式に変換
- ONNX モデルの最適化（冗長演算削除、量子化など）
- 最適化後の精度と推論速度を比較
- imagesファイル内の画像を推論（犬、猫、象、馬、鶏、蝶、羊、蜘蛛、牛、リスの10種類、.jpegのみ）

## 使用方法

> [!NOTE]
> Python 3.12.4で動作確認済み（他のライブラリのバージョンはrequirements.txtに記載）

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

### モデルの変換
onnx-optimizer-tool > models内の.pth(または.h5)ファイルが.onnxファイルに変換されます。  
onnx-optimizer-toolディレクトリ内で以下のコードを実行してください。
```
# Pytorchの場合
python convert_to_onnx.py --input models/<your/model/path>.pth --output models/<output/model/path>.onnx --input-shape <your/model/input_shape>

# TensorFlowの場合
python convert_to_onnx.py --input models/<your/model/path>.h5 --output models/<output/model/path>.onnx --input-shape <your/model/input_shape>
```

### モデルの最適化
`fuse_consecutive_transposes`,`eliminate_deadend`,`fuse_add_bias_into_conv`,`eliminate_identity`を適用しモデルを最適化します。  
最適化されたモデルとベースモデルの推論速度を比較します。  
onnx-optimizer-toolディレクトリ内で以下のコードを実行してください。

> [!WARNING]
> onnx-optimizer-tool > models内に`pt_model.onnx`または`tf_model.onnx`があることが前提になります


```
# Pytorchの場合
python scripts/optimize_onnx.py --framework pytorch --input models/<your/model/path>.onnx --output models/<output/model/path> --input-shape <your/model/input_shape>

# TensorFlowの場合
python scripts/optimize_onnx.py --framework tensorflow --input models/<your/model/path>.onnx --output models/<output/model/path>.onnx --input-shape <your/model/input_shape>

```

### 任意の画像を推論
images内に配置された任意の.jpeg画像を指定したモデルで推論します。  
tools内から以下のコードを実行してください。(.pth,.onnxファイルのみ実装)  
ルートディレクトリ配下のmodelsフォルダ内のモデルについて利用できます。  
デフォルトでは鶏の画像が8枚入っています。種類に制限はありますが、任意の種類、枚数を推論させることができます。
> [!WARNING]
> ファイルを.zipでダウンロードするとモデルの重みをロードできないので、学習し直すか、git cloneでダウンロードすると動きます。

```
# 使用例
python inference.py --model vit_onnx --modelname vit_cls.onnx
python inference.py --model vit --modelname vit_cls.pth
```

## 学習からモデル変換までの流れ
### モデルの学習
model_validation > learn_vit.ipynbを参照

**使用データ**：archive内の10種類の動物の画像データ（.jpeg）約24,000枚  
　　　　　　tools > _make_dataloader.pyを使用しデータローダーを作成  
**使用モデル**：事前学習済みViT-B/16  
　　　　　　出力層を10分類になるように調整し、出力層のみ学習させる（条件はnotebookに記載）

→学習させたモデルのテストデータ(test_loader)での精度は**97.66%**

### モデルを.onnx形式へ変換
model_validation > validation_onnx.ipynbを参照

**使用データ**：tools > _make_dataloader.pyを使用しデータローダーを作成しtest_loaderのみで検証  
**使用モデル**：上記で学習させた.pthファイルをconvert_to_onnx.pyで.onnxファイルに変換したもの

~~→.onnxモデルのテストデータ(test_loader)での精度は**8.31%**~~

モデルを変換する際に、正しくロードできていなかったため重みが初期化されていた。

正しく変換した後の精度は**97.92%**



> [!NOTE]
> ~~入力する画像データに関しては同様の処理を行い、データ型についてもfloat32だった。~~  
> ~~Rumtime上で推論させるときの最適化を外したが、精度は変わらなかった。~~
>
> ~~変換時の処理に原因がある可能性が高いが、[ドキュメント](https://pytorch.org/docs/stable/onnx.html)と大きく異なることもなさそう。~~ 
>


