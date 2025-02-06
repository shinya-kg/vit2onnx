from setuptools import setup, find_packages

setup(
    name="onnx_optimizer_tool",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "onnx",
        "onnxruntime",
        "onnxoptimizer",
        "tensorflow",
        "torch",
        "tf2onnx",
    ],
    entry_points = {
        "console_scripts": [
            "convert-to-onnx=scripts.convert_to_onnx:main",
            "optimize-onnx=scripts.optimize_onnx:main",
        ],
    },
    author="Shinya Koga",
    description="ONNX モデルの最適化と変換ツール",
    # long_description=open("./workspace/README.md").read(),
    # long_description_content_type="text/markdown",
    url="https://github.com/your-repo/onnx-optimizer-tool",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)